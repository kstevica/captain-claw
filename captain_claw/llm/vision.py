"""The agent's own model sees images: capability, selection, encoding, cost.

A multimodal chat model gets a turn's pictures inline, as image content parts
of its own request, instead of a text description from a separate vision
model. Session history keeps only path markers — ``[Attached image: <path>]``
(a user attachment) and ``[Tool image: <path>]`` (a picture a tool returned) —
and the agent picks which markers of the current turn ride on a call
(:func:`plan_inline_images`); the provider converters turn those into image
parts, so no image bytes are ever stored. A model that can't see keeps the
old path: the server-side description and the separate ``image_vision`` model.

Whether a model sees is :func:`resolve_sees_images`: an explicit switch
(``model.vision`` / an allowed model's ``vision``) wins; a model that rejected
image input this process is never sent one again; otherwise a short list of
model families known to be multimodal decides. Local builds of open models
(Qwen 3.x quants, GGUFs, MLX conversions) ship with and without their vision
tower under near-identical names, so they need the explicit switch.
"""

from __future__ import annotations

import base64
import io
import json
import math
import mimetypes
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

# ``[Attached image: p]`` (the chat pipeline) and ``[Tool image: p]`` (a tool's
# picture, appended to its result by the tool loop).
IMAGE_MARKER_RE = re.compile(r"\[(Attached|Tool) image:\s*([^\]\n]+?)\s*\]")


def tool_image_marker(path: str) -> str:
    return f"[Tool image: {path}]"


# ── Capability ───────────────────────────────────────────────────────

# Runtimes that take text only, whatever model they run.
_TEXT_ONLY_PROVIDERS = frozenset({"claude-cli", "antigravity-cli", "litert", "browser"})

# Families that are multimodal in every build they ship as. Matched on the
# model name without its provider prefix.
_KNOWN_VISION_RE = re.compile(
    r"(?:^|/)(?:gpt-4o|gpt-4\.1|gpt-[5-9]|claude-|gemini-)"
    r"|llava|pixtral|internvl|minicpm-?v|llama-?4"
    r"|(?:^|[-_./:])vl(?:$|[-_./:])|qwen\d(?:\.\d)?-?vl|vision",
    re.IGNORECASE,
)
_TEXT_ONLY_NAME_RE = re.compile(r"text-?only|textonly", re.IGNORECASE)

# provider|base_url|model → when its server refused image input (the self-heal
# in llm/__init__.py). Not sent images again for a day: kept in
# ~/.captain-claw/vision_rejected.json so an agent restart doesn't pay the
# refused call again, and forgotten after _REJECTED_TTL so a server that gains
# vision (llama.cpp restarted with its mmproj, a model swapped under the same
# name) is tried again. An explicit `vision: on` doesn't clear it — that is how
# local models are configured; delete the file to retry sooner.
_VISION_REJECTED: dict[str, float] = {}
_REJECTED_TTL = 24 * 3600.0
_REJECTED_STORE: Path | None = None  # None → ~/.captain-claw/vision_rejected.json
_rejected_loaded = False


def _bare_model(provider: str, model: str) -> str:
    """The model name without the LiteLLM provider prefix, lowercased."""
    name = str(model or "").strip().lower()
    prefix = f"{str(provider or '').strip().lower()}/"
    if prefix != "/" and name.startswith(prefix):
        name = name[len(prefix):]
    return name


def _vision_key(provider: str, model: str, base_url: str) -> str:
    return "|".join((
        str(provider or "").strip().lower(),
        str(base_url or "").strip().rstrip("/").lower(),
        _bare_model(provider, model),
    ))


def _rejected_store() -> Path:
    if _REJECTED_STORE is not None:
        return _REJECTED_STORE
    from captain_claw.config import _HOME_DIR

    return _HOME_DIR / "vision_rejected.json"


def _read_rejected() -> dict[str, float]:
    """The stored refusals still within the TTL."""
    try:
        data = json.loads(_rejected_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    if not isinstance(data, dict):
        return {}
    return {str(key): float(when) for key, when in data.items()
            if isinstance(when, (int, float)) and now - float(when) < _REJECTED_TTL}


def _load_rejected() -> None:
    global _rejected_loaded
    if not _rejected_loaded:
        _rejected_loaded = True
        for key, when in _read_rejected().items():
            _VISION_REJECTED.setdefault(key, when)


def _save_rejected() -> None:
    """Write the refusals, merged with what other agents on this home stored."""
    path = _rejected_store()
    merged = {**_read_rejected(), **_VISION_REJECTED}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.debug("Could not store the image-input refusal", error=str(exc))


def remember_vision_rejected(provider: str, model: str, base_url: str = "") -> None:
    if vision_rejected(provider, model, base_url):
        return
    _VISION_REJECTED[_vision_key(provider, model, base_url)] = time.time()
    _save_rejected()
    log.warning("Model refused image input; using the text fallback for a day",
                provider=provider, model=model, base_url=base_url or None)


def vision_rejected(provider: str, model: str, base_url: str = "") -> bool:
    _load_rejected()
    key = _vision_key(provider, model, base_url)
    when = _VISION_REJECTED.get(key)
    if when is None:
        return False
    if time.time() - when >= _REJECTED_TTL:
        _VISION_REJECTED.pop(key, None)
        return False
    return True


def normalize_vision_switch(value: Any) -> str:
    """``on`` / ``off`` / ``auto`` from a config value (YAML reads on/off as bools)."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    text = str(value or "").strip().lower()
    if text in {"on", "true", "yes", "1", "enabled"}:
        return "on"
    if text in {"off", "false", "no", "0", "disabled"}:
        return "off"
    return "auto"


@lru_cache(maxsize=256)
def known_vision_model(provider: str, model: str) -> bool:
    name = _bare_model(provider, model)
    if not name or _TEXT_ONLY_NAME_RE.search(name):
        return False
    if _KNOWN_VISION_RE.search(name):
        return True
    try:
        import litellm

        return bool(litellm.supports_vision(model=str(model or "")))
    except Exception:
        return False


def resolve_sees_images(provider: str, model: str, base_url: str = "", switch: Any = "auto") -> bool:
    """Whether *model* on *provider* gets images inline. An explicit ``off``
    and a refusal seen this process always win; ``on`` turns it on for any
    runtime that can carry images; ``auto`` asks the known-model list — and on
    Ollama keeps the long-standing default of sending images to every model."""
    provider = str(provider or "").strip().lower()
    mode = normalize_vision_switch(switch)
    if mode == "off" or provider in _TEXT_ONLY_PROVIDERS:
        return False
    if vision_rejected(provider, model, base_url):
        return False
    if mode == "on":
        return True
    if _TEXT_ONLY_NAME_RE.search(_bare_model(provider, model)):
        return False
    if provider == "ollama":
        return True
    return known_vision_model(provider, model)


# ── Encoding ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EncodedImage:
    """One picture ready for the wire: base64 bytes (resized), MIME type, size."""

    path: str
    mime: str
    data: str
    width: int
    height: int

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.data}"


_WIRE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
_ENCODE_CACHE: OrderedDict[tuple[Any, ...], EncodedImage] = OrderedDict()
_ENCODE_CACHE_MAX = 32


def _image_size(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(io.BytesIO(data)) as img:
            return int(img.size[0]), int(img.size[1])
    except Exception:
        return None


def _to_jpeg(data: bytes, quality: int) -> bytes | None:
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(io.BytesIO(data)) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue() or None
    except Exception:
        return None


def resolve_image_path(path: str) -> Path | None:
    candidate = Path(str(path or "").strip()).expanduser()
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def encode_image(path: str, max_pixels: int = 1280, jpeg_quality: int = 85) -> EncodedImage | None:
    """Read, downscale (longest edge *max_pixels*) and base64-encode *path*.
    None when it isn't a readable image. Cached on path + mtime + size, so a
    tool loop doesn't re-encode the same picture on every call."""
    file_path = resolve_image_path(path)
    if file_path is None:
        return None
    try:
        stat = file_path.stat()
    except OSError:
        return None
    key = (str(file_path), stat.st_mtime_ns, stat.st_size, int(max_pixels), int(jpeg_quality))
    cached = _ENCODE_CACHE.get(key)
    if cached is not None:
        _ENCODE_CACHE.move_to_end(key)
        return cached
    try:
        raw = file_path.read_bytes()
    except OSError:
        return None
    from captain_claw.image_mime import sniff_image_mime
    from captain_claw.tools.image_ocr import _maybe_resize_image

    data, mime = _maybe_resize_image(raw, int(max_pixels), int(jpeg_quality))
    # The bytes decide, not the name: a JPEG saved as .png sent as image/png
    # was refused by the provider and ended the turn (image_mime).
    mime = mime or sniff_image_mime(data) or mimetypes.guess_type(file_path.name)[0] or "image/png"
    if mime not in _WIRE_MIMES:
        # BMP and the like: providers take JPEG/PNG/WebP/GIF only.
        converted = _to_jpeg(data, int(jpeg_quality))
        if converted is None:
            return None
        data, mime = converted, "image/jpeg"
    size = _image_size(data) or (int(max_pixels), int(max_pixels) * 3 // 4)
    image = EncodedImage(
        path=str(file_path), mime=mime, data=base64.b64encode(data).decode("ascii"),
        width=size[0], height=size[1],
    )
    _ENCODE_CACHE[key] = image
    while len(_ENCODE_CACHE) > _ENCODE_CACHE_MAX:
        _ENCODE_CACHE.popitem(last=False)
    return image


def as_encoded(item: Any, max_pixels: int = 1280, jpeg_quality: int = 85) -> EncodedImage | None:
    """An ``EncodedImage`` from a converter's input (already encoded, or a path)."""
    if isinstance(item, EncodedImage):
        return item
    if isinstance(item, (str, Path)):
        return encode_image(str(item), max_pixels, jpeg_quality)
    return None


# ── Cost ─────────────────────────────────────────────────────────────

# What one image costs when only its presence is known (no size): a 1568 px
# picture on the Qwen grid, rounded up. Errs high, like the text estimates.
IMAGE_TOKEN_ALLOWANCE = 2500
_OPENAI_GPT_RE = re.compile(r"(?:^|/)(?:gpt-|o\d)", re.IGNORECASE)


def estimate_image_tokens(width: int, height: int, provider: str = "", model: str = "") -> int:
    """Input tokens one *width* x *height* image costs on *provider*/*model*.

    Anthropic bills about w·h/750; Gemini 258 per 768-px tile; OpenAI's GPT
    models 85 + 170 per 512-px tile of the image scaled into 2048 / 768.
    Everything else — local and open models — is counted on the Qwen-VL grid
    (a token per 28-px patch after the 2x2 merge), the costliest common one.
    """
    w, h = max(1, int(width)), max(1, int(height))
    provider = str(provider or "").lower()
    name = _bare_model(provider, model)
    if provider == "anthropic" or "claude" in name:
        return max(1, math.ceil(w * h / 750))
    if provider == "gemini" or "gemini" in name:
        return 258 * math.ceil(w / 768) * math.ceil(h / 768)
    if provider in {"openai", "openrouter"} and _OPENAI_GPT_RE.search(name):
        scale = min(1.0, 2048 / max(w, h))
        sw, sh = w * scale, h * scale
        scale = min(1.0, 768 / min(sw, sh))
        sw, sh = sw * scale, sh * scale
        return 85 + 170 * math.ceil(sw / 512) * math.ceil(sh / 512)
    return math.ceil(w / 28) * math.ceil(h / 28) + 2


def typical_image_tokens(max_pixels: int, provider: str = "", model: str = "") -> int:
    """The cost of a 4:3 picture at the inline size cap."""
    edge = max(1, int(max_pixels))
    return estimate_image_tokens(edge, edge * 3 // 4, provider, model)


def tool_image_count(text: str) -> int:
    return sum(1 for m in IMAGE_MARKER_RE.finditer(str(text or "")) if m.group(1) == "Tool")


# ── Selection ────────────────────────────────────────────────────────


@dataclass
class InlinePlan:
    """The images one request carries, by candidate position."""

    images: dict[int, list[EncodedImage]] = field(default_factory=dict)
    tokens: int = 0
    # Markers of the current turn left as text: over the count or token cap,
    # or not a readable image. The user's own attachments among them are also
    # in ``skipped_attached`` (the model is told it didn't get those).
    skipped: list[str] = field(default_factory=list)
    skipped_attached: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.images.values())


def plan_inline_images(
    turn_messages: list[tuple[int, str]],
    *,
    max_images: int = 4,
    max_tool_images: int = 2,
    max_pixels: int = 1280,
    jpeg_quality: int = 85,
    token_cap: int = 0,
    provider: str = "",
    model: str = "",
) -> InlinePlan:
    """Pick the current turn's images for one call.

    *turn_messages* are ``(position, content)`` of the turn's messages, oldest
    first. The user's attachments come first (what was asked about), then the
    newest tool pictures — a browser loop keeps only its last screenshots —
    up to *max_images* in all and *token_cap* tokens (0 = no cap). A path
    named twice is sent once: where the user attached it, else at the tool
    result that named it last.
    """
    attached: list[tuple[int, str]] = []  # the user's, where first named
    tools: list[tuple[int, str]] = []     # tools', newest first
    seen: set[str] = set()
    for pos, content in turn_messages:
        for match in IMAGE_MARKER_RE.finditer(str(content or "")):
            path = match.group(2).strip()
            if match.group(1) == "Attached" and path not in seen:
                seen.add(path)
                attached.append((pos, path))
    for pos, content in reversed(turn_messages):
        for match in reversed(list(IMAGE_MARKER_RE.finditer(str(content or "")))):
            path = match.group(2).strip()
            if match.group(1) == "Tool" and path not in seen:
                seen.add(path)
                tools.append((pos, path))
    ordered = attached + tools[: max(0, int(max_tool_images))]
    plan = InlinePlan()
    plan.skipped = [path for _pos, path in tools[max(0, int(max_tool_images)):]]
    chosen: dict[tuple[int, str], EncodedImage] = {}
    user_paths = {path for _pos, path in attached}

    def _skip(path: str) -> None:
        plan.skipped.append(path)
        if path in user_paths:
            plan.skipped_attached.append(path)

    for pos, path in ordered:
        if len(chosen) >= max(0, int(max_images)):
            _skip(path)
            continue
        image = encode_image(path, max_pixels, jpeg_quality)
        if image is None:
            _skip(path)
            continue
        cost = estimate_image_tokens(image.width, image.height, provider, model)
        if token_cap and plan.tokens + cost > token_cap:
            _skip(path)
            continue
        chosen[(pos, path)] = image
        plan.tokens += cost
    # Each message carries its images in the order it names them.
    for pos, content in turn_messages:
        for match in IMAGE_MARKER_RE.finditer(str(content or "")):
            image = chosen.pop((pos, match.group(2).strip()), None)
            if image is not None:
                plan.images.setdefault(pos, []).append(image)
    return plan


# ── Wire helpers (converters and the self-heal) ──────────────────────

TOOL_IMAGES_LEAD = "(The picture(s) the tool call(s) above returned.)"


def tool_images_lead(images: list[EncodedImage]) -> str:
    """The text of the user message that carries tool pictures — it names their
    paths, so a refused picture can still be described (see the self-heal)."""
    return " ".join([TOOL_IMAGES_LEAD] + [tool_image_marker(img.path) for img in images])
NOT_RECEIVED_NOTE = (
    "(This model could not receive the attached image(s) — it does not accept image "
    "input. Call image_vision on the image path for a description; do not guess what "
    "it shows.)"
)


def openai_image_part(image: EncodedImage) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image.data_url}}


def responses_image_part(image: EncodedImage) -> dict[str, Any]:
    return {"type": "input_image", "image_url": image.data_url}


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in {"image_url", "input_image", "image"}


def has_image_parts(messages: Any) -> bool:
    """Whether a wire payload (OpenAI-style, Responses or Ollama) carries an image."""
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("images"):
            return True
        content = msg.get("content")
        if isinstance(content, list) and any(_is_image_part(p) for p in content):
            return True
    return False


def strip_image_parts(messages: Any) -> list[Any]:
    """A copy of a wire payload with every image removed and a note in its
    place, so the model is told it didn't get the picture."""
    result: list[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if msg.get("images"):
            msg = {key: value for key, value in msg.items() if key != "images"}
            msg["content"] = f"{content or ''}\n{NOT_RECEIVED_NOTE}".strip()
        elif isinstance(content, list) and any(_is_image_part(p) for p in content):
            kept = [dict(p) if isinstance(p, dict) else p for p in content if not _is_image_part(p)]
            text_type = "input_text" if any(
                isinstance(p, dict) and p.get("type") == "input_image" for p in content
            ) else "text"
            kept.append({"type": text_type, "text": NOT_RECEIVED_NOTE})
            msg = {**msg, "content": kept}
        result.append(msg)
    return result


def _message_has_images(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    return bool(msg.get("images")) or (
        isinstance(content, list) and any(_is_image_part(p) for p in content))


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def refused_image_paths(messages: Any) -> list[tuple[int, list[str]]]:
    """``(index, paths)`` of each wire message that carries images, the paths
    read from the markers in its text."""
    found: list[tuple[int, list[str]]] = []
    for index, msg in enumerate(messages or []):
        if _message_has_images(msg):
            paths = list(dict.fromkeys(m.group(2).strip() for m in IMAGE_MARKER_RE.finditer(
                _message_text(msg))))
            if paths:
                found.append((index, paths))
    return found


def replace_refused_note(msg: Any, text: str) -> Any:
    """*msg* (from :func:`strip_image_parts`) with its note replaced by *text*."""
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    if isinstance(content, str):
        return {**msg, "content": content.replace(NOT_RECEIVED_NOTE, text)}
    if isinstance(content, list):
        return {**msg, "content": [
            {**p, "text": text} if isinstance(p, dict) and p.get("text") == NOT_RECEIVED_NOTE else p
            for p in content]}
    return msg


def count_payload_images(messages: Any) -> int:
    total = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        total += len(msg.get("images") or [])
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(1 for p in content if _is_image_part(p))
    return total


def text_view_of_payload(messages: Any) -> list[Any]:
    """The payload with image bytes dropped — for size estimates only."""
    result: list[Any] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        content = msg.get("content")
        if msg.get("images"):
            msg = {key: value for key, value in msg.items() if key != "images"}
        if isinstance(content, list):
            msg = {**msg, "content": [p for p in content if not _is_image_part(p)]}
        result.append(msg)
    return result


_IMAGE_REJECTED_RE = re.compile(
    r"(?:image|vision|multimodal|multi-modal|image_url|mmproj)[^.\n]{0,80}?"
    r"(?:not supported|unsupported|does not support|doesn't support|do not support|"
    r"not enabled|is only supported|only supported by|cannot be used|not allowed)"
    r"|(?:not support|unsupported|does not support|doesn't support|no endpoints found that support)"
    r"[^.\n]{0,40}?(?:image|vision|multimodal)"
    r"|not a (?:multimodal|vision)"
    r"|missing data required for image input"
    r"|provide the mmproj"
    # Servers that take text content only and choke on the parts list.
    r"|content[^.\n]{0,40}?(?:must be|expected)(?: a| type)? (?:string|str)\b"
    r"|expected (?:a )?string[^.\n]{0,40}?(?:got|received) (?:an? )?(?:array|list)",
    re.IGNORECASE,
)


def is_image_input_rejected_error(msg: str) -> bool:
    """Whether *msg* is a server refusing image input. Only consulted when the
    request carried images, so the wording can stay broad."""
    return bool(_IMAGE_REJECTED_RE.search(str(msg or "")))
