"""The name-recognition pack: a local model for the deterministic lane.

Code finds names by their context (a label, a greeting, a sign-off, a
title); a bare name in a sentence ("we met Marko") needs a model. This pack
runs a token-classification model on this machine — nothing leaves it — and
adds the people and companies it finds (score ≥ ``detectors.NER_MIN_SCORE``;
places and the rest are left) to every run, alongside the code.

Measured on ``anonymize/name_cases.json`` with ``scripts/measure_name_models.py``
(2026-11-12, see docs/anonymization-plan.md): BERTić finds about nine in ten
bare names (Croatian and English) and breaks almost nothing, so it is the
default. The multilingual XLM-R model labels days, holidays and whole phrases
as companies, so only its people are taken.

It is optional and an admin's choice (Admin console → Settings →
Anonymization): it needs ``transformers`` and ``torch`` installed (not in the
slim desktop app or on a Pi by default) and a model download. Only the
models listed in :data:`MODELS` can be chosen, and only their ``safetensors``
weights are loaded — no pickled code, no remote code.

Setting: ``system_settings["anonymize:ner"] = {enabled, model}``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import threading
from typing import Any

from captain_claw.anonymize import detectors
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING_KEY = "anonymize:ner"
MODELS: dict[str, str] = {
    "classla/bcms-bertic-ner": "Recommended · Croatian, Bosnian, Serbian, Montenegrin, and English names too · about 440 MB",
    "Davlan/xlm-roberta-base-wikiann-ner": "Other languages · people only, finds fewer · about 1.1 GB",
}
DEFAULT_MODEL = "classla/bcms-bertic-ner"
# Which of a model's labels are taken (its others are left).
MODEL_LABELS: dict[str, frozenset[str]] = {
    "classla/bcms-bertic-ner": frozenset({"PER", "ORG"}),
    "Davlan/xlm-roberta-base-wikiann-ner": frozenset({"PER"}),
}
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "sentencepiece*"]
CHUNK_CHARS = 1500

_state: dict[str, Any] = {"status": "off", "error": "", "model": ""}
_lock = threading.Lock()
_task: asyncio.Task | None = None


def runtime_available() -> bool:
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None


def downloaded(model: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(try_to_load_from_cache(model, f), str) for f in ("config.json", "model.safetensors"))
    except Exception:  # noqa: BLE001
        return False


async def load_config(db) -> dict[str, Any]:
    raw = await db.get_system_setting(SETTING_KEY)
    try:
        data = json.loads(raw) if raw else {}
    except ValueError:
        data = {}
    model = str(data.get("model") or DEFAULT_MODEL)
    return {"enabled": bool(data.get("enabled")), "model": model if model in MODELS else DEFAULT_MODEL}


async def save_config(db, enabled: bool, model: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"model is one of: {', '.join(MODELS)}")
    await db.set_system_setting(SETTING_KEY, json.dumps({"enabled": bool(enabled), "model": model}))
    return {"enabled": bool(enabled), "model": model}


def status(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    if not config["enabled"]:
        state = "off"
    elif not runtime_available():
        state = "not_installed"
    elif _state["model"] == model and _state["status"] in ("downloading", "loading", "ready", "error"):
        state = _state["status"]
    elif not downloaded(model):
        state = "not_downloaded"
    else:
        state = "not_loaded"
    return {
        **config, "status": state, "error": _state["error"] if state == "error" else "",
        "runtime": runtime_available(), "downloaded": downloaded(model),
        "models": [{"id": k, "label": v} for k, v in MODELS.items()],
    }


def _chunks(text: str) -> list[tuple[int, str]]:
    """(offset, piece) — pieces of at most CHUNK_CHARS, cut at line or sentence ends."""
    out, pos = [], 0
    while pos < len(text):
        end = min(len(text), pos + CHUNK_CHARS)
        if end < len(text):
            window = text[pos:end]
            cut = max(window.rfind("\n"), window.rfind(". "))
            if cut > CHUNK_CHARS // 3:
                end = pos + cut + 1
        out.append((pos, text[pos:end]))
        pos = end
    return out


def _make_backend(pipe: Any, model: str = DEFAULT_MODEL):
    labels = MODEL_LABELS.get(model, frozenset({"PER"}))

    def backend(text: str) -> list[tuple[int, int, str, float]]:
        found: list[tuple[int, int, str, float]] = []
        with _lock:  # one model, many engine threads
            for offset, piece in _chunks(text):
                if not piece.strip():
                    continue
                for ent in pipe(piece):
                    start, end = ent.get("start"), ent.get("end")
                    if start is None or end is None:
                        continue
                    label = re.sub(r"^[BI]-", "", str(ent.get("entity_group") or ent.get("entity") or ""))
                    if label.upper() not in labels:
                        continue
                    found.append((offset + int(start), offset + int(end), label, float(ent.get("score") or 0)))
        return found
    return backend


def _load_pipeline(model: str) -> Any:
    """From the downloaded files only (the local snapshot folder: no network)."""
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    path = snapshot_download(model, allow_patterns=ALLOW_PATTERNS, local_files_only=True)
    return pipeline("token-classification", model=path, aggregation_strategy="simple", device=-1,
                    model_kwargs={"use_safetensors": True}, trust_remote_code=False)


def _download(model: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model, allow_patterns=ALLOW_PATTERNS)


async def _bring_up(model: str, *, download: bool) -> None:
    _state.update(model=model, error="")
    try:
        if download and not downloaded(model):
            _state["status"] = "downloading"
            await asyncio.to_thread(_download, model)
        if not downloaded(model):
            raise RuntimeError("the model isn't downloaded")
        _state["status"] = "loading"
        pipe = await asyncio.to_thread(_load_pipeline, model)
        detectors.set_ner_backend(_make_backend(pipe, model))
        _state["status"] = "ready"
        log.info("Anonymization name-recognition pack ready", model=model)
    except Exception as exc:  # noqa: BLE001 — shown to the admin; code keeps working
        detectors.set_ner_backend(None)
        _state.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        log.warning("Anonymization name-recognition pack failed", model=model, error=type(exc).__name__)


def apply(config: dict[str, Any], *, download: bool = False) -> None:
    """Bring the pack up (or down) to match *config*, in the background."""
    global _task
    if not config["enabled"] or not runtime_available():
        detectors.set_ner_backend(None)
        _state.update(status="off", model=config["model"], error="")
        return
    if _state["model"] == config["model"] and _state["status"] in ("downloading", "loading", "ready"):
        return
    detectors.set_ner_backend(None)
    _task = asyncio.get_running_loop().create_task(_bring_up(config["model"], download=download))


async def boot(db) -> None:
    """At Flight Deck start: load the pack if an admin turned it on (never downloads)."""
    try:
        config = await load_config(db)
        if config["enabled"] and runtime_available() and downloaded(config["model"]):
            apply(config)
    except Exception as exc:  # noqa: BLE001
        log.debug("Name-recognition pack not started", error=type(exc).__name__)
