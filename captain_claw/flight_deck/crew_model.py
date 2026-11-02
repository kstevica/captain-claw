"""Crew — a worker on a model that isn't its CLI's own account: one of the person's agents',
a tier of the deck's, or one typed in for the task.

The key stays here. A worker can read its environment and print anything it likes, and the team's
keys don't reach anyone who isn't an admin (``server._sees_agent_secrets``). So the CLI is pointed
at ``/fd/crew/llm`` on this deck, with a key good only for this turn of this task. Flight Deck puts
the real one on the way out, asks for the chosen model whatever the CLI asked for, and counts the
tokens against the person's day, as an agent's are.

A task keeps what was chosen (``model_source``), and that's looked up again for every call: a tier
an admin changes or an agent given another model is what its next call runs on, and a person who
loses their approval or their agent loses this with it.

What counts is what an agent's call would count (``llm.uncached_tokens``): input that missed the
prompt cache, and output. A long session re-reads its whole history on every call, and those reads
would swallow the day.

A server failing before it has said anything is asked again here, and the worker never hears of it.
A reply cut off part way can't be picked up: it's ended, and the CLI asks again itself, as all four
do. A server that goes quiet is hung up on. The runner watches for a worker that doesn't get past it.

A CLI can only talk the way it was built to: Claude Code in Anthropic's API, Codex in OpenAI's
Responses API, OpenCode, Crush and Pi in those and OpenAI's chat API. Which one a model's server has is
known for the big providers and not for anything else, so a server of someone's own is given the
benefit of the doubt, and says so itself with a 404 if it hasn't got it.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from captain_claw.flight_deck import crew_harness, crew_logins, crew_store, fd_crypto, usage_quota
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.flight_deck.crew_harness import Route
from captain_claw.flight_deck.endpoints import endpoint_of
from captain_claw.llm import _is_codex_family_model, _normalize_provider_name
from captain_claw.logging import get_logger

log = get_logger(__name__)

ANTHROPIC = "anthropic"
OPENAI = "openai"
RESPONSES = "responses"
_EVERY_API = (ANTHROPIC, OPENAI, RESPONSES)
# What each CLI can talk, the one it does best first.
_CLI_APIS = {"claude": (ANTHROPIC,), "codex": (RESPONSES,), "opencode": (OPENAI, ANTHROPIC), "crush": (OPENAI, ANTHROPIC), "pi": (OPENAI, ANTHROPIC)}
_OWN_ENDPOINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "anthropic": ("https://api.anthropic.com/v1", (ANTHROPIC,)),
    "openai": ("https://api.openai.com/v1", (OPENAI, RESPONSES)),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", (OPENAI,)),
    "xai": ("https://api.x.ai/v1", (OPENAI, RESPONSES)),
    "openrouter": ("https://openrouter.ai/api/v1", _EVERY_API),
    "deepseek": ("https://api.deepseek.com/v1", (OPENAI,)),
}
_OUTPUT_MAX = 65536
_BODY_MAX = 64_000_000
_FORWARDED = ("content-type", "accept", "anthropic-version", "anthropic-beta", "openai-beta", "user-agent")
_PATHS = ("messages", "messages/count_tokens", "chat/completions", "responses", "responses/compact", "models")
_SCAN_MAX = 8_000_000
_CLOSING: set[asyncio.Future] = set()
_CALL_TRIES = 3
_TRY_AGAIN_AFTER = 1.0
_LONGEST_WAIT = 15.0  # a server asking for longer gets that from the CLI, not from a call held open here
_BUSY = (408, 429, 500, 502, 503, 504, 520, 522, 524, 529)
_CALLS: dict[str, dict[str, float]] = {}  # per task: its calls under way, what's come back, and how big its last call was
_CALLS_KEPT = 1000


@dataclass(frozen=True)
class Upstream:
    """Where a call goes, and on whose account. Never leaves this deck."""

    url: str  # ends where the API's paths start: https://api.anthropic.com/v1
    key: str
    model: str
    context: int = 0
    output: int = 0
    apis: tuple[str, ...] = _EVERY_API
    own_keys: bool = False  # spent on the person's own account, so not against their plan
    native_key: bool = False  # Anthropic's own endpoint takes its key in x-api-key, and only there


def _whole(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _api_url(provider: str, base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if provider in ("anthropic", "ollama") and not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _upstream(provider: str, model: str, base_url: str, key: str, *, context: int = 0, output: int = 0, own_keys: bool = False) -> Upstream:
    """A provider and an address, as a tier or an agent's config has them. A refusal says why."""
    name = _normalize_provider_name(provider)
    if name == "ollama":
        base = base_url or os.environ.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        url, apis = _api_url(name, base), _EVERY_API
    elif name in _OWN_ENDPOINTS:
        _, custom = endpoint_of(name, base_url)
        url, apis = _api_url(name, base_url), (ANTHROPIC,) if name == "anthropic" else _EVERY_API
        if not custom:
            url, apis = _OWN_ENDPOINTS[name]
    else:
        raise HTTPException(400, f"A worker can't run on {provider or 'that'}: it isn't a model server Captain can reach.")
    if name == "openai" and not key and _is_codex_family_model(model):
        raise HTTPException(400, f"{model} signs in through ChatGPT, so it has no key to run a worker on.")
    clean = model.strip().removeprefix(f"{name}/")
    if not crew_logins.MODEL_ID_RE.match(clean):
        raise HTTPException(400, "That model's name can't be sent to a server.")
    return Upstream(url=url, key=key, model=clean, context=context, output=output, apis=apis, own_keys=own_keys,
                    native_key=name == "anthropic" and not endpoint_of(name, base_url)[1])


# ── What can be chosen ──────────────────────────────────────────────────


def works_with(upstream: Upstream) -> list[str]:
    return [cli for cli, apis in _CLI_APIS.items() if set(apis) & set(upstream.apis)]


def route_for(harness: str, upstream: Upstream, deck_url: str) -> Route:
    api = next((a for a in _CLI_APIS.get(harness, ()) if a in upstream.apis), "")
    if not api:
        label = crew_harness.HARNESSES[harness].label
        raise HTTPException(400, f"{label} can't talk to that model's server. " + ("OpenCode, Crush or Pi can."
                            if harness in ("claude", "codex") else ""))
    return Route(url=f"{deck_url.rstrip('/')}/fd/crew/llm/v1", model=upstream.model, api=api, context=upstream.context,
                 output=upstream.output)


def deck_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('FD_PORT') or '25080'}"


def _key_names(provider: str) -> tuple[str, ...]:
    """The variables a provider's key is in: for one Flight Deck has no entry for, by the usual name."""
    from captain_claw.flight_deck import server

    return server._provider_key_env_names(provider) or (f"{provider.upper()}_API_KEY",)


def _tier_key(set_: dict, tier: dict, grant: Grant) -> str:
    """A tier's key: its own or, behind "@system", the team's. Blank, its set's environment has it,
    or for an admin the deck's own, as an agent of theirs would go on. Never the deck's for anyone
    else (``server._withhold_deck_keys``), and never to an address the tier names: that's how a key
    would leave the team."""
    from captain_claw.flight_deck.basna_routes import tier_api_key

    key = tier_api_key(tier) or ""
    provider = _normalize_provider_name(str(tier.get("provider") or ""))
    if key or endpoint_of(provider, tier.get("base_url"))[1]:
        return key
    names = _key_names(provider)
    for item in set_.get("env_vars") or []:
        if isinstance(item, dict) and item.get("key") in names and str(item.get("value") or "").strip():
            return str(item["value"]).strip()
    if grant.trusted:
        return next((os.environ[n] for n in names if os.environ.get(n)), "")
    return ""


async def _tier_upstream(db, uid: str, grant: Grant, name: str) -> Upstream:
    from captain_claw.flight_deck import basna_routes

    await basna_routes._refresh_system_provider_keys(db)
    set_ = await basna_routes.owner_tier_set(db, uid)
    tier = (set_.get("tiers") or {}).get(name)
    if not isinstance(tier, dict) or not str(tier.get("model") or "").strip():
        raise HTTPException(400, f"Your models have no {name} tier any more.")
    return _upstream(str(tier.get("provider") or ""), str(tier["model"]), str(tier.get("base_url") or ""),
                     _tier_key(set_, tier, grant), context=_whole(tier.get("input_ctx")),
                     output=_whole(tier.get("output_ctx")), own_keys=set_.get("source") == "byok")


def _agent_entries(uid: str) -> dict[str, dict]:
    from captain_claw.flight_deck import server
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    return {slug: entry for slug, entry in server._load_process_registry().items()
            if isinstance(entry, dict) and (not _fd_auth_enabled() or entry.get("owner") == uid)}


def _agent_context(slug: str) -> int:
    """The context it's set to run on, where it says: as the agent reads them, the last copy wins."""
    import yaml

    from captain_claw.flight_deck import server

    found = 0
    for path in server._agent_config_paths(server.DATA_DIR / slug):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, ValueError, yaml.YAMLError):
            continue
        section = data.get("context") if isinstance(data, dict) else None
        found = _whole(section.get("max_tokens")) if isinstance(section, dict) and section.get("max_tokens") else found
    return found


async def _agent_upstream(uid: str, grant: Grant, slug: str) -> Upstream:
    """Its model as the agent runs it: its config, then its key from where it looks. Only an agent
    that's the person's own: a shared one is on its owner's account."""
    from captain_claw.config import Config
    from captain_claw.flight_deck import server

    entry = _agent_entries(uid).get(slug)
    if entry is None:
        raise HTTPException(400, f"You have no agent called {slug} on this deck.")
    loaded = server._agent_config_model(slug)
    if not loaded or not str(loaded[0].get("model") or "").strip():
        raise HTTPException(400, f"{slug} has no model set.")
    model, provider_keys = loaded
    provider = _normalize_provider_name(str(model.get("provider") or ""))
    try:
        dotenv = Config._read_dotenv_file(server.DATA_DIR / slug / ".env")
    except OSError:
        dotenv = {}
    names = _key_names(provider)
    named_key = str(model.get("api_key") or "").strip()
    key = ((named_key if named_key != "@system" else "") or str(provider_keys.get(provider) or "").strip()
           or next((str(dotenv[n]).strip() for n in names if str(dotenv.get(n) or "").strip()), ""))
    base = str(model.get("base_url") or "").strip()
    if not base:
        named = [str(dotenv.get(n) or "").strip() for n in server._PROVIDER_BASE_ENV.get(provider, ())]
        base = next((u for u in named if u), "")
    # Asked of the key going out, wherever it was found: its config can say where it goes and name
    # no key of its own, with the team's sitting in its .env.
    the_teams = key in await server._org_keys()
    if the_teams and await server._team_key(provider, base) != key:
        raise HTTPException(400, f"{slug}'s key is the team's, and its model is somewhere the team doesn't send it.")
    if not key and not base and grant.trusted:
        key = next((os.environ[n] for n in names if os.environ.get(n)), "")
    return _upstream(provider, str(model["model"]), base, key, context=_agent_context(slug), output=_whole(model.get("max_tokens")),
                     own_keys=entry.get("key_source") == "byok" and not the_teams)


async def choices(db, user: dict, grant: Grant) -> dict[str, Any]:
    """What this person can choose from. Never a key, or an address that a key of the team's goes to."""
    from captain_claw.flight_deck import basna_routes

    uid = user["id"]
    await basna_routes._refresh_system_provider_keys(db)
    set_ = await basna_routes.owner_tier_set(db, uid)
    registry = basna_routes._load_registry().get("tiers") or {}

    def about(upstream: Upstream) -> dict[str, Any]:
        return {"model": upstream.model, "context": upstream.context, "works_with": works_with(upstream), "why_not": ""}

    tiers = []
    for name, tier in (set_.get("tiers") or {}).items():
        if not isinstance(tier, dict) or not str(tier.get("model") or "").strip() or name == "micro":
            continue
        label = str((registry.get(name) or {}).get("label") or name.capitalize())
        try:
            found = about(await _tier_upstream(db, uid, grant, name))
        except HTTPException as exc:
            found = {"model": str(tier["model"]), "context": 0, "works_with": [], "why_not": str(exc.detail)}
        tiers.append({"id": name, "label": label, **found})
    agents = []
    for slug, entry in _agent_entries(uid).items():
        label = str(entry.get("name") or slug)
        try:
            found = about(await _agent_upstream(uid, grant, slug))
        except HTTPException as exc:
            found = {"model": "", "context": 0, "works_with": [], "why_not": str(exc.detail)}
        agents.append({"id": slug, "label": label, **found})
    agents.sort(key=lambda a: a["label"].lower())
    return {"tiers": tiers, "agents": agents, "tier_set": str(set_.get("name") or "")}


async def choose(db, user: dict, grant: Grant, harness: str, raw: Any) -> tuple[dict[str, Any], Upstream | None]:
    """What a task keeps of a choice from the page, and where that goes now. Nothing chosen: its CLI's own."""
    if not isinstance(raw, dict) or not raw.get("kind"):
        return {}, None
    kind = str(raw["kind"])
    uid = user["id"]
    if kind == "tier":
        name = str(raw.get("tier") or "").strip()
        source = {"kind": kind, "tier": name}
    elif kind == "agent":
        slug = str(raw.get("agent") or "").strip()
        source = {"kind": kind, "agent": slug}
    elif kind == "typed":
        source = await _typed(db, uid, raw)
    else:
        raise HTTPException(400, "A model comes from an agent, a tier, or is typed in.")
    upstream = await resolve(db, user, grant, source)
    route_for(harness, upstream, "")  # one its CLI can't talk to is refused to the one asking
    await must_have_tokens(uid, upstream)
    return {**source, "label": await _label(db, uid, source)}, upstream


async def _typed(db, uid: str, raw: dict) -> dict[str, Any]:
    url = crew_logins.clean_url(str(raw.get("url") or ""))
    model = str(raw.get("model") or "").strip()
    if not crew_logins.MODEL_ID_RE.match(model):
        raise HTTPException(400, "Give the model's name, as the server calls it.")
    try:
        context = int(raw.get("context") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "Its context size is a number of tokens.") from None
    if context and not 1024 <= context <= 100_000_000:
        raise HTTPException(400, "Its context size is a number of tokens.")
    key = str(raw.get("key") or "").strip()
    if len(key) > crew_logins.MAX_SECRET:
        raise HTTPException(400, "That is too long to be a key.")
    if not key:
        # Blank, on the server someone saved their own model for, is the key they saved for it.
        own = await crew_logins.read_own_model(db, uid)
        key = str(own.get("key") or "") if own.get("url") == url else ""
    source: dict[str, Any] = {"kind": "typed", "url": url, "model": model, "context": context}
    if key:
        try:
            source["key"] = fd_crypto.seal({"key": key})
        except fd_crypto.CryptoUnavailable as exc:
            raise HTTPException(503, f"A key can't be kept safely on this deck: {exc}") from None
    return source


async def _label(db, uid: str, source: dict[str, Any]) -> str:
    if source["kind"] == "agent":
        return f"{(_agent_entries(uid).get(source['agent']) or {}).get('name') or source['agent']}'s model"
    if source["kind"] == "typed":
        return "Typed in for this task"
    from captain_claw.flight_deck import basna_routes

    label = (basna_routes._load_registry().get("tiers") or {}).get(source["tier"], {}).get("label") or source["tier"]
    return f"{label} tier"


async def resolve(db, user: dict, grant: Grant, source: dict[str, Any]) -> Upstream:
    uid = user["id"]
    kind = source.get("kind")
    if kind == "tier":
        return await _tier_upstream(db, uid, grant, str(source.get("tier") or ""))
    if kind == "agent":
        return await _agent_upstream(uid, grant, str(source.get("agent") or ""))
    if kind == "typed":
        key = ""
        if source.get("key"):
            try:
                key = str(fd_crypto.unseal(source["key"]).get("key") or "")
            except (fd_crypto.CryptoUnavailable, AttributeError, ValueError):
                raise HTTPException(503, "This task's key can't be read on this deck any more.") from None
        url = str(source.get("url") or "")
        anthropics = not endpoint_of("anthropic", url)[1]
        return Upstream(url=url, key=key, model=str(source.get("model") or ""), context=_whole(source.get("context")),
                        own_keys=True, apis=(ANTHROPIC,) if anthropics else _EVERY_API, native_key=anthropics)
    raise HTTPException(400, "That isn't a model a task can run on.")


def same(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """The same choice, as far as a retry goes: a sealed key is different each time it's sealed."""
    return {k: v for k, v in (a or {}).items() if k != "key"} == {k: v for k, v in (b or {}).items() if k != "key"}


def public(source: dict[str, Any] | None) -> dict[str, str]:
    return {"kind": str(source["kind"]), "label": str(source.get("label") or "")} if source and source.get("kind") else {}


def forget(source: dict[str, Any] | None) -> dict[str, Any]:
    """What a finished task keeps of it: no way to run on it again."""
    return {k: v for k, v in (source or {}).items() if k != "key"}


async def must_have_tokens(uid: str, upstream: Upstream) -> None:
    if upstream.own_keys:
        return
    spent = await usage_quota.automation_blocked(uid)
    if spent:
        raise HTTPException(429, usage_quota.limit_text(spent))


def new_token() -> str:
    return f"crew-{secrets.token_urlsafe(32)}"


# ── The calls ───────────────────────────────────────────────────────────


def _missed_input(usage: dict) -> int:
    """The input that missed the prompt cache. Anthropic's API keeps cache reads apart from its input,
    and cache writes are new input. The others fold what they read into theirs and say how much."""
    if "prompt_tokens" not in usage and ("cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage):
        return _whole(usage.get("input_tokens")) + _whole(usage.get("cache_creation_input_tokens"))
    details = next((d for d in (usage.get("prompt_tokens_details"), usage.get("input_tokens_details")) if isinstance(d, dict)), {})
    read = _whole(details.get("cached_tokens")) or _whole(usage.get("prompt_cache_hit_tokens")) or _whole(usage.get("cache_read_input_tokens"))
    return max(0, (_whole(usage.get("prompt_tokens")) or _whole(usage.get("input_tokens"))) - read)


class _Count:
    """The tokens a reply says it used: in one of a stream's events, or in the whole body."""

    def __init__(self, *, stream: bool, asked: int = 0) -> None:
        self.stream = stream
        self.asked = asked  # what's new in this call since its last, in bytes
        self.final = False
        self.tail = b""
        self.taken = 0
        self.body = bytearray()
        self.input = 0
        self.output = 0

    def _read(self, raw: bytes) -> None:
        try:
            obj = json.loads(raw)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        inner = [obj.get("usage"), *(o.get("usage") for o in (obj.get("message"), obj.get("response")) if isinstance(o, dict))]
        usage = next((u for u in inner if isinstance(u, dict)), None)
        if usage is None:
            return
        self.input = max(self.input, _missed_input(usage))
        self.output = max(self.output, _whole(usage.get("completion_tokens")) or _whole(usage.get("output_tokens")))
        # Anthropic's first event says what went in, with a stand-in for what comes out. Its last says the rest.
        self.final = self.final or obj.get("type") != "message_start"

    def feed(self, chunk: bytes) -> None:
        self.taken += len(chunk)
        if not self.stream:
            if self.taken <= _SCAN_MAX:
                self.body += chunk
            return
        lines = (self.tail + chunk).split(b"\n")
        self.tail = lines.pop()[-_SCAN_MAX:]
        for line in lines:
            if line.startswith(b"data:") and b"usage" in line:
                self._read(line[5:].strip())

    def finish(self) -> int:
        """What it used. A reply cut off before the event that says, or a server that never does: roughly
        what went through, or hanging up early would be a way to use a model without it counting. A CLI
        hanging up after that event is only being quick, and it's taken at its word."""
        if not self.stream and self.taken <= _SCAN_MAX:
            self._read(bytes(self.body))
        said = self.input + self.output
        return said if self.final else max(said, (self.asked + self.taken // 8) // 4)


def _silence_seconds() -> float:
    """How long a server can say nothing, first byte or next, before it's taken for gone. A slow local
    model reading a long prompt can take minutes over its first word."""
    try:
        return max(5.0, float(os.environ.get("FD_CREW_MODEL_SILENCE_SECONDS", 300)))
    except ValueError:
        return 300.0


def last_heard(task_id: str) -> float:
    """When a call of this task's last went out or a reply last moved (0: not since this deck started).
    Crew's watcher reads it: a CLI can say nothing for minutes while its model writes a long reply."""
    return float((_CALLS.get(task_id) or {}).get("heard") or 0.0)


def call_under_way(task_id: str) -> dict[str, float]:
    """What a task's model is doing now: ``open`` calls, ``since`` when, ``got`` bytes of reply, last ``heard`` when."""
    return dict(_CALLS.get(task_id) or {})


def _call_of(task_id: str) -> dict[str, float]:
    if task_id not in _CALLS and len(_CALLS) >= _CALLS_KEPT:
        _CALLS.clear()
    return _CALLS.setdefault(task_id, {"asked": 0.0, "open": 0.0, "got": 0.0})


def _began(task_id: str, asked: int) -> None:
    now = time.time()
    call = _call_of(task_id)
    call.update(asked=float(asked), since=now, heard=now, got=0.0, open=call["open"] + 1)


def _heard(task_id: str, size: int = 0) -> None:
    call = _call_of(task_id)
    call.update(heard=time.time(), got=call["got"] + size)


def _ended(task_id: str) -> None:
    call = _call_of(task_id)
    call["open"] = max(0.0, call["open"] - 1)


def _refusal(status: int, message: str) -> JSONResponse:
    """In the shape both kinds of client read an error from. Not a 429: they'd keep trying."""
    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": message}}, status_code=status)


def _token_of(request: Request) -> str:
    bearer = request.headers.get("authorization", "")
    return (bearer[7:] if bearer.lower().startswith("bearer ") else "") or request.headers.get("x-api-key", "")


def _running_with(token: str) -> dict[str, Any] | None:
    if not token.startswith("crew-"):
        return None
    for task in crew_store.get_store().with_status("running"):
        held = str(task.get("model_token") or "")
        if held and secrets.compare_digest(held.encode(), token.encode()):
            return task
    return None


def _from_this_computer(request: Request) -> bool:
    """A worker runs here, so a call that came from anywhere else isn't one. Asked through a reverse
    proxy, the address alone says nothing."""
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1") and not any(h.startswith("x-forwarded") or h == "forwarded" for h in request.headers)


async def _call(client: httpx.AsyncClient, url: str, headers: dict[str, str], sent: dict) -> httpx.Response:
    """The server's first answer to a call. One that fails before saying anything gets asked again, so
    the worker needn't hear about a blip. One that said nothing for this long has had its chance."""
    for left in range(_CALL_TRIES - 1, -1, -1):
        wait = _TRY_AGAIN_AFTER * (_CALL_TRIES - left)
        try:
            reply = await client.send(client.build_request("POST", url, headers=headers, json=sent), stream=True)
        except httpx.ReadTimeout:
            raise
        except httpx.TransportError:
            if not left:
                raise
        else:
            if reply.status_code not in _BUSY or not left:
                return reply
            asked_for = reply.headers.get("retry-after", "")
            if asked_for.isdigit() and int(asked_for) > _LONGEST_WAIT:
                return reply
            wait = float(asked_for) if asked_for.isdigit() else wait
            await reply.aclose()
        await asyncio.sleep(wait)
    raise httpx.TransportError("no tries left")  # the last one returned or raised


def _scrubbed(body: bytes, key: str) -> bytes:
    return body.replace(key.encode(), b"[its key]") if key else body


async def forward(request: Request, path: str) -> Response:
    """One call of a worker's, to its task's model on its task's account."""
    from captain_claw.flight_deck import crew_access, crew_runner
    from captain_claw.flight_deck.auth import get_db

    if not _from_this_computer(request):
        return _refusal(403, "This is only for Crew's workers, on this computer.")
    task = _running_with(_token_of(request))
    if task is None:
        return _refusal(401, "That key is for a turn of a Crew task, and no such turn is running.")
    rest = path
    while rest.startswith("v1/"):
        rest = rest[3:]
    if rest not in _PATHS:
        return _refusal(404, f"A worker's model has no {rest}.")
    db = get_db()
    user = await crew_runner._user_of(task["user_id"])
    grant = crew_access.grant_of(user)
    if not user or not grant.allowed or task["harness"] not in grant.harnesses:
        return _refusal(403, "Crew approval was withdrawn.")
    try:
        upstream = await resolve(db, user, grant, task.get("model_source") or {})
    except HTTPException as exc:
        return _refusal(403, str(exc.detail))
    try:
        await must_have_tokens(task["user_id"], upstream)
    except HTTPException as exc:
        return _refusal(402, str(exc.detail))  # and not a 403: Claude Code calls that failing to authenticate
    if rest == "models":
        return JSONResponse({"object": "list", "data": [{"id": upstream.model, "object": "model", "type": "model",
                                                         "display_name": upstream.model, "owned_by": "captain"}]})
    body = await request.body()
    if len(body) > _BODY_MAX:
        return _refusal(413, "That call is too big to send on.")
    try:
        sent = json.loads(body)
    except ValueError:
        return _refusal(400, "A model call is a JSON body.")
    if not isinstance(sent, dict):
        return _refusal(400, "A model call is a JSON object.")
    sent["model"] = upstream.model
    # OpenRouter's ways of naming other models to run on, on the same key.
    sent.pop("models", None)
    sent.pop("route", None)
    if rest == "chat/completions" and sent.get("stream"):
        # Or a stream doesn't say what it used.
        given = sent.get("stream_options")
        sent["stream_options"] = {**(given if isinstance(given, dict) else {}), "include_usage": True}
    headers = {name: request.headers[name] for name in _FORWARDED if name in request.headers}
    headers["accept-encoding"] = "identity"  # the tokens are read off it on the way past
    if upstream.key:
        if rest.startswith("messages") or upstream.native_key:
            headers["x-api-key"] = upstream.key
        if not upstream.native_key:
            headers["authorization"] = f"Bearer {upstream.key}"
    url = f"{upstream.url}/{rest}" + (f"?{request.url.query}" if request.url.query else "")
    grew = max(0, len(body) - int((_CALLS.get(task["id"]) or {}).get("asked") or 0))
    _began(task["id"], len(body))
    client = httpx.AsyncClient(timeout=httpx.Timeout(_silence_seconds(), connect=15.0), follow_redirects=False)
    try:
        reply = await _call(client, url, headers, sent)
    except httpx.ReadTimeout:
        _ended(task["id"])
        await client.aclose()
        return _refusal(504, f"The model's server said nothing for {_silence_seconds():.0f} seconds.")
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        _ended(task["id"])
        await client.aclose()
        return _refusal(502, f"Captain couldn't reach the model's server: {type(exc).__name__}.")
    media = reply.headers.get("content-type") or "application/json"
    if reply.status_code >= 300:
        _ended(task["id"])
        said = await reply.aread()
        await reply.aclose()
        await client.aclose()
        wait = {name: reply.headers[name] for name in ("retry-after", "x-should-retry") if name in reply.headers}
        return Response(_scrubbed(said, upstream.key), status_code=reply.status_code, media_type=media, headers=wait)
    count = _Count(stream="event-stream" in media, asked=grew)

    async def finish() -> None:
        _ended(task["id"])
        try:
            await reply.aclose()
            await client.aclose()
            await usage_quota.record_tokens(db, task["user_id"], count.finish(), byok=upstream.own_keys)
        except Exception as exc:  # noqa: BLE001 - the call is over; its count is all that's lost
            log.warning("A worker's tokens weren't counted", task=task["id"], error=str(exc))

    async def relay():
        try:
            async for chunk in reply.aiter_raw():
                count.feed(chunk)
                _heard(task["id"], len(chunk))
                yield chunk
        except httpx.HTTPError as exc:
            # All a client can be told now is that the reply stopped short. It asks again.
            log.warning("A model's reply was cut off", task=task["id"], error=type(exc).__name__)
        finally:
            # Not awaited here: a worker that hangs up cancels this, and anything it's waiting for.
            job = asyncio.ensure_future(finish())
            _CLOSING.add(job)
            job.add_done_callback(_CLOSING.discard)

    return StreamingResponse(relay(), status_code=reply.status_code, media_type=media)
