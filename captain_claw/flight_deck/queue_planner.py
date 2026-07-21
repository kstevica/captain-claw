"""Queue Task Planner — one description becomes a reviewed list of queue tasks.

A batch job like "enrich fund_portfolio" is twenty-five near-identical queue
messages that differ only in an id range. Writing them by hand is how batches
get skipped or overlapped.

The design decision that shapes this module is in
docs/queue-task-planner-plan.md: the model is asked for a TEMPLATE plus a list
of ranges, never for the messages themselves. That message is ~90% standing
rules and ~10% batch specifics, and a model asked to reproduce those rules
twenty-five times will paraphrase, compress and eventually drop the clause
that looked redundant — here, ``never do +1 on the id!``, which corrupts a
table. Expanding template × batches in Python instead means every message is
byte-identical except its range, at the cost of one LLM call whether there
are three batches or two hundred.

Nothing here has side effects. The result is a proposal the user reviews.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/queue", tags=["queue"])

# A fumbled range must not be able to enqueue the whole table.
MAX_TASKS_CEILING = 200
DEFAULT_MAX_TASKS = 50
MIN_BATCH = 1
MAX_BATCH = 50
# One small planning call — file extracts and fact payloads stay bounded.
_FACTS_TIMEOUT = 10.0


# ── Expansion ────────────────────────────────────────────────────────

_PLACEHOLDERS = {
    "from": ("{from}", "{from_id}", "{start}"),
    "to": ("{to}", "{to_id}", "{end}"),
    "index": ("{index}", "{n}"),
    "total": ("{total}",),
}


def expand_template(template: str, batches: list[dict[str, Any]]) -> list[str]:
    """Render one message per batch. Pure, deterministic, no model involved.

    Accepts the obvious placeholder spellings rather than insisting on one —
    the model writes the template, and rejecting `{start}` because we wanted
    `{from}` would cost a retry for nothing.
    """
    out: list[str] = []
    total = len(batches)
    for i, b in enumerate(batches, start=1):
        msg = template
        values = {
            "from": b.get("from", b.get("start", "")),
            "to": b.get("to", b.get("end", "")),
            "index": i,
            "total": total,
        }
        for field, spellings in _PLACEHOLDERS.items():
            for token in spellings:
                msg = msg.replace(token, str(values[field]))
        # Any extra keys the model invented (e.g. {table}) resolve from the
        # batch itself, so a richer batch shape still works.
        for k, v in b.items():
            if k in ("from", "to", "start", "end"):
                continue
            msg = msg.replace("{" + str(k) + "}", str(v))
        out.append(msg.strip())
    return out


def unresolved_placeholders(messages: list[str]) -> list[str]:
    """Placeholders left in the expansion — a template/batch mismatch.

    Worth surfacing rather than shipping: `_id from {from} to {to}` reaching
    the agent verbatim is a task that cannot succeed.
    """
    found: list[str] = []
    for m in messages:
        for tok in re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", m):
            if tok not in found:
                found.append(tok)
    return found


def clamp_batches(batches: list[dict[str, Any]], max_tasks: int) -> tuple[list[dict], str | None]:
    """Trim to *max_tasks*, saying so — a silent cap reads as 'that's all of it'."""
    if len(batches) <= max_tasks:
        return batches, None
    dropped = len(batches) - max_tasks
    return batches[:max_tasks], (
        f"The plan covered {len(batches)} batches; kept the first {max_tasks} and "
        f"dropped {dropped}. Raise max_tasks, or plan the rest afterwards."
    )


def _as_int(value: Any) -> int | None:
    """Models write numbers as strings often enough to matter."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def make_batches(start: int, size: int, count: int,
                 key_max: int | None = None) -> tuple[list[dict[str, int]], str | None]:
    """Consecutive ranges from *start* — the arithmetic half of a plan.

    Continuing a job needs no model at all: the template is already written and
    approved, and "the next 25 batches of 10 from 491" is arithmetic. Doing it
    here rather than in the client keeps one implementation, and lets it stop
    at the real end of the table instead of planning rows that don't exist.
    """
    size = max(MIN_BATCH, min(MAX_BATCH, int(size)))
    batches: list[dict[str, int]] = []
    note: str | None = None
    cur = int(start)
    for _ in range(max(0, int(count))):
        if key_max is not None and cur > key_max:
            note = (f"Stopped at {key_max}, the last row in the table — asked for "
                    f"{count} batches, made {len(batches)}.")
            break
        hi = cur + size - 1
        if key_max is not None and hi > key_max:
            hi = key_max
        batches.append({"from": cur, "to": hi})
        cur = hi + 1
    return batches, note


# ── Facts ────────────────────────────────────────────────────────────

async def gather_datastore_facts(
    host: str, port: int, auth: str, table: str = "", key_column: str = "_id",
) -> dict[str, Any]:
    """What the target agent's datastore actually contains.

    Ranges must be facts, not guesses: a plan for `_id 1..500` against a table
    whose ids stop at 318 produces 18 tasks that can only fail. Best-effort —
    a planner without facts is worse, not broken.
    """
    facts: dict[str, Any] = {"tables": [], "key": key_column}
    params = {"token": auth} if auth else {}
    base = f"http://{host}:{port}/api/datastore"
    try:
        async with httpx.AsyncClient(timeout=_FACTS_TIMEOUT) as client:
            resp = await client.get(f"{base}/tables", params=params)
            if resp.status_code != 200:
                facts["error"] = f"tables: HTTP {resp.status_code}"
                return facts
            tables = resp.json()
            facts["tables"] = [
                {
                    "name": t.get("name"),
                    "rows": t.get("row_count"),
                    "columns": [c.get("name") for c in (t.get("columns") or [])],
                }
                for t in (tables if isinstance(tables, list) else [])
            ]
            target = table or (facts["tables"][0]["name"] if facts["tables"] else "")
            if not target:
                return facts
            facts["table"] = target
            # min/max of the batching key, in one SELECT.
            col = "_id" if key_column == "_id" else re.sub(r"[^A-Za-z0-9_]", "", key_column)
            sql = (f'SELECT MIN("{col}") AS lo, MAX("{col}") AS hi, COUNT(*) AS n '
                   f'FROM "ds_{re.sub(r"[^A-Za-z0-9_]", "", target)}"')
            r2 = await client.post(f"{base}/sql", params=params, json={"sql": sql})
            if r2.status_code == 200:
                data = r2.json()
                rows = data.get("rows") or []
                if rows and len(rows[0]) >= 3:
                    facts["key_min"], facts["key_max"], facts["key_count"] = rows[0][:3]
    except httpx.HTTPError as e:
        facts["error"] = str(e)
    return facts


def facts_summary(facts: dict[str, Any]) -> str:
    """The facts, as the compact block that goes into the planning prompt."""
    if facts.get("error") and not facts.get("tables"):
        return "(the agent's datastore could not be read — do not invent ranges)"
    lines: list[str] = []
    for t in facts.get("tables", [])[:12]:
        cols = ", ".join((t.get("columns") or [])[:40])
        lines.append(f"- {t.get('name')} ({t.get('rows')} rows): {cols}")
    if facts.get("table"):
        lo, hi = facts.get("key_min"), facts.get("key_max")
        if lo is not None and hi is not None:
            lines.append(
                f"\nBatching key `{facts.get('key')}` in `{facts['table']}`: "
                f"min={lo}, max={hi}, {facts.get('key_count')} rows. "
                "Every batch MUST stay inside this range."
            )
    return "\n".join(lines) or "(no tables)"


# ── Attachments ──────────────────────────────────────────────────────
#
# A file the user attaches serves two different purposes, and conflating them
# is how this goes wrong:
#
#   1. The TASKS need its path, so the agent can open it at run time.
#   2. The PLANNER needs a peek INSIDE it — "how many rows?", "what are the id
#      columns called?" — or it is guessing at the very ranges we went to the
#      trouble of grounding in the datastore.
#
# So the path always goes into the prompt; the content goes in only as a small
# preview, capped hard, because this is one small call and a 170 KB
# spreadsheet would swamp it.

_PREVIEW_CHARS = 3000
_PREVIEW_ROWS = 25
_MAX_PREVIEW_BYTES = 8 * 1024 * 1024


async def _fetch_agent_file(host: str, port: int, auth: str, path: str) -> bytes | None:
    """Pull an uploaded file back from the agent that stored it."""
    import urllib.parse

    qs = f"path={urllib.parse.quote(path)}"
    if auth:
        qs += f"&token={urllib.parse.quote(auth)}"
    url = f"http://{host}:{port}/api/files/download?{qs}"
    try:
        async with httpx.AsyncClient(timeout=_FACTS_TIMEOUT) as client:
            resp = await client.get(url)
            if resp.status_code != 200 or len(resp.content) > _MAX_PREVIEW_BYTES:
                return None
            return resp.content
    except httpx.HTTPError:
        return None


def preview_file(name: str, blob: bytes) -> str:
    """A short, text-shaped look inside an attachment.

    Reuses the extractors the document tools already own rather than growing a
    second understanding of what an .xlsx is. Anything unreadable degrades to
    a size note — the planner still gets the path, which is the part the tasks
    actually need.
    """
    import tempfile

    suffix = pathlib.Path(name).suffix.lower()
    try:
        if suffix in (".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".sql"):
            text = blob.decode("utf-8", errors="replace")
            head = "\n".join(text.splitlines()[:_PREVIEW_ROWS])
            return head[:_PREVIEW_CHARS]
        if suffix in (".xlsx", ".docx", ".pdf", ".pptx"):
            from captain_claw.tools import document_extract as de

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(blob)
                tmp.flush()
                target = pathlib.Path(tmp.name)
                if suffix == ".xlsx":
                    md = de._extract_xlsx_markdown(target, _PREVIEW_ROWS)
                elif suffix == ".docx":
                    md = de._extract_docx_markdown(target)
                elif suffix == ".pptx":
                    md = de._extract_pptx_markdown(target, 5)
                else:
                    md, _ = de._extract_pdf_markdown(target, 3)
                return (md or "")[:_PREVIEW_CHARS]
    except Exception as e:  # a preview is a nicety; never fail the plan for it
        log.debug("attachment preview failed", file=name, error=str(e))
    return f"(binary or unreadable, {len(blob):,} bytes)"


async def build_file_notes(host: str, port: int, auth: str,
                           files: list[dict[str, Any]]) -> str:
    """The attachments block for the prompt: path first, then a peek inside."""
    notes: list[str] = []
    for f in files[:8]:
        path = str(f.get("path") or "").strip()
        name = str(f.get("filename") or path.rsplit("/", 1)[-1] or "file")
        if not path:
            continue
        blob = await _fetch_agent_file(host, port, auth, path)
        preview = preview_file(name, blob) if blob else "(could not be read back)"
        notes.append(
            f"- `{path}` ({name})\n"
            f"  The agent can open this path directly. First rows / first page:\n"
            f"  {preview.strip()[:_PREVIEW_CHARS]}"
        )
    return "\n".join(notes)


# ── Whose model does the planning? ───────────────────────────────────
#
# The agent you are planning FOR already has a working model and the key to
# call it. Falling back to a registry tier means the planner fails with
# "Missing Anthropic API Key" on a machine where nothing is broken — the keys
# simply live with the agent, in fd-data/<slug>/.env, not in FD's environment.
#
# Same principle the flow engine already uses when it spawns a specialist:
# use the keys of the agent you're talking to.

# provider name → the env var its key lives under. The provider name is what
# matters, not the base_url: an agent on "openai" pointed at api.deepseek.com
# still keeps its key in OPENAI_API_KEY.
_PROVIDER_KEY_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def _safe_alive(is_alive: Any, slug: str) -> bool:
    try:
        return bool(is_alive(slug))
    except Exception:
        return False


def resolve_agent_model(port: int) -> dict[str, str]:
    """The target agent's own provider, model, base_url and key.

    Empty dict when the agent can't be identified — the caller then falls back
    to the registry tier, which is right for a Flight Deck that does have its
    own key configured.
    """
    if not port:
        return {}
    try:
        from captain_claw.flight_deck.server import (
            DATA_DIR, _load_process_registry, _process_is_alive,
        )
    except Exception:  # pragma: no cover — server not importable in isolation
        return {}

    # A port is not unique in the registry: a dead agent's entry lingers, and a
    # new agent takes the freed port. Observed live — 24101 held both a stale
    # `gpt-test` and the running `deep-researcher-xik6`, and taking the first
    # match planned with the wrong agent's model and no key at all. Prefer the
    # entry whose process is actually alive, exactly as _resolve_agent_auth does.
    matches = [(s_, e) for s_, e in (_load_process_registry() or {}).items()
               if int(e.get("web_port") or 0) == int(port)]
    if not matches:
        return {}
    slug, entry = next(
        ((s_, e) for s_, e in matches if _safe_alive(_process_is_alive, s_)), matches[0])

    out: dict[str, str] = {
        "provider": str(entry.get("provider") or ""),
        "model": str(entry.get("model") or ""),
    }
    # config.yaml carries base_url (and sometimes the key itself).
    try:
        import yaml

        cfg = yaml.safe_load((DATA_DIR / slug / "config.yaml").read_text()) or {}
        model_cfg = cfg.get("model") or {}
        out["provider"] = str(model_cfg.get("provider") or out["provider"])
        out["model"] = str(model_cfg.get("model") or out["model"])
        out["base_url"] = str(model_cfg.get("base_url") or "")
        if model_cfg.get("api_key"):
            out["api_key"] = str(model_cfg["api_key"])
    except Exception as e:
        log.debug("agent config unreadable for planner", slug=slug, error=str(e))

    if not out.get("api_key"):
        env = _read_env_file(DATA_DIR / slug / ".env")
        for var in _PROVIDER_KEY_VARS.get(out["provider"].lower(), ()):
            if env.get(var):
                out["api_key"] = env[var]
                break
    return {k: v for k, v in out.items() if v}


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You turn one description of a repetitive job into a PLAN for a task queue.\n\n"
    "The queue runs each task in its own fresh session, one after another, with no "
    "memory of the others. So every task must be self-contained: all the standing "
    "rules, every time, plus the one thing that differs (usually an id range).\n\n"
    "You do NOT write the tasks, and you do NOT enumerate the batches. You write "
    "ONE template plus the OVERALL range and the batch size; the caller slices "
    "that range and expands the template over it. This is deliberate: it keeps "
    "every task identical except its range, no rule can be dropped or reworded "
    "between batch 3 and batch 19, and the slicing is arithmetic that cannot "
    "produce an overlap, a gap, or an off-by-one.\n\n"
    "Rules for the template:\n"
    "- Carry EVERY standing rule the user gave, in their words. Do not summarize, "
    "shorten, or 'improve' them. A rule you drop is a rule the agent breaks.\n"
    "- Use {from} and {to} placeholders for the batch bounds.\n"
    "- State the range as a hard boundary: work only these ids, do not continue to "
    "the next batch, do not touch ids outside the range.\n"
    "- Write it as an instruction to the agent, in the user's language of choice.\n\n"
    "If files are attached: their paths are given. A task that needs a file must "
    "name its path so the agent can open it — the agent cannot see your preview, "
    "only the file itself.\n\n"
    "Rules for the range:\n"
    "- Stay strictly inside the real min/max you are given. Never invent rows.\n"
    "- Give the FIRST and LAST value to cover, and the batch size. Nothing else.\n"
    "- If the user asks for more than exists, cover what exists and say so in "
    "warnings.\n"
    "- Only if the work is NOT a contiguous range (e.g. a specific list of ids "
    "from an attached file) may you enumerate `batches` instead.\n\n"
    "Return ONLY a JSON object, no prose."
)

_SHAPE = (
    "{\n"
    '  "template": "<the full task text, with {from} and {to}>",\n'
    '  "range": {"start": <first value>, "end": <last value>},\n'
    '  "batch_size": <rows per task>,\n'
    '  "rationale": "1-2 sentences on how you split it",\n'
    '  "warnings": ["…"]\n'
    "}\n"
    "Do NOT list the individual batches — `range` + `batch_size` is the whole plan, "
    "and enumerating 50 of them is how this reply gets truncated."
)


def build_user_prompt(intent: str, facts: dict[str, Any], batch_size: int,
                      key_column: str, file_notes: str = "") -> str:
    parts = [f"The job:\n{intent}\n"]
    parts.append(f"The target agent's datastore:\n{facts_summary(facts)}\n")
    if file_notes:
        parts.append(f"Attached files:\n{file_notes}\n")
    parts.append(
        f"Batch size: {batch_size} rows per task, batching on `{key_column}`.\n")
    parts.append(f"Return exactly this JSON shape:\n{_SHAPE}")
    return "\n".join(parts)


def parse_plan(raw: str) -> dict[str, Any]:
    """Pull the JSON object out of a model reply that may be fenced or chatty."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # An unterminated object means the reply was cut off mid-write, which
        # says something quite different from "the model ignored the format".
        cut_off = text.count("{") > text.count("}") or text.rstrip().endswith((",", '"', ":"))
        if cut_off:
            raise HTTPException(502, (
                "The planner's reply was cut off before the JSON finished "
                f"({e}). The template is probably very long — shorten the "
                "standing rules, or plan fewer tasks at a time."))
        raise HTTPException(502, f"The planner did not return JSON: {e}")
    if not isinstance(data, dict):
        raise HTTPException(502, "The planner returned JSON that is not an object")
    return data


# ── Route ────────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    intent: str
    host: str = "localhost"
    port: int = 0
    auth: str = ""
    table: str = ""
    key_column: str = "_id"
    batch_size: int = 10
    max_tasks: int = DEFAULT_MAX_TASKS
    file_notes: str = ""
    # Files already uploaded to the agent: [{"path": …, "filename": …}]
    files: list[dict] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


class PlanResponse(BaseModel):
    template: str
    batches: list[dict] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    rationale: str = ""
    warnings: list[str] = Field(default_factory=list)
    facts: dict = Field(default_factory=dict)


class ExpandRequest(BaseModel):
    template: str
    batches: list[dict] = Field(default_factory=list)
    max_tasks: int = DEFAULT_MAX_TASKS


@router.post("/expand")
async def expand_plan(body: ExpandRequest, user: dict = Depends(get_current_user)):
    """Re-render messages from an edited template. No model, no cost.

    The review UI edits the template and needs to show the result immediately.
    This keeps ONE implementation of the expansion — a second copy in
    TypeScript would drift from the one that actually produced the plan.
    """
    template = (body.template or "").strip()
    if not template:
        raise HTTPException(400, "template is required")
    max_tasks = max(1, min(MAX_TASKS_CEILING, int(body.max_tasks or DEFAULT_MAX_TASKS)))
    batches, cap_note = clamp_batches(
        [b for b in (body.batches or []) if isinstance(b, dict)], max_tasks)
    messages = expand_template(template, batches)
    warnings = [cap_note] if cap_note else []
    left = unresolved_placeholders(messages)
    if left:
        warnings.append(
            "These placeholders had nothing to fill them and would reach the agent "
            f"as literal text: {', '.join(left)}."
        )
    return {"messages": messages, "batches": batches, "warnings": warnings}


class ContinueRequest(BaseModel):
    template: str
    start: int
    batch_size: int = 10
    count: int = 10
    max_tasks: int = DEFAULT_MAX_TASKS
    # Optional, but worth passing: it stops the continuation at the real end
    # of the table instead of inventing rows past it.
    host: str = "localhost"
    port: int = 0
    auth: str = ""
    table: str = ""
    key_column: str = "_id"


@router.post("/continue")
async def continue_plan(body: ContinueRequest, user: dict = Depends(get_current_user)):
    """The next N batches from where a previous plan stopped. No model call.

    The template was written and approved once; carrying on is arithmetic. A
    continuation that costs an LLM call would also risk a *different* template
    for the second half of the same job.
    """
    template = (body.template or "").strip()
    if not template:
        raise HTTPException(400, "template is required")
    max_tasks = max(1, min(MAX_TASKS_CEILING, int(body.max_tasks or DEFAULT_MAX_TASKS)))

    facts: dict[str, Any] = {}
    key_max: int | None = None
    if body.port:
        facts = await gather_datastore_facts(
            body.host, body.port, body.auth, body.table, body.key_column)
        raw_max = facts.get("key_max")
        if isinstance(raw_max, (int, float)):
            key_max = int(raw_max)

    batches, note = make_batches(body.start, body.batch_size,
                                 min(int(body.count or 10), max_tasks), key_max)
    warnings = [note] if note else []
    if not batches:
        warnings.append(
            f"Nothing to do: {body.start} is past the last row"
            + (f" ({key_max})." if key_max is not None else ".")
        )
    messages = expand_template(template, batches)
    left = unresolved_placeholders(messages)
    if left:
        warnings.append(
            "These placeholders had nothing to fill them: " + ", ".join(left) + "."
        )
    return {"template": template, "batches": batches, "messages": messages,
            "rationale": "", "warnings": warnings, "facts": facts}


@router.post("/plan", response_model=PlanResponse)
async def plan_tasks(body: PlanRequest, user: dict = Depends(get_current_user)):
    """Turn one description into a reviewable list of queue tasks.

    One small LLM call. Creates no session, spawns nothing, enqueues nothing —
    the caller shows the plan and only then sends it to a lane.
    """
    intent = (body.intent or "").strip()
    if not intent:
        raise HTTPException(400, "intent is required")
    batch_size = max(MIN_BATCH, min(MAX_BATCH, int(body.batch_size or 10)))
    max_tasks = max(1, min(MAX_TASKS_CEILING, int(body.max_tasks or DEFAULT_MAX_TASKS)))

    facts: dict[str, Any] = {}
    if body.port:
        facts = await gather_datastore_facts(
            body.host, body.port, body.auth, body.table, body.key_column)

    # Precedence: what the caller asked for, then the AGENT'S own model and
    # key, then Flight Deck's registry tier.
    agent_model = resolve_agent_model(body.port)
    from captain_claw.flight_deck.basna_routes import _load_registry

    registry = _load_registry()
    tiers = registry.get("tiers", {})
    fast = tiers.get("reason", {}) or tiers.get("fast", {})
    provider = body.provider or agent_model.get("provider") or fast.get("provider", "anthropic")
    model = body.model or agent_model.get("model") or fast.get("model", "")
    base_url = body.base_url or agent_model.get("base_url") or fast.get("base_url", "")
    api_key = body.api_key or agent_model.get("api_key") or None

    file_notes = body.file_notes or ""
    if body.files and body.port:
        fetched = await build_file_notes(body.host, body.port, body.auth, body.files)
        file_notes = f"{file_notes}\n{fetched}".strip() if file_notes else fetched

    user_prompt = build_user_prompt(
        intent, facts, batch_size, body.key_column, file_notes)
    try:
        from captain_claw.llm import Message, create_provider

        prov = create_provider(provider=provider, model=model,
                               api_key=api_key, base_url=base_url or None,
                               temperature=0.2, max_tokens=8000)
        resp = await prov.complete(
            messages=[Message(role="system", content=_SYSTEM),
                      Message(role="user", content=user_prompt)],
            temperature=0.2, max_tokens=8000)
    except Exception as e:
        log.warning("queue planner call failed", error=str(e),
                    provider=provider, model=model, used_agent_key=bool(agent_model.get("api_key")))
        raise HTTPException(502, f"Planner call failed ({provider}:{model}): {e}")

    data = parse_plan(getattr(resp, "content", "") or "")
    template = str(data.get("template") or "").strip()
    if not template:
        raise HTTPException(502, "The planner returned no template")
    warnings = [str(w) for w in (data.get("warnings") or [])]
    # An explicit list wins (a non-contiguous job — specific ids from a file),
    # but the normal answer is a range we slice ourselves.
    batches = [b for b in (data.get("batches") or []) if isinstance(b, dict)]
    if not batches:
        rng = data.get("range") if isinstance(data.get("range"), dict) else {}
        start, end = _as_int(rng.get("start")), _as_int(rng.get("end"))
        if start is None or end is None:
            raise HTTPException(
                502, "The planner returned neither a range nor any batches")
        size = _as_int(data.get("batch_size")) or batch_size
        size = max(MIN_BATCH, min(MAX_BATCH, size))
        key_max = _as_int(facts.get("key_max"))
        if key_max is not None:
            end = min(end, key_max)
        count = max(0, (end - start + size) // size)
        batches, note = make_batches(start, size, count, key_max)
        if note:
            warnings.append(note)
    batches, cap_note = clamp_batches(batches, max_tasks)
    if cap_note:
        warnings.append(cap_note)

    messages = expand_template(template, batches)
    left = unresolved_placeholders(messages)
    if left:
        warnings.append(
            "These placeholders had nothing to fill them and would reach the agent "
            f"as literal text: {', '.join(left)}. Edit the template before sending."
        )

    log.info("queue plan built", tasks=len(messages), batch_size=batch_size,
             table=facts.get("table", ""))
    return PlanResponse(
        template=template,
        batches=batches,
        messages=messages,
        rationale=str(data.get("rationale") or ""),
        warnings=warnings,
        facts=facts,
    )
