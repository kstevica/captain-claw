"""App-authoring planner: turns an app idea into a structured spec + plan.

This is the bridge between the existing :mod:`captain_claw.plan_mode`
(generic step-by-step planner) and the new code-app runtime
(``captain_claw.flight_deck.app_runtime``).

Why a separate planner instead of stuffing app-authoring into the
generic plan-mode prompt? Two reasons:

1. The output shape matters. For an app the agent doesn't need a
   list of free-form prose tasks — it needs a *spec*: what data
   types exist, what HTTP endpoints does the backend expose, what
   screens does the frontend render. That structure is what the
   downstream scaffolding tool (``app_runner.scaffold``) consumes.

2. The downstream tasks are mechanical. Once the spec is decided,
   "write backend.py", "write frontend.html", "smoke-test", and
   "fix-on-error" are the same four tasks for every app — the
   generic planner doesn't need to invent them.

Flow::

    user request
        │
        ▼
    AppPlanGenerator.generate()  ── LLM call → AppSpec (JSON)
        │
        ▼
    spec_to_plan(spec)           ── deterministic → Plan (workflow tasks)
        │
        ▼
    SessionOrchestrator.load_workflow + execute
        │
        ▼
    agent runs app_runner.scaffold(slug, backend, frontend)
        │
        ▼
    smoke test → on 5xx, app_runner.get_logs + edit + restart

The system prompt is inlined here on purpose. Plan-mode's MD
templates are user-tunable; app-authoring is a tighter contract
between the planner and the runtime, so we keep the contract in
code where it can be refactored in lockstep with the runtime.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from captain_claw.llm import LLMProvider, Message
from captain_claw.logging import get_logger
from captain_claw.plan_mode import Plan, parse_json_response
from captain_claw.task_graph import OrchestratorTask


log = get_logger(__name__)


_APP_PLAN_TIMEOUT_SECONDS = 120.0
_APP_PLAN_MAX_TOKENS = 8000

# Slug rules mirror ``app_runtime._safe_slug``: alphanumerics, dashes,
# underscores. We additionally enforce lowercase and a leading letter
# so the slug reads naturally in URLs and file paths.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{1,47}$")


# ── data shape ────────────────────────────────────────────────────────


@dataclass
class FieldSpec:
    """One field on an entity (a column, basically)."""

    name: str
    type: str            # 'string' | 'number' | 'boolean' | 'datetime' | 'json'
    required: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FieldSpec":
        return cls(
            name=str(d.get("name") or "").strip(),
            type=str(d.get("type") or "string").strip().lower(),
            required=bool(d.get("required", False)),
            description=str(d.get("description") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class EntitySpec:
    """An entity type stored via the shared FD datastore."""

    name: str                       # python identifier, e.g. "note"
    description: str = ""
    fields: list[FieldSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EntitySpec":
        raw_fields = d.get("fields") or []
        return cls(
            name=str(d.get("name") or "").strip(),
            description=str(d.get("description") or "").strip(),
            fields=[
                FieldSpec.from_dict(f) for f in raw_fields if isinstance(f, dict)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
        }


@dataclass
class EndpointSpec:
    """One HTTP endpoint exposed by the backend."""

    method: str               # 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
    path: str                 # leading slash, e.g. '/items' or '/items/{id}'
    description: str = ""
    entity: str = ""          # references EntitySpec.name when applicable
    action: str = ""          # 'list' | 'get' | 'create' | 'update' | 'delete' | 'custom'

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EndpointSpec":
        return cls(
            method=str(d.get("method") or "GET").strip().upper(),
            path=str(d.get("path") or "/").strip(),
            description=str(d.get("description") or "").strip(),
            entity=str(d.get("entity") or "").strip(),
            action=str(d.get("action") or "").strip().lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "entity": self.entity,
            "action": self.action,
        }


@dataclass
class ScreenSpec:
    """One screen (visual surface) rendered by the frontend HTML.

    The current renderer is a single-page bundle, so multiple screens
    are realized as conditional sections within ``frontend.html``. The
    spec still lists them separately so the agent can keep the layout
    organized.
    """

    id: str
    title: str = ""
    description: str = ""
    reads: list[str] = field(default_factory=list)   # endpoint paths it reads
    writes: list[str] = field(default_factory=list)  # endpoint paths it calls

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScreenSpec":
        return cls(
            id=str(d.get("id") or "").strip(),
            title=str(d.get("title") or "").strip(),
            description=str(d.get("description") or "").strip(),
            reads=[str(x) for x in (d.get("reads") or []) if str(x).strip()],
            writes=[str(x) for x in (d.get("writes") or []) if str(x).strip()],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "reads": list(self.reads),
            "writes": list(self.writes),
        }


@dataclass
class AppSpec:
    """Structured app design returned by :class:`AppPlanGenerator`.

    This is the contract between the planner and the code-app runtime:
    everything downstream — the scaffolder, the smoke test, the
    self-repair prompt — reads from this shape.
    """

    name: str                          # human-readable, e.g. "Notes demo"
    slug: str                          # identifier, e.g. "notes_demo"
    summary: str                       # 1–2 sentence pitch
    entities: list[EntitySpec] = field(default_factory=list)
    endpoints: list[EndpointSpec] = field(default_factory=list)
    screens: list[ScreenSpec] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # extra pip packages

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppSpec":
        return cls(
            name=str(d.get("name") or "").strip(),
            slug=str(d.get("slug") or "").strip().lower(),
            summary=str(d.get("summary") or "").strip(),
            entities=[
                EntitySpec.from_dict(e) for e in (d.get("entities") or []) if isinstance(e, dict)
            ],
            endpoints=[
                EndpointSpec.from_dict(e) for e in (d.get("endpoints") or []) if isinstance(e, dict)
            ],
            screens=[
                ScreenSpec.from_dict(s) for s in (d.get("screens") or []) if isinstance(s, dict)
            ],
            dependencies=[
                str(x).strip() for x in (d.get("dependencies") or []) if str(x).strip()
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "summary": self.summary,
            "entities": [e.to_dict() for e in self.entities],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "screens": [s.to_dict() for s in self.screens],
            "dependencies": list(self.dependencies),
        }

    def render_markdown(self) -> str:
        """Human-readable rendering for the chat UI / plan preview."""
        lines: list[str] = []
        lines.append(f"**App:** {self.name}  (`{self.slug}`)")
        if self.summary:
            lines.append(f"_{self.summary}_")
        lines.append("")
        if self.entities:
            lines.append("**Data model**")
            for e in self.entities:
                fields_str = ", ".join(
                    f"{f.name}: {f.type}{'*' if f.required else ''}" for f in e.fields
                )
                lines.append(f"- `{e.name}` — {e.description or 'no description'}")
                if fields_str:
                    lines.append(f"  - fields: {fields_str}")
            lines.append("")
        if self.endpoints:
            lines.append("**Endpoints**")
            for ep in self.endpoints:
                lines.append(f"- `{ep.method} {ep.path}` — {ep.description or ep.action}")
            lines.append("")
        if self.screens:
            lines.append("**Screens**")
            for s in self.screens:
                lines.append(f"- `{s.id}`: {s.title}")
                if s.description:
                    lines.append(f"  - {s.description}")
            lines.append("")
        if self.dependencies:
            lines.append("**Dependencies:** " + ", ".join(self.dependencies))
        return "\n".join(lines).rstrip()


# ── validation ────────────────────────────────────────────────────────


class AppSpecError(ValueError):
    """The planner emitted a spec we can't safely scaffold from."""


def validate_app_spec(spec: AppSpec) -> list[str]:
    """Return a list of human-readable validation errors (empty on success).

    Validation is intentionally light — the planner's LLM gets the
    schema in the prompt, and the scaffolder will catch syntax issues
    on its own. This pass only enforces shape we *can't* recover from.
    """
    errors: list[str] = []
    if not spec.name:
        errors.append("`name` is required")
    if not spec.slug:
        errors.append("`slug` is required")
    elif not _SLUG_RE.match(spec.slug):
        errors.append(
            f"`slug` {spec.slug!r} must match {_SLUG_RE.pattern} "
            "(lowercase letters/digits/_-, leading letter, ≤48 chars)"
        )
    seen_ep: set[tuple[str, str]] = set()
    for ep in spec.endpoints:
        if not ep.path.startswith("/"):
            errors.append(f"endpoint path must start with '/': {ep.method} {ep.path}")
        key = (ep.method.upper(), ep.path)
        if key in seen_ep:
            errors.append(f"duplicate endpoint: {ep.method} {ep.path}")
        seen_ep.add(key)
    seen_entity: set[str] = set()
    for e in spec.entities:
        if not e.name:
            errors.append("entity missing `name`")
            continue
        if e.name in seen_entity:
            errors.append(f"duplicate entity: {e.name}")
        seen_entity.add(e.name)
    return errors


# ── planner ───────────────────────────────────────────────────────────


_APP_PLAN_SYSTEM_PROMPT = """\
You are an app architect for Captain Claw's code-app runtime. Given a
user description, produce a JSON spec for a small self-contained
single-page app. The spec will be fed to a code generator that writes
``backend.py`` (Python, async ``handle(method, path, headers, body)``)
and ``frontend.html`` (vanilla HTML + JS, no build step).

Output rules:
- Respond with a single JSON object, no prose, no code fences.
- Field names are exactly as below — extras are ignored.
- Keep the design *small*: one screen unless the user really needs more,
  at most ~5 entities, at most ~12 endpoints.
- Persistence uses the shared FD datastore (no SQL). Each entity maps
  to one ``datastore("entity_name")`` collection.
- Endpoints follow REST conventions (``GET /items``, ``POST /items``,
  ``DELETE /items/{id}``) unless the use case demands something
  different.
- Slugs are lowercase letters/digits/underscores/dashes, must start
  with a letter, ≤48 chars. Example: ``notes_demo``.
- Only list ``dependencies`` if the backend genuinely needs them; the
  default stdlib + the FD datastore client are always available.

Schema::

  {
    "name":      "Display name",
    "slug":      "url_safe_slug",
    "summary":   "One- or two-sentence pitch.",
    "entities":  [
      {
        "name": "note",
        "description": "A single sticky note.",
        "fields": [
          {"name":"title","type":"string","required":true,"description":""},
          {"name":"body","type":"string","required":false,"description":""}
        ]
      }
    ],
    "endpoints": [
      {"method":"GET","path":"/items","description":"...","entity":"note","action":"list"},
      {"method":"POST","path":"/items","description":"...","entity":"note","action":"create"},
      {"method":"DELETE","path":"/items/{id}","description":"...","entity":"note","action":"delete"}
    ],
    "screens":   [
      {"id":"main","title":"Notes","description":"List + add form",
       "reads":["/items"],"writes":["/items","/items/{id}"]}
    ],
    "dependencies": []
  }

Allowed field types: ``string``, ``number``, ``boolean``, ``datetime``, ``json``.
Allowed endpoint actions: ``list``, ``get``, ``create``, ``update``, ``delete``, ``custom``.
"""


def _user_prompt(user_request: str) -> str:
    return f"User request:\n\n{user_request.strip()}\n\nReturn only the JSON spec."


class AppPlanGenerator:
    """LLM-driven generator that turns a user request into an :class:`AppSpec`.

    Mirrors the shape of :class:`captain_claw.plan_mode.PlanGenerator` so
    callers can mix-and-match: use ``PlanGenerator`` for generic work,
    ``AppPlanGenerator`` for app authoring.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float = _APP_PLAN_TIMEOUT_SECONDS,
        max_tokens: int = _APP_PLAN_MAX_TOKENS,
    ):
        self._provider = provider
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    async def generate(self, user_request: str) -> AppSpec | None:
        """Generate an :class:`AppSpec` from ``user_request``.

        Returns ``None`` on LLM failure or parse failure. Callers should
        treat ``None`` as "ask the user to clarify" rather than retrying
        blindly — the planner is deterministic enough that a second call
        won't usually fix a bad first response.
        """
        if not user_request or not user_request.strip():
            log.warning("AppPlanGenerator.generate called with empty user_request")
            return None

        messages = [
            Message(role="system", content=_APP_PLAN_SYSTEM_PROMPT),
            Message(role="user", content=_user_prompt(user_request)),
        ]
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    messages=messages, tools=None, max_tokens=self._max_tokens,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            log.error("App-plan generation timed out", timeout=self._timeout)
            return None
        except Exception as e:
            log.error(
                "App-plan LLM call failed",
                error=str(e), error_type=type(e).__name__,
            )
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        if not raw:
            log.error("App planner returned empty content")
            return None

        parsed = parse_json_response(raw)
        if parsed is None:
            log.error("Failed to parse app-plan JSON", raw_preview=raw[:500])
            return None

        try:
            spec = AppSpec.from_dict(parsed)
        except (TypeError, ValueError) as e:
            log.error("App-plan JSON didn't match schema", error=str(e))
            return None

        errors = validate_app_spec(spec)
        if errors:
            log.error("App-plan spec failed validation", errors=errors)
            return None
        return spec


# ── spec → executable workflow ────────────────────────────────────────


def spec_to_plan(spec: AppSpec, *, user_input: str = "") -> Plan:
    """Translate an :class:`AppSpec` into a :class:`Plan` of orchestrator tasks.

    The four tasks are the same for every app:

    1. ``scaffold`` — agent writes ``backend.py`` and ``frontend.html``
       to disk via the ``scaffold_app`` tool (calls ``POST /fd/code-apps/{slug}/scaffold``).
    2. ``smoke`` — agent hits a couple of routes via the proxy and
       verifies the responses match the spec's endpoint shape.
    3. ``inspect_logs`` — verifier step that reads ``/fd/code-apps/{slug}/logs``
       to confirm no tracebacks landed in stderr.
    4. ``revise`` — only fires if (2) or (3) failed; reads the logs
       and patches ``backend.py``, then loops back to ``smoke``.

    The agent inside each task interprets the spec to generate concrete
    code — we don't try to template-generate Python here because that
    would limit what apps can look like.
    """
    spec_json = json.dumps(spec.to_dict(), indent=2)
    slug = spec.slug

    tasks: list[OrchestratorTask] = [
        OrchestratorTask(
            id="scaffold",
            title=f"Scaffold code-app {slug!r}",
            description=(
                f"Generate `backend.py` and `frontend.html` for the app described "
                f"below, then call the `scaffold_app` tool to persist them as a "
                f"code-app under slug `{slug}`.\n\n"
                "Backend contract:\n"
                "- Define `async def handle(method, path, headers, body) -> dict` "
                "returning `{status, headers, body}`.\n"
                "- Import `from captain_claw.flight_deck.app_datastore_client "
                "import datastore` and use `datastore('<entity>')` per entity.\n"
                "- All endpoints under `/api/` arrive without the prefix — "
                "match the `path` fields below verbatim.\n\n"
                "Frontend contract:\n"
                "- Self-contained HTML+JS. No external bundlers.\n"
                "- Call API as relative paths under `./api/` (the parent FD "
                "proxy rewrites them to the subprocess).\n\n"
                f"App spec (JSON):\n\n```json\n{spec_json}\n```"
            ),
            acceptance_criteria=(
                f"`POST /fd/code-apps/{slug}/scaffold` returned 200 and the "
                "manifest contains `name`, `slug`, `version`."
            ),
            step_kind="atomic",
        ),
        OrchestratorTask(
            id="smoke",
            title=f"Smoke-test code-app {slug!r}",
            description=(
                f"Hit each endpoint in the spec via `/fd/code-apps/{slug}/api/...` "
                "and check the status codes are sensible (2xx for the happy "
                "path, 4xx for invalid inputs). Use the `proxy_app` tool. "
                "Capture any failure for the next step."
            ),
            acceptance_criteria=(
                "Every endpoint in the spec returned a status code consistent "
                "with the spec — no 5xx, no connection failures."
            ),
            depends_on=["scaffold"],
            step_kind="atomic",
        ),
        OrchestratorTask(
            id="inspect_logs",
            title=f"Inspect subprocess logs for {slug!r}",
            description=(
                f"Call `/fd/code-apps/{slug}/logs?n=200`. If `last_error` is "
                "non-empty or stderr contains a traceback, surface the most "
                "recent traceback to the next step."
            ),
            acceptance_criteria=(
                "`last_error` is empty and stderr contains no Python traceback."
            ),
            depends_on=["smoke"],
            step_kind="verify",
        ),
        OrchestratorTask(
            id="revise",
            title=f"Repair {slug!r} if smoke / logs failed",
            description=(
                f"If the previous step found an error, edit `backend.py` "
                f"(or `frontend.html`) to fix it. Call "
                f"`POST /fd/code-apps/{slug}/restart` afterward so the next "
                "iteration picks up the fix. Re-run smoke."
            ),
            acceptance_criteria=(
                "Smoke test passes after the revision and logs are clean."
            ),
            depends_on=["inspect_logs"],
            step_kind="revise",
        ),
    ]

    summary = (
        f"Scaffold and self-test code-app **{spec.name}** "
        f"(`{spec.slug}`): {spec.summary}".strip()
    )
    return Plan(summary=summary, user_input=user_input or spec.summary, tasks=tasks)


# ── convenience entry point ───────────────────────────────────────────


async def plan_app(
    provider: LLMProvider,
    user_request: str,
) -> tuple[AppSpec, Plan] | None:
    """One-shot: generate spec + workflow plan, or return ``None`` on failure.

    Convenience for callers that want the whole pipeline in one call.
    Both pieces are surfaced because the FD UI usually wants to render
    the spec for the user *before* showing the plan steps.
    """
    generator = AppPlanGenerator(provider)
    spec = await generator.generate(user_request)
    if spec is None:
        return None
    plan = spec_to_plan(spec, user_input=user_request)
    return spec, plan
