"""The browser tool's ``batch``: several steps in one call.

Each tool call is a whole round of the model — seconds, often tens of them —
so a form filled field by field, or a page opened and then read, cost a round
per step. A batch runs up to :data:`STEPS_MAX` steps in order and answers once,
with what each step said.

Every step goes the very way a call of its own goes
(``Agent._execute_tool_with_guard``): the guard, approval cards, the privacy
gate, a shared agent's member rules, a rehearsal, the deck's path limits, and
the tool-call register — each step is in Activity as a browser call of its
own. A step of a kind the person may approve first (a click on "Pay now",
storing a password), or one that waits for them or runs steps of its own, is
never part of a batch: the steps are all checked before the first one runs,
and a batch with such a step is refused whole, so that call stands on its own.

It stops at the first step that fails, and says what the steps before it did
and which didn't run; nothing runs twice. A step that moves the work to the
person's Chrome (a site walls the headless browser) ends the batch too: refs
and tabs from before don't hold there.
"""

from __future__ import annotations

import json
from typing import Any

from captain_claw.tools.registry import ToolResult

STEPS_MAX = 10
PICTURES_MAX = 3
# Each of these gives a picture the model looks at.
PICTURE_ACTIONS = frozenset({"screenshot", "zoom", "observe", "act"})
# Waits for the person (an extension install, pairing with Flight Deck), runs
# steps of its own (a recording, a replay, a sign-in), or is a batch itself.
NOT_IN_BATCH = frozenset({
    "batch", "login", "extension_install", "extension_connect",
    "workflow_record_start", "workflow_record_stop", "workflow_run",
})
TIMEOUT_MAX = 600.0


def _steps(raw: Any) -> list[Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, list) else None


def check(raw: Any, actions: frozenset[str] | set[str]) -> tuple[list[dict[str, Any]], str]:
    """The steps, cleaned (no internal ``_`` arguments, no nested ``steps``),
    or ``([], why)`` when the batch can't run as it is."""
    from captain_claw.approval_kinds import KIND_LABELS, kind_for

    steps = _steps(raw)
    if not steps:
        return [], ("batch needs steps: a list of browser calls, each with its action and arguments, e.g. "
                    '[{"action": "navigate", "url": "…"}, {"action": "page_text"}].')
    if len(steps) > STEPS_MAX:
        return [], f"A batch takes at most {STEPS_MAX} steps ({len(steps)} given): split it."
    out: list[dict[str, Any]] = []
    pictures = 0
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            return [], f"Step {i} isn't a browser call: each step is an object with an action and its arguments."
        clean = {k: v for k, v in step.items() if isinstance(k, str) and not k.startswith("_") and k != "steps"}
        action = str(clean.get("action") or "").strip().lower()
        if not action:
            return [], f"Step {i} has no action."
        if action not in actions:
            return [], f"Step {i}: '{action}' isn't a browser action."
        if action in NOT_IN_BATCH:
            return [], (f"Step {i}: {action} can't be part of a batch (it waits for the person, or runs steps of "
                        "its own) — call it on its own.")
        kind = kind_for("browser", {**clean, "action": action})
        if kind:
            return [], (f"Step {i} ({action}) is a kind the person may want to approve first "
                        f"({KIND_LABELS.get(kind, kind).lower()}) — call it on its own, not in a batch.")
        if action in PICTURE_ACTIONS:
            pictures += 1
            if pictures > PICTURES_MAX:
                return [], f"A batch gives at most {PICTURES_MAX} pictures (screenshot, zoom, observe): split it."
        clean["action"] = action
        out.append(clean)
    return out, ""


def timeout(tool: Any, raw: Any) -> float:
    """Room for every step, as each would have as a call of its own."""
    steps = [s for s in (_steps(raw) or []) if isinstance(s, dict)][:STEPS_MAX]
    total = sum(float(tool.timeout_for(s)) for s in steps) + 15.0
    return min(max(total, float(tool.timeout_seconds)), TIMEOUT_MAX)


def _not_run(i: int, n: int) -> str:
    if i >= n:
        return ""
    return f" Step {n} wasn't run." if i + 1 == n else f" Steps {i + 1}–{n} weren't run."


def _moved_to_chrome(args: dict[str, Any], before: str | None, after: str | None) -> bool:
    """The step went to the person's Chrome without being asked to (a wall)."""
    return (args.get("action") in ("open", "navigate") and str(args.get("target") or "").lower() != "chrome"
            and before != "chrome" and after == "chrome")


async def run(tool: Any, kwargs: dict[str, Any]) -> ToolResult:
    """Run the batch in ``kwargs["steps"]``, each step as a call of its own."""
    from captain_claw.tools import browser as browser_mod

    actions = frozenset(tool.parameters["properties"]["action"]["enum"])
    steps, why = check(kwargs.get("steps"), actions)
    if why:
        return ToolResult(success=False, error=why)
    agent = kwargs.get("_agent")
    call = getattr(agent, "_execute_tool_with_guard", None)
    if call is None:
        return ToolResult(success=False, error="batch runs only in an agent's turn — call the actions one by one.")
    sid = str(kwargs.get("_session_id") or "").strip() or "default"
    target = str(kwargs.get("target") or "").strip().lower()
    n = len(steps)
    parts: list[str] = []
    images: list[str] = []
    for i, step in enumerate(steps, 1):
        args = dict(step)
        if target and not args.get("target"):
            args["target"] = target
        before = browser_mod._TARGETS.get(sid)
        try:
            result = await call("browser", args, interaction_label="tool_call:browser",
                                session_id_override=str(kwargs.get("_session_id") or "").strip() or None)
        except Exception as exc:  # noqa: BLE001 — a guard or a policy said no: that step failed
            result = ToolResult(success=False, error=str(exc) or type(exc).__name__)
        label = f"Step {i}/{n} ({args['action']})"
        images.extend(getattr(result, "images", None) or [])
        if not result.success:
            done = "\n\n".join(parts)
            return ToolResult(success=False, images=images, error=(
                (done + "\n\n" if done else "") + f"{label} failed: {result.error}{_not_run(i, n)}"))
        parts.append(f"{label}:\n{(result.content or 'Done.').strip()}")
        if i < n and _moved_to_chrome(args, before, browser_mod._TARGETS.get(sid)):
            parts.append(f"The work moved to the person's Chrome at step {i}, so the rest stopped there "
                         f"(refs and tabs from before don't hold in it).{_not_run(i, n)}")
            break
    return ToolResult(success=True, content="\n\n".join(parts), images=images)
