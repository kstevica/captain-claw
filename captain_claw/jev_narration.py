"""Jev's decisions as narration lines the person reads in the chat's step list.

Jev (TypeSafe) doesn't write text; it answers typed questions with
probabilities. Those probabilities are its "thinking": which options it
weighed and how sure it was. Each decision Jev makes inside a turn — the
course check, how a new message relates to the open task, more steps or stop
at the step budget, what follows a refused call — becomes one plain sentence
here, with the runner-up options and their share.

The lines reach the web chat as narration marked ``source: "jev"``
(``Agent._emit_jev_narration``): the chat shows them in the step list with
Jev's mark, while the channel relay (WhatsApp, glasses, devices), Telegram and
Basna/Vatra's team board leave them out. They are never saved into the
session, so the model never reads them as its own words.
"""

from __future__ import annotations

from typing import Any

STAGE_WORDS = {
    "orienting": "orienting",
    "gathering": "gathering",
    "producing": "producing",
    "checking": "checking",
    "delivered": "delivered",
}
NEXT_WORDS = {
    "continue": "keep going",
    "refocus": "refocus",
    "finish": "wrap up",
    "ask_person": "ask you",
}
RELATION_WORDS = {
    "new_task": "a new task",
    "continues": "the same task",
    "amends": "an addition to the task",
    "not_a_task": "not a task",
}
LINE_WORDS = {
    "refocus": "Reminded the agent to get back to the task.",
    "finish": "Told the agent the task looks done and to give its answer.",
    "ask_person": "Told the agent to ask you if it needs something only you can give.",
}
BLOCK_WORDS = {
    "verify": "read the file once before deciding",
    "continue": "go on with the rest of the request",
    "finish": "give the final answer",
    "ask_user": "ask you",
    "repeat_ok": "let the call run once more",
}


def pct(p: Any) -> str:
    try:
        return f"{round(float(p) * 100)}%"
    except (TypeError, ValueError):
        return "?"


def _others(probs: dict[str, float] | None, chosen: str | None, words: dict[str, str], n: int = 2) -> str:
    """The runner-up options and their share: "producing 30%, gathering 6%"."""
    if not isinstance(probs, dict):
        return ""
    ranked = sorted(
        ((k, v) for k, v in probs.items() if k != chosen and isinstance(v, (int, float)) and v >= 0.05),
        key=lambda kv: -kv[1],
    )[:n]
    return ", ".join(f"{words.get(k, k)} {pct(v)}" for k, v in ranked)


def course_check(check: Any, line_key: str = "") -> str:
    """After a course check: where the work stands, on track or not, what
    comes next — and what the agent was told, if anything."""
    parts: list[str] = []
    step = getattr(check, "step", None)
    head = f"Jev checked the course after step {step}" if step else "Jev checked the course"
    on_track = getattr(check, "on_track", None)
    if isinstance(on_track, (int, float)):
        parts.append(f"on track {pct(on_track)}")
    stage, name = getattr(check, "stage", None), getattr(check, "stage_name", None)
    if isinstance(stage, (int, float)) and name:
        n = min(5, max(1, int(round(stage))))
        others = _others(getattr(check, "stage_probs", None), name, STAGE_WORDS)
        parts.append(f"stage {n} of 5, {STAGE_WORDS.get(name, name)}" + (f" (also {others})" if others else ""))
    nxt = getattr(check, "next", None)
    if nxt:
        conf = getattr(check, "next_confidence", None)
        others = _others(getattr(check, "next_probs", None), nxt, NEXT_WORDS)
        detail = ", ".join(x for x in (pct(conf) if conf is not None else "", others) if x)
        parts.append(f"next: {NEXT_WORDS.get(nxt, nxt)}" + (f" ({detail})" if detail else ""))
    text = head + (": " + "; ".join(parts) + "." if parts else ".")
    if line_key:
        text += " " + LINE_WORDS.get(line_key, "")
    else:
        text += " No reminder needed."
    return text.strip()


def relation(choice: str, confidence: float | None, probs: dict[str, float] | None,
             guess: str, applied: bool, threshold: float) -> str:
    """At turn start: how Jev read the new message against the open task."""
    word = RELATION_WORDS.get(choice, choice)
    others = _others(probs, choice, RELATION_WORDS)
    sure = f"{pct(confidence)}" + (f"; {others}" if others else "")
    if (confidence or 0.0) < threshold:
        return (f"Jev wasn't sure how your message relates to the open task ({word} {sure}), "
                f"so the usual rules decide: {RELATION_WORDS.get(guess, guess)}.")
    if applied:
        return f"Jev read your message as {word} ({sure}) and updated the task reminder."
    return f"Jev read your message as {word} ({sure}), as the usual rules did."


def budget(p: float | None, extended: bool, new_limit: int | None = None, reused: bool = False) -> str:
    """At the step budget: more steps or stop."""
    if p is None:
        return ""
    if reused:
        why = f"Jev's last course check says the work is on track ({pct(p)})"
    else:
        why = f"Jev says the work is still moving ({pct(p)})" if extended \
            else f"Jev says the work has stalled (only {pct(p)} that it's still moving)"
    if extended:
        more = f", so it gets more steps (up to {new_limit})" if new_limit else ", so it gets more steps"
        return f"Step budget used up. {why}{more}."
    if reused:
        return f"Step budget used up. Jev's last course check doesn't say keep going ({pct(p)} on track), so the turn stops here."
    return f"Step budget used up. {why}, so the turn stops here."


def block_review(tool: str, choice: str, p: float, applied: bool, released: bool) -> str:
    """After a refused repeat call: what Jev picked to come next."""
    what = BLOCK_WORDS.get(choice, choice)
    if released:
        return f"A repeated {tool} call was held back; Jev judged something changed since ({pct(p)}) and let it run once."
    if applied:
        return f"A repeated {tool} call was held back; Jev picked what comes next: {what} ({pct(p)})."
    return f"A repeated {tool} call was held back; Jev wasn't sure what comes next ({what} {pct(p)}), so the usual message stands."
