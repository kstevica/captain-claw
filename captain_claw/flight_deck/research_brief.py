"""R12 — intent brief (faithful prompt rephrase) for Basna/Vatra.

A terse or ambiguous task ("do a ROPA for X") under-specifies the work. A strong
model silently fills the gaps in its head; weaker models don't — they guess
scope, miss deliverable shape, and drift. This is a preprocessing step that turns
the raw intent into ONE clear, structured task brief BEFORE routing, so:

* the router / Lead selects the team against the clarified task, not the terse one;
* every worker builds against the same explicit scope + deliverable definition;
* the user can read and EDIT the brief first — a Code-plan-style review gate.

The single hard rule is **faithfulness**: the brief clarifies and structures, it
never adds goals, narrows scope, or drops a constraint. To make drift impossible
even if the model over-reaches, the brief is always carried ALONGSIDE the
original intent, and the original is declared authoritative on any conflict
(:func:`brief_task`). This keeps the feature safe for *any* research task —
nothing here is domain-specific.

Pure + model-free so it stays unit-testable; the caller runs the one LLM call.
"""

from __future__ import annotations

# Cap the stored/echoed brief so a runaway generation can't bloat every prompt.
_MAX_BRIEF_CHARS = 4000


def derive_brief_prompt(intent: str) -> str:
    """Prompt a reason-tier model to restate the task as a faithful, structured brief."""
    return (
        "You are scoping a task for a team that will carry it out. Rewrite the request "
        "below as ONE clear, well-structured brief — this is a faithful CLARIFICATION, "
        "not a reinterpretation.\n\n"
        "Hard rules:\n"
        "- Preserve every constraint, entity, and requirement in the original. Add NO new "
        "goals and do NOT narrow the scope.\n"
        "- Make implicit structure explicit: the objective, what's in scope vs out, the "
        "subjects/entities involved, the shape of the expected deliverable, and any "
        "definitions or standards the task refers to.\n"
        "- Where the request is genuinely ambiguous, state the most reasonable reading as an "
        "explicit **Assumption** rather than silently deciding — so a human can correct it.\n"
        "- If the request is already precise, restate it concisely; don't pad it.\n"
        "- Do not answer the task or start the work. Produce only the brief.\n\n"
        "Format as short markdown with these sections (omit any that don't apply):\n"
        "**Objective** · **In scope** · **Out of scope** · **Subjects / entities** · "
        "**Deliverable** · **Key constraints & definitions** · **Assumptions**\n\n"
        "## Original request\n"
        f"{intent}"
    )


def derive_brief_with_files_prompt(intent: str, file_names: list[str] | None) -> str:
    """Brief prompt for an agent that ALSO has the user's attached files in hand.

    Extends :func:`derive_brief_prompt` with an instruction to open and read the
    attachments and fold their substance into the brief — so the team plans against
    what the files actually contain, not just their names. Domain-agnostic: it says
    "reflect the files", never what the files are about.
    """
    listed = ", ".join(f for f in (file_names or []) if f) or "the attached file(s)"
    return (
        derive_brief_prompt(intent)
        + "\n\n## Attached files\n"
        + f"The user attached these file(s) to the task: {listed}. They are provided to "
        "you directly — OPEN and read them before writing the brief. Fold what they hold "
        "into it: what data or content each file contains, how it relates to the task, and "
        "what the team must actually DO with it. If the task's specifics live in the files "
        "(a dataset to analyse, a document to work from, an image to build against), the "
        "brief must make that explicit so no downstream worker treats the files as optional "
        "context. Do not invent file contents you cannot see; describe only what is there."
    )


def parse_brief(output: str) -> str:
    """Clean the model's brief: strip code fences, trim, cap length. May return ""."""
    if not output:
        return ""
    text = output.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```")).strip()
    return text[:_MAX_BRIEF_CHARS].strip()


def brief_task(intent: str, brief: str | None) -> str:
    """The effective task string fed to routing + workers.

    With no brief this is exactly the raw intent (so an off/empty brief reproduces
    today's behaviour byte-for-byte). With a brief, the ORIGINAL intent is kept
    verbatim and declared authoritative, and the brief follows as a labelled
    clarification — so a rephrase can never silently become the source of truth.
    """
    intent = intent or ""
    brief = (brief or "").strip()
    if not brief:
        return intent
    return (
        f"{intent}\n\n---\n## Structured brief (clarification of the request above)\n"
        "This restates and structures the request to remove ambiguity. It introduces no new "
        "goals; if anything here conflicts with the original request above, the ORIGINAL "
        "request governs.\n\n"
        f"{brief}"
    )
