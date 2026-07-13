"""Phase B — decomposed, dependency-layered team build for Code.

Today Code's big-job build is one ``code-implementer`` doing the whole plan in a
single context (code_routes.py). On weak models that context is where the plan
gets lost. Vatra's answer — a team of focused agents coordinated by a plan and a
shared ledger — is what let a weak-model ensemble beat a strong single model
(docs/code-from-vatra-plan.md). This module ports that structure to Code.

The plan is decomposed once (a reason-tier "Group 0" call) into **slices**: each
a focused unit of work with an owner archetype, the files it owns, and the slices
it ``depends_on``. Slices are then arranged into **layers** by their dependency
DAG (foundations first), and built layer by layer — a focused implementer per
slice, sharing an interface ledger (the ``facts`` tool) so parallel work agrees
on signatures. A barrier between layers means a slice never starts before the
foundations it builds on exist.

Why DAG layering and not ``vatra_groups.resolve_groups``: Vatra phases by
archetype role (research→A, review→C). Code slices are almost all
``code-implementer``, so role-based grouping would collapse them into one phase.
The dependency graph is the real ordering here, so we layer on it directly (and
reuse ``vatra_groups.group_label`` only for display).

Pure module: prompts, parser, DAG layering, renderers. The route owns the model
calls and the git/dispatch wiring.
"""

from __future__ import annotations

import json
import re

from captain_claw.flight_deck import vatra_groups
from captain_claw.logging import get_logger

log = get_logger(__name__)

_MAX_SLICES = 8          # hard ceiling regardless of the knob (cost guard)
_MAX_LAYERS = 6          # a pathological chain can't produce endless phases
_CODE_OWNERS_DEFAULT = "code-implementer"


# ── decomposition prompt ─────────────────────────────────────────────

def decompose_prompt(intent: str, plan_text: str, roster: list[str],
                     max_slices: int) -> str:
    roster_str = ", ".join(roster) if roster else _CODE_OWNERS_DEFAULT
    plan_block = f"\n\n## Approved plan\n{plan_text[:9000]}" if plan_text.strip() else ""
    return (
        "You are the build coordinator. Split the approved plan into a small set of "
        "independent BUILD SLICES that a team can implement with minimal collision, "
        "then declare the dependencies between them. Aim for slices that own "
        "DISJOINT files. Put shared foundations others depend on (data models, "
        "schema, types, shared interfaces, config) in their OWN early slice so the "
        "rest can build against them.\n\n"
        f"Use at most {max_slices} slices — fewer is fine; a tiny task may need "
        "just one. Reply ONLY with JSON, no prose:\n"
        '{"slices": [{"id": "s1", "title": "<short>", '
        '"brief": "<what to build, 1-3 sentences, concrete>", '
        f'"owner": "<one of: {roster_str}>", '
        '"files": ["<repo-relative path this slice owns>", ...], '
        '"depends_on": ["<id of a slice whose output this needs>", ...]}]}\n'
        "Rules: ids are short and unique; depends_on lists ONLY other slice ids; no "
        "cycles; a slice's files should not overlap another's; a foundations slice "
        "has an empty depends_on. If the plan is genuinely one indivisible unit, "
        'return a single slice.\n\n'
        f"## Task\n{intent}{plan_block}"
    )


# ── parsing ──────────────────────────────────────────────────────────

def parse_slices(output: str, arch_by_id: dict, cap: int = _MAX_SLICES) -> list[dict]:
    """Normalize the decompose reply into slice dicts, then layer them. Tolerant;
    never raises. Returns [] when there's nothing usable (caller falls back to the
    single-implementer build)."""
    if not output:
        return []
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        blob = text[start:end + 1] if 0 <= start < end else None
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return []

    cap = max(1, min(int(cap or _MAX_SLICES), _MAX_SLICES))
    slices: list[dict] = []
    seen_ids: set[str] = set()
    for i, item in enumerate((raw.get("slices") or [])[:cap], 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()[:120]
        brief = str(item.get("brief") or "").strip()[:800]
        if not title and not brief:
            continue
        sid = str(item.get("id") or f"s{i}").strip()[:20] or f"s{i}"
        if sid in seen_ids:
            sid = f"{sid}_{i}"
        seen_ids.add(sid)
        owner = str(item.get("owner") or "").strip()
        if owner not in (arch_by_id or {}):
            owner = _CODE_OWNERS_DEFAULT
        files = [_relpath(f) for f in (item.get("files") or []) if _relpath(f)][:20]
        deps = [str(d).strip()[:20] for d in (item.get("depends_on") or [])
                if str(d).strip()]
        slices.append({"id": sid, "title": title or sid, "brief": brief,
                       "owner_archetype_id": owner, "files": files,
                       "depends_on": deps})

    if not slices:
        return []
    # Drop dependency refs that don't point at a real sibling (and self-refs).
    ids = {s["id"] for s in slices}
    for s in slices:
        s["depends_on"] = [d for d in s["depends_on"] if d in ids and d != s["id"]]
    assign_layers(slices)
    return slices


def _relpath(p) -> str:
    s = str(p or "").strip().replace("\\", "/")
    if not s or s.startswith("/") or s.startswith("~"):
        return ""
    parts = [seg for seg in s.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        return ""
    return "/".join(parts)[:200]


# ── dependency-DAG layering ──────────────────────────────────────────

def assign_layers(slices: list[dict]) -> int:
    """Set each slice's ``layer`` (1-based) = longest dependency chain to a root.
    Foundations (no deps) are layer 1; a slice is one past the deepest slice it
    depends on. Cycle-safe: a back-edge that would exceed the layer cap is treated
    as absent (the slice lands as early as its acyclic deps allow), so the pass
    always terminates. Returns the number of layers."""
    by_id = {s["id"]: s for s in slices}
    layer: dict[str, int] = {}

    def depth(sid: str, stack: frozenset) -> int:
        if sid in layer:
            return layer[sid]
        if sid in stack or len(stack) > _MAX_LAYERS:
            return 1                       # cycle / too deep → treat as a root
        s = by_id.get(sid)
        deps = [d for d in (s.get("depends_on") if s else []) if d in by_id]
        d = 1 if not deps else 1 + max(depth(dp, stack | {sid}) for dp in deps)
        d = min(d, _MAX_LAYERS)
        layer[sid] = d
        return d

    for s in slices:
        s["layer"] = depth(s["id"], frozenset())
        s["group_resolved"] = vatra_groups.group_label(s["layer"])  # display only
    return max((s["layer"] for s in slices), default=1)


def layers(slices: list[dict]) -> list[tuple[int, list[dict]]]:
    """Slices grouped by layer, ascending — the phases to run, in order."""
    out: dict[int, list[dict]] = {}
    for s in slices:
        out.setdefault(int(s.get("layer", 1)), []).append(s)
    return [(lyr, out[lyr]) for lyr in sorted(out)]


# ── slice prompt + renderers ─────────────────────────────────────────

def slice_prompt(intent: str, s: dict, by_id: dict, *, facts: bool = False) -> str:
    """The focused build prompt for one slice: its mandate, the files it owns, the
    already-built slices it depends on (read them, don't rebuild), and — when the
    interface ledger is armed — the discipline for sharing/reading signatures."""
    files = s.get("files") or []
    files_block = ("\n\nFiles this slice owns (create/edit these; do not rewrite "
                   "other slices' files):\n" + "\n".join(f"- {f}" for f in files)) \
        if files else ""
    deps = [by_id[d] for d in (s.get("depends_on") or []) if d in by_id]
    deps_block = ""
    if deps:
        listed = "\n".join(
            f"- {d['title']}" + (f" (files: {', '.join(d['files'])})" if d.get("files") else "")
            for d in deps)
        deps_block = (
            "\n\nAlready built — your foundations (read these files, build ON them, "
            "do NOT reimplement them):\n" + listed)
    ledger_block = (_INTERFACE_LEDGER_DIRECTIVE if facts else "")
    return (
        f"You are building ONE slice of a larger, already-approved plan — stay "
        f"strictly within your slice.\n\n## Your slice — {s['title']}\n{s['brief']}"
        f"{files_block}{deps_block}\n\n"
        "Implement it fully with write/edit tool calls (files exist only when "
        "written to disk), install deps and verify via your shell. Do not touch "
        "files another slice owns; if you need something not yet built, code "
        "against the interface your foundations expose and note the assumption.\n\n"
        f"Original request for context:\n{intent}"
        + ledger_block
    )


#: Injected into slice prompts when the interface ledger (facts tool) is armed —
#: the discipline that keeps parallel coders' interfaces consistent.
_INTERFACE_LEDGER_DIRECTIVE = (
    "\n\nINTERFACE LEDGER: the team shares a ledger of interface decisions (tool: "
    "`facts`). When your slice defines something others build against — a function "
    "or endpoint signature, a data model / table field, a shared type or constant "
    "name — record it: `facts` action=set, a short snake_case key (e.g. "
    "user_api_create_signature), the value (the signature/shape), status, and "
    "provenance (the file). Before you invent a name or shape another slice may "
    "already own, `facts` action=get/list and reuse theirs — never guess a "
    "teammate's interface. If set reports a CONFLICT, adopt the existing value "
    "rather than overwriting."
)


def coordination_markdown(slices: list[dict]) -> str:
    """The human-readable coordination plan for the report + chat."""
    lines = ["# Build coordination plan", "",
             f"{len(slices)} slice(s) across {len(layers(slices))} layer(s):", ""]
    for lyr, group in layers(slices):
        lines.append(f"## Layer {lyr} ({vatra_groups.group_label(lyr)})")
        for s in group:
            deps = ", ".join(s.get("depends_on") or []) or "—"
            files = ", ".join(s.get("files") or []) or "—"
            lines.append(f"- **{s['title']}** _({s['owner_archetype_id']})_ · "
                         f"files: {files} · depends on: {deps}")
        lines.append("")
    return "\n".join(lines).strip()


def schedule_progress(lyr: int, total_layers: int, group: list[dict]) -> str:
    titles = ", ".join(s["title"] for s in group)
    return f"Layer {lyr}/{total_layers}: {len(group)} slice(s) — {titles}"
