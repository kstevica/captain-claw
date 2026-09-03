"""Two-axis (function × domain) archetype composition — PROTOTYPE, flag-gated.

A *grid* agent is a pair ``function.domain`` (e.g. ``reviewer.legal``). Instead
of storing F×M flat leaf archetypes, we keep two small base registries —
``instructions/functions.json`` (role / SOP / base tools / cognitive_mode /
recall_mode) and ``instructions/domains.json`` (instruction overlay / extra
tools / tier_floor / recall_override) — and compose a leaf on the fly at spawn
time. Editing one function updates every domain's variant of it, and editing one
domain updates every function's variant: a "living" grid on both axes, whereas a
flattened leaf is only living along whichever single id it was stored under.

The composed dict has the SAME shape the base archetype registry yields, so
every downstream consumer (``server._resolve_archetype``'s fold-in, dubina's
``spawn_archetype_agent``) treats it identically — no downstream changes.

Gated by the ``FD_ARCHETYPE_GRID`` env flag. When off (the default), or when an
id is not a resolvable pair, callers fall back to the existing single-id
archetype lookup — so this is purely additive and non-breaking. Base archetype
ids never contain a dot, so a dotted id is unambiguously a grid selector.

PROTOTYPE SCOPE / follow-ups (not in this module):
- Composition reads the BASE function/domain files only; per-user function or
  domain overlays (the equivalent of ``user_archetypes`` shadowing a base id)
  are future work.
- The composed dict carries ``recall_mode`` and ``memory_tags`` for the
  deep-memory pooling design, but WRITING those tags into deep memory (threading
  them into ``DeepMemoryIndex.add`` / ``search``) is a separate wiring step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_INSTR = Path(__file__).parent.parent / "instructions"
_FUNCTIONS_FILE = _INSTR / "functions.json"
_DOMAINS_FILE = _INSTR / "domains.json"

_GRID_FLAG = "FD_ARCHETYPE_GRID"

# Tier capability ladder, low → high. A domain's ``tier_floor`` can only RAISE a
# function's tier, never lower it. Tiers outside this ladder (coding / vision —
# specialist, not strictly "higher") are treated as no-floor.
_TIER_ORDER = ["micro", "fast", "balanced", "longctx", "reason"]


def grid_enabled() -> bool:
    """True when the function×domain grid is switched on via ``FD_ARCHETYPE_GRID``."""
    return os.environ.get(_GRID_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _load(path: Path, key: str) -> dict[str, dict]:
    """Load a registry file into an ``id → entry`` map.

    Best-effort: a missing or invalid file yields an empty map (logged for the
    invalid case) so a broken grid file can never break the normal spawn path.
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Invalid grid registry; ignoring", file=str(path), error=str(exc))
        return {}
    out: dict[str, dict] = {}
    for entry in data.get(key, []):
        eid = str(entry.get("id") or "").strip()
        if eid:
            out[eid] = entry
    return out


def load_functions() -> dict[str, dict]:
    """The function axis, keyed by id. Empty when the file is absent/invalid."""
    return _load(_FUNCTIONS_FILE, "functions")


def load_domains() -> dict[str, dict]:
    """The domain axis, keyed by id. Empty when the file is absent/invalid."""
    return _load(_DOMAINS_FILE, "domains")


def parse_pair(aid: str) -> tuple[str, str | None]:
    """``"reviewer.legal"`` → ``("reviewer", "legal")``; a plain id → ``(aid, None)``.

    Splits on the FIRST dot only (a function id may not contain a dot; a domain
    id may). Any ``@tier`` suffix has already been stripped by the caller, which
    partitions on ``@`` before resolving the id.
    """
    aid = (aid or "").strip()
    if "." not in aid:
        return aid, None
    fid, _, did = aid.partition(".")
    fid, did = fid.strip(), did.strip()
    if not fid or not did:
        return aid, None
    return fid, did


def _max_tier(fn_tier: str | None, dom_floor: str | None) -> str:
    """Return the function's tier raised to the domain's floor when the floor is
    higher on the capability ladder. Unknown tiers → no raise (return fn_tier)."""
    fn_tier = (fn_tier or "balanced").strip()
    if not dom_floor:
        return fn_tier
    try:
        higher = _TIER_ORDER.index(fn_tier) >= _TIER_ORDER.index(str(dom_floor).strip())
    except ValueError:
        return fn_tier
    return fn_tier if higher else str(dom_floor).strip()


def compose_archetype(fn: dict, dm: dict) -> dict:
    """Compose a function entry and a domain entry into a spawn-ready archetype.

    Merge rules:
      - ``fleet_instructions`` — function (role + SOP), then the domain overlay
        appended after a blank line.
      - ``tools`` — union of the function's tools and the domain's ``tools_add``.
      - ``tier`` — the function's tier, raised to the domain's ``tier_floor`` if
        that floor is higher (:func:`_max_tier`).
      - ``cognitive_mode`` — the function's (how it thinks).
      - ``recall_mode`` — the domain's ``recall_override`` if set, else the
        function's (drives deep-memory recall breadth: pool | domain | self).
      - ``memory_tags`` — ``["agent:<fn>", "domain:<dm>"]`` for deep-memory
        pooling (write both; recall filters on them).
    """
    fid, did = str(fn["id"]), str(dm["id"])
    label = str(dm.get("label") or did)
    role = str(fn.get("role") or fid)
    fn_instr = str(fn.get("fleet_instructions") or "").rstrip()
    overlay = str(dm.get("overlay") or "").strip()
    tools = sorted(set(fn.get("tools") or []) | set(dm.get("tools_add") or []))
    keywords = sorted(set(fn.get("keywords") or []) | set(dm.get("keywords") or []))
    return {
        "id": f"{fid}.{did}",
        "role": f"{label} {role}",
        "family": label,
        "description": str(fn.get("description") or role),
        "cognitive_mode": str(fn.get("cognitive_mode") or "neutra"),
        "tier": _max_tier(fn.get("tier"), dm.get("tier_floor")),
        "tools": tools,
        "keywords": keywords,
        "reliability_seed": fn.get("reliability_seed", 0.7),
        "runtime": str(fn.get("runtime") or ""),
        "fleet_instructions": (fn_instr + ("\n\n" + overlay if overlay else "")).strip(),
        # ── grid extensions (consumed by the deep-memory pooling design) ──
        "recall_mode": str(dm.get("recall_override") or fn.get("recall_mode") or "pool"),
        "memory_tags": [f"agent:{fid}", f"domain:{did}"],
        "composed": True,
        "source": "grid",
    }


def recall_filter(recall_mode: str, memory_tags: list[str] | None) -> str:
    """Build a Typesense ``filter_by`` fragment for an agent's recall breadth.

    - ``pool`` (or unknown/empty) → ``""`` — read the whole owner pool, no tag
      narrowing (the safe default; the owner scope is ANDed on server-side).
    - ``domain`` → restrict to the agent's ``domain:<x>`` tag.
    - ``self`` → restrict to the agent's own ``agent:<x>`` tag.

    Degrades to ``""`` (pool) when the mode needs a tag the agent doesn't carry.

    NOTE: a ``domain``/``self`` filter also excludes pool rows that carry *no*
    tags (e.g. auto-indexed VFS files). That is intended for a specialist that
    wants high-signal recall; the follow-up to also see shared/general context is
    to stamp a ``domain:general`` sentinel on untagged writes and OR it in here.
    """
    mode = (recall_mode or "pool").strip().lower()
    if mode not in ("domain", "self"):
        return ""
    prefix = "domain:" if mode == "domain" else "agent:"
    tag = next((t for t in (memory_tags or []) if t.startswith(prefix)), "")
    return f"tags:=`{tag}`" if tag else ""


def resolve_pair(aid: str) -> dict | None:
    """Resolve a ``function.domain`` id into a composed archetype, or ``None``.

    Returns ``None`` — deferring to the normal single-id lookup — when the grid
    is disabled, the id is not a pair, or either axis is unknown. Never raises:
    a bad grid file must degrade gracefully to base behaviour.
    """
    if not grid_enabled():
        return None
    fid, did = parse_pair(aid)
    if did is None:
        return None
    try:
        fn = load_functions().get(fid)
        dm = load_domains().get(did)
    except Exception as exc:  # defensive: file races, unexpected shapes
        log.warning("Grid registry load failed", pair=aid, error=str(exc))
        return None
    if not fn or not dm:
        return None
    try:
        composed = compose_archetype(fn, dm)
    except Exception as exc:
        log.warning("Archetype composition failed", pair=aid, error=str(exc))
        return None
    log.info("Composed grid archetype", pair=aid, tier=composed["tier"],
             recall_mode=composed["recall_mode"], tags=composed["memory_tags"])
    return composed
