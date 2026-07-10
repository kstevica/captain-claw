"""Cross-mode quality/cost governor — the opt-in safety envelope.

Every cross-pollination quality feature (Code's test gate + deep build +
coverage check; Basna/Vatra's acted-gate, research map, delta rounds, worker
escalation, per-round git snapshots) is gated by a flag here and draws from ONE
per-run token budget with a hard ceiling.

The whole point of this module is a single guarantee:

    An empty / absent ``quality`` config == the systems' CURRENT behaviour.

``QualityProfile.from_dict(None)`` returns all-features-off with an unbounded
budget, so nothing new runs and nothing extra is spent. That is what makes it
impossible for these additions to regress the existing Code / Basna / Vatra
behaviour or to inflate token spend unless a human explicitly opts in.

ONE deliberate, documented exception (locked 2026-07-10, see
docs/vatra-quality-tightening-plan.md): ``honesty_guard`` defaults to **True**.
It is free and prompt-only — it appends the anti-fabrication directives to
research workers, the Vatra reporter, and the Basna synthesizer. An explicit
``{"honesty_guard": false}`` restores the pre-guard prompts byte-for-byte. It is
independent of presets and deliberately excluded from ``any_enabled`` (which
keeps meaning "any opt-in lever on").

It rides in the same session/project ``config`` JSON as ``HorizonConfig`` (under
a ``quality`` key), so wiring it needs no DB migration.

Presets (``quality.profile``) flip sensible groups so a user doesn't hand-set
ten booleans:

* ``"off"``       — everything off. Identical to today. (default)
* ``"balanced"``  — the token-neutral / token-saving wins only: acted-gate,
  test gate, delta rounds, research map, worker escalation. Quality goes up,
  spend stays flat or drops (tests cost zero LLM tokens; delta rounds and the
  research map cut re-reads). Safe to leave on.
* ``"thorough"``  — ``balanced`` plus the genuinely expensive levers (deep
  build, coverage check). These can add tokens, so ``thorough`` is only ever
  meaningful together with a ``token_budget`` — the governor refuses to start a
  paid lever once the budget is exhausted and records why.

Individual flags in the dict override the preset, so ``{"profile": "balanced",
"deep_build": true}`` is valid.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, fields

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Which flags each preset turns ON. "off" is the empty set (all defaults False).
# A preset only enables flags whose feature is actually WIRED, so turning on a
# preset does exactly what it advertises.
#  * ``deep_build`` (C3) IS wired, but it is the one genuinely expensive lever (N
#    build attempts), so no preset enables it — it must be turned on explicitly
#    (ideally with a ``token_budget``) so a preset can never surprise-spend.
_PRESETS: dict[str, set[str]] = {
    "off": set(),
    "balanced": {
        "acted_gate",       # R2 — retry a worker that wrote nothing (saver)
        "test_gate",        # C1 — run repo tests, feed failures to triage (free)
        "research_map",     # R1 — index research folders, cut re-reads (saver on chains)
        "delta_rounds",     # R4 — continuation rounds inline less prior text (saver)
        "critic_triage",    # R3 — actionable closer revisions (free, opt-in path)
        "worker_escalate",  # R5 — an overwhelmed worker escalates instead of emitting junk
        "judgment_ledger",  # R11 — force explicit hard-call resolution (free, prompt-only)
    },
    "thorough": {
        "acted_gate", "test_gate", "research_map", "delta_rounds", "critic_triage",
        "worker_escalate", "judgment_ledger",
        "coverage_check",   # C5 — judge final state vs the approved plan (one LLM call)
        "git_snapshots",    # R6 — git init + per-round commits for research folders (free)
        "source_corpus",    # R10 — save + index full fetched pages (depth without context blowup)
        "rubric_contract",  # R9 — derive a completeness checklist + score coverage against it
        "intent_brief",     # R12 — clarify the task into an editable brief before routing (one LLM call)
        "consistency_check",  # cross-section identity/arithmetic verify (one extract call + pure code)
        "facts_ledger",     # shared canonical-values store + `facts` tool (free, no LLM)
        "constraints_contract",  # hard-rules contract: derive once, persist, validate (1-2 LLM calls)
    },
    # claim_check (R8) is the paid research lever (spawns a web-tool verifier + a
    # revision), like deep_build on the code side — no preset enables it; turn it
    # on explicitly, ideally with a token_budget.
}

_VALID_PROFILES = frozenset(_PRESETS)


@dataclass
class QualityProfile:
    """Per-run feature flags + cost knobs. All defaults reproduce today's behaviour."""

    profile: str = "off"

    # ── Code-side levers ──
    test_gate: bool = False        # C1: run the repo's tests after build/fix
    test_command: str = ""         # explicit test command; "" → auto-detect
    deep_build: bool = False       # C3: wrap the build in the Dubina escalation ladder
    coverage_check: bool = False   # C5: judge the final state against the approved plan

    # ── Research-side levers (Basna / Vatra) ──
    acted_gate: bool = False       # R2: one corrective retry when a worker wrote nothing
    research_map: bool = False     # R1: FTS index + cartographer pass over the VFS folder
    delta_rounds: bool = False     # R4: continuation rounds inline less prior text (search instead)
    critic_triage: bool = False    # R3: closer distils critic findings into an ordered checklist
    worker_escalate: bool = False  # R5: worker can flag ESCALATE → higher-tier re-dispatch
    git_snapshots: bool = False    # R6: git init the research folder + commit each round
    judgment_ledger: bool = False  # R11: force explicit enumeration+resolution of the hard calls
    source_corpus: bool = False    # R10: web_fetch saves full page text to the VFS, returns head+ptr
    claim_check: bool = False      # R8: tool-enabled fact-checker verifies the deliverable's claims
    rubric_contract: bool = False  # R9: derive a completeness checklist + score coverage against it
    intent_brief: bool = False     # R12: clarify the task into an editable brief before routing

    # ── Calibration posture (default-on guard + user-selectable mode) ──
    honesty_guard: bool = True     # anti-fabrication directives; the ONE default-on flag
    output_mode: str = ""          # "" (today) | "complete" | "conservative"

    # ── Deterministic consistency (docs/vatra-quality-tightening-plan.md §2) ──
    consistency_check: bool = False  # extract figures → verify identity/arithmetic in code

    # ── Shared facts ledger (plan §4 / increment 4) ──
    facts_ledger: bool = False     # canonical key→value store + `facts` tool for all workers

    # ── Constraints contract (plan §5 / increment 5) ──
    constraints_contract: bool = False  # derive hard rules once, persist per folder, validate

    # ── Cost discipline (shared) ──
    token_budget: int = 0          # <= 0 → unbounded (i.e. current behaviour)
    deep_build_samples: int = 2    # C3 pool size — kept small on purpose
    deep_build_fix_attempts: int = 1
    escalate_max: int = 2          # R5: cap re-dispatch escalations per run
    claim_check_max: int = 8       # R8: how many top claims the fact-checker live-verifies
    consistency_max_values: int = 40  # consistency_check: extraction cap (value occurrences)

    @classmethod
    def from_dict(cls, d: dict | None) -> QualityProfile:
        """Parse a ``quality`` config block. ``None``/``{}`` → all-off (today)."""
        d = d or {}
        profile = str(d.get("profile") or "off").lower()
        if profile not in _VALID_PROFILES:
            log.warning("unknown quality profile; treating as off", profile=profile)
            profile = "off"

        # Apply the preset, then let explicit keys override it either way.
        on = _PRESETS[profile]
        bool_flags = {
            "test_gate", "deep_build", "coverage_check", "acted_gate",
            "research_map", "delta_rounds", "critic_triage", "worker_escalate",
            "git_snapshots", "judgment_ledger", "source_corpus", "claim_check",
            "rubric_contract", "intent_brief", "consistency_check", "facts_ledger",
            "constraints_contract",
        }
        kw: dict = {"profile": profile}
        for name in bool_flags:
            kw[name] = bool(d[name]) if name in d else (name in on)

        # honesty_guard is the one default-ON flag: presets never touch it, only
        # an explicit key flips it (false → pre-guard prompts, byte-for-byte).
        kw["honesty_guard"] = bool(d["honesty_guard"]) if "honesty_guard" in d else True

        mode = str(d.get("output_mode") or "").lower()
        if mode not in ("", "complete", "conservative"):
            log.warning("unknown output_mode; ignoring", output_mode=mode)
            mode = ""
        kw["output_mode"] = mode

        def _int(key: str, default: int) -> int:
            # Use the default only when the key is ABSENT — an explicit 0/-1 must
            # clamp, not silently jump back to the default (the ``x or default`` trap).
            v = d.get(key, default)
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        kw["test_command"] = str(d.get("test_command") or "")
        kw["token_budget"] = max(0, _int("token_budget", 0))
        kw["deep_build_samples"] = max(1, _int("deep_build_samples", 2))
        kw["deep_build_fix_attempts"] = max(0, _int("deep_build_fix_attempts", 1))
        kw["escalate_max"] = max(0, _int("escalate_max", 2))
        kw["claim_check_max"] = max(1, _int("claim_check_max", 8))
        kw["consistency_max_values"] = max(1, _int("consistency_max_values", 40))
        return cls(**kw)

    _BOOL_FLAGS = (
        "test_gate", "deep_build", "coverage_check", "acted_gate",
        "research_map", "delta_rounds", "critic_triage", "worker_escalate",
        "git_snapshots", "judgment_ledger", "source_corpus", "claim_check",
        "rubric_contract", "intent_brief", "consistency_check", "facts_ledger",
        "constraints_contract",
    )

    @property
    def any_enabled(self) -> bool:
        """True if any feature flag is on — lets callers skip all new work cheaply."""
        return any(getattr(self, name) for name in self._BOOL_FLAGS)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class TokenBudget:
    """A per-run output-token ceiling shared by every quality lever in a run.

    ``total <= 0`` means unbounded — the default, i.e. exactly today's behaviour
    (no lever is ever refused for cost). When a positive total is set, a lever
    calls :meth:`can_afford` before doing paid work and :meth:`add` after; once
    spending reaches the ceiling further paid levers are refused and
    :attr:`stopped_reason` explains why. This never *interrupts* work in flight —
    it only prevents *starting* new paid work past the ceiling, so partial
    results are never lost.
    """

    def __init__(self, total: int = 0):
        self.total = max(0, int(total or 0))
        self._spent = 0
        self.stopped_reason = ""

    @property
    def unbounded(self) -> bool:
        return self.total <= 0

    def add(self, tokens: int) -> None:
        self._spent += max(0, int(tokens or 0))

    def spent(self) -> int:
        return self._spent

    def remaining(self) -> float:
        if self.unbounded:
            return math.inf
        return max(0, self.total - self._spent)

    def can_afford(self, estimate: int = 0) -> bool:
        """True if a lever costing ~``estimate`` output tokens may start."""
        if self.unbounded:
            return True
        if self._spent + max(0, int(estimate or 0)) > self.total:
            if not self.stopped_reason:
                self.stopped_reason = (
                    f"token budget reached ({self._spent}/{self.total}); "
                    "skipping further paid quality levers")
            return False
        return True

    def over(self) -> bool:
        return not self.unbounded and self._spent >= self.total


# ── R2: acted-gate for ensemble/collaborative workers ─────────────────
# A worker that narrates ("I'll analyse this…") but writes no file and returns
# no text burns its slot and gets judged as a fail — the weak/fast-tier failure
# mode Code already guards against in its build loop. This ports that gate to
# Basna/Vatra worker dispatch: one corrective retry when a worker produced
# literally nothing. Basna already backfills empty output *from generated files*,
# so this only fires when there is no file AND no text — genuinely nothing.

_WRITE_TOOLS = frozenset({
    "write", "edit", "file_write", "file_edit", "str_replace", "create_file", "apply_patch",
})

ACTED_CORRECTIVE = (
    "\n\n=== CORRECTIVE ===\n"
    "Your previous attempt produced NOTHING: no file was written and you "
    "returned no answer — only narration. Narration is not a contribution. "
    "Do the actual work now and return your real answer as text, or write your "
    "deliverable to a file in your working directory. Do not describe what you "
    "would do; do it.\n=== end corrective ===\n\n"
)


def worker_produced_nothing(result: dict | None) -> bool:
    """True when a dispatch returned empty text AND made no file-writing tool call.

    This is the only case the acted-gate retries: a pure-narration no-op. A worker
    that wrote a file (even with empty chat) is handled by the existing generated-
    file backfill, so it is NOT flagged here.
    """
    if not result:
        return True
    if (result.get("output") or "").strip():
        return False
    for a in result.get("actions") or []:
        if str(a.get("tool", "")).lower() in _WRITE_TOOLS:
            return False
    return True


# ── R5: worker escalation ─────────────────────────────────────────────
# Code's small→big promotion, generalized to research workers: a worker that
# recognizes the slice exceeds it flags ``ESCALATE: <reason>`` instead of emitting
# junk the merge must absorb. Here we re-drive the SAME agent once with a
# corrective that pushes it to deliver its best focused analysis (a full tier
# re-spawn is a heavier future refinement); the escalation also reads as a soft
# signal for reliability learning.

_ESCALATE_RE = re.compile(r"(?mi)^\s*ESCALATE\s*[:\-]\s*(.+)$")

ESCALATE_DIRECTIVE = (
    "\n\nSCOPE CHECK: if this slice is genuinely beyond a single focused pass — "
    "too broad, needs capabilities you lack, or you cannot do it justice — reply "
    "with one line `ESCALATE: <one-sentence reason>` INSTEAD of a half-answer. "
    "Only for a real overload; for normal work, just do it."
)

ESCALATE_CORRECTIVE = (
    "\n\n=== CORRECTIVE ===\nYou flagged this as too big. There is no bigger agent "
    "coming — YOU are the specialist. Narrow to the highest-value part you CAN do "
    "well and deliver concrete, substantive findings on that, clearly noting what "
    "you scoped out and why. Do not flag again; produce the best real contribution "
    "you can now.\n=== end corrective ===\n\n"
)


def escalate_reason(output: str | None) -> str | None:
    """The reason from an `ESCALATE: ...` line in a worker's output, or None."""
    m = _ESCALATE_RE.search(output or "")
    return m.group(1).strip()[:300] if m else None


# ── R11: judgment ledger ──────────────────────────────────────────────
# Weak models CAN make the hard boundary calls a strong model makes — they just
# don't volunteer them. Forcing each specialist to enumerate and resolve the
# hardest judgment calls in its scope is what turns implicit hedging into
# explicit, defensible determinations (the Fable-vs-Vatra gap was mostly this).
# Free — it's a prompt directive; the Deep critics then attack the ledger.

JUDGMENT_LEDGER_DIRECTIVE = (
    "\n\nHARD CALLS: end your contribution with a short **Judgment calls** section. "
    "List the 2–5 genuinely difficult or contestable decisions in your scope — "
    "boundary questions, close classifications, which of several plausible options "
    "applies, gaps where the evidence is thin — and for EACH state the call you made "
    "and one line of reasoning. Do not hide these in prose; make the determinations "
    "explicit so they can be checked. If a call is genuinely uncertain, say so and "
    "give your best-supported answer rather than omitting it."
)


# ── Honesty guard: unconfirmable specifics stay unconfirmed ────────────
# The failure mode that a "be thorough" push creates: a model rewarded for
# completeness states a specific it cannot support — a named individual, a
# role-holder, an origin, an exact figure or attribution — as established fact.
# The careful-model behaviour is to keep such a claim qualified. This is a free,
# prompt-only guard, ON by default via ``honesty_guard`` (the pipeline's own
# completeness pressure is always on, so the counterweight must be too; an
# explicit ``honesty_guard: false`` restores the old prompts). It complements
# the judgment ledger (R11): the ledger stops hedging-by-omission; this stops
# asserting-by-invention. Deliberately domain-agnostic — it names a CLASS of
# specific (any named office-holder), never a particular field.

UNVERIFIED_GUARD_DIRECTIVE = (
    "\n\nDO NOT ASSERT THE UNCONFIRMABLE: state a specific as established fact ONLY "
    "when you can support it from a source or the given inputs. For any load-bearing "
    "specific you cannot confirm — a named individual or role-holder (e.g. a person "
    "presented as an appointed officer), an origin or affiliation, an exact "
    "date/figure/identifier, or a specific attribution — either qualify it honestly "
    "(\"unconfirmed\", \"not independently verified\", \"reportedly\", or attributed to "
    "whoever claims it) or leave it out. Completeness never justifies inventing a "
    "specific: if no source names the holder of a role, write that the role is "
    "unconfirmed rather than supplying a name.\n"
    "PLACEHOLDERS & ESTIMATES: a fact only the requester can supply (their internal "
    "data, identifiers, financials, decisions) is never guessed — write "
    "[TO BE PROVIDED: <what>] so it is findable. Estimates are welcome but must be "
    "labeled with their basis: \"(estimate — basis: <one line>)\". Never invent an "
    "identifier, name, figure, or third-party entity to make a section look "
    "finished — a correct placeholder beats a plausible fabrication."
)

# ── Reporter/synthesis honesty overlay ─────────────────────────────────
# The assembly prompts push "resolve it and move on" ("don't narrate the
# disagreement", "finish it — completely") — exactly the pressure that converts a
# team's honest uncertainty into one confident wrong answer. This overlay is the
# exception clause: resolve what the evidence resolves, and surface what it
# doesn't in ONE labeled section instead of absorbing it silently. Appended at
# runtime when ``honesty_guard`` is on — the template files are not edited, so
# turning the guard off restores the original assembly prompts byte-for-byte.

REPORTER_HONESTY_DIRECTIVE = (
    "\n\nUNRESOLVED & ASSUMPTIONS: reconcile what the evidence actually resolves. "
    "When contributions contradict each other and the evidence does NOT settle it, "
    "or a part rests on an assumption, a labeled estimate, or a "
    "[TO BE PROVIDED: …] placeholder, make the best-supported call in the body "
    "text AND record it in a short final **Unresolved & assumptions** section — "
    "the item, the call you made, and what would confirm it. Do not silently "
    "absorb disagreements between specialists, and never pad a section with "
    "invented specifics to look finished. If nothing is unresolved, omit the "
    "section entirely."
)

# ── Output modes: the completeness-vs-correctness trade, user-selectable ─
# ``output_mode`` makes the scaffold's posture explicit instead of implicit:
# "conservative" is a review copy (fabricates nothing, reports what is unfilled),
# "complete" is a full draft (maximal coverage, every unknown labeled). The empty
# default emits nothing — today's behaviour.

CONSERVATIVE_MODE_DIRECTIVE = (
    "\n\nOUTPUT MODE — REVIEW COPY: correctness outranks completeness. State as "
    "fact only what a source or the given inputs support; everything else is a "
    "labeled estimate or a [TO BE PROVIDED: …] placeholder. Do not stretch thin "
    "evidence to fill a section — a shorter, verifiable deliverable is the goal, "
    "and unfilled items are reported openly, not hidden."
)

COMPLETE_MODE_DIRECTIVE = (
    "\n\nOUTPUT MODE — FULL DRAFT: deliver the most complete draft you can. Cover "
    "every part of the task; where a needed fact is unknown, keep momentum with a "
    "labeled estimate (\"estimate — basis: …\") or a [TO BE PROVIDED: …] "
    "placeholder rather than stopping — but completeness never justifies stating "
    "an unverified specific as fact."
)


def output_mode_directive(mode: str) -> str:
    """The prompt block for an ``output_mode`` value ("" → nothing, i.e. today)."""
    if mode == "conservative":
        return CONSERVATIVE_MODE_DIRECTIVE
    if mode == "complete":
        return COMPLETE_MODE_DIRECTIVE
    return ""


# ── Facts ledger directives (paired with the `facts` tool + facts_ledger.py) ─
# Workers get the write/read discipline; the reporter/synthesizer gets the
# "must match" rule plus the actual ledger dump (appended by the routes, since
# only they know the folder).

FACTS_LEDGER_DIRECTIVE = (
    "\n\nFACTS LEDGER: this run keeps a shared ledger of canonical values "
    "(tool: `facts`). Any load-bearing number, date, or identifier your piece "
    "establishes and others may reuse → `facts` action=set with a short "
    "snake_case key (e.g. total_budget_eur), the value, unit, status "
    "(verified|derived|estimated|assumed|to_be_completed) and provenance (URL, "
    "file, or 'derived from <keys>'). Any value another piece owns → `facts` "
    "action=get (or list) — NEVER restate a teammate's number from memory. If "
    "set reports a CONFLICT, do not overwrite: the ledger value stays canonical "
    "until you reconcile (check the source, or flag it in your Judgment calls / "
    "output). Ledger only load-bearing values, not trivia."
)

REPORTER_FACTS_DIRECTIVE = (
    "\n\n## Canonical facts ledger\n"
    "The table below is the run's shared ledger — the single source of truth "
    "for these values. Every figure, date, and identifier in the deliverable "
    "MUST match it. A value with status estimated/assumed is presented as such "
    "(never as plain fact); a to_be_completed value stays a labeled "
    "placeholder; an unresolved conflict goes to the Unresolved & assumptions "
    "section — never silently pick a side.\n"
)


# ── Per-run quality metrics (persisted into the session's analysis JSON) ─
# Everything below is already computed somewhere during a run — claim-check
# verdicts, consistency tallies, coverage gaps, retry counts — but nothing was
# stored where trends can be seen. One flat record per run, assembled by both
# engines through THIS function so the schema cannot drift between them.
# A lever that didn't run contributes no keys: the stored record shows exactly
# which checks a run actually had.

def build_quality_metrics(
    *,
    claim_findings: list[dict] | None = None,
    consistency: dict | None = None,
    gaps: list[dict] | None = None,
    contract: dict | None = None,
    acted_retries: int | None = None,
    escalations: int | None = None,
    budget: TokenBudget | None = None,
) -> dict:
    """The per-run quality tally for ``analysis.quality_metrics``. Pure; every
    part optional (pass ``None`` for a lever that didn't run). Returns {} when
    nothing ran, so callers can skip persisting an empty record."""
    out: dict = {}
    if claim_findings is not None:
        n = len(claim_findings)
        c = sum(1 for f in claim_findings if f.get("verdict") == "confirmed")
        r = sum(1 for f in claim_findings if f.get("verdict") == "refuted")
        # Mirrors research_verify.unconfirmed(): asserted-but-unconfirmable
        # specifics that carried a hedge rewrite.
        h = sum(1 for f in claim_findings
                if f.get("verdict") == "unverifiable" and f.get("hedge"))
        out.update(claims_checked=n, claims_confirmed=c, claims_refuted=r,
                   claims_unverifiable=n - c - r, claims_hedged=h)
    if consistency is not None:
        out.update(
            consistency_critical=int(consistency.get("critical", 0)),
            consistency_major=int(consistency.get("major", 0)),
            consistency_initial_critical=int(consistency.get("initial_critical", 0)),
            consistency_revised=bool(consistency.get("revised")),
        )
    if gaps is not None:
        out.update(
            gaps_major=sum(1 for g in gaps if str(g.get("severity")) == "major"),
            gaps_minor=sum(1 for g in gaps if str(g.get("severity")) != "major"),
        )
    if contract is not None:
        out.update(
            contract_checked=int(contract.get("checked", 0)),
            contract_failed_critical=int(contract.get("failed_critical", 0)),
            contract_failed_major=int(contract.get("failed_major", 0)),
            contract_unclear=int(contract.get("unclear", 0)),
        )
    if acted_retries is not None:
        out["acted_retries"] = int(acted_retries)
    if escalations is not None:
        out["escalations"] = int(escalations)
    if budget is not None:
        if budget.spent() > 0:
            out["budget_spent_tokens"] = budget.spent()
        if budget.stopped_reason:
            out["budget_stopped_reason"] = budget.stopped_reason
    return out


# ── R10: source corpus directive (paired with the web_fetch behaviour) ─
SOURCE_CORPUS_DIRECTIVE = (
    "\n\nSOURCES: this run keeps a shared source corpus. When you `web_fetch` a "
    "primary source, its FULL text is saved to `vfs:<project>/sources/` and you get "
    "a preview + pointer — read or `researchmap`-search the saved file for the rest "
    "rather than re-fetching. Fetch full pages one at a time for anything you will "
    "actually rely on or cite; use batch fetch only to triage many links quickly.\n"
    "CONTEXT DISCIPLINE: to read a web page ALWAYS use `web_fetch` — it offloads the "
    "full page to the corpus and returns only a lean preview, keeping your context "
    "small. Do NOT pull web content into your context with `shell` (curl / wget): "
    "piping raw page HTML into the conversation bloats your context fast and can "
    "overflow the model, losing the whole turn. Use `shell` for local files and "
    "commands, not to download web pages."
)
