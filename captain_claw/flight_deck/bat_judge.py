"""Bat — the honest done-judge (Invariant B).

"Finish at any cost" rewards *faking* done — stubbing a result, shrinking the
goal, claiming success — unless the done-verdict is something the working loop
cannot overrule. This module is that verdict. It is deliberately austere:

  * **Fail-closed.** Anything ambiguous — an unparseable panel vote, a timeout,
    an exception, a missing deliverable — counts as NOT done. Done is only ever
    returned on positive evidence.
  * **Evidence over narrative.** The panel judges the *deliverable* (and the
    deterministic check results), never the worker's own claim that it finished.
    The caller passes real evidence (file contents, command output, verifier
    findings); the worker's "all criteria satisfied" summary carries no weight.
  * **Independent.** The panel runs on a model tier the caller resolves
    separately from the workers (never self-grading). Model calls are injected,
    so this module stays pure and unit-testable.
  * **Two layers.** Layer 1 — deterministic checks (deliverable gate, tests,
    acceptance contract, consistency). A single *critical* failure blocks done
    outright, no panel needed. Layer 2 — an independent vote panel, aggregated
    with a tally that ignores abstentions and requires a real agree-majority.
  * **Anti-deadlock.** A stubborn loop + a flaky model veto could run to the
    spend cap forever. So: only deterministic hard failures may block done
    indefinitely. When the deterministic layer is all-green but the model panel
    keeps refusing, the veto is overridden after ``max_model_veto_rounds`` (the
    caller tracks the streak and passes it in), and the override is recorded.

This mirrors the house pattern (``research_consistency.run_check`` /
``quality_findings.run_gate``): the model is injected, the policy is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# A panel member: (deliverable, task, lens_index) -> {vote, reason}
PanelVoteFn = Callable[[str, str, int], Awaitable[dict]]

_VOTES = ("agree", "disagree", "abstain")


@dataclass
class Check:
    """One deterministic check result (Layer 1). ``critical`` failures block
    done; non-critical failures are advisory (surfaced, but don't block)."""

    name: str
    passed: bool
    critical: bool = True
    detail: str = ""


@dataclass
class Verdict:
    done: bool
    reason: str
    det_criticals: list[str] = field(default_factory=list)
    det_advisories: list[str] = field(default_factory=list)
    panel: dict[str, Any] = field(default_factory=dict)
    overridden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "reason": self.reason,
            "det_criticals": list(self.det_criticals),
            "det_advisories": list(self.det_advisories),
            "panel": dict(self.panel),
            "overridden": self.overridden,
        }


def _norm_vote(v: Any) -> str:
    s = str((v or {}).get("vote", "")).strip().lower() if isinstance(v, dict) else ""
    return s if s in _VOTES else "abstain"


def tally(votes: list[dict]) -> dict:
    """Aggregate panel votes. Abstentions never decide; a real agree-majority is
    required (ties and no-quorum are NOT done). Mirrors
    quality_profile.tally_votes semantics, kept local so this module is
    import-light and self-contained."""
    counts = {"agree": 0, "disagree": 0, "abstain": 0}
    for v in votes:
        counts[_norm_vote(v)] += 1
    agree, disagree = counts["agree"], counts["disagree"]
    if agree == 0 and disagree == 0:
        verdict = "no_quorum"
    elif agree > disagree:
        verdict = "agree"
    elif disagree > agree:
        verdict = "disagree"
    else:
        verdict = "tie"
    return {"verdict": verdict, "counts": counts, "margin": agree - disagree,
            "total": agree + disagree}


async def _safe_vote(fn: PanelVoteFn, deliverable: str, task: str, idx: int) -> dict:
    """Fail-closed wrapper: any error / missing / unparseable vote becomes a
    'disagree' (NOT done), never an abstain and never a silent agree."""
    try:
        res = await fn(deliverable, task, idx)
    except Exception as e:  # noqa: BLE001  — a judge that errors must not pass
        return {"vote": "disagree", "reason": f"judge error (fail-closed): {e}", "idx": idx}
    vote = _norm_vote(res)
    reason = str((res or {}).get("reason", "")).strip() if isinstance(res, dict) else ""
    # An abstention is not evidence of done; treat a bare abstain as a soft
    # disagree so it cannot help reach an agree-majority by reducing the
    # disagree count. (It already can't — abstains are dropped by tally — but we
    # also never let an all-abstain panel read as done: tally → no_quorum.)
    return {"vote": vote, "reason": reason, "idx": idx}


async def evaluate(
    *,
    task: str,
    deliverable: str,
    deterministic: list[Check] | None = None,
    panel_vote_fn: PanelVoteFn | None = None,
    panel_size: int = 3,
    prior_model_veto_rounds: int = 0,
    max_model_veto_rounds: int = 2,
    work_remaining: bool = True,
) -> Verdict:
    """Decide whether the task is genuinely done.

    Order:
      1. If any *critical* deterministic check failed → NOT done (fail-closed),
         no panel call.
      2. If a deliverable is required but empty/missing → NOT done.
      3. Run the independent panel (``panel_size`` members, each fail-closed) and
         tally. An agree-majority → done.
      4. Anti-deadlock: when the deterministic layer is all-green (at least one
         check, no criticals failing) but the panel refused, the veto is
         overridden — the panel may demand *more work* only while more work is
         possible. The override fires when the model-veto streak reaches
         ``max_model_veto_rounds``, OR immediately when ``work_remaining`` is
         False (every step is done and there is nothing left to retry, so a
         deterministically-complete deliverable must not be blocked forever).
         ``overridden=True`` records it. A critical deterministic failure (e.g. a
         failed step) is NEVER overridden — the run still fails closed.
    """
    deterministic = deterministic or []
    det_criticals = [c.name + (f" ({c.detail})" if c.detail else "")
                     for c in deterministic if c.critical and not c.passed]
    det_advisories = [c.name + (f" ({c.detail})" if c.detail else "")
                      for c in deterministic if not c.critical and not c.passed]

    if det_criticals:
        return Verdict(
            done=False,
            reason="deterministic critical check(s) failed: " + "; ".join(det_criticals),
            det_criticals=det_criticals, det_advisories=det_advisories,
        )

    if not (deliverable or "").strip():
        return Verdict(
            done=False, reason="no deliverable produced yet",
            det_criticals=det_criticals, det_advisories=det_advisories,
        )

    det_all_green = bool(deterministic) and not det_criticals

    if panel_vote_fn is None:
        # No independent panel configured. The only safe "done" is when a
        # non-empty deterministic layer is fully green; otherwise fail closed.
        if det_all_green:
            return Verdict(done=True, reason="deterministic checks all green (no panel configured)",
                           det_advisories=det_advisories)
        return Verdict(done=False, reason="no panel and no deterministic evidence of completion",
                       det_advisories=det_advisories)

    votes = [await _safe_vote(panel_vote_fn, deliverable, task, i) for i in range(max(1, panel_size))]
    t = tally(votes)
    panel = {"votes": votes, **t}

    if t["verdict"] == "agree":
        return Verdict(done=True, reason=f"panel agrees ({t['counts']['agree']}/{t['total']} with margin {t['margin']})",
                       det_advisories=det_advisories, panel=panel)

    # Panel refused. Anti-deadlock override when the deterministic layer is fully
    # green AND either the model has vetoed long enough OR there's no more work to
    # do (nothing the loop could change to satisfy the panel).
    if det_all_green and (not work_remaining or (prior_model_veto_rounds + 1) >= max_model_veto_rounds):
        why = ("no steps left to try" if not work_remaining
               else f"overridden after {prior_model_veto_rounds + 1} round(s)")
        return Verdict(
            done=True, overridden=True,
            reason=f"deterministic checks all green; model veto ({t['verdict']}) {why}",
            det_advisories=det_advisories, panel=panel,
        )

    return Verdict(
        done=False,
        reason=f"panel did not agree (verdict={t['verdict']}, margin={t['margin']})",
        det_advisories=det_advisories, panel=panel,
    )
