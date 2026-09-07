"""Deterministic blast-radius classifier (R5) — token-free, no LLM.

The autonomous loop already refuses to auto-run irreversible / outward-facing
actions (action_catalog.py risk/reversibility metadata, fd_dispatch.
should_auto_dispatch, config.AUTONOMY_HARD_EXCLUDE). This lifts that same
instinct OUTSIDE the loop: a small deterministic classifier that flags a
*high-blast-radius* operation — destruction / data loss, database migrations,
production / infra writes, git history rewrites (force-push), and money-like or
outward-facing actions — so an OPT-IN gate can route it to a human before it
runs (interactive tool guard) or before a plan auto-approves (Code studio
agent-initiated run).

Design:
  * Pure and deterministic: no LLM call, no side effects, no network. It never
    changes an output on its own — a caller decides what to do with a verdict.
  * LAYERS ALONGSIDE the existing LLM "is this suspicious?" guard; never
    replaces or weakens it.
  * The money / outward-facing vocabulary is drawn from
    ``config.AUTONOMY_HARD_EXCLUDE`` (a test asserts containment) so the two
    walls stay in lockstep; the destructive-command patterns are ones the loop
    never had to name (it can never reach raw shell).
  * Bias: when in doubt, FLAG. Both callers route a hit to human approval, so a
    false positive costs one confirmation; a false negative lets a high-impact
    op through unreviewed. For prose (plan text) that bias is acceptable
    precisely because the gate is opt-in and non-destructive.
"""

from __future__ import annotations

import re
from typing import Any

# Money / outward-facing subset of the autonomy hard-exclude vocabulary. These
# name-fragments are high blast radius wherever they run. Kept a subset of
# AUTONOMY_HARD_EXCLUDE (test-asserted) so the two safety walls can't drift.
_MONEY_OUTWARD: tuple[str, ...] = (
    "pay", "payment", "stripe", "checkout", "transfer", "wire",
    "tweet", "post_to", "social",
)

# Tool-argument fields carrying an executable command / query worth scanning.
# NOT content/body fields — a file write whose CONTENT mentions "DROP TABLE" is
# not itself a migration; scanning only command-shaped fields keeps the
# structured (tool-call) path low on false positives.
_COMMAND_FIELDS: tuple[str, ...] = (
    "command", "cmd", "query", "sql", "script", "statement",
)

# (regex, human-readable reason). Case-insensitive, most-specific first.
_PATTERN_SPECS: tuple[tuple[str, str], ...] = (
    # ── destruction / data loss ──
    (r"\brm\s+-[a-z]*[rf]", "recursive/forced file deletion (rm -rf)"),
    (r"\bgit\s+clean\s+-[a-z]*[dfx]", "git clean (deletes untracked files)"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard (discards work)"),
    (r"\bmkfs\b", "filesystem format (mkfs)"),
    (r"\bdd\b[^\n]*\bof=", "raw disk write (dd)"),
    (r"\btruncate\b", "truncate (file/table)"),
    (r">\s*/dev/(sd|nvme|disk)", "raw device write"),
    # ── git history rewrite / force-push ──
    (r"\bgit\s+push\b[^\n]*(--force\b|--force-with-lease\b|\s-f\b)",
     "git force-push (rewrites published history)"),
    # ── database migrations / destructive SQL ──
    (r"\bdrop\s+(table|database|schema|index)\b", "SQL DROP"),
    (r"\btruncate\s+table\b", "SQL TRUNCATE TABLE"),
    (r"\bdelete\s+from\b", "SQL DELETE"),
    (r"\balter\s+table\b", "SQL schema change (ALTER TABLE)"),
    (r"\b(?:db[:_ ]?migrate|migrat[a-z]*)\b", "database migration"),
    (r"\b(alembic|flyway|liquibase|prisma\s+migrate|knex\s+migrate|"
     r"rails\s+db:migrate|goose\s+up|dbmate)\b", "migration tool"),
    # ── production / infrastructure writes ──
    (r"\bkubectl\s+(delete|apply|drain|scale|cordon|rollout)\b",
     "kubectl cluster write"),
    (r"\bterraform\s+(apply|destroy)\b", "terraform infrastructure change"),
    (r"\bhelm\s+(install|upgrade|uninstall|delete|rollback)\b",
     "helm release change"),
    (r"\baws\s+[a-z0-9-]+\s+(delete|terminate|remove|put|create|update|modify)\b",
     "AWS resource write/delete"),
    (r"\bgcloud\s+[a-z0-9-]+\s+(delete|create|update)\b", "gcloud resource write"),
    # ── money / payments (prose + commands) ──
    (r"\b(wire|bank|funds?)\s+transfer\b", "money transfer"),
    (r"\b(payment|payout|refund|charge\s+the\s+card|stripe|paypal)\b",
     "payment / money movement"),
)

_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rx, re.IGNORECASE), reason) for rx, reason in _PATTERN_SPECS
)


def _scan_text(text: str) -> tuple[bool, str]:
    """Scan a free-text blob (a shell command or a plan's prose) for a
    high-blast-radius signature. Returns (hit, reason); (False, "") if none."""
    t = text or ""
    if not t.strip():
        return False, ""
    for rx, reason in _COMPILED:
        if rx.search(t):
            return True, reason
    return False, ""


def classify_tool(tool_name: str, arguments: dict[str, Any] | None) -> tuple[bool, str]:
    """Classify a concrete tool call for the interactive guard path. Flags
    (a) any tool whose NAME carries a money/outward fragment (pay, stripe, wire,
    tweet ...) and (b) any command-bearing tool whose command/query text matches
    a destructive / migration / infra / money pattern. Returns (hit, reason)."""
    name = (tool_name or "").lower()
    for frag in _MONEY_OUTWARD:
        if frag in name:
            return True, f"outward/irreversible tool ({frag})"
    args = arguments if isinstance(arguments, dict) else {}
    blobs: list[str] = []
    for key in _COMMAND_FIELDS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            blobs.append(v)
        elif isinstance(v, (list, tuple)):
            blobs.append(" ".join(str(x) for x in v))
    return _scan_text("\n".join(blobs))


def classify_plan(text: str) -> tuple[bool, str]:
    """Classify a Code-studio plan's text for the auto-approve path. Prose, so it
    leans toward flagging — a hit only means 'a human should approve this before
    it builds', never a hard refusal."""
    return _scan_text(text or "")
