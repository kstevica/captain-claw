"""Anonymization: find private data and replace it with tokens.

* :mod:`.detectors` — the deterministic lane (patterns, checksums, labels).
* :mod:`.engine` — what every lane shares: spans → entities → tokens →
  output → verify, and restore.
* :mod:`.formats` — tables and JSON cell by cell, written back as they came.

The vault (where the real values are kept, encrypted) lives in Flight Deck:
:mod:`captain_claw.flight_deck.anonymize_vault`. Plan: docs/anonymization-plan.md.
"""

from captain_claw.anonymize.detectors import KINDS
from captain_claw.anonymize.engine import (
    TOKEN_RE,
    Entity,
    Options,
    RunResult,
    Segment,
    VaultState,
    VerifyFailed,
    anonymize_filename,
    anonymize_text,
    make_token,
    restore,
    run,
    summary,
)

__all__ = [
    "KINDS", "TOKEN_RE", "Entity", "Options", "RunResult", "Segment", "VaultState", "VerifyFailed",
    "anonymize_filename", "anonymize_text", "make_token", "restore", "run", "summary",
]
