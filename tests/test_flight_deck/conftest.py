"""Shared fixtures for the Flight Deck test package.

Production defaults new beings to the decomposed 'faculties' tick
(docs/being-faculties-plan.md). The existing tick tests, though, were written
against the single-prompt 'monolith' path — they assert the shape of that one
prompt and its exact gate call-counts. So pin conceive to 'monolith' for the
whole package; tests that exercise the faculties pipeline opt in explicitly via
``store.set_cognition(..., "faculties")``.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import beings as _beings


@pytest.fixture(autouse=True)
def _default_monolith_cognition(monkeypatch):
    # Read at call time by conceive/conceive_offspring, so this cleanly pins the
    # legacy path without touching every being-creation helper in the suite.
    monkeypatch.setattr(_beings, "DEFAULT_COGNITION", "monolith")
