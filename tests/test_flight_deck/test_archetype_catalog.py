"""The base archetype catalog: its shape, its shelves, and what planners get to see of it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from captain_claw.flight_deck.archetype_routes import ArchetypeBody, _validate
from captain_claw.flight_deck.archetypes import (
    CATEGORIES,
    archetype_category,
    disclaimer_rule,
    disclaimer_text,
    load_base_registry,
    shortlist_archetypes,
    with_disclaimer_rule,
)
from captain_claw.flight_deck.basna_routes import _build_catalog

REPO = Path(__file__).resolve().parents[2]
LOOK = (REPO / "flight-deck/src/components/layout/archetypeLook.ts").read_text(encoding="utf-8")

REGISTRY = load_base_registry()
ARCHETYPES = REGISTRY["archetypes"]
BY_ID = {a["id"]: a for a in ARCHETYPES}
MODES = {"neutra", "ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"}
SECTIONS = ["## Hard limits", "## Before you start", "## Tool usage", "## Standard Operating Procedure",
            "## Domain knowledge", "## Quality bar", "## Collaboration", "## Output"]
CRISIS_FAMILIES = {"Health & Wellbeing", "Growth & Relationships"}
CRISIS_LINES = ("112 in the EU", "999 or 112 in the UK", "911 in the US", "988 in the US",
                "116 123 Samaritans", "findahelpline.com")
# Written with the catalog's first 31, in the older, shorter form.
WRITTEN_SHORT = {
    "deep-researcher", "fact-checker", "market-scanner", "data-analyst", "report-builder", "editor-writer",
    "comms-outbound", "social-repurposer", "project-coordinator", "inbox-manager", "triage-router",
    "monitor-watchdog", "concierge", "code-implementer", "code-reviewer", "code-cartographer", "debugger",
    "git-operator", "light-planner", "long-horizon-planner", "qa-engineer", "quick-dirty", "simplifier",
    "security-reviewer", "architect", "brand-design-reviewer", "ui-reviewer", "visual-extractor",
    "deal-screener", "ic-prep", "portfolio-watch",
}
DETAILED = [a for a in ARCHETYPES if a["id"] not in WRITTEN_SHORT]


def _section(text: str, name: str) -> str:
    start = text.find(name)
    rest = text[start + len(name):]
    end = re.search(r"^## ", rest, re.M)
    return rest[:end.start()] if end else rest


def test_the_catalog_has_all_three_shelves():
    assert len(BY_ID) == len(ARCHETYPES), "duplicate ids"
    counts = {c: sum(a.get("category") == c for a in ARCHETYPES) for c in CATEGORIES}
    assert sum(counts.values()) == len(ARCHETYPES), "an archetype has no valid category"
    assert counts["specialist"] >= 70 and counts["personal"] >= 20
    assert len(DETAILED) == 100


def test_the_first_31_are_the_generalists_and_the_100_after_them_specialists():
    # The gallery's first split: Generalists and Specialists, which a person may want
    # for work or for their own life. Company archetypes are an admin's, never built in.
    assert {a["id"] for a in ARCHETYPES if a["category"] == "generalist"} == WRITTEN_SHORT
    assert len(WRITTEN_SHORT) == 31
    assert all(a["category"] in ("specialist", "personal") for a in DETAILED)
    assert not any(a["category"].startswith("group-") for a in ARCHETYPES)
    # The current families stay: a shelf is above a family, never instead of one.
    assert all(a.get("family") for a in ARCHETYPES)
    for family in ("Data & Analysis", "Engineering", "Ops & Coordination", "Research & Intelligence",
                   "Vision & Multimodal", "Writing & Comms", "Investment & VC"):
        assert any(a["family"] == family for a in ARCHETYPES), family


@pytest.mark.parametrize("a", ARCHETYPES, ids=lambda a: a["id"])
def test_every_archetype_can_be_spawned(a):
    assert a["tier"] in REGISTRY["tiers"]
    assert a["cognitive_mode"] in MODES
    assert a["role"] and a["description"] and a["fleet_instructions"]
    # Read-only roles among the first 31 go without write, edit and the like on purpose.
    assert {"shell", "glob", "web_search", "web_fetch", "insights"} <= set(a["tools"])
    assert a["id"] in WRITTEN_SHORT or set(REGISTRY["base_tools"]) <= set(a["tools"])
    assert "gws" not in a["tools"]
    assert a["lead"] in (True, False)
    assert 0 < a["reliability_seed"] <= 1


@pytest.mark.parametrize("a", DETAILED, ids=lambda a: a["id"])
def test_detailed_instructions_keep_their_shape(a):
    text = a["fleet_instructions"]
    assert re.match(r"You are an? ", text)
    positions = [text.find(s) for s in SECTIONS]
    assert -1 not in positions and positions == sorted(positions), "sections missing or out of order"
    assert text.find("## Hard limits") < 1200, "weak models get this clipped from the end: limits go first"
    assert 2500 <= len(text) <= 6100
    assert "—" not in text and "—" not in a["description"]
    assert not re.search(r"claude-|gpt-\d|gemini-\d", text, re.I), "tiers pick the model, never the text"
    assert len(a["description"]) <= 100
    # Every tool it names, it has.
    tool_names = {t for x in ARCHETYPES for t in x["tools"]} | {"knowledge"}
    for name in re.findall(r"`([a-z][a-z0-9_-]*)`", text):
        if name in tool_names:
            assert name in a["tools"] or name == "knowledge" or name in REGISTRY["base_tools"], name
    # Every colleague it names exists.
    for name in re.findall(r"`([a-z][a-z0-9_-]*)`", _section(text, "## Collaboration")):
        assert name in BY_ID or name in tool_names, name


@pytest.mark.parametrize("a", DETAILED, ids=lambda a: a["id"])
def test_keywords_are_specific_enough_to_match_on(a):
    vague = {"report", "reports", "review", "plan", "planning", "data", "email", "team", "tax", "help", "write",
             "analysis", "business", "project", "work", "research", "strategy", "marketing", "sales", "money"}
    assert 6 <= len(a["keywords"]) <= 12
    for keyword in a["keywords"]:
        assert keyword == keyword.lower() and len(keyword) >= 3 and keyword not in vague, keyword


@pytest.mark.parametrize("a", [x for x in ARCHETYPES if x["family"] in CRISIS_FAMILIES], ids=lambda a: a["id"])
def test_wellbeing_archetypes_lead_with_the_crisis_rule(a):
    limits = _section(a["fleet_instructions"], "## Hard limits")
    first = next(line for line in limits.splitlines() if line.strip().startswith(("-", "*")))
    for line in CRISIS_LINES:
        assert line in first, line


def test_the_reflection_companion_says_what_it_is_not():
    assert "not therapy" in BY_ID["reflection-companion"]["fleet_instructions"][:900]


@pytest.mark.parametrize("a", [x for x in DETAILED if {"google_mail", "microsoft_mail", "send_mail"} & set(x["tools"])],
                         ids=lambda a: a["id"])
def test_mail_stays_the_owners_to_start(a):
    assert re.search(r"only when|never (send|start)", a["fleet_instructions"], re.I)


@pytest.mark.parametrize("a", ARCHETYPES, ids=lambda a: a["id"])
def test_every_archetype_has_its_own_look(a):
    assert f"'{a['id']}':" in LOOK, "no icon in archetypeLook.ts"
    assert f"'{a['family']}':" in LOOK, "no family look in archetypeLook.ts"


# ── Disclaimers ──

LEGAL_OUTSIDE_THE_FAMILY = {"life-admin", "hr-advisor", "ai-act-officer", "term-sheet-analyst", "lease-abstractor"}


def test_every_legal_and_health_archetype_has_its_disclaimer():
    for a in ARCHETYPES:
        expected = ("legal" if a["family"] == "Legal" or a["id"] in LEGAL_OUTSIDE_THE_FAMILY
                    else "health" if a["family"] == "Health & Wellbeing"
                    else "clinical" if a["family"] == "Healthcare & Life Sciences"
                    else None)
        assert a.get("disclaimer") == expected, a["id"]


@pytest.mark.parametrize("a", [x for x in ARCHETYPES if x.get("disclaimer")], ids=lambda a: a["id"])
def test_the_agent_says_the_same_line_the_deck_shows(a):
    text = REGISTRY["disclaimers"][a["disclaimer"]]
    instructions = a["fleet_instructions"]
    assert disclaimer_rule(text) in _section(instructions, "## Hard limits")
    # Weak models get the first 1,500 characters: the whole line has to be in them.
    assert instructions.find(text) + len(text) < 1500
    # Never stamped on what goes out: a redline or a complaint letter stays clean.
    assert "Keep it out of documents meant for others" in instructions


def test_the_lines_say_what_they_must():
    lines = REGISTRY["disclaimers"]
    assert "Not legal advice" in lines["legal"] and "lawyer" in lines["legal"]
    assert "Not medical" in lines["health"] and "112" in lines["health"]
    assert "qualified professional" in lines["clinical"] and "Not medical advice" in lines["clinical"]
    assert all("—" not in line for line in lines.values())


def test_an_edited_archetype_gets_the_rule_back_and_an_intact_one_is_left_alone():
    line = disclaimer_text("legal")
    assert with_disclaimer_rule("Do the work.", "legal") == f"{disclaimer_rule(line)}\n\nDo the work."
    assert with_disclaimer_rule(f"Say: {line}", "legal") == f"Say: {line}"
    assert with_disclaimer_rule("Do the work.", "") == "Do the work."
    assert with_disclaimer_rule("Do the work.", "nonsense") == "Do the work."
    assert disclaimer_text("nonsense") == ""


def test_a_disclaimer_is_kept_on_save_and_an_unknown_one_is_dropped():
    assert _validate(ArchetypeBody(role="X", disclaimer="health"))["disclaimer"] == "health"
    assert _validate(ArchetypeBody(role="X", disclaimer="nonsense"))["disclaimer"] == ""
    assert _validate(ArchetypeBody(role="X"))["disclaimer"] == ""


def _spawn_stubs(monkeypatch, archetypes):
    import types

    import captain_claw.flight_deck.archetypes as arch_mod
    import captain_claw.flight_deck.auth as fd_auth
    import captain_claw.flight_deck.basna_routes as basna_mod

    async def fake_merged(db, uid):
        return list(archetypes)

    async def fake_owner_tiers(db, uid):
        return {}, []

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    monkeypatch.setattr(fd_auth, "get_db", lambda: object())
    monkeypatch.setattr(basna_mod, "_load_owner_tiers", fake_owner_tiers)
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=""))


async def test_a_spawn_records_the_disclaimer_and_restores_a_dropped_rule(monkeypatch):
    from captain_claw.flight_deck import server

    request = _spawn_stubs(monkeypatch, [
        {"id": "my-lawyer", "role": "My Lawyer", "tools": [], "fleet_instructions": "Do the work.", "disclaimer": "legal"},
        {"id": "my-writer", "role": "My Writer", "tools": [], "fleet_instructions": "Write."},
    ])
    with_one = server.AgentConfig(name="x", archetype="my-lawyer", provider="anthropic", model="m")
    await server._resolve_archetype(with_one, request, None)
    assert with_one._disclaimer == "legal"
    assert disclaimer_text("legal") in with_one.fleet_instructions
    assert with_one.fleet_instructions.endswith("Do the work.")

    without = server.AgentConfig(name="y", archetype="my-writer", provider="anthropic", model="m")
    await server._resolve_archetype(without, request, None)
    assert without._disclaimer == ""
    assert without.fleet_instructions == "Write."


async def test_a_built_in_archetypes_instructions_are_not_doubled_at_spawn(monkeypatch):
    from captain_claw.flight_deck import server

    built_in = BY_ID["contract-reviewer"]
    request = _spawn_stubs(monkeypatch, [built_in])
    config = server.AgentConfig(name="x", archetype="contract-reviewer", provider="anthropic", model="m")
    await server._resolve_archetype(config, request, None)
    assert config.fleet_instructions == built_in["fleet_instructions"]


# ── What planners see ──

GENERALISTS = {a["id"] for a in ARCHETYPES if a["category"] == "generalist"}


def _ids(intent: str, **kw) -> set[str]:
    return {a["id"] for a in shortlist_archetypes(intent, [{**a, "source": "base"} for a in ARCHETYPES], **kw)}


@pytest.mark.parametrize("intent, expected", [
    ("Redline this NDA against our contract playbook", "contract-reviewer"),
    ("Reconcile our bank statement for October", "bookkeeper"),
    ("We got a data subject access request, what is the GDPR deadline?", "privacy-officer"),
    ("Plan a two week trip to Japan with the kids", "trip-planner"),
    ("Is this message a scam? It says my parcel is held", "tech-helper"),
    ("Check our website against WCAG 2.2 accessibility", "accessibility-auditor"),
])
def test_a_request_reaches_its_specialist(intent, expected):
    assert expected in _ids(intent), intent


def test_planners_keep_the_generalists_and_a_bounded_number_of_specialists():
    ids = _ids("contract review, gdpr privacy, bank reconciliation, vat return, seo audit, recruiter, hr policy, "
               "lease abstract, medical coding, trademark search, cap table, press release, grant proposal, "
               "meal plan, fitness plan, tutor, budget, tax documents, career change, translation")
    assert GENERALISTS <= ids
    assert len(ids - GENERALISTS) <= 15


def test_a_request_with_no_match_falls_back_to_the_generalists():
    assert _ids("Pripremi mi sazetak sastanka") == GENERALISTS


def test_kept_and_custom_archetypes_are_always_shown():
    custom = {"id": "my-own", "role": "My Own", "family": "Custom", "description": "x", "source": "user"}
    base = [{**a, "source": "base"} for a in ARCHETYPES]
    ids = {a["id"] for a in shortlist_archetypes("nothing relevant", [*base, custom], keep=["game-master"])}
    assert {"my-own", "game-master"} <= ids


def test_the_planner_prompt_stays_about_the_size_it_was():
    base = [{**a, "source": "base"} for a in ARCHETYPES]
    everything = _build_catalog(base, {})
    assert "contract-reviewer" not in everything
    assert "contract-reviewer" in _build_catalog(base, {}, "review this contract")
    assert "game-master" in _build_catalog(base, {}, "zzz", ["game-master"])
    assert len(everything) < 12_000


# ── The field itself ──

def test_a_category_is_kept_on_save_and_an_unknown_one_becomes_a_specialist():
    assert _validate(ArchetypeBody(role="X", category="personal"))["category"] == "personal"
    assert _validate(ArchetypeBody(role="X", category="nonsense"))["category"] == "specialist"
    assert _validate(ArchetypeBody(role="X"))["category"] == "specialist"
    assert archetype_category({}) == "specialist"
    assert archetype_category({"category": "generalist"}) == "generalist"


def test_a_row_saved_when_specialists_were_called_focused_is_still_a_specialist():
    assert archetype_category({"category": "focused"}) == "specialist"
    assert _validate(ArchetypeBody(role="X", category="focused"))["category"] == "specialist"


def test_a_persons_own_archetype_never_sits_in_a_company_group():
    # It would read as the company's to them, and nobody else would see it there.
    assert _validate(ArchetypeBody(role="X", category="group-dental"))["category"] == "specialist"
    assert _validate(ArchetypeBody(role="X", category="group-dental"), company=True)["category"] == "group-dental"


def test_a_base_archetype_on_any_tier_can_be_saved():
    for tier in REGISTRY["tiers"]:
        assert _validate(ArchetypeBody(role="X", tier=tier))["tier"] == tier


def test_the_catalog_file_is_plain_utf8_json():
    data = json.loads((REPO / "captain_claw/instructions/archetypes.json").read_text(encoding="utf-8"))
    assert data["version"] >= 3
