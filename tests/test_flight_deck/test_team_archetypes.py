"""Archetypes an admin generates and publishes for the whole deck."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import agent_forge as af
from captain_claw.flight_deck import team_archetypes as ta
from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.archetype_routes import router as archetype_router
from captain_claw.flight_deck.archetypes import (
    CRISIS_RULE,
    TEAM_SETTING,
    archetype_category,
    disclaimer_text,
    load_base_registry,
    merge_archetypes,
    merged_archetypes,
    place_safety_rules,
    shortlist_archetypes,
)
from captain_claw.flight_deck.auth import get_current_user

pytestmark = pytest.mark.asyncio

REPO = Path(__file__).resolve().parents[2]
ADMIN = {"id": "admin-1", "role": "admin", "email": "a@x.co"}
MEMBER = {"id": "member-1", "role": "user", "email": "m@x.co"}
WRITTEN = "You are a Treatment Plan Explainer.\n\n## Hard limits\n- Never advise a treatment.\n\n" + "## Output\n" + ("A clear letter. " * 80)
OUTLINE = {
    "group": {"label": "Dental Practice", "hint": "For the people running a practice."},
    "family": {"name": "Practice Administration", "icon": "Stethoscope", "color": "teal"},
    "archetypes": [
        {"role": "Treatment Plan Explainer", "description": "Explains plans to patients.", "tier": "balanced",
         "cognitive_mode": "dorian", "icon": "FileText", "keywords": ["Treatment Plan Letter"], "tools": ["docx", "made_up_tool"],
         "disclaimer": "clinical"},
        {"role": "Claims Letter Drafter", "description": "Drafts insurer letters.", "tier": "nonsense",
         "cognitive_mode": "nonsense", "icon": "NotAnIcon", "keywords": [], "tools": [], "disclaimer": "nonsense"},
    ],
}


class Deck:
    def __init__(self):
        self.settings: dict[str, str] = {}
        self.user = ADMIN
        self.reply = json.dumps(OUTLINE)
        self.cut = False
        self.prompts: list[str] = []

    async def get_system_setting(self, key):
        return self.settings.get(key)

    async def set_system_setting(self, key, value):
        self.settings[key] = value

    async def list_user_archetypes(self, uid):
        return []

    async def list_shared_archetypes(self, uid):
        return []


@pytest.fixture
def deck(monkeypatch):
    d = Deck()

    class Provider:
        async def complete(self, messages, **kw):
            d.prompts.append(messages[-1].content)
            return type("R", (), {"content": d.reply, "finish_reason": "length" if d.cut else "stop"})()

    async def tiers(db, uid):
        return {}, []

    async def not_spent(uid):
        return None

    for module in ("captain_claw.flight_deck.team_archetypes", "captain_claw.flight_deck.archetype_routes"):
        monkeypatch.setattr(f"{module}.get_db", lambda: d)
    monkeypatch.setattr(vr, "_load_owner_tiers", tiers)
    monkeypatch.setattr(vr, "_load_registry", lambda: {})
    monkeypatch.setattr(vr, "_resolve_creds", lambda *a, **k: {"provider": "x", "model": "m"})
    monkeypatch.setattr(vr, "_provider_call", lambda *a, **k: (Provider(), 8192))
    monkeypatch.setattr(af, "_refuse_when_spent", not_spent)
    return d


def _client(deck) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(archetype_router)
    app.include_router(ta.router)
    app.dependency_overrides[get_current_user] = lambda: deck.user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test")


def _pack(**over) -> dict:
    return {
        "group": {"label": "Dental Practice", "hint": "For the practice."},
        "family": {"name": "Practice Administration", "icon": "Stethoscope", "color": "teal"},
        "archetypes": [{"role": "Treatment Plan Explainer", "description": "Explains plans.", "tier": "balanced",
                        "cognitive_mode": "dorian", "tools": ["docx"], "fleet_instructions": WRITTEN,
                        "keywords": ["treatment plan letter"], "icon": "FileText", "disclaimer": "clinical"}],
        **over,
    }


# ── Who can do it ──

@pytest.mark.parametrize("method, path, kw", [
    ("post", "/fd/archetypes/team/outline", {"data": {"instructions": "x"}}),
    ("post", "/fd/archetypes/team/write", {"json": {"role": "x"}}),
    ("post", "/fd/archetypes/team", {"json": _pack()}),
    ("put", "/fd/archetypes/team/x", {"json": {"role": "x"}}),
    ("delete", "/fd/archetypes/team/x", {}),
    ("delete", "/fd/archetypes/team/groups/x", {}),
])
async def test_only_an_admin_can_change_what_everyone_sees(deck, method, path, kw):
    deck.user = MEMBER
    async with _client(deck) as c:
        resp = await getattr(c, method)(path, **kw)
    assert resp.status_code == 403
    assert TEAM_SETTING not in deck.settings
    assert deck.prompts == []


# ── Generating ──

async def test_an_outline_is_made_safe_and_saves_nothing(deck):
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team/outline", data={"instructions": "Dental practices", "count": "2"})
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["group"]["label"] == "Dental Practice" and out["family"]["icon"] == "Stethoscope"
    first, second = out["archetypes"]
    assert first["keywords"] == ["treatment plan letter"]
    assert "docx" in first["tools"] and "made_up_tool" not in first["tools"]
    assert set(load_base_registry()["base_tools"]) <= set(first["tools"])
    assert second["tier"] == "balanced" and second["cognitive_mode"] == "neutra"
    assert second["icon"] == "" and second["disclaimer"] == ""
    assert TEAM_SETTING not in deck.settings
    assert "Design exactly 2 archetypes." in deck.prompts[0] and "Stethoscope" in deck.prompts[0]


async def test_an_outline_needs_something_to_go_on(deck):
    async with _client(deck) as c:
        assert (await c.post("/fd/archetypes/team/outline", data={"instructions": "  "})).status_code == 400


@pytest.mark.parametrize("reply, cut, says", [
    ("not json", False, "unreadable"), ('{"group":', True, "cut off"), ('{"archetypes": []}', False, "no archetypes"),
])
async def test_a_bad_outline_says_what_went_wrong(deck, reply, cut, says):
    deck.reply, deck.cut = reply, cut
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team/outline", data={"instructions": "x"})
    assert resp.status_code == 502 and says in resp.json()["detail"]


async def test_an_outline_never_exceeds_a_set(deck):
    deck.reply = json.dumps({**OUTLINE, "archetypes": [{"role": f"Role {i}"} for i in range(40)]})
    async with _client(deck) as c:
        out = (await c.post("/fd/archetypes/team/outline", data={"instructions": "x"})).json()
    assert len(out["archetypes"]) == ta.MAX_PACK


async def test_writing_places_the_rules_a_model_does_not_get_to_word(deck):
    deck.reply = WRITTEN
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team/write", json={"role": "Explainer", "disclaimer": "health", "others": [{"role": "Claims Drafter", "description": "x"}]})
    assert resp.status_code == 200, resp.text
    text = resp.json()["fleet_instructions"]
    limits = text[text.index("## Hard limits"):]
    assert limits.index(CRISIS_RULE) < limits.index(disclaimer_text("health")) < limits.index("Never advise")
    assert TEAM_SETTING not in deck.settings
    assert "`claims-drafter`" in deck.prompts[0]


@pytest.mark.parametrize("reply, cut", [("too short", False), (("No limits section. " * 80), False), (WRITTEN, True)])
async def test_a_bad_write_is_refused_so_it_can_be_tried_again(deck, reply, cut):
    deck.reply, deck.cut = reply, cut
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team/write", json={"role": "Explainer"})
    assert resp.status_code == 502 and "Try again" in resp.json()["detail"]


async def test_rules_already_there_are_not_doubled():
    once = place_safety_rules(WRITTEN, "legal")
    assert place_safety_rules(once, "legal") == once
    assert place_safety_rules(WRITTEN, "") == WRITTEN


# ── Publishing, and everyone seeing it ──

async def test_a_published_group_reaches_every_user(deck):
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team", json=_pack())
        assert resp.status_code == 201, resp.text
        assert resp.json()["group"] == "group-dental-practice"
        deck.user = MEMBER
        reg = (await c.get("/fd/archetypes")).json()
        light = (await c.get("/fd/archetypes?light=true")).json()
    assert reg["groups"] == [{"id": "group-dental-practice", "label": "Dental Practice", "hint": "For the practice."}]
    assert reg["family_looks"] == [{"name": "Practice Administration", "icon": "Stethoscope", "color": "teal"}]
    made = next(a for a in reg["archetypes"] if a["id"] == "treatment-plan-explainer")
    assert made["source"] == "team" and made["category"] == "group-dental-practice"
    assert made["family"] == "Practice Administration" and made["icon"] == "FileText" and made["disclaimer"] == "clinical"
    assert made["fleet_instructions"] == place_safety_rules(WRITTEN, "clinical")
    # The built-in catalog is all still there.
    assert len(reg["archetypes"]) == len(load_base_registry()["archetypes"]) + 1
    assert "fleet_instructions" not in next(a for a in light["archetypes"] if a["id"] == "treatment-plan-explainer")
    assert light["groups"] == reg["groups"]


async def test_a_spawn_can_use_it(deck):
    async with _client(deck) as c:
        await c.post("/fd/archetypes/team", json=_pack())
    archetypes = await merged_archetypes(deck, MEMBER["id"])
    assert next(a for a in archetypes if a["id"] == "treatment-plan-explainer")["source"] == "team"


async def test_publishing_into_what_exists(deck):
    async with _client(deck) as c:
        into_shelf = await c.post("/fd/archetypes/team", json=_pack(group={"id": "focused"}, family={"name": "Legal"}))
        assert into_shelf.json()["group"] == "focused" and into_shelf.json()["family"] == "Legal"
        stored = json.loads(deck.settings[TEAM_SETTING])
        # A built-in family keeps its look, and no shelf was made.
        assert stored["groups"] == [] and stored["families"] == []
        same_name = await c.post("/fd/archetypes/team", json=_pack(group={"label": "focused"}))
        assert same_name.json()["group"] == "focused"
        first = await c.post("/fd/archetypes/team", json=_pack(group={"label": "Dental"}))
        again = await c.post("/fd/archetypes/team", json=_pack(group={"label": "dental"}))
    assert first.json()["group"] == again.json()["group"] == "group-dental"
    assert len(json.loads(deck.settings[TEAM_SETTING])["groups"]) == 1


async def test_a_publish_never_takes_a_built_in_or_published_id(deck):
    built_in = load_base_registry()["archetypes"][0]["role"]
    pack = _pack()
    pack["archetypes"] = [{**pack["archetypes"][0], "role": built_in}, {**pack["archetypes"][0], "role": built_in}]
    async with _client(deck) as c:
        ids = (await c.post("/fd/archetypes/team", json=pack)).json()["ids"]
    base_id = load_base_registry()["archetypes"][0]["id"]
    assert base_id not in ids and len(set(ids)) == 2


@pytest.mark.parametrize("change, status, says", [
    ({"group": {"label": ""}}, 400, "group needs a name"),
    ({"group": {"label": "!!!"}}, 400, "letters or numbers"),
    ({"family": {"name": " "}}, 400, "family needs a name"),
    ({"archetypes": []}, 400, "nothing to publish"),
])
async def test_a_bad_publish_is_refused_and_leaves_nothing(deck, change, status, says):
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team", json=_pack(**change))
    assert resp.status_code == status and says in resp.json()["detail"]
    assert TEAM_SETTING not in deck.settings


async def test_an_unwritten_archetype_cannot_be_published(deck):
    pack = _pack()
    pack["archetypes"][0]["fleet_instructions"] = "short"
    async with _client(deck) as c:
        resp = await c.post("/fd/archetypes/team", json=pack)
    assert resp.status_code == 400 and "no instructions" in resp.json()["detail"]
    assert TEAM_SETTING not in deck.settings


async def test_tools_and_looks_are_checked_at_publish_too(deck):
    pack = _pack(family={"name": "New One", "icon": "NotAnIcon", "color": "not-a-colour"})
    pack["archetypes"][0].update(tools=["docx", "made_up_tool", "gws"], icon="NotAnIcon")
    async with _client(deck) as c:
        assert (await c.post("/fd/archetypes/team", json=pack)).status_code == 201
    stored = json.loads(deck.settings[TEAM_SETTING])
    made = stored["archetypes"][0]
    assert "made_up_tool" not in made["tools"] and "gws" not in made["tools"] and "docx" in made["tools"]
    assert made["icon"] == ""
    assert stored["families"] == [{"name": "New One", "icon": "Sparkles", "color": "violet"}]


# ── Changing and removing ──

async def test_an_edit_reaches_everyone_and_keeps_its_id_and_icon(deck):
    async with _client(deck) as c:
        await c.post("/fd/archetypes/team", json=_pack())
        resp = await c.put("/fd/archetypes/team/treatment-plan-explainer", json={
            "role": "Renamed", "family": "Practice Administration", "category": "group-dental-practice",
            "fleet_instructions": "New text.", "tier": "reason",
        })
        assert resp.status_code == 200, resp.text
        deck.user = MEMBER
        made = next(a for a in (await c.get("/fd/archetypes")).json()["archetypes"] if a["id"] == "treatment-plan-explainer")
    assert made["role"] == "Renamed" and made["fleet_instructions"] == "New text." and made["icon"] == "FileText"


async def test_a_disclaimer_chosen_after_writing_still_brings_its_rules(deck):
    """The admin can change it on the review screen or in the editor, after the writer ran."""
    pack = _pack()
    pack["archetypes"][0]["disclaimer"] = "health"
    async with _client(deck) as c:
        assert (await c.post("/fd/archetypes/team", json=pack)).status_code == 201
        made = next(a for a in (await c.get("/fd/archetypes")).json()["archetypes"] if a["id"] == "treatment-plan-explainer")
        limits = made["fleet_instructions"].split("## Hard limits", 1)[1]
        assert limits.index(CRISIS_RULE) < limits.index(disclaimer_text("health"))
        resp = await c.put("/fd/archetypes/team/treatment-plan-explainer", json={
            "role": "Renamed", "family": "Practice Administration", "category": "group-dental-practice",
            "fleet_instructions": WRITTEN, "tier": "reason", "disclaimer": "legal",
        })
    assert resp.status_code == 200, resp.text
    edited = resp.json()["fleet_instructions"]
    assert disclaimer_text("legal") in edited.split("## Hard limits", 1)[1]
    assert CRISIS_RULE not in edited


async def test_removing_the_last_one_takes_its_group_and_family(deck):
    async with _client(deck) as c:
        await c.post("/fd/archetypes/team", json=_pack())
        assert (await c.delete("/fd/archetypes/team/treatment-plan-explainer")).status_code == 200
        assert (await c.delete("/fd/archetypes/team/treatment-plan-explainer")).status_code == 404
        reg = (await c.get("/fd/archetypes")).json()
    assert reg["groups"] == [] and reg["family_looks"] == []
    assert not any(a["id"] == "treatment-plan-explainer" for a in reg["archetypes"])


async def test_removing_a_group_takes_only_what_is_in_it(deck):
    async with _client(deck) as c:
        await c.post("/fd/archetypes/team", json=_pack())
        other = _pack(group={"label": "Other"}, family={"name": "Other Family"})
        other["archetypes"][0]["role"] = "Another Role"
        await c.post("/fd/archetypes/team", json=other)
        resp = await c.delete("/fd/archetypes/team/groups/group-dental-practice")
        assert resp.json() == {"removed": 1}
        assert (await c.delete("/fd/archetypes/team/groups/generalist")).status_code == 404
    stored = json.loads(deck.settings[TEAM_SETTING])
    assert [g["id"] for g in stored["groups"]] == ["group-other"]
    assert [f["name"] for f in stored["families"]] == ["Other Family"]
    assert [a["id"] for a in stored["archetypes"]] == ["another-role"]


# ── How it sits with everything else ──

async def test_a_persons_own_archetype_of_the_same_id_wins():
    base = [{"id": "a", "role": "Base"}]
    team = [{"id": "t", "role": "Team"}]
    rows = [{"archetype_id": "t", "data": json.dumps({"role": "Mine"})}]
    merged = {a["id"]: a for a in merge_archetypes(base, rows, None, team)}
    assert merged["t"]["role"] == "Mine" and merged["t"]["source"] == "user"
    assert merged["a"]["source"] == "base"


async def test_an_unreadable_setting_never_breaks_the_gallery(deck):
    deck.settings[TEAM_SETTING] = "{not json"
    assert len(await merged_archetypes(deck, MEMBER["id"])) == len(load_base_registry()["archetypes"])


async def test_planners_see_a_team_archetype_only_when_the_request_is_about_it():
    team = [{"id": "explainer", "role": "Treatment Plan Explainer", "family": "Practice", "category": "group-dental",
             "keywords": ["treatment plan letter"], "source": "team"}]
    mine = [{"id": "mine", "role": "Mine", "family": "Custom", "category": "focused", "source": "user"}]
    assert "explainer" in {a["id"] for a in shortlist_archetypes("write a treatment plan letter", [*team, *mine])}
    unrelated = {a["id"] for a in shortlist_archetypes("summarise this meeting", [*team, *mine])}
    assert "explainer" not in unrelated and "mine" in unrelated


async def test_a_team_group_is_a_shelf_and_a_typo_is_not():
    assert archetype_category({"category": "group-dental"}) == "group-dental"
    assert archetype_category({"category": "dental"}) == "focused"


async def test_the_deck_can_draw_every_look_it_offers():
    look = (REPO / "flight-deck/src/components/layout/archetypeLook.ts").read_text(encoding="utf-8")
    colors = re.search(r"COLOR_LOOKS: Record<string, Omit<FamilyLook, 'icon'>> = \{(.*?)\n\}", look, re.S).group(1)
    assert set(re.findall(r"^\s*(\w+): \{ text:", colors, re.M)) == set(ta.COLORS)
    icons = re.search(r"TEAM_ICONS: Record<string, LucideIcon> = \{([^}]*)\}", look).group(1)
    assert {i.strip() for i in icons.split(",") if i.strip()} == set(ta.ICONS)


async def test_the_prompts_ship_and_keep_their_rules():
    root = REPO / "captain_claw/instructions/archetype_pack"
    outline, write = (root / "outline.md").read_text(encoding="utf-8"), (root / "write.md").read_text(encoding="utf-8")
    assert '"group"' in outline and '"family"' in outline and '"archetypes"' in outline
    for heading in ("## Hard limits", "## Before you start", "## Standard Operating Procedure", "## Domain knowledge", "## Output"):
        assert heading in write
    assert "You can't look anything up" in write and "Never invent a statute" in write
    assert "—" not in outline and "—" not in write
