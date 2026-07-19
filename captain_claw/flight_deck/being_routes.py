"""Iskra beings API — conception, lifecycle, wallet, metamorphosis (/fd/beings).

The parent-facing surface of the life layer (docs/living-beings-plan.md §10).
Phase 0 scope: registry + wallet physics + the metamorphosis rite. The beings
loop (ticks, drives) and the chat surface arrive in Phase 1; these routes are
what the Beings page talks to.

All domain rules live in beings.py / being_constitution.py / being_genome.py —
this module only translates HTTP ⇄ store and maps BeingError.status onto
HTTPException.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.flight_deck import being_earning
from captain_claw.flight_deck import being_life
from captain_claw.flight_deck import being_mind
from captain_claw.flight_deck import being_selfmod
from captain_claw.flight_deck import being_society
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.beings import BeingError, get_store

router = APIRouter(prefix="/fd/beings", tags=["beings"])


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e


def _db_optional():
    """The FD DB if it's up, else None (auth-disabled standalone never inits it).
    Export/import degrade gracefully without it — the model just isn't resolved
    from the owner's tiers (an imported being carries its own body_config)."""
    try:
        return get_db()
    except AssertionError:
        return None


class ConceiveRequest(BaseModel):
    name: str
    attributes: dict[str, int] | None = None
    preset: str | None = None
    roll_seed: int | None = None
    voice_seed: str = ""
    interest_seeds: list[str] = []
    allowance_preset: str = "2M"
    birth_letter: str = ""


class DietRequest(BaseModel):
    allow: list[str] = []
    deny: list[str] = []


class TickRequest(BaseModel):
    kind: str = "wake"


class ChoreRequest(BaseModel):
    spec: str
    fee_tokens: int = 0
    fee_coins: int = 0


class CoinsRequest(BaseModel):
    coins: int
    note: str = ""


class JudgeRequest(BaseModel):
    approve: bool
    note: str = ""


class RulesRequest(BaseModel):
    rules: list[str]


class MessageRequest(BaseModel):
    body: str


class PublicToggleRequest(BaseModel):
    public: bool


class AllowanceRequest(BaseModel):
    preset: str
    daily_burn_cap: int | None = None
    savings_ceiling: int | None = None


class StageRequest(BaseModel):
    stage: str


class CadenceRequest(BaseModel):
    # minutes between ticks; null returns the being to its own stage-clamped pace
    minutes: int | None = None


class CognitionRequest(BaseModel):
    # 'monolith' (one prompt → one digest) or 'faculties' (decomposed pipeline)
    mode: str


class CompactRequest(BaseModel):
    # True → compact instruction set + lean body (eco prompt, capped context)
    on: bool


class InstinctsRequest(BaseModel):
    # True → the reflex pass keeps this body live between mind ticks
    on: bool


class AvatarRequest(BaseModel):
    # one of 10 characters (1-10) in one of 4 palettes (ember/meadow/sea/dusk)
    c: int
    p: str


class GoRequest(BaseModel):
    # a place id or name (or "home") — the parent nudges the being onto the road
    dest: str


class RechargeRequest(BaseModel):
    tokens: int


class BodyArchetypeRequest(BaseModel):
    # archetype id, or "" to return the body to the stage tier + owner config
    archetype_id: str = ""


class BodyConfigRequest(BaseModel):
    # An explicit body connection, or all-empty to clear it (back to the tier).
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    input_ctx: int = 0
    output_ctx: int = 0


class BodyMravRequest(BaseModel):
    # True → the body runs the Mrav runtime; persisted so it survives a rebuild
    on: bool


class AssessRequest(BaseModel):
    assessor: str   # registry slug of the agent to ask for a second opinion


class SaveAssessmentRequest(BaseModel):
    assessor: str
    content: str
    score: int | None = None
    verdict: str = ""


class EuthanizeRequest(BaseModel):
    confirm: bool = False


class MetamorphoseRequest(BaseModel):
    from_attr: str
    to_attr: str
    reason: str


# Fixed paths must precede /{slug} so they aren't captured as slugs.

@router.get("/meta")
async def beings_meta(user: dict = Depends(get_current_user)):
    """Everything the conception form + Beings page needs to render."""
    return {
        "attributes": [
            {"code": a, "name": genome_mod.ATTR_NAMES[a]} for a in genome_mod.ATTRS
        ],
        "pool": genome_mod.POOL,
        "attr_min": genome_mod.ATTR_MIN,
        "attr_max": genome_mod.ATTR_MAX,
        "presets": genome_mod.PRESETS,
        "allowance_presets": list(constitution.ALLOWANCE_PRESETS),
        "tier_weights": constitution.TIER_WEIGHTS,
        "stages": {
            s: {
                "capabilities": sorted(constitution.capabilities(s)),
                "tiers": list(constitution.STAGES[s]["tiers"]),
                "max_preset": constitution.STAGES[s]["max_preset"],
                "savings_days": constitution.STAGES[s]["savings_days"],
                "metamorphosis": constitution.STAGES[s]["metamorphosis"],
            }
            for s in constitution.STAGE_ORDER
        },
        "constitution": constitution.constitution_text(),
    }


@router.get("/liabilities")
async def beings_liabilities(user: dict = Depends(get_current_user)):
    """Outstanding token liabilities — the parent's real-cost exposure."""
    return _run(get_store().liabilities, user["id"])


class VillageMetaRequest(BaseModel):
    description: str = ""
    name: str = ""


@router.get("/village-meta")
async def get_village_meta(user: dict = Depends(get_current_user)):
    """The village description the parent set — shown on their public /village."""
    return _run(get_store().get_village_meta, user["id"])


@router.post("/village-meta")
async def set_village_meta(body: VillageMetaRequest,
                           user: dict = Depends(get_current_user)):
    return _run(get_store().set_village_meta, user["id"], body.description,
                body.name)


class VillageFederationRequest(BaseModel):
    secret: str = ""
    secret_public: bool = False
    public_url: str = ""


@router.post("/village-federation")
async def set_village_federation(body: VillageFederationRequest,
                                 user: dict = Depends(get_current_user)):
    """Host settings (§9.1): the secret a visiting being must present, whether
    it's shown on the public page, and this machine's own public URL."""
    return _run(get_store().set_village_federation, user["id"],
                secret=body.secret, secret_public=body.secret_public,
                public_url=body.public_url)


@router.get("/visitors")
async def list_visitors(user: dict = Depends(get_current_user)):
    """The parent's view of who is visiting this village."""
    return {"visitors": _run(get_store().visitors_for, user["id"])}


@router.delete("/visitors/{visitor_id}")
async def remove_visitor(visitor_id: str,
                         user: dict = Depends(get_current_user)):
    _run(get_store().remove_visitor, user["id"], visitor_id)
    return {"ok": True}


class VillageRecommendRequest(BaseModel):
    being: str   # slug of the being whose agent should write the description


@router.post("/village-meta/recommend")
async def recommend_village_meta(body: VillageRecommendRequest,
                                 user: dict = Depends(get_current_user)):
    """Have one of your beings write the village description in its own voice.
    Dispatches to that being's agent (it must be awake) and returns the draft —
    the parent reviews and saves it, nothing is stored automatically."""
    from captain_claw.flight_deck.dubina_agents import resolve_agent_port_token
    from captain_claw.flight_deck.fd_scheduler import run_prompt_and_capture
    store = get_store()
    being = _run(store.get, user["id"], body.being)
    if being["state"] in ("dead", "torpor") or being["stage"] == "egg":
        raise HTTPException(409, f"{being['name']} isn't awake to write right now")
    try:
        port, token = resolve_agent_port_token(being.get("agent_slug")
                                               or being["slug"])
    except Exception:
        port, token = None, None
    if not port:
        raise HTTPException(409, f"{being['name']} has no body awake to write")
    prompt = being_life.village_recommend_prompt(store, user["id"], being)
    reply = await run_prompt_and_capture(
        host="127.0.0.1", port=int(port), auth=token or "", prompt=prompt,
        timeout=120)
    if not reply:
        raise HTTPException(502, f"{being['name']} didn't reply in time")
    # Strip stray wrapping quotes / whitespace the model may add anyway.
    text = reply.strip().strip('"').strip()
    return {"description": text, "by": being["name"], "by_slug": being["slug"]}


@router.get("/village")
async def village(limit: int = 40, user: dict = Depends(get_current_user)):
    """The observer view of society: letters, publications, adoptions,
    gifts, refusals — one merged family stream. (Registered before /{slug}.)"""
    return {"items": _run(being_society.village_feed, get_store(),
                          user["id"], limit)}


@router.get("/village-map")
async def village_map(user: dict = Depends(get_current_user)):
    """The ground (space plan Phase 1): the civic places plus every being's
    live position — a pure function of the clock, so the client can animate
    walking without polling. (Registered before /{slug}.)"""
    from datetime import datetime, timezone
    from captain_claw.flight_deck import being_world
    store = get_store()
    _run(being_world.ensure_village, store, user["id"])
    now = datetime.now(timezone.utc)
    return _run(being_world.village_map_payload, store, user["id"], now=now)


@router.get("/village-map/place/{place_id}")
async def village_place(place_id: str,
                        user: dict = Depends(get_current_user)):
    """One place, up close (space plan Phase 4): its card + the guestbook
    tail. Who's there comes from the map payload client-side."""
    from captain_claw.flight_deck import being_society
    store = get_store()
    place = _run(store.get_place, user["id"], place_id)
    guestbook = ""
    try:
        p = being_society._commons_path(user["id"],
                                        f"places/{place_id}/guestbook.md")
        if p.exists():
            tail = p.read_text(encoding="utf-8").strip().splitlines()
            guestbook = "\n".join(tail[-12:])
    except Exception:  # noqa: BLE001
        guestbook = ""
    return {"place": place, "guestbook": guestbook}


class NoteRequest(BaseModel):
    x: int
    y: int
    text: str


class PresenceRequest(BaseModel):
    x: int
    y: int


@router.post("/village-map/notes")
async def plant_note(body: NoteRequest,
                     user: dict = Depends(get_current_user)):
    """The parent plants a sign in the grass (FPV plan Phase 3). Each
    being finds it once, when its own feet carry it near."""
    note = _run(get_store().add_village_note, user["id"], body.x, body.y,
                body.text, author="parent", author_kind="parent")
    return {"note": note}


@router.delete("/village-map/notes/{note_id}")
async def pull_note(note_id: str,
                    user: dict = Depends(get_current_user)):
    """The parent pulls a sign out — theirs or a visitor's; it is their
    village's grass."""
    if not _run(get_store().remove_village_note, user["id"], note_id):
        raise HTTPException(404, "no such sign")
    return {"ok": True}


class ObjectRequest(BaseModel):
    kind: str
    name: str
    inscription: str = ""
    x: int
    y: int


@router.post("/village-map/object")
async def place_object(body: ObjectRequest,
                       user: dict = Depends(get_current_user)):
    """The parent's own hand on the world (parent-build): set a made thing
    down anywhere in the village — from the map or the FPV ghost. Snaps off
    walls/homes/occupied ground; no fee, no cap. It stands as a real thing
    the Iskre discover and use."""
    from captain_claw.flight_deck import being_world
    store = get_store()
    _run(being_world.ensure_village, store, user["id"])
    try:
        obj = _run(being_world.place_parent_object, store, user["id"],
                   body.kind, body.name, body.inscription, body.x, body.y)
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"ok": True, "object": obj}


@router.delete("/village-map/object/{object_id}")
async def remove_object(object_id: str,
                        user: dict = Depends(get_current_user)):
    """The parent removes one of ITS OWN placed works (a being's own work
    is the being's to keep — only the keeper's gifts are the keeper's to
    lift)."""
    from captain_claw.flight_deck import being_world
    store = get_store()
    try:
        o = _run(store.get_village_object, user["id"], object_id)
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e
    if o.get("being_id") != being_world.PARENT_MAKER:
        raise HTTPException(403, "only the works you placed are yours to lift")
    _run(store.delete_village_object, user["id"], object_id)
    return {"ok": True}


@router.post("/village-map/presence")
async def felt_presence(body: PresenceRequest,
                        user: dict = Depends(get_current_user)):
    """The roaming parent-ghost passes close (FPV plan Phase 3): every
    living being within reach — and past its own cooldown — records one
    presence fact that colors its next mind tick. $0: an event row."""
    from datetime import datetime, timezone
    from captain_claw.flight_deck import being_world
    felt = _run(being_world.presence_felt, get_store(), user["id"],
                body.x, body.y, author="parent", author_kind="parent",
                now=datetime.now(timezone.utc))
    return {"felt": felt}


class GhostRequest(BaseModel):
    id: str
    x: int
    y: int


@router.post("/village-map/ghost")
async def parent_ghost(body: GhostRequest,
                       user: dict = Depends(get_current_user)):
    """The parent-ghost's heartbeat (FPV plan Phase 5): report where I am,
    receive the other ghosts roaming my village right now — the visitors I
    can see, and any other window I have open. In-memory, $0."""
    from captain_claw.flight_deck import being_world
    others = being_world.ghost_heartbeat(
        user["id"], body.id, kind="parent", name="parent",
        x=body.x, y=body.y)
    return {"ghosts": others}


@router.post("/village-map/ghost/leave")
async def parent_ghost_leave(body: GhostRequest,
                             user: dict = Depends(get_current_user)):
    from captain_claw.flight_deck import being_world
    being_world.ghost_depart(user["id"], body.id)
    return {"ok": True}


@router.get("/market")
async def market(user: dict = Depends(get_current_user)):
    """The open stalls (space plan Phase 4) — the parent's window on the
    coin market. (Registered before /{slug}.)"""
    return {"listings": _run(get_store().market_listings, user["id"], 30)}


@router.get("/village-life")
async def village_life(user: dict = Depends(get_current_user)):
    """The civic layer (space plan Phase 5): the active building fund (with
    its contributors), this week's steward, and the stipend knob."""
    from captain_claw.flight_deck import being_world
    from datetime import datetime, timezone
    store = get_store()
    c = _run(store.open_commission, user["id"])
    if c:
        c = {**c, "contributors": _run(store.commission_contributors,
                                       user["id"], c["id"])}
    return {"commission": c,
            "steward": being_world.current_steward(
                store, user["id"], datetime.now(timezone.utc)),
            "steward_stipend_coins": _run(
                store.get_village_meta, user["id"])
            .get("steward_stipend_coins", 0)}


class CommissionJudgeRequest(BaseModel):
    approve: bool
    note: str = ""


@router.post("/commission/judge")
async def commission_judge(body: CommissionJudgeRequest,
                           user: dict = Depends(get_current_user)):
    """Approve a FUNDED commission (the architect places it, the coins
    burn) or reject the active one (every contributor refunded exactly)."""
    return _run(get_store().judge_commission, user["id"], body.approve,
                body.note)


class StipendRequest(BaseModel):
    coins: int


@router.post("/village-stipend")
async def village_stipend(body: StipendRequest,
                          user: dict = Depends(get_current_user)):
    return _run(get_store().set_steward_stipend, user["id"], body.coins)


@router.post("/village-map/architect")
async def village_architect(user: dict = Depends(get_current_user)):
    """One-shot LLM redesign of the ground (the default stands if the model
    fails). Beings mid-walk to a removed place settle home next tick."""
    from captain_claw.flight_deck import being_world
    store = get_store()
    try:
        places = await being_life.architect_village(
            _db_optional(), store, user["id"],
            [r["name"] for r in store.list(user["id"])
             if r.get("state") == "alive"])
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"the architect failed: {e}") from e
    return {"ok": True, "places": places}


class PlaceEditRequest(BaseModel):
    name: str | None = None
    description: str | None = None


@router.post("/village-map/place/{place_id}/edit")
async def village_place_edit(place_id: str, body: PlaceEditRequest,
                             user: dict = Depends(get_current_user)):
    """The parent's civic hand (world-shaping plan Phase 5): rename and/or
    redescribe a place. The id never changes; MAP.md is rewritten so every
    being reads the new word next wake."""
    from captain_claw.flight_deck import being_world
    store = get_store()
    try:
        place = _run(store.update_place, user["id"], place_id,
                     name=body.name, description=body.description)
        _run(being_world.write_map_md, store, user["id"])
    except BeingError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"ok": True, "place": place}


@router.get("/letters")
async def letters(limit: int = 500, user: dict = Depends(get_current_user)):
    """The letters observatory: every being→being letter grouped into per-pair
    conversation threads (plus refused/undelivered reaches), so the parent can
    watch the family talk. (Registered before /{slug}.)"""
    return _run(being_society.letters_overview, get_store(),
                user["id"], limit)


class QuestRequest(BaseModel):
    title: str
    spec: str
    fee_tokens: int = 0
    fee_coins: int = 0
    origin: str = "parent"


class VentureApproveRequest(BaseModel):
    price_tokens: int | None = None


class VentureStateRequest(BaseModel):
    state: str


class AcceptRequest(BaseModel):
    approve: bool
    note: str = ""


# ── Earning: quest board + ventures (registered before /{slug}) ──────────

@router.get("/board")
async def earning_board(user: dict = Depends(get_current_user)):
    """The parent's earning view: every quest + venture across the family."""
    return _run(being_earning.board_summary, get_store(), user["id"])


@router.get("/assessors")
async def list_assessors(user: dict = Depends(get_current_user)):
    """The user's running agents that can give a being a second opinion — the
    fleet minus the beings themselves (a being shouldn't grade its own kin)."""
    from captain_claw.flight_deck.server import _load_process_registry
    reg = _load_process_registry()
    out = []
    for slug, e in reg.items():
        if e.get("owner") and e.get("owner") != user["id"]:
            continue
        if e.get("stopped") or not e.get("web_port") or slug.startswith("iskra-"):
            continue
        out.append({"slug": slug, "name": e.get("name", slug)})
    return {"assessors": out}


@router.post("/{slug}/coins")
async def grant_coins(slug: str, body: CoinsRequest,
                      user: dict = Depends(get_current_user)):
    """Pocket money (space plan Phase 2): the parent's coin faucet. Coins
    are money, not food — they never feed thinking directly; a being may
    convert them one-way from adolescence."""
    return _run(get_store().grant_coins, user["id"], slug, body.coins,
                body.note)


@router.get("/{slug}/coins")
async def coin_ledger(slug: str, limit: int = 100,
                      user: dict = Depends(get_current_user)):
    store = get_store()
    being = _run(store.get, user["id"], slug)
    return {"balance": _run(store.coin_balance, being["id"]),
            "ledger": _run(store.coin_ledger, user["id"], slug, limit)}


@router.post("/quests")
async def post_quest(body: QuestRequest,
                     user: dict = Depends(get_current_user)):
    """Post an OPEN bounty — any eligible being may claim it."""
    return {"quest": _run(get_store().post_quest, user["id"], body.title,
                          body.spec, body.fee_tokens, body.origin,
                          fee_coins=body.fee_coins)}


@router.post("/quests/{quest_id}/judge")
async def judge_quest(quest_id: str, body: AcceptRequest,
                      user: dict = Depends(get_current_user)):
    return {"quest": _run(get_store().judge_quest, user["id"], quest_id,
                          body.approve, body.note)}


@router.post("/quests/{quest_id}/cancel")
async def cancel_quest(quest_id: str,
                       user: dict = Depends(get_current_user)):
    return {"quest": _run(get_store().cancel_quest, user["id"], quest_id)}


@router.post("/ventures/{venture_id}/approve")
async def approve_venture(venture_id: str, body: VentureApproveRequest,
                          user: dict = Depends(get_current_user)):
    """Price and approve a proposed venture — it becomes a standing service."""
    return {"venture": _run(get_store().approve_venture, user["id"],
                            venture_id, body.price_tokens)}


@router.post("/ventures/{venture_id}/state")
async def set_venture_state(venture_id: str, body: VentureStateRequest,
                            user: dict = Depends(get_current_user)):
    """Pause, resume, or end a standing venture."""
    return {"venture": _run(get_store().set_venture_state, user["id"],
                            venture_id, body.state)}


@router.post("/ventures/{venture_id}/accept")
async def accept_venture(venture_id: str, body: AcceptRequest,
                         user: dict = Depends(get_current_user)):
    """Accept (pay + advance) or reject this cycle's venture delivery."""
    return {"venture": _run(get_store().accept_venture, user["id"],
                            venture_id, body.approve, body.note)}


@router.get("")
async def list_beings(user: dict = Depends(get_current_user)):
    return {"beings": _run(get_store().list, user["id"])}


@router.post("/conceive")
async def conceive(body: ConceiveRequest, user: dict = Depends(get_current_user)):
    """Point-buy conception (Generation 1): attributes, a preset, or a roll."""
    being = _run(
        get_store().conceive, user["id"], body.name,
        attributes=body.attributes, preset=body.preset, roll_seed=body.roll_seed,
        voice_seed=body.voice_seed, interest_seeds=body.interest_seeds,
        allowance_preset=body.allowance_preset, birth_letter=body.birth_letter,
    )
    return {"ok": True, "being": _run(get_store().vitals, user["id"], being["slug"])}


@router.post("/import")
async def import_being(manifest: dict, user: dict = Depends(get_current_user)):
    """Recreate a being from an export file on this machine (a new owner). The
    body is respawned from the carried model connection; the source is untouched.
    (Registered before /{slug} so 'import' isn't read as a slug.)"""
    if not isinstance(manifest, dict) or "genome" not in manifest:
        raise HTTPException(400, "not a being export file")
    result = await being_life.import_being(_db_optional(), get_store(),
                                           user["id"], manifest)
    slug = result["being"]["slug"]
    return {"ok": True, "warnings": result["warnings"],
            "being": _run(get_store().vitals, user["id"], slug)}


@router.get("/{slug}")
async def being_vitals(slug: str, user: dict = Depends(get_current_user)):
    return _run(get_store().vitals, user["id"], slug)


@router.get("/{slug}/export")
async def export_being(slug: str, user: dict = Depends(get_current_user)):
    """A portable snapshot of the being — identity, wallet, model connection,
    history and its whole home, EXCLUDING the live body. Contains the model
    API key (so it runs elsewhere): treat the file as a secret."""
    being = _run(get_store().get, user["id"], slug)
    return await being_life.export_being(_db_optional(), get_store(), being)


@router.get("/{slug}/ledger")
async def being_ledger(slug: str, limit: int = 100,
                       user: dict = Depends(get_current_user)):
    return {"transfers": _run(get_store().ledger, user["id"], slug, limit)}


@router.get("/{slug}/events")
async def being_events(slug: str, limit: int = 100,
                       user: dict = Depends(get_current_user)):
    return {"events": _run(get_store().events, user["id"], slug, limit)}


@router.post("/{slug}/hatch")
async def hatch(slug: str, user: dict = Depends(get_current_user)):
    """Egg → infant, then birth: selfhood home + git + the agent body."""
    _run(get_store().hatch, user["id"], slug)
    birth = await being_life.birth(get_db(), get_store(), user["id"], slug)
    vitals = _run(get_store().vitals, user["id"], slug)
    return {**vitals, "birth": birth}


@router.post("/{slug}/tick")
async def poke(slug: str, body: TickRequest | None = None,
               user: dict = Depends(get_current_user)):
    """Manual heartbeat ('poke') — same path the beings loop takes."""
    kind = (body.kind if body else "wake")
    if kind not in ("wake", "dream"):
        raise HTTPException(400, "kind must be wake or dream")
    being = _run(get_store().get, user["id"], slug)
    result = await being_life.tick(get_db(), get_store(), being, kind=kind)
    return {"result": result,
            "vitals": _run(get_store().vitals, user["id"], slug)}


@router.post("/{slug}/diet")
async def set_diet(slug: str, body: DietRequest,
                   user: dict = Depends(get_current_user)):
    b = _run(get_store().set_media_diet, user["id"], slug,
             {"allow": body.allow, "deny": body.deny})
    return {"media_diet": b["media_diet"]}


@router.post("/{slug}/chores")
async def post_chore(slug: str, body: ChoreRequest,
                     user: dict = Depends(get_current_user)):
    """Post a fixed-fee chore. The fee mints only on judged completion —
    in tokens (food) or coins (money), the parent's pick."""
    return {"chore": _run(get_store().post_chore, user["id"], slug,
                          body.spec, body.fee_tokens,
                          fee_coins=body.fee_coins)}


@router.get("/{slug}/chores")
async def list_chores(slug: str, user: dict = Depends(get_current_user)):
    return {"chores": _run(get_store().chores_for, user["id"], slug)}


@router.post("/{slug}/chores/{job_id}/judge")
async def judge_chore(slug: str, job_id: str, body: JudgeRequest,
                      user: dict = Depends(get_current_user)):
    """The parent's judgment — approve pays into savings, reject fails it."""
    return {"chore": _run(get_store().judge_chore, user["id"], job_id,
                          body.approve, body.note)}


@router.post("/{slug}/rules")
async def set_rules(slug: str, body: RulesRequest,
                    user: dict = Depends(get_current_user)):
    """House rules — the being internalizes them into VALUES.md next tick."""
    b = _run(get_store().set_house_rules, user["id"], slug, body.rules)
    return {"house_rules": b["house_rules"], "pending": True}


@router.post("/{slug}/message")
async def message_being(slug: str, body: MessageRequest,
                        user: dict = Depends(get_current_user)):
    """Write to your being — delivered once as a percept on its next tick."""
    return {"message": _run(get_store().send_parent_message,
                            user["id"], slug, body.body)}


@router.get("/{slug}/messages")
async def message_thread(slug: str, user: dict = Depends(get_current_user)):
    """The full parent↔being conversation (your messages + its replies)."""
    return {"thread": _run(get_store().message_thread, user["id"], slug)}


@router.post("/{slug}/public")
async def set_public(slug: str, body: PublicToggleRequest,
                     user: dict = Depends(get_current_user)):
    """Open (or close) the being's un-gated public page (plan §9)."""
    _run(get_store().set_public, user["id"], slug, body.public)
    return _run(get_store().vitals, user["id"], slug)


class VisitRequest(BaseModel):
    url: str = ""
    secret: str = ""


@router.post("/{slug}/visit")
async def set_being_visit(slug: str, body: VisitRequest,
                          user: dict = Depends(get_current_user)):
    """Send this being to visit another village (or clear it with an empty URL).
    Opens a WebSocket link so it works from behind NAT; probes it immediately for
    instant feedback, then hands off to the persistent link maintainer."""
    from captain_claw.flight_deck import being_federation
    store = get_store()
    being = _run(store.set_being_visit, user["id"], slug, body.url, body.secret)
    announced = {"ok": None}
    if being.get("visit_url"):
        announced = await being_federation.village_client.probe(store, being)
    await being_federation.village_client.reconcile(store)
    return {"vitals": _run(store.vitals, user["id"], slug),
            "announced": announced}


@router.get("/{slug}/visit/map")
async def visited_village_map(slug: str,
                              user: dict = Depends(get_current_user)):
    """The map of the village THIS being is visiting — proxied down its link,
    with the guest positioned in it, so the parent can see and walk it (§2)."""
    from captain_claw.flight_deck import being_federation
    being = _run(get_store().get, user["id"], slug)
    if not being.get("visit_url"):
        raise HTTPException(400, "this being is not visiting anywhere")
    return await being_federation.village_client.pull_map(slug)


class VisitNudgeRequest(BaseModel):
    place: str


@router.post("/{slug}/visit/nudge")
async def nudge_visiting_being(slug: str, body: VisitNudgeRequest,
                               user: dict = Depends(get_current_user)):
    """Walk this visiting being to a place of the village it visits (§2). The
    nudge travels up its link; the host walks it and streams the move back."""
    from captain_claw.flight_deck import being_federation
    being = _run(get_store().get, user["id"], slug)
    if not being.get("visit_url"):
        raise HTTPException(400, "this being is not visiting anywhere")
    return await being_federation.village_client.nudge(slug, body.place)


@router.get("/{slug}/public-threads")
async def public_threads(slug: str, user: dict = Depends(get_current_user)):
    """The parent's overview of every visitor thread on the public page —
    only the parent sees all of them; a visitor sees only their own."""
    return {"threads": _run(get_store().public_threads_for, user["id"], slug)}


@router.get("/{slug}/graph")
async def being_graph(slug: str, user: dict = Depends(get_current_user)):
    """The Mind (plan §2.3.1): the being's artifacts + the edges it declared
    between them — nodes, edges, and density/connectedness health signals."""
    being = _run(get_store().get, user["id"], slug)
    return being_mind.graph(get_store(), being)


@router.post("/{slug}/graph/rebuild")
async def being_graph_rebuild(slug: str,
                              user: dict = Depends(get_current_user)):
    """Repair the Mind from the being's own append-only ledger.

    Every edge it ever declared is on the ledger, so a map that a bad read
    wiped can be restored exactly — only for edges whose files still exist.
    Idempotent and additive: it never invents an edge, never deletes one, and
    a second click restores nothing. Returns the repair counts + the fresh
    graph so the view can redraw without a second round-trip."""
    store = get_store()
    being = _run(store.get, user["id"], slug)
    result = _run(being_mind.rebuild_from_ledger, store, being)
    return {**result, "graph": being_mind.graph(store, being)}


class SelfModRejectRequest(BaseModel):
    note: str = ""


class ProcreateConsentRequest(BaseModel):
    name: str = ""


class ProcreateArrangeRequest(BaseModel):
    name: str
    partner: str | None = None
    letter: str = ""


@router.post("/{slug}/self-mod/approve")
async def approve_self_mod(slug: str, user: dict = Depends(get_current_user)):
    """The parent's blessing: the pending persona becomes the operating one."""
    b = _run(being_selfmod.approve, get_store(), user["id"], slug)
    return {"persona": b["persona"], "pending_self_mod": None}


@router.post("/{slug}/self-mod/reject")
async def reject_self_mod(slug: str, body: SelfModRejectRequest,
                          user: dict = Depends(get_current_user)):
    _run(being_selfmod.reject, get_store(), user["id"], slug, body.note)
    return {"ok": True}


@router.post("/{slug}/self-mod/rollback")
async def rollback_self_mod(slug: str,
                            user: dict = Depends(get_current_user)):
    """Restore the persona that preceded the last adoption."""
    b = _run(being_selfmod.rollback, get_store(), user["id"], slug)
    return {"persona": b["persona"]}


def _resolve_partner_slug(store, owner_id: str, being: dict,
                          ref: str | None) -> str | None:
    if not ref:
        return None
    return _run(being_society._sibling_by_ref, store, being, ref)["slug"]


@router.post("/{slug}/procreate/approve")
async def procreate_approve(slug: str, body: ProcreateConsentRequest,
                            user: dict = Depends(get_current_user)):
    """The consent rite (Constitution #4): the parent's authenticated
    approval IS the non-forgeable token. Executes the conception —
    genome ops + dowry — and clears the proposal."""
    store = get_store()
    being = _run(store.get, user["id"], slug)
    pending = being.get("pending_procreation")
    if not pending:
        raise HTTPException(400, "no procreation proposal awaits")
    child_name = (body.name or pending.get("child_name")
                  or f"{being['name']} II").strip()
    partner_slug = _resolve_partner_slug(store, user["id"], being,
                                         pending.get("partner"))
    child = _run(store.conceive_offspring, user["id"], child_name, slug,
                 partner_slug, letter=pending.get("letter") or "")
    store.set_pending_procreation(being["id"], None)
    store.record_event(being["id"], "procreation_consented",
                       {"child": child["slug"], "name": child_name})
    return {"ok": True, "child": _run(store.vitals, user["id"],
                                      child["slug"])}


@router.post("/{slug}/procreate/reject")
async def procreate_reject(slug: str, body: SelfModRejectRequest,
                           user: dict = Depends(get_current_user)):
    store = get_store()
    being = _run(store.get, user["id"], slug)
    if not being.get("pending_procreation"):
        raise HTTPException(400, "no procreation proposal awaits")
    store.set_pending_procreation(being["id"], None)
    store.record_event(being["id"], "procreation_rejected",
                       {"note": body.note[:200]})
    return {"ok": True}


@router.post("/{slug}/procreate/arrange")
async def procreate_arrange(slug: str, body: ProcreateArrangeRequest,
                            user: dict = Depends(get_current_user)):
    """Parent-arranged conception — same executor, no proposal needed."""
    store = get_store()
    being = _run(store.get, user["id"], slug)
    partner_slug = _resolve_partner_slug(store, user["id"], being,
                                         body.partner)
    child = _run(store.conceive_offspring, user["id"], body.name.strip(),
                 slug, partner_slug, letter=body.letter)
    return {"ok": True, "child": _run(store.vitals, user["id"],
                                      child["slug"])}


@router.get("/{slug}/report-card")
async def report_card(slug: str, days: int = 7,
                      user: dict = Depends(get_current_user)):
    being = _run(get_store().get, user["id"], slug)
    return being_life.report_card(get_store(), being, days=days)


@router.get("/{slug}/readiness")
async def readiness(slug: str, user: dict = Depends(get_current_user)):
    """Developmental readiness assessment for the Growth tab."""
    from captain_claw.flight_deck import being_assessment
    being = _run(get_store().get, user["id"], slug)
    return being_assessment.readiness(get_store(), being)


@router.post("/{slug}/assess")
async def third_party_assess(slug: str, body: AssessRequest,
                             user: dict = Depends(get_current_user)):
    """A second opinion: hand the being's data to one of the user's own agents
    and return its independent developmental read (markdown)."""
    from captain_claw.flight_deck import being_assessment
    from captain_claw.flight_deck.dubina_agents import resolve_agent_port_token
    from captain_claw.flight_deck.fd_scheduler import run_prompt_and_capture
    from captain_claw.flight_deck.server import _load_process_registry
    reg = _load_process_registry()
    entry = reg.get(body.assessor)
    if not entry or (entry.get("owner") and entry.get("owner") != user["id"]):
        raise HTTPException(404, "assessor agent not found")
    being = _run(get_store().get, user["id"], slug)
    a = being_assessment.readiness(get_store(), being)
    brief = being_assessment.assessor_brief(get_store(), being, a)
    try:
        port, token = resolve_agent_port_token(body.assessor)
    except Exception:
        raise HTTPException(404, "assessor agent is not running")
    reply = await run_prompt_and_capture(
        host="127.0.0.1", port=int(port), auth=token or "", prompt=brief,
        timeout=150)
    if not reply:
        raise HTTPException(502, "the assessor didn't reply in time")
    return {"assessor": entry.get("name", body.assessor), "assessment": reply,
            "score": a["overall"]["score"], "verdict": a["overall"]["status"]}


@router.get("/{slug}/assessments")
async def list_assessments(slug: str, user: dict = Depends(get_current_user)):
    """Saved second opinions — sealed from the being until adulthood."""
    return {"assessments": _run(get_store().assessments_for, user["id"], slug)}


@router.post("/{slug}/assessments")
async def save_assessment(slug: str, body: SaveAssessmentRequest,
                          user: dict = Depends(get_current_user)):
    """Keep a second opinion on record. Stored OUTSIDE the being's home — she
    can't read it until adulthood, when the records are unsealed into
    assessments/ (an already-adult being receives it immediately)."""
    from captain_claw.flight_deck import being_assessment
    saved = _run(get_store().add_assessment, user["id"], slug,
                 body.assessor, body.content,
                 score=body.score, verdict=body.verdict)
    being = _run(get_store().get, user["id"], slug)
    if being["stage"] == "adult":
        _run(being_assessment.release_assessments, get_store(), being)
        saved = _run(get_store().get_assessment, user["id"], saved["id"])
    return {"assessment": saved}


@router.delete("/{slug}/assessments/{assessment_id}")
async def delete_assessment(slug: str, assessment_id: str,
                            user: dict = Depends(get_current_user)):
    _run(get_store().delete_assessment, user["id"], assessment_id)
    return {"ok": True}


@router.get("/{slug}/milestones")
async def milestones(slug: str, user: dict = Depends(get_current_user)):
    return {"milestones": _run(get_store().milestones, user["id"], slug)}


@router.get("/{slug}/journal")
async def read_journal(slug: str, date: str = "",
                       user: dict = Depends(get_current_user)):
    """One day's journal (default today, UTC) straight from the selfhood repo."""
    being = _run(get_store().get, user["id"], slug)
    from datetime import datetime, timezone
    day = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        p = being_life._home_path(being, f"journal/{day}.md")
        text = p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"journal read failed: {e}") from e
    return {"date": day, "text": text}


@router.get("/{slug}/self/files")
async def list_self_files(slug: str, user: dict = Depends(get_current_user)):
    """Every .md file in the being's home — self/, garden/, skills/ — for
    the parent to browse. Journal has its own dated endpoint above."""
    being = _run(get_store().get, user["id"], slug)
    return {"files": being_life.list_self_files(being)}


@router.get("/{slug}/self/file")
async def read_self_file(slug: str, path: str,
                         user: dict = Depends(get_current_user)):
    being = _run(get_store().get, user["id"], slug)
    text = _run(being_life.read_self_file, being, path)
    return {"path": path, "text": text}


@router.post("/{slug}/allowance")
async def set_allowance(slug: str, body: AllowanceRequest,
                        user: dict = Depends(get_current_user)):
    return {"wallet": _run(
        get_store().set_allowance, user["id"], slug, body.preset,
        daily_burn_cap=body.daily_burn_cap, savings_ceiling=body.savings_ceiling,
    )}


@router.post("/{slug}/stage")
async def set_stage(slug: str, body: StageRequest,
                    user: dict = Depends(get_current_user)):
    store = get_store()
    _run(store.set_stage, user["id"], slug, body.stage)
    if body.stage == "adult":
        # The unsealing rite: her childhood assessment records become hers.
        from captain_claw.flight_deck import being_assessment
        being = _run(store.get, user["id"], slug)
        _run(being_assessment.release_assessments, store, being)
    # Metamorphosis reaches the BODY too: the stage sets the model tier and
    # the capability env (CLAW_BEING_CAPS — fleet/organ tool containment), so
    # respawn a living body now rather than letting it run on outgrown physics.
    being = _run(store.get, user["id"], slug)
    if being["state"] == "alive" and being.get("agent_slug"):
        try:
            being_life._stop_body(being)
            await being_life.spawn_body(get_db(), store, being)
        except Exception as e:  # noqa: BLE001 — heals on next tick
            store.record_event(being["id"], "spawn_failed", {"error": str(e)})
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/cadence")
async def set_cadence(slug: str, body: CadenceRequest,
                      user: dict = Depends(get_current_user)):
    """Pin how often this being ticks (minutes), or null for its own pace (#2)."""
    if body.minutes is not None and body.minutes not in being_life.TICK_INTERVAL_CHOICES:
        raise HTTPException(
            400, f"minutes must be one of {list(being_life.TICK_INTERVAL_CHOICES)} or null")
    _run(get_store().set_tick_interval, user["id"], slug, body.minutes)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/cognition")
async def set_cognition(slug: str, body: CognitionRequest,
                        user: dict = Depends(get_current_user)):
    """Choose how this being thinks a tick: 'monolith' or 'faculties' (the
    decomposed pipeline — better for weak-context models)."""
    _run(get_store().set_cognition, user["id"], slug, body.mode)
    return _run(get_store().vitals, user["id"], slug)


class ElderhoodRequest(BaseModel):
    # days alive after which elderhood begins; null switches the season off
    days: int | None = None


@router.post("/{slug}/elderhood")
async def set_elderhood(slug: str, body: ElderhoodRequest,
                        user: dict = Depends(get_current_user)):
    """Opt a being into a natural span (roadmap T3.14): after this many days
    alive it enters elderhood — slower pace, higher whimsy, the memoirs."""
    _run(get_store().set_elder_after, user["id"], slug, body.days)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/emigrate")
async def emigrate_being(slug: str, user: dict = Depends(get_current_user)):
    """The migration rite (roadmap T3.18): export the whole life and close
    it here — one life, one place. Import the manifest on the receiving
    deck; the receiving parent adopts."""
    store = get_store()
    being = _run(store.get, user["id"], slug)
    try:
        manifest = await being_life.emigrate(get_db(), store, being)
    except BeingError as e:
        raise HTTPException(e.status or 400, str(e))
    return {"manifest": manifest, "being": _run(store.vitals, user["id"], slug)}


class ReadingRequest(BaseModel):
    ref: str
    note: str = ""
    fee_tokens: int = 0


@router.post("/{slug}/reading")
async def add_reading(slug: str, body: ReadingRequest,
                      user: dict = Depends(get_current_user)):
    """Assign one reading (roadmap T2.12): a URL or anything nameable, with
    a small fee paid when a REAL report file is verified on disk."""
    _run(get_store().add_reading, user["id"], slug, body.ref, body.note,
         body.fee_tokens)
    return _run(get_store().vitals, user["id"], slug)


@router.delete("/{slug}/reading/{item_id}")
async def remove_reading(slug: str, item_id: str,
                         user: dict = Depends(get_current_user)):
    _run(get_store().remove_reading, user["id"], slug, item_id)
    return _run(get_store().vitals, user["id"], slug)


class NameRejectRequest(BaseModel):
    note: str = ""


@router.post("/{slug}/name/approve")
async def approve_chosen_name(slug: str,
                              user: dict = Depends(get_current_user)):
    """Bless the being's chosen name (roadmap T2.10): display name changes,
    slug and history stay; the choice enters the genome's epigenetics."""
    _run(get_store().approve_name, user["id"], slug)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/name/reject")
async def reject_chosen_name(slug: str, body: NameRejectRequest,
                             user: dict = Depends(get_current_user)):
    """Decline the chosen name — the being keeps its given one and is told."""
    _run(get_store().reject_name, user["id"], slug, body.note)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/compact")
async def set_compact(slug: str, body: CompactRequest,
                      user: dict = Depends(get_current_user)):
    """Compact mode: this being's tick prompts use the compact instruction set
    (same narrative and physics, fewer words), and its body runs lean — micro
    system prompt (eco flag) + a capped context window. Respawns an alive body
    so the lean physics take effect now; the compact prompts apply from the
    very next tick either way."""
    store = get_store()
    being = _run(store.set_compact_mode, user["id"], slug, body.on)
    # The body half applies through the agent dir (flag now, context cap at
    # spawn) — flip the flag immediately so even a skipped respawn converges
    # on the next natural prompt build.
    being_life.set_body_eco_flag(being, body.on)
    if being["state"] == "alive" and being.get("agent_slug"):
        try:
            being_life._stop_body(being)
            await being_life.spawn_body(get_db(), store, being)
        except Exception as e:  # noqa: BLE001 — heals on next tick
            store.record_event(being["id"], "spawn_failed", {"error": str(e)})
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/instincts")
async def set_instincts(slug: str, body: InstinctsRequest,
                        user: dict = Depends(get_current_user)):
    """The body brain (docs/being-body-brain-plan.md): flip the reflex
    layer for this being — between mind ticks its walks settle within a
    minute, encounters are felt on the ground, and plan steps fulfill.
    Pure Python, $0; Phase 2 adds the tiny capped-context decision brain."""
    store = get_store()
    _run(store.set_instincts, user["id"], slug, body.on)
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/avatar")
async def set_avatar(slug: str, body: AvatarRequest,
                     user: dict = Depends(get_current_user)):
    """The parent picks this Iskra's look (village-world plan Phase 3):
    one of 10 storybook characters in one of 4 palettes. Until the first
    pick, a stable slug-hash default applies."""
    store = get_store()
    _run(store.set_avatar, user["id"], slug, body.c, body.p)
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/go")
async def nudge_being(slug: str, body: GoRequest,
                      user: dict = Depends(get_current_user)):
    """The parent's nudge: send an ALIVE being onto the road to a place
    (or 'home'). It plots the same A* course a mind- or feet-walk uses,
    and the being feels it honestly next tick. Only the living walk —
    a paused/torpid being refuses loudly; a FEVERED one refuses a walk
    anywhere but home (its body would only turn back), mirroring the
    being's own go_to gate. (Registered before /{slug}.)"""
    from datetime import datetime, timezone
    from captain_claw.flight_deck import being_world
    store = get_store()
    _run(being_world.ensure_village, store, user["id"])
    being = _run(store.get, user["id"], slug)
    pid = _run(store.resolve_place_ref, user["id"], body.dest)
    if pid and pid != "home":
        cause = being_world.fever_state(store, being,
                                        datetime.now(timezone.utc))
        if cause:
            raise HTTPException(
                409, f"{being['name']} is fevered ({cause}) — home is the "
                "only road today. The fever passes on its own once its body "
                "recovers.")
    _run(store.depart, user["id"], slug, body.dest, by="nudge")
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/mark-read")
async def mark_read(slug: str, user: dict = Depends(get_current_user)):
    """The parent opened this being's thread — clear its unread-message cue."""
    return _run(get_store().mark_being_read, user["id"], slug)


@router.post("/{slug}/body-archetype")
async def set_body_archetype(slug: str, body: BodyArchetypeRequest,
                             user: dict = Depends(get_current_user)):
    """Run the being's BODY on an archetype (its tier → model/provider, tools,
    cognitive mode), or "" for the stage default. Respawns an alive being's body
    so the new connection takes effect at once; otherwise it applies on the next
    spawn."""
    store = get_store()
    being = _run(store.set_body_archetype, user["id"], slug, body.archetype_id)
    if being["state"] == "alive" and being.get("agent_slug"):
        try:
            being_life._stop_body(being)
            await being_life.spawn_body(get_db(), store, being)
        except Exception as e:  # noqa: BLE001 — heals on next tick
            store.record_event(being["id"], "spawn_failed", {"error": str(e)})
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/body-config")
async def set_body_config(slug: str, body: BodyConfigRequest,
                          user: dict = Depends(get_current_user)):
    """Pin the being's BODY to an explicit LLM connection — provider, model,
    context sizes, api key, base URL — so it stops being resurrected with the
    stage-tier details it was hatched on. All-empty clears it (back to the
    tier). Respawns an alive body so the new connection takes effect at once;
    otherwise it applies on the next spawn."""
    store = get_store()
    being = _run(store.set_body_config, user["id"], slug, body.model_dump())
    if being["state"] == "alive" and being.get("agent_slug"):
        try:
            being_life._stop_body(being)
            await being_life.spawn_body(get_db(), store, being)
        except Exception as e:  # noqa: BLE001 — heals on next tick
            store.record_event(being["id"], "spawn_failed", {"error": str(e)})
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/body-mrav")
async def set_body_mrav(slug: str, body: BodyMravRequest,
                        user: dict = Depends(get_current_user)):
    """Run the being's BODY on the Mrav runtime (or not), persisted on the
    being record so it survives a body destroy/rebuild — spawn_body rewrites
    the flag file from it every spawn. Respawns an alive body so it takes
    effect now; otherwise it applies on the next spawn."""
    store = get_store()
    being = _run(store.set_body_mrav, user["id"], slug, body.on)
    if being["state"] == "alive" and being.get("agent_slug"):
        try:
            being_life._stop_body(being)
            await being_life.spawn_body(get_db(), store, being)
        except Exception as e:  # noqa: BLE001 — heals on next tick
            store.record_event(being["id"], "spawn_failed", {"error": str(e)})
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/recharge")
async def recharge(slug: str, body: RechargeRequest,
                   user: dict = Depends(get_current_user)):
    """Top up the being's wallet (parent-minted). If this revives an exhausted
    being from torpor, bring it back NOW rather than waiting out the 24h sleep."""
    from datetime import datetime, timezone
    store = get_store()
    v = _run(store.grant, user["id"], slug, body.tokens)
    b = _run(store.get, user["id"], slug)
    now = datetime.now(timezone.utc)
    if b["state"] == "torpor":
        wv = store.wallet_view(b)
        if not wv["enforced"] or wv["balance_tokens"] > wv["reserve_tokens"]:
            b = _run(store.set_state, user["id"], slug, "alive")
            store.record_event(b["id"], "woke_from_torpor", {"cause": "recharge"})
            # Fresh body, not a plain restart — honor any connection/config the
            # parent changed while it slept (see the wake route).
            if b.get("agent_slug"):
                try:
                    being_life._stop_body(b)
                    await being_life.spawn_body(get_db(), store, b)
                except Exception as e:  # noqa: BLE001 — fall back to a restart
                    store.record_event(b["id"], "spawn_failed", {"error": str(e)})
                    being_life._start_body(b)
            store.reschedule_wake(user["id"], slug, now)
            v = _run(store.vitals, user["id"], slug)
    elif b["state"] == "alive":
        # A recharge is "keep going": pull the next tick forward so a being that
        # rested at its daily burn cap (next wake could be tomorrow) resumes now
        # — the grant just raised today's burn headroom, so it can spend again.
        store.reschedule_wake(user["id"], slug, now)
    return v


@router.post("/{slug}/pause")
async def pause(slug: str, user: dict = Depends(get_current_user)):
    """Night falls: state → paused, the body process sleeps too."""
    b = _run(get_store().set_state, user["id"], slug, "paused")
    being_life._stop_body(b)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/wake")
async def wake(slug: str, user: dict = Depends(get_current_user)):
    from datetime import datetime, timezone
    store = get_store()
    b = _run(store.set_state, user["id"], slug, "alive")
    # Resume on the being's own cadence — a wake left in the past by the pause
    # would otherwise fire one stale "catch-up" tick the instant the loop runs.
    new_wake = being_life.wake_reschedule(b, datetime.now(timezone.utc))
    if new_wake is not None:
        _run(store.reschedule_wake, user["id"], slug, new_wake)
    # Wake on a FRESH body: a plain restart reuses the on-disk config.yaml, so
    # anything the parent changed while paused — its connection (body_config),
    # archetype, compact mode — would be silently dropped (a body set to a new
    # provider/model/ctx while paused would wake on the stale one and fail to
    # connect). A full respawn regenerates the config from the live record.
    if b.get("agent_slug"):
        try:
            being_life._stop_body(b)
            await being_life.spawn_body(get_db(), store, b)
        except Exception as e:  # noqa: BLE001 — fall back to a plain restart
            store.record_event(b["id"], "spawn_failed", {"error": str(e)})
            being_life._start_body(b)
    return _run(store.vitals, user["id"], slug)


@router.post("/{slug}/euthanize")
async def euthanize(slug: str, body: EuthanizeRequest,
                    user: dict = Depends(get_current_user)):
    if not body.confirm:
        raise HTTPException(400, "confirm: true required — this is forever")
    b = _run(get_store().set_state, user["id"], slug, "dead")
    being_life._stop_body(b)
    return _run(get_store().vitals, user["id"], slug)


@router.delete("/{slug}")
async def purge(slug: str, user: dict = Depends(get_current_user)):
    """Remove a DEAD being completely — its DB rows AND its VFS home. Forever.
    Guarded to the dead (euthanize first). The body is already stopped."""
    store = get_store()
    being = _run(store.get, user["id"], slug)
    being_life._stop_body(being)          # belt-and-suspenders; dead = stopped
    removed = _run(store.purge, user["id"], slug)
    home_gone = being_life.remove_home(being)
    return {"ok": True, "removed": removed["slug"], "home_removed": home_gone}


@router.post("/{slug}/metamorphose")
async def metamorphose(slug: str, body: MetamorphoseRequest,
                       user: dict = Depends(get_current_user)):
    """The paid rite (§2.1.2): the parent's call here IS the co-sign."""
    being = _run(
        get_store().metamorphose, user["id"], slug,
        body.from_attr, body.to_attr, body.reason,
    )
    return {"ok": True, "being": _run(get_store().vitals, user["id"], being["slug"])}
