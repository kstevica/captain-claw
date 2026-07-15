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
    fee_tokens: int


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


@router.get("/village-meta")
async def get_village_meta(user: dict = Depends(get_current_user)):
    """The village description the parent set — shown on their public /village."""
    return _run(get_store().get_village_meta, user["id"])


@router.post("/village-meta")
async def set_village_meta(body: VillageMetaRequest,
                           user: dict = Depends(get_current_user)):
    return _run(get_store().set_village_meta, user["id"], body.description)


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


class QuestRequest(BaseModel):
    title: str
    spec: str
    fee_tokens: int
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


@router.post("/quests")
async def post_quest(body: QuestRequest,
                     user: dict = Depends(get_current_user)):
    """Post an OPEN bounty — any eligible being may claim it."""
    return {"quest": _run(get_store().post_quest, user["id"], body.title,
                          body.spec, body.fee_tokens, body.origin)}


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
    """Post a fixed-fee chore. The fee mints only on judged completion."""
    return {"chore": _run(get_store().post_chore, user["id"], slug,
                          body.spec, body.fee_tokens)}


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
    _run(get_store().set_stage, user["id"], slug, body.stage)
    if body.stage == "adult":
        # The unsealing rite: her childhood assessment records become hers.
        from captain_claw.flight_deck import being_assessment
        being = _run(get_store().get, user["id"], slug)
        _run(being_assessment.release_assessments, get_store(), being)
    return _run(get_store().vitals, user["id"], slug)


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


@router.post("/{slug}/pause")
async def pause(slug: str, user: dict = Depends(get_current_user)):
    """Night falls: state → paused, the body process sleeps too."""
    b = _run(get_store().set_state, user["id"], slug, "paused")
    being_life._stop_body(b)
    return _run(get_store().vitals, user["id"], slug)


@router.post("/{slug}/wake")
async def wake(slug: str, user: dict = Depends(get_current_user)):
    b = _run(get_store().set_state, user["id"], slug, "alive")
    being_life._start_body(b)
    return _run(get_store().vitals, user["id"], slug)


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
