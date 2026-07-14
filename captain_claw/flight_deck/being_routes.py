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


class AllowanceRequest(BaseModel):
    preset: str
    daily_burn_cap: int | None = None
    savings_ceiling: int | None = None


class StageRequest(BaseModel):
    stage: str


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


@router.get("/{slug}")
async def being_vitals(slug: str, user: dict = Depends(get_current_user)):
    return _run(get_store().vitals, user["id"], slug)


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


@router.post("/{slug}/metamorphose")
async def metamorphose(slug: str, body: MetamorphoseRequest,
                       user: dict = Depends(get_current_user)):
    """The paid rite (§2.1.2): the parent's call here IS the co-sign."""
    being = _run(
        get_store().metamorphose, user["id"], slug,
        body.from_attr, body.to_attr, body.reason,
    )
    return {"ok": True, "being": _run(get_store().vitals, user["id"], being["slug"])}
