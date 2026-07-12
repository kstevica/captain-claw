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
from captain_claw.flight_deck import being_life
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
