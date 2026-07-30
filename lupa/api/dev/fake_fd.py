"""A fake Flight Deck for Lupa development — run the whole product without Captain.

Implements the engine contract the BFF consumes (auth, Vatra lifecycle, facts,
VFS, costs) with realistic fixtures and a time-driven run simulation:

    planning ──(~2s)──▶ awaiting_plan ──(approve)──▶ running ──(~6s)──▶ done

Mints REAL HS256 JWTs with the same secret the BFF verifies against
(``FD_JWT_SECRET``, default ``lupa-dev-secret``), so the full login → refresh →
commission → receipts flow works end-to-end in a browser.

Run:  FD_JWT_SECRET=lupa-dev-secret uvicorn --app-dir lupa/api dev.fake_fd:app --port 25081
Then: FD_JWT_SECRET=lupa-dev-secret LUPA_FD_URL=http://127.0.0.1:25081 <run the BFF>
"""

from __future__ import annotations

import os
import time
import uuid

import jwt as pyjwt
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile

SECRET = os.environ.get("FD_JWT_SECRET", "lupa-dev-secret")
PLAN_SECONDS = 2.0
RUN_SECONDS = 6.0

app = FastAPI(title="Fake Flight Deck (Lupa dev)")

# sid → {created, approved, cancelled, intent, title, round, parent}
_sessions: dict[str, dict] = {}

_REPORT_MD = """# EU heat-pump market — first read

**Bottom line:** the EU heat-pump market reached **€14.2B in 2024** and slows
to ~9% CAGR through 2030 as subsidy programs normalize.

## What the desk verified

| Value | Status |
|---|---|
| 2024 market size €14.2B | verified against EHPA data |
| 2030 outlook €24.1B | derived from the CAGR model |
| DE share 31% | verified |

## Where the team disagrees

One specialist read the 2023 dip as structural; the closer kept the cyclical
reading after the consistency pass — see the receipts panel for the trail.

## Next questions worth a round

- Nordics: heat-pump-ready housing stock vs. grid constraints.
- The subsidy cliff scenario: what happens if Germany's BEG is cut in 2027?
"""

_REPORT_MD_R2 = """# EU heat-pump market — deepened (round 2)

**Bottom line:** the EU heat-pump market reached **€14.2B in 2024** and slows
to ~9% CAGR through 2030 — but the Nordics carry more of that growth than the
round-1 model assumed.

## What the desk verified

| Value | Status |
|---|---|
| 2024 market size €14.2B | verified against EHPA data |
| 2030 outlook €24.1B | derived from the CAGR model |
| DE share 31% | verified |
| Nordics share 2024 18% | verified (new this round) |

## Nordics deep-dive (new)

Housing stock is heat-pump-ready in **74%** of SE/FI detached homes; the
binding constraint is winter grid peaks, not adoption willingness. NO subsidy
dependence is the lowest in the EU.

## Where the team disagrees

The structural-vs-cyclical read on the 2023 dip stands as cyclical; round 2's
Nordics evidence weakens the structural case further.

## Next questions worth a round

- The subsidy cliff scenario: what happens if Germany's BEG is cut in 2027?
"""


def _token(sub: str) -> str:
    # Dev sandbox: everyone is an admin, so the Pack Studio is reachable.
    return pyjwt.encode({"sub": sub, "role": "admin", "type": "access",
                         "iat": int(time.time()), "exp": int(time.time()) + 900},
                        SECRET, algorithm="HS256")


def _user_from(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing token")
    try:
        return pyjwt.decode(authorization[7:], SECRET, algorithms=["HS256"])["sub"]
    except pyjwt.PyJWTError:
        raise HTTPException(401, "bad token")


def _status(s: dict) -> str:
    if s.get("cancelled"):
        return "cancelled"
    age = time.time() - s["created"]
    if s.get("approved") is None and s.get("round", 1) == 1:
        return "planning" if age < PLAN_SECONDS else "awaiting_plan"
    started = s.get("approved") or s["created"]
    return "running" if time.time() - started < RUN_SECONDS else "done"


# ── auth ─────────────────────────────────────────────────────────────


def _login_response(sub: str) -> Response:
    import json as _json
    resp = Response(
        content=_json.dumps({"access_token": _token(sub),
                             "user": {"id": sub, "email": f"{sub}@dev.local",
                                      "display_name": sub, "role": "user"}}),
        media_type="application/json")
    resp.headers.append(
        "set-cookie",
        f"fd_refresh={sub}; HttpOnly; Max-Age=604800; Path=/fd/auth; SameSite=lax")
    return resp


@app.post("/fd/auth/login")
@app.post("/fd/auth/register")
async def login(body: dict):
    email = str(body.get("email", "dev@dev.local"))
    return _login_response(email.split("@")[0] or "dev")


@app.post("/fd/auth/refresh")
async def refresh(cookie: str | None = Header(default=None)):
    sub = ""
    for part in (cookie or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == "fd_refresh":
            sub = v
    if not sub:
        raise HTTPException(401, "no refresh cookie")
    return _login_response(sub)


@app.post("/fd/auth/logout")
async def logout():
    return {"ok": True}


@app.get("/fd/auth/me")
async def me(authorization: str | None = Header(default=None)):
    sub = _user_from(authorization)
    return {"id": sub, "email": f"{sub}@dev.local", "role": "user"}


# ── vatra lifecycle ──────────────────────────────────────────────────


@app.post("/fd/vatra/start")
async def vatra_start(body: dict, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {"created": time.time(), "approved": None,
                      "intent": body.get("intent", ""),
                      "title": body.get("title", ""), "round": 1}
    return {"session_id": sid, "title": body.get("title", ""), "status": "planning"}


@app.post("/fd/vatra/plan/approve")
async def plan_approve(body: dict, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    s = _sessions.get(body.get("session_id", ""))
    if not s:
        raise HTTPException(404, "session not found")
    s["approved"] = time.time()
    return {"ok": True, "status": "running"}


@app.post("/fd/vatra/plan/cancel")
async def plan_cancel(body: dict, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    s = _sessions.get(body.get("session_id", ""))
    if not s:
        raise HTTPException(404, "session not found")
    if _status(s) != "awaiting_plan":
        raise HTTPException(409, "not at the plan gate")
    s["cancelled"] = True
    return {"ok": True}


@app.post("/fd/vatra/sessions/{sid}/continue")
async def vatra_continue(sid: str, body: dict,
                         authorization: str | None = Header(default=None)):
    _user_from(authorization)
    parent = _sessions.get(sid)
    if not parent:
        raise HTTPException(404, "session not found")
    new_sid = uuid.uuid4().hex[:12]
    _sessions[new_sid] = {"created": time.time(), "approved": time.time(),
                          "intent": body.get("instruction", ""),
                          "title": parent["title"],
                          "round": parent.get("round", 1) + 1, "parent": sid}
    return {"ok": True, "session_id": new_sid, "round": _sessions[new_sid]["round"]}


@app.get("/fd/basna/sessions/{sid}")
async def session_detail(sid: str, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    # A Kalup factory run: the "team" configures a desk instead of reporting.
    if s["intent"].startswith("KALUP PACK DRAFT"):
        status = _status(s)
        return {"id": sid, "status": status, "title": "pack draft",
                "intent": s["intent"],
                "truth": _KALUP_TRUTH if status == "done" else "",
                "config": {"mode": "vatra"},
                "route": {"group0_plan": {"steps": [
                    {"agent": "vertical-architect", "does": "derive vocabulary + intake"},
                    {"agent": "editor-writer", "does": "onboarding + golden eval task"}]}},
                "analysis": None}
    # A Basna (second-opinion) session carries its own truth once executed.
    if s.get("basna"):
        done = bool(s.get("basna_done"))
        return {"id": sid, "status": "done" if done else "running",
                "title": s["title"], "intent": s["intent"],
                "truth": _SECOND_OPINION_MD if done else "",
                "confidence": 0.79 if done else 0.0,
                "config": {"mode": "basna"}, "route": {}, "analysis": {}}
    status = _status(s)
    report = _REPORT_MD if s.get("round", 1) == 1 else _REPORT_MD_R2
    detail: dict = {
        "id": sid, "status": status, "title": s["title"], "intent": s["intent"],
        "truth": report if status == "done" else "",
        "config": {"mode": "vatra", "round": s.get("round", 1)},
        "route": {"group0_plan": {
            "steps": [
                {"agent": "deep-researcher", "does": "market sizing: EHPA + Eurostat, 2019-2024"},
                {"agent": "analyst", "does": "CAGR model to 2030 + subsidy-cliff scenario"},
                {"agent": "editor-writer", "does": "assemble the report, resolve conflicts"},
            ]}},
        "analysis": None,
    }
    if status == "done":
        detail["analysis"] = {
            "quality_verdict": "pass",
            "blocking": {"rounds": 1, "verdict": "pass"},
            "quality_metrics": {
                "claims_checked": 14, "claims_confirmed": 11, "claims_refuted": 1,
                "claims_unverifiable": 2, "claims_hedged": 2,
                "consistency_critical": 0, "consistency_major": 1,
                "consistency_initial_critical": 2, "consistency_revised": True,
                "gaps_major": 1, "gaps_minor": 2,
                "contract_checked": 4, "contract_failed_critical": 0,
                "contract_failed_major": 0, "contract_unclear": 1,
                "block_rounds": 1, "quality_verdict": "pass",
                "acted_retries": 1, "budget_spent_tokens": 184_000,
            },
            "consistency": {"values_checked": 21, "relations_checked": 6,
                            "initial_critical": 2, "initial_major": 3,
                            "critical": 0, "major": 1, "revised": True},
            "gaps": [
                {"severity": "major", "text": "Nordics housing-stock readiness not covered"},
                {"severity": "minor", "text": "No FR regional split"},
                {"severity": "minor", "text": "2021 base-year sensitivity untested"},
            ],
        }
    return detail


@app.get("/fd/basna/sessions/{sid}/progress")
async def progress(sid: str, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    status = _status(s)
    started = s.get("approved") or s["created"]
    steps = [
        ("phase", "Planning"),
        ("plan", "Lead decomposed the brief into 3 slices"),
        ("phase", "Main"),
        ("dispatch", "deep-researcher: sizing the market (EHPA, Eurostat)"),
        ("dispatch", "analyst: building the CAGR model"),
        ("note", "facts ledger: 9 value(s) recorded"),
        ("verify", "consistency: 2 critical findings → correction applied"),
        ("phase", "Synthesizing"),
        ("verify", "contract: 4 checked, 0 failed"),
    ]
    n = len(steps) if status == "done" else min(
        len(steps), int((time.time() - started) / (RUN_SECONDS / len(steps))) + 1)
    events = [{"i": i, "stage": st, "message": msg}
              for i, (st, msg) in enumerate(steps[:n])]
    if status == "done":
        events.append({"i": len(events), "stage": "cost",
                       "message": "$0.42 · effective $5.04/hr", "usd": 0.42})
    return {"events": events, "active": status == "running"}


@app.get("/fd/basna/sessions/{sid}/facts")
async def facts(sid: str, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    return {"project": f"vatra-{sid[:8]}", "count": 5, "facts": [
        {"key": "eu_hp_market_2024_eur_b", "value": "14.2", "unit": "B EUR",
         "status": "verified", "updated_by": "deep-researcher"},
        {"key": "eu_hp_market_2030_eur_b", "value": "24.1", "unit": "B EUR",
         "status": "derived", "updated_by": "analyst"},
        {"key": "cagr_2024_2030_pct", "value": "9.2", "unit": "%",
         "status": "derived", "updated_by": "analyst"},
        {"key": "de_share_2024_pct", "value": "31", "unit": "%",
         "status": "verified", "updated_by": "deep-researcher"},
        {"key": "units_sold_2024_m", "value": "2.6", "unit": "M",
         "status": "estimated", "updated_by": "deep-researcher"},
    ], "conflicts": [
        {"key": "units_sold_2024_m", "value": "3.1",
         "existing": "2.6", "by": "analyst"},
    ]}


# ── vfs + costs ──────────────────────────────────────────────────────


_CONTRACT = {
    "constraints": [
        {"id": "c1", "text": "All monetary values in EUR", "severity": "critical",
         "status": "pass"},
        {"id": "c2", "text": "2030 projection must state its CAGR assumption",
         "severity": "major", "status": "pass"},
        {"id": "c3", "text": "Cover at least DE, FR, IT, Nordics", "severity": "major",
         "status": "pass"},
        {"id": "c4", "text": "Cite a primary source for the 2024 base year",
         "severity": "critical", "status": "unclear"},
    ]
}


@app.get("/fd/vfs/list")
async def vfs_list(project: str, path: str = "",
                   authorization: str | None = Header(default=None)):
    _user_from(authorization)
    return {"entries": [
        {"name": "r1-report.md", "dir": False, "size": len(_REPORT_MD)},
        {"name": "r1-market-model.md", "dir": False, "size": 800},
        {"name": "sources", "dir": True},
    ]}


@app.get("/fd/vfs/read")
async def vfs_read(project: str, path: str,
                   authorization: str | None = Header(default=None)):
    import json as _json
    _user_from(authorization)
    if path == ".contract.json":
        return {"text": _json.dumps(_CONTRACT), "binary": False}
    if path.endswith("market-model.md"):
        return {"text": "# Market model\n\nCAGR 9.2% on a €14.2B 2024 base → €24.1B in 2030.",
                "binary": False}
    return {"text": _REPORT_MD, "binary": False}


@app.get("/fd/costs")
async def costs(limit: int = 200, run_kind: str = "",
                authorization: str | None = Header(default=None)):
    _user_from(authorization)
    rows = []
    for sid, s in _sessions.items():
        if _status(s) == "done":
            rows.append({"run_kind": "vatra", "run_id": sid, "usd": 0.42,
                         "elapsed_seconds": 300.0,
                         "usage": {"prompt_tokens": 152_000, "completion_tokens": 32_000},
                         "at": "2026-07-30T10:00:00+00:00"})
    return {"costs": rows, "count": len(rows), "priced": len(rows),
            "total_usd": round(0.42 * len(rows), 2) or None}


# ── house style: forge + archetypes ──────────────────────────────────

_ARCHETYPES: dict[str, dict] = {}


@app.post("/fd/archetypes/forge")
async def forge(instructions: str = Form(""), count: str = Form("0"),
                files: list[UploadFile] = File(default=[]),
                authorization: str | None = Header(default=None)):
    _user_from(authorization)
    names = ", ".join(f.filename or "doc" for f in files) or "your instructions"
    return {"archetypes": [
        {"id": "house-lead-analyst", "role": "House Lead Analyst",
         "tier": "reason",
         "instructions": f"Lead the analysis in the house style, grounded in {names}."},
        {"id": "house-fact-checker", "role": "House Fact-Checker",
         "tier": "balanced",
         "instructions": "Verify every figure against primary sources; house rigor."},
        {"id": "house-writer", "role": "House Writer",
         "tier": "balanced",
         "instructions": "Assemble the memo in the house voice: terse, sourced, decision-first."}]}


@app.put("/fd/archetypes/{aid}")
async def put_archetype(aid: str, body: dict,
                        authorization: str | None = Header(default=None)):
    _user_from(authorization)
    _ARCHETYPES[aid] = {**body, "id": aid, "source": "user"}
    return _ARCHETYPES[aid]


@app.get("/fd/archetypes/mine")
async def my_archetypes(authorization: str | None = Header(default=None)):
    _user_from(authorization)
    return list(_ARCHETYPES.values())


# ── second opinion (Basna ensemble) ──────────────────────────────────


@app.post("/fd/basna/route")
async def basna_route(body: dict, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    sid = "basna-" + uuid.uuid4().hex[:8]
    _sessions[sid] = {"created": time.time(), "approved": time.time(),
                      "intent": body.get("intent", ""), "title": "second opinion",
                      "round": 99, "basna": True}
    return {"session_id": sid, "selected": ["deep-researcher", "analyst", "skeptic"]}


@app.post("/fd/basna/execute")
async def basna_execute(body: dict, authorization: str | None = Header(default=None)):
    _user_from(authorization)
    s = _sessions.get(body.get("session_id", ""))
    if not s:
        raise HTTPException(404, "session not found")
    s["basna_done"] = True
    return {"session_id": body["session_id"], "confidence": 0.79,
            "truth": _SECOND_OPINION_MD}


_SECOND_OPINION_MD = """# Independent second read

An independent ensemble re-ran the same brief from scratch. **It broadly agrees
with the desk**, with two differences worth your attention:

- **2030 outlook.** The desk's €24.1B rests on a 9.2% CAGR; this ensemble reads
  the subsidy normalization as steeper and lands at **€22.4B** (≈7% below).
- **DE share.** Both put Germany near a third of the market — no disagreement.

The ensemble did *not* find the Nordics under-weighting the desk flagged, so
treat that as the desk's own signal, not a cross-confirmed finding.
"""


_KALUP_TRUTH = """The team configured the desk. Manifest:

```json
{
  "name": "Tender Desk",
  "tagline": "Public-sector RFPs, decoded",
  "theme": {
    "accent": "#2fb2a0",
    "accent_soft": "#7ad4c8",
    "bg": "#0e1413",
    "surface": "#17201e",
    "border": "#26332f",
    "text": "#e6ece9",
    "text_dim": "#93a39d"
  },
  "vocabulary": {
    "stream": "Tender",
    "streams": "Tenders",
    "commission": "Analyze",
    "brief": "Tender brief",
    "round": "Pass",
    "report": "Assessment",
    "plan_gate_title": "Review the analysis plan",
    "plan_gate_hint": "Your bid team drafted this plan. Approve to start, or cancel and rephrase.",
    "composer_placeholder": "Paste the RFP scope or describe the tender: issuer, deadline, lot structure.",
    "continue_placeholder": "Deepen this tender: compliance matrix, pricing angle, incumbent risk…",
    "empty_streams": "No tenders yet. Add the first RFP to analyze.",
    "new_stream": "New tender",
    "receipts_title": "Receipts",
    "receipts_hint": "How this assessment was verified, and what it cost.",
    "facts_title": "Facts ledger",
    "cost_title": "Cost",
    "brief_title": "Tender watch",
    "brief_hint": "The desk re-checks this tender on a schedule and reports only changes.",
    "brief_placeholder": "What should the desk keep watching? (amendments, Q&A, deadlines)",
    "inbox_title": "Watch inbox"
  },
  "intake": {
    "types": [{"id": "tender", "label": "Tender analysis",
               "description": "Eligibility, compliance matrix, scoring odds.",
               "default_max_agents": 5}]
  },
  "quality": {"profile": "thorough"},
  "briefs": {"presets": [
    {"id": "daily", "label": "Daily", "hours": 24},
    {"id": "weekly", "label": "Weekly", "hours": 168}
  ]},
  "roi": {"analyst_hourly_usd": 90, "analyst_label": "a bid consultant"},
  "evals": [{"brief": "Analyze a sample municipal IT-services RFP: eligibility, compliance matrix, three scoring risks."}],
  "onboarding_md": "# Welcome to Tender Desk\\n\\nPaste an RFP and the desk returns an assessment with receipts: eligibility, a compliance matrix, and scoring risks — each figure verified.\\n\\nSet a **Tender watch** and the desk re-checks amendments and Q&A on a schedule."
}
```
"""


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("FAKE_FD_PORT", "25081")))


if __name__ == "__main__":
    main()
