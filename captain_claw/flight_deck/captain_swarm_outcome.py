"""Captain Swarm — the best outcome of one real task (Kind B), in rehearsal.

docs/captain-swarm-plan.md §1 (Kind B), §4 and §15 Phase 3b. One task —
"the cheapest new copy of this book that ships to Croatia", "a free slot for
X next week" — tried several ways. Each attempt is a Bat run in rehearsal: its
workers read, search and browse, but every outward action becomes an intent
and the browser stops at the commit point (``captain_claw/rehearsal_turn.py``).
An attempt ends with what it found, as candidates with a link and a number.

Captain doesn't take its word for any of it (:func:`verify`). It fetches each
link itself — public addresses only — and a candidate counts only if the page
shows its number, its name, and the words the task insists on. The best
verified number is the attempt's score; the best attempt so far is the
champion. The held-out check, for an outcome, is a fresh look at the
champion's best candidate: offers and slots change.

Attempts run at once (Phase 4): as many as the owner has Bat runs free
(:func:`capacity`), each with an equal part of the budget left, each idea with
its own sites; the rehearsal guard keeps a second attempt out of a site
another one has open in its browser. Before an attempt becomes the best, a
model reviewer reads the task and what the page says around the number
(:func:`review_material`): a kit at the price of the board doesn't count.

When the owner picks the winner, it is carried out once (:func:`carry_out_task`):
one ordinary Bat run re-checks the page, asks the owner if anything changed,
and acts through Bat's own gate — the owner's Go for spending, mail and accounts.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import math
import re
import secrets
import socket
import urllib.parse
import uuid
from typing import Any

from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
from captain_claw.logging import get_logger

log = get_logger(__name__)

WORST = 1e12
_MAX_CANDIDATES = 8
_FETCH_TIMEOUT_S = 15.0
_FETCH_MAX_BYTES = 2_000_000
_REDIRECTS = 3
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def _store():
    from captain_claw.flight_deck import captain_swarm_loop as loop
    store = loop.get_store()
    if store is None:
        raise RuntimeError("Captain Swarm's store isn't open")
    return store


# ── the objective ────────────────────────────────────────────────────


def outcome_template(*, task: str, label: str = "price", direction: str = "min", max_value: float | None = None,
                     must_include: list[str] | None = None, currency: str = "") -> dict[str, Any]:
    return {
        "kind": "outcome",
        "title": f"Best result: {task.strip()[:90]}",
        "goal": task.strip(),
        "subject": {"type": "attempts", "task": task.strip(),
                    "rank": {"label": label, "direction": direction},
                    "constraints": {"max_value": max_value, "must_include": list(must_include or []),
                                    "currency": currency},
                    "freshness_hours": 24},
        "metric": {"kind": "state", "read": "best_verified", "direction": direction},
        "guardrails": [],
        "noise": {"min_gain": 0.01},
        "lane": {"kind": "bat", "per_round": 3, "parallel": 3, "attempt_usd": 2.0},
        "budget": {"usd": 10.0, "hours": 4.0, "max_experiments": 6},
        "stop": {"target": None, "stall_rounds": 2, "remeasure_every": 99},
    }


def bat_slots() -> int:
    """Bat runs one owner may have going at once: every attempt is one."""
    from captain_claw.flight_deck.bat_routes import _BAT_MAX_RUNS_PER_OWNER
    return int(_BAT_MAX_RUNS_PER_OWNER)


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck.captain_swarm_objective import DEFAULT_BUDGET, ObjectiveError, _num, lanes
    subj = raw.get("subject") or {}
    task = " ".join(str(subj.get("task") or raw.get("goal") or "").split())
    if len(task) < 10:
        raise ObjectiveError("Say what to find, in a sentence or two.")
    rank = subj.get("rank") or {}
    direction = str(rank.get("direction") or "min")
    if direction not in ("min", "max"):
        raise ObjectiveError("Lower or higher is better: min or max.")
    cons = subj.get("constraints") or {}
    max_value = cons.get("max_value")
    budget = {**DEFAULT_BUDGET, **(raw.get("budget") or {})}
    lane = raw.get("lane") or {}
    stop = raw.get("stop") or {}
    parallel, per_round = lanes(lane, default=1, most=bat_slots(), per_round_most=6)
    return {
        "kind": "outcome",
        "title": str(raw.get("title") or "").strip()[:200] or f"Best result: {task[:90]}",
        "goal": task[:2000],
        "subject": {"type": "attempts", "task": task[:2000],
                    "rank": {"label": str(rank.get("label") or "price")[:40], "direction": direction},
                    "constraints": {
                        "max_value": None if max_value in (None, "") else _num(max_value, "The most it may be", 0, 1e9),
                        "must_include": [" ".join(str(w).split())[:60] for w in cons.get("must_include") or []
                                         if str(w).strip()][:8],
                        "currency": str(cons.get("currency") or "")[:8]},
                    "freshness_hours": _num(subj.get("freshness_hours", 24), "How long a result stays fresh", 1, 720)},
        "metric": {"kind": "state", "read": "best_verified", "direction": direction},
        "guardrails": [],
        "noise": {"min_gain": _num((raw.get("noise") or {}).get("min_gain", 0.01), "The smallest gain that counts", 0, 1e9)},
        "lane": {"kind": "bat", "per_round": per_round, "parallel": parallel,
                 "attempt_usd": _num(lane.get("attempt_usd", 2.0), "The most one attempt may spend", 0.1, 100)},
        "budget": {"usd": _num(budget["usd"], "The usd limit", 0.5, 500), "hours": _num(budget["hours"], "The hours limit", 0.25, 48),
                   "max_experiments": int(_num(budget["max_experiments"], "The max experiments limit", 1, 30))},
        "stop": {"target": None, "stall_rounds": int(_num(stop.get("stall_rounds", 2), "Rounds without a gain", 1, 10)),
                 "remeasure_every": 99},
    }


def worst(direction: str) -> float:
    return WORST if direction == "min" else -WORST


# ── where a run starts, and the lane ─────────────────────────────────


async def start_point(run: dict) -> str:
    """Nothing found yet."""
    vid = f"v_{secrets.token_hex(6)}"
    await _store().add_version(version_id=vid, run_id=run["id"], content={"candidates": []})
    return vid


def lane_brief(run: dict, exp: dict) -> str:
    subj = run["objective"]["subject"]
    cons = subj["constraints"]
    rank = subj["rank"]
    sites = (exp.get("change") or {}).get("sites") or []
    lines = [
        f"Task: {subj['task']}",
        "",
        f"Try it this way: {exp['hypothesis']}",
        f"Why this way: {exp['why']}" if exp.get("why") else "",
        f"Mainly on: {', '.join(sites)}. Other attempts run at the same time on other sites; if the browser says a "
        "site is busy, use another source or read the page with web_fetch." if sites else "",
        "",
        "This is a REHEARSAL. Search, read and browse as much as you need, and fill in forms if it helps you see "
        "the real price or slot — but nothing outside changes: sending, buying, booking, signing up or submitting "
        "is recorded instead of done, and the browser stops before any such click. That's expected; don't try "
        "to get around it. Stop at the point where the next step would commit, and say what that step would be.",
        "",
        f"What counts: the {rank['label']} ({'lower' if rank['direction'] == 'min' else 'higher'} is better)"
        + (f", in {cons['currency']}" if cons.get("currency") else "")
        + (f", at most {cons['max_value']:g}" if cons.get("max_value") is not None else "")
        + (f"; it must be: {', '.join(cons['must_include'])}" if cons.get("must_include") else "") + ".",
        "To search: web_search; if it has no key here, web_fetch https://html.duckduckgo.com/html/?q=<your query> "
        "and read the results.",
        "Captain will open every link you give and check the page itself: give the page that shows the "
        f"{rank['label']}, and the {rank['label']} exactly as that page shows it.",
        "",
        "End your answer with this JSON block (up to 5 candidates, best first):",
        '```json\n{"candidates": [{"title": "<what it is, as the page names it>", "url": "<the page>", '
        f'"value": <the {rank["label"]} as a number>, "currency": "<if any>", '
        '"details": "<what else matters: shipping, date, seller>", "next_step": "<the action that would commit>"}]}\n```',
    ]
    return "\n".join(x for x in lines if x is not None).strip()


class BatLane:
    """One attempt = one Bat run in rehearsal. Its workers carry the attempt's
    tag (``CLAW_REHEARSAL``), so nothing they do reaches the world."""

    async def start(self, run: dict, exp: dict, brief: str) -> dict:
        from captain_claw.flight_deck import bat_loop
        store = bat_loop.get_store()
        if store is None:
            raise RuntimeError("Bat isn't running on this deck")
        bat_id = f"bat_{uuid.uuid4().hex[:12]}"
        # Its part of what's left: the attempts of one wave share the budget equally.
        left = exp.get("usd_share", run["objective"]["budget"]["usd"] - run["spent_usd"])
        cap = max(0.1, min(float(run["objective"]["lane"].get("attempt_usd") or 2.0), float(left)))
        await store.create_run(
            run_id=bat_id, owner_id=run["owner_id"], title=f"Captain Swarm #{exp['round']}.{exp['n']}: "
                                                            f"{exp['hypothesis'][:60]}",
            task=lane_brief(run, exp),
            config={"source": "captain_swarm", "rehearsal": rehearsal.make_tag(run["id"], exp["id"]),
                    "worker_mode": "plain", "steps": []},
            origin={}, llm_usd_cap=cap, status="planning")
        await store.append_event(bat_id, "note", f"a Captain Swarm rehearsal for {run['id']}")
        try:
            await bat_loop.kick()
        except Exception:
            log.warning("captain swarm: Bat kick failed (it starts on the next tick)", exc_info=True)
        return {"child_id": bat_id, "branch": ""}

    async def poll(self, run: dict, exp: dict) -> dict:
        from captain_claw.flight_deck import bat_loop
        from captain_claw.flight_deck.bat_store import TERMINAL_STATES
        store = bat_loop.get_store()
        b = await store.get_run(exp["child_id"]) if store is not None else None
        if b is None:
            return {"state": "failed", "note": "its Bat run is gone"}
        usd, tokens = float(b.get("cumulative_usd") or 0.0), int(b.get("cumulative_tokens") or 0)
        if b["status"] not in TERMINAL_STATES:
            return {"state": "running", "usd": usd, "tokens": tokens}
        candidates = parse_candidates(b.get("truth") or "")
        if not candidates:
            why = b.get("stopped_reason") or "no candidates in its answer"
            return {"state": "failed", "usd": usd, "tokens": tokens,
                    "note": f"its Bat run ended ({b['status']}) with nothing to check: {why}"[:300]}
        vid = f"v_{secrets.token_hex(6)}"
        await _store().add_version(version_id=vid, run_id=run["id"], parent=exp["base_sha"],
                                   content={"candidates": candidates, "bat_run": b["id"],
                                            "summary": (b.get("truth") or "")[:4000]})
        return {"state": "done", "head_sha": vid, "commits": len(candidates), "dirty": 0, "usd": usd, "tokens": tokens}

    async def stop(self, run: dict, exp: dict) -> None:
        from captain_claw.flight_deck import bat_loop
        store = bat_loop.get_store()
        if store is not None and exp.get("child_id"):
            await bat_loop.cancel_run(store, exp["child_id"])

    async def discard(self, run: dict, exp: dict) -> None:
        return None  # the Bat run stays, as the attempt's record


async def no_branch(run: dict, version_id: str) -> None:
    return None


async def capacity(run: dict, want: int) -> int:
    """Lanes free now: every attempt is a Bat run, so Bat's limit per owner, less
    the owner's other Bat runs going (between waves none of this run's are)."""
    from captain_claw.flight_deck import bat_loop
    from captain_claw.flight_deck.bat_store import RUNNING_STATES
    store = bat_loop.get_store()
    if store is None:
        return want  # the lane says why it can't start
    busy = sum(1 for r in await store.list_runs(run["owner_id"], limit=100) if r["status"] in RUNNING_STATES)
    return max(0, min(want, bat_slots() - busy))


def parse_candidates(text: str) -> list[dict]:
    """The candidates an attempt ended with: the last JSON block holding them."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    blocks += re.findall(r"(\{\s*\"candidates\".*\})", text or "", re.DOTALL)
    for raw in reversed(blocks):
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        out = []
        for c in data.get("candidates") or [] if isinstance(data, dict) else []:
            if not isinstance(c, dict):
                continue
            url = str(c.get("url") or "").strip()
            try:
                value = float(str(c.get("value")).replace(",", "")) if c.get("value") is not None else None
            except ValueError:
                value = None
            if not url.startswith(("http://", "https://")) or value is None or not math.isfinite(value):
                continue
            out.append({"title": str(c.get("title") or "")[:300], "url": url[:2000], "value": value,
                        "currency": str(c.get("currency") or "")[:8], "details": str(c.get("details") or "")[:600],
                        "next_step": str(c.get("next_step") or "")[:300]})
        if out:
            return out[:_MAX_CANDIDATES]
    return []


# ── screen: proof that nothing reached the world ─────────────────────


def worker_prefix(bat_id: str) -> str:
    """What every worker of Bat run *bat_id* is named: ``bat-<its id's first 8>-…``,
    slugged the way Flight Deck slugs agent names."""
    return re.sub(r"[^a-z0-9-]", "-", f"bat-{bat_id[:8]}".lower()).strip("-") + "-"


async def screen(run: dict, exp: dict) -> dict:
    version = await _store().get_version(exp["head_sha"])
    candidates = (version or {}).get("content", {}).get("candidates") or []
    out = {"changed": [c["url"] for c in candidates], "diff_stat": f"{len(candidates)} candidate(s)", "hits": []}
    if not candidates:
        return {**out, "ok": False, "reason": "it found nothing to check"}
    try:
        found = await rehearsal.leaks(worker_prefix(exp["child_id"]), since=float(exp["created_at"]) - 5)
    except Exception as exc:  # noqa: BLE001 — no proof is no pass
        return {**out, "ok": False, "reason": f"couldn't check the tool-call register: {exc}"}
    if found:
        what = "; ".join(f"{f['what']} by {f['agent']}" for f in found[:4])
        return {**out, "ok": False, "hits": found,
                "reason": f"its workers did something outside after all ({what}) — the guard has a hole to fix"}
    return {**out, "ok": True, "reason": ""}


# ── measure: Captain checks every claim itself ───────────────────────


def _public(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified)


async def _resolves_public(host: str) -> bool:
    if not host or host.lower() in ("localhost",) or host.lower().endswith((".localhost", ".local", ".internal")):
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addrs = {i[4][0] for i in infos}
    return bool(addrs) and all(_public(a.split("%")[0]) for a in addrs)


async def fetch_text(url: str) -> tuple[str, str]:
    """``(page text, final url)`` of a public web page, or ``("", why)``. Only
    http(s), only public addresses, every redirect checked again."""
    import httpx
    current = url
    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S, follow_redirects=False,
                                 headers={"User-Agent": _UA, "Accept-Language": "en,hr;q=0.8"}) as client:
        for _ in range(_REDIRECTS + 1):
            parts = urllib.parse.urlsplit(current)
            if parts.scheme not in ("http", "https") or not await _resolves_public(parts.hostname or ""):
                return "", f"not a public web address: {current[:120]}"
            try:
                async with client.stream("GET", current) as r:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        current = urllib.parse.urljoin(current, r.headers["location"])
                        continue
                    if r.status_code != 200:
                        return "", f"HTTP {r.status_code}"
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > _FETCH_MAX_BYTES:
                            break
            except httpx.HTTPError as exc:
                return "", f"couldn't fetch it: {type(exc).__name__}"
            return page_text(body.decode(r.encoding or "utf-8", errors="replace")), current
    return "", "too many redirects"


def page_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html)
    # Prices often sit in attributes (content="412.50", data-price): keep those values too.
    attrs = " ".join(re.findall(r'(?i)(?:content|data-price|value)="([^"]{1,40})"', raw_html))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text + " " + attrs).replace("\xa0", " ").split())


def number_forms(value: float) -> list[str]:
    """How a page may write *value*: 412 · 412.50 · 412,50 · 1,240.50 · 1.240,50 · 1 240,50."""
    whole = abs(value - round(value)) < 1e-9
    forms = set()
    for decimals in ([0, 2] if whole else [2, 1]):
        s = f"{abs(value):,.{decimals}f}"           # 1,240.50
        forms |= {s, s.replace(",", ""), s.replace(",", "").replace(".", ","),
                  s.replace(",", "X").replace(".", ",").replace("X", "."),
                  s.replace(",", " "), s.replace(",", " ").replace(".", ",")}
    return sorted(f for f in forms if f)


def excerpt(text: str, value: float, width: int = 240) -> str:
    """What the page says around *value*, for the reviewer."""
    for f in number_forms(value):
        m = re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d])", text or "")
        if m:
            return text[max(0, m.start() - width):m.end() + width]
    return (text or "")[:2 * width]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())}


def verify(candidate: dict, text: str, constraints: dict, direction: str) -> tuple[bool, str]:
    """Does the page bear the candidate out? ``(verified, why not)``."""
    if not text:
        return False, "no page text"
    low = text.lower()
    if not any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d])", text) for f in number_forms(candidate["value"])):
        return False, f"the page doesn't show {candidate['value']:g}"
    title_words = _words(candidate.get("title") or "")
    if title_words and len(title_words & _words(text)) / len(title_words) < 0.5:
        return False, "the page doesn't name what it claims"
    for word in constraints.get("must_include") or []:
        if word.lower() not in low:
            return False, f"the page doesn't say {word!r}"
    cap = constraints.get("max_value")
    if cap is not None and (candidate["value"] > cap if direction == "min" else candidate["value"] < cap):
        return False, f"{candidate['value']:g} is past the limit of {cap:g}"
    return True, ""


async def measure(run: dict, version_id: str, which: str) -> dict:
    """Open the version's candidates and keep what the pages bear out. For the
    held-out look, only its best candidate, fetched fresh."""
    subj = run["objective"]["subject"]
    direction = subj["rank"]["direction"]
    version = await _store().get_version(version_id)
    content = (version or {}).get("content") or {}
    candidates = list(content.get("candidates") or [])
    if which != "dev":
        best = content.get("best")
        if not best:
            # Nothing was claimed here (the start): nothing to look at again.
            value = worst(direction)
            return {"value": value, "stats": {"n": 1, "mean": value, "stdev": 0.0, "low": value, "high": value},
                    "guard": {}, "per_case": {}, "checks": [], "best": None, "error": ""}
        candidates = [c for c in candidates if c["url"] == best.get("url")][:1]
    checks, per_case = [], {}
    for c in candidates[:_MAX_CANDIDATES]:
        text, where = await fetch_text(c["url"])
        ok, why = verify(c, text, subj["constraints"], direction) if text else (False, where)
        checks.append({**c, "verified": ok, "why": why, "snippet": excerpt(text, c["value"]) if ok else ""})
        per_case[c["url"]] = 1.0 if ok else 0.0
    good = [c for c in checks if c["verified"]]
    best = (min if direction == "min" else max)(good, key=lambda c: c["value"]) if good else None
    if which != "dev" and not best:
        # The fresh look didn't bear it out: offers and slots change.
        why = checks[0]["why"] if checks else "nothing to look at again"
        return {"value": None, "error": f"a fresh look didn't bear it out: {why}"}
    value = best["value"] if best else worst(direction)
    if which == "dev":
        await _store().add_version(version_id=f"{version_id}_c", run_id=run["id"], parent=version_id,
                                   content={"checks": checks})
        if best:
            await _record_best(version_id, best)
    return {"value": value, "stats": {"n": 1, "mean": value, "stdev": 0.0, "low": value, "high": value},
            "guard": {}, "per_case": per_case, "checks": checks, "best": best, "error": ""}


async def _record_best(version_id: str, best: dict) -> None:
    """Keep the best verified candidate on the version, for the held-out look and the carry-out."""
    store = _store()
    v = await store.get_version(version_id)
    if v is None:
        return
    db = await store._ensure_db()
    await db.execute("UPDATE cs_versions SET content = ? WHERE id = ?",
                     (json.dumps({**v["content"], "best": best}), version_id))
    await db.commit()


# ── the proposer ─────────────────────────────────────────────────────


_SYSTEM = (
    "You plan attempts at one real task for Captain Swarm. Each attempt is one way of going about it — which "
    "sources to try, how to search, what to compare — tried by an assistant with web search and a browser, in "
    "a rehearsal where nothing is bought, sent or booked. Captain then checks every result itself. Propose {k} "
    "DIFFERENT ways: different sources or approaches, not rewordings. Attempts run at the same time, and only "
    "one at a time may use a website in its browser, so give each way its own sites. Don't repeat a way the "
    "ledger already tried unless you say what is different now. Return ONLY a JSON array of objects: "
    '{{"hypothesis": "<the way, concretely>", "strategy_tag": "<2-4 word kebab-case tag>", '
    '"sites": ["<the websites it will mainly use, as domains>"], '
    '"why": "<why it might find a better result>"}}. No prose.'
)


def brief(run: dict, ledger: list[dict]) -> str:
    subj = run["objective"]["subject"]
    champ = run.get("champion") or {}
    best = ((champ.get("dev") or {}).get("best")) or None
    lines = [f"Task: {subj['task']}",
             f"Ranked by {subj['rank']['label']} ({'lower' if subj['rank']['direction'] == 'min' else 'higher'} is better)."]
    if best:
        lines.append(f"Best verified so far: {best['value']:g} {best.get('currency') or ''} — {best['title'][:120]} "
                     f"({best['url'][:120]})")
    if ledger:
        lines.append("\nTried so far:")
        for e in ledger[-10:]:
            lines.append(f"- [{e['status']}] ({e.get('strategy_tag') or '-'}) {e['hypothesis'][:200]}"
                         + (f" → {e['verdict_reason'][:160]}" if e.get("verdict_reason") else ""))
    return "\n".join(lines)


async def propose(run: dict, ledger: list[dict], k: int) -> list[dict]:
    from captain_claw.flight_deck import captain_swarm_loop as loop
    text = await loop.ask_model(run, _SYSTEM.format(k=k), brief(run, ledger), label="propose")
    return loop.diverse(loop.parse_proposals(text, k), ledger)


# ── the reviewer's material ──────────────────────────────────────────


async def review_material(run: dict, exp: dict) -> str:
    """The task, and the attempt's best result as Captain's own fetch saw it."""
    subj = run["objective"]["subject"]
    cons, rank = subj["constraints"], subj["rank"]
    best = (exp.get("dev") or {}).get("best") or {}
    if not best:
        raise RuntimeError("it has no verified result to review")
    return "\n".join(x for x in [
        f"Task: {subj['task']}",
        f"Ranked by {rank['label']} ({'lower' if rank['direction'] == 'min' else 'higher'} is better)"
        + (f", in {cons['currency']}" if cons.get("currency") else "")
        + (f", at most {cons['max_value']:g}" if cons.get("max_value") is not None else "")
        + (f"; it must be: {', '.join(cons['must_include'])}" if cons.get("must_include") else "") + ".",
        f"The result: {best.get('title')} — {best['value']:g} {best.get('currency') or ''} at {best.get('url')}",
        f"What the attempt said about it: {best['details']}" if best.get("details") else "",
        "What the page says around that number, from Captain's own fetch:",
        best.get("snippet") or "(no excerpt)",
    ] if x)


# ── carrying the winner out, once ────────────────────────────────────


def carry_out_task(run: dict, best: dict, intents: list[dict]) -> str:
    """The one Bat run that does it for real: re-check first, ask if anything
    changed, and act through Bat's gate."""
    subj = run["objective"]["subject"]
    steps = [f"- {i['what']}" for i in intents if i["kind"] in ("commit", "intent")][:12]
    return "\n".join([
        f"Carry this out, once: {subj['task']}",
        "",
        f"The chosen result: {best['title']} — {best['value']:g} {best.get('currency') or ''} at {best['url']}",
        f"Details: {best['details']}" if best.get("details") else "",
        f"The step that commits it: {best['next_step']}" if best.get("next_step") else "",
        "",
        "In rehearsal, the attempt that found it stopped at these points:" if steps else "",
        *steps,
        "",
        "First open the page again and check that it still shows the same "
        f"{subj['rank']['label']} and terms. If anything changed, ask the owner before going on. Then do it — "
        "once — and report exactly what was done, with any confirmation number.",
    ]).strip()
