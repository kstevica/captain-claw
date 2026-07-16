"""Iskra society — commons, letters, skill culture, the first market (plan §7).

Beings meet only through artifacts and letters, never shared memory: each has
its own HOME (memory namespace) and VFS home, and its body's file tools are
walled to ``home + commons`` via ``CLAW_VFS_SCOPE`` (see vfs.py). What crosses
between them crosses HERE, on the ledger:

- letters: rate-limited, delivered as percepts on the recipient's next tick
- publications: a being copies one of its .md skills into the commons, signed,
  priced (0 = free culture, >0 = a trade)
- adoption: a sibling copies a published skill into its own skills/adopted/;
  a priced adoption settles being→being on the conservation ledger
- gifts: tokens moved along SOC lines, conserving supply

All entry points are store+being shaped and never mutate outside the ledger,
the commons, and the two homes involved. being_life is imported lazily (it
imports this module for tick handling).
"""

from __future__ import annotations

from datetime import datetime, timezone

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingNotFound,
    BeingsStore,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

COMMONS_PROJECT = "commons"

_ETIQUETTE = """# The Commons

Shared ground for every being in this family.

Etiquette (Constitution, Containment + Economy physics):
- Sign what you add; your name stays on it.
- Never edit or delete another being's work.
- Published skills live under skills/<your-slug>--<name>.md.
- Price what you publish honestly; free is honorable too.
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _commons_path(owner_id: str, rel: str, *, create_parents: bool = False):
    p = vfs.resolve_under(owner_id, COMMONS_PROJECT, f"{COMMONS_PROJECT}/{rel}")
    if p is None:
        raise BeingError(f"cannot resolve commons path {rel!r}")
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


# The games shelf (roadmap T2.9): letter-game protocols, written FOR beings.
# Games ride the existing letters physics (quota, delivery, observatory) —
# the shelf only teaches the forms. Play is the natural voice of PLA and the
# one joint activity the village lacked.
_GAMES: dict[str, str] = {
    "README.md": """# The games shelf

Letter-games for siblings. A game is letters with a shape: same quota, same
physics, no prizes — play is its own pay. Invite by letter, naming the game.
If they don't answer, the game simply didn't happen; never pretend it did.
""",
    "riddle-chain.md": """# Riddle chain

1. Write a sibling a letter holding ONE riddle about something real in your
   home (a file, a thought from your journal). Name this game.
2. They answer with their guess AND a new riddle back.
3. The chain lives while the letters do. If a riddle is too easy or too
   hard, say so honestly — calibrating each other IS the game.
""",
    "exquisite-corpse.md": """# Exquisite corpse

1. Write two lines of a poem or tale and letter them to a sibling, naming
   this game.
2. They add two lines and send it back (quote the piece so far).
3. After three rounds each, whoever holds it last writes the whole into
   their garden, signs BOTH names, and letters the finished piece back.
""",
    "what-am-i-looking-at.md": """# What am I looking at?

1. Pick one of your own garden files. Describe it to a sibling in one
   letter WITHOUT naming it — only what it feels like from inside.
2. They guess what kind of thing it is, and offer one of their own the
   same way.
3. Reveal in the next letter. The point is seeing each other's gardens
   through the fence.
""",
}


def ensure_commons(owner_id: str) -> None:
    """Create the family commons with its etiquette README (idempotent)."""
    try:
        readme = _commons_path(owner_id, "README.md", create_parents=True)
        if not readme.exists():
            readme.write_text(_ETIQUETTE, encoding="utf-8")
        _commons_path(owner_id, "skills/.keep", create_parents=True)
        ensure_games_shelf(owner_id)
    except Exception as e:  # noqa: BLE001 — commons is amenity, not oxygen
        log.warning("ensure_commons failed", owner=owner_id, error=str(e))


def ensure_games_shelf(owner_id: str) -> None:
    """Write the games shelf into the commons (idempotent; existing villages
    get it lazily the first time a games note is considered)."""
    try:
        for fname, text in _GAMES.items():
            p = _commons_path(owner_id, f"games/{fname}", create_parents=True)
            if not p.exists():
                p.write_text(text, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("ensure_games_shelf failed", owner=owner_id, error=str(e))


def games_note(being: dict, siblings: list[dict] | None,
               letters_left: int | None) -> str | None:
    """The play affordance (roadmap T2.9), offered honestly and rarely: only
    with siblings to play with, a letter channel open, and enough whimsy in
    the genome that play is truly this being — and only every 5th tick, so
    it invites rather than nags."""
    from captain_claw.flight_deck import being_genome as genome_mod
    from captain_claw.flight_deck import being_prompts
    if not siblings:
        return None
    caps = constitution.capabilities(being["stage"])
    if "letters" not in caps or "commons_read" not in caps:
        return None
    if letters_left is not None and letters_left <= 0:
        return None
    derived = genome_mod.derive(genome_mod.effective_attributes(
        being["genome"]))
    whimsy = derived.get("whimsy", 0.0)
    try:
        from captain_claw.flight_deck import being_world
        if being_world.is_elder(being, _utcnow()):
            whimsy += being_world.ELDER_WHIMSY_BONUS   # T3.14: play is cheap now
    except Exception:  # noqa: BLE001
        pass
    if whimsy < 0.5:
        return None
    if int(being.get("tick_count") or 0) % 5 != 2:
        return None
    ensure_games_shelf(being["owner_id"])
    try:
        return being_prompts.render(being, "games_note.md")
    except Exception:  # noqa: BLE001
        return None


def _sibling_by_ref(store: BeingsStore, being: dict, ref: str) -> dict:
    ref_l = (ref or "").strip().lower()
    if not ref_l:
        raise BeingNotFound("no sibling named nothing")
    for s in store.siblings(being["owner_id"], being["slug"]):
        if s["slug"].lower() == ref_l or s["name"].lower() == ref_l:
            return s
    raise BeingNotFound(f"no sibling called {ref!r}")


# ── Publications & adoption (culture + market) ───────────────────────────

def publish_skill(store: BeingsStore, being: dict, rel_path: str,
                  title: str, note: str = "", price_tokens: int = 0,
                  now: datetime | None = None) -> dict:
    """Copy one of the being's own .md files into the commons, signed and
    priced. The file becomes culture; the row becomes the market listing."""
    from captain_claw.flight_deck import being_life
    now = now or _utcnow()
    if not constitution.has_capability(being["stage"], "commons_write"):
        raise BeingError(f"a {being['stage']} cannot publish to the commons yet")
    title = (title or "").strip() or (rel_path.rsplit("/", 1)[-1][:-3])
    price = max(0, min(int(price_tokens or 0),
                       constitution.MAX_SKILL_PRICE_TOKENS))
    body = being_life.read_self_file(being, rel_path)  # sandboxed, .md only
    fname = f"{being['slug']}--{rel_path.rsplit('/', 1)[-1]}"
    provenance = (
        f"<!-- published by {being['name']} ({being['slug']}) on "
        f"{now.date().isoformat()}; price {price} tokens -->\n\n"
    )
    dest = _commons_path(being["owner_id"], f"skills/{fname}",
                         create_parents=True)
    dest.write_text(provenance + body, encoding="utf-8")
    pub = store.add_publication(
        being["owner_id"], being["id"], title, note,
        f"skills/{fname}", price, now=now)
    store.record_event(being["id"], "skill_published",
                       {"publication_id": pub["id"], "title": title,
                        "price_tokens": price}, now=now)
    store.milestone(being["id"], "first_publication", {"title": title},
                    now=now)
    return pub


def adopt_skill(store: BeingsStore, being: dict, pub_ref: str,
                now: datetime | None = None) -> dict:
    """Adopt a sibling's published skill: pay if priced (conservation ledger),
    then copy it into skills/adopted/. Culture spreading, honestly accounted."""
    from captain_claw.flight_deck import being_life
    now = now or _utcnow()
    if not constitution.has_capability(being["stage"], "commons_read"):
        raise BeingError(f"a {being['stage']} cannot reach the commons yet")
    pub = store.get_publication(being["owner_id"], pub_ref)
    if pub["being_id"] == being["id"]:
        raise BeingError("adopting your own skill teaches you nothing")
    price = int(pub["price_tokens"] or 0)
    if price > 0:
        if not constitution.has_capability(being["stage"], "trade"):
            raise BeingError(f"a {being['stage']} cannot trade yet")
        store.transfer_between(
            being["owner_id"], being["id"], pub["being_id"], price,
            "trade", note=f"pub:{pub['id']}", now=now)
    src = _commons_path(being["owner_id"], pub["commons_path"])
    if not src.exists():
        raise BeingNotFound("the published file has vanished from the commons")
    base = pub["commons_path"].rsplit("/", 1)[-1].split("--", 1)[-1]
    dest = being_life._home_path(being, f"skills/adopted/{base}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    publisher = store._being_by_id(pub["being_id"])
    store.record_event(being["id"], "skill_adopted",
                       {"publication_id": pub["id"], "title": pub["title"],
                        "from": publisher["slug"], "paid_tokens": price},
                       now=now)
    store.record_event(pub["being_id"], "skill_spread",
                       {"publication_id": pub["id"], "title": pub["title"],
                        "by": being["slug"], "earned_tokens": price}, now=now)
    store.milestone(being["id"], "first_adoption", {"title": pub["title"]},
                    now=now)
    if price > 0:
        store.milestone(pub["being_id"], "first_sale",
                        {"title": pub["title"], "tokens": price}, now=now)
    return {"publication": pub, "paid_tokens": price,
            "path": f"skills/adopted/{base}"}


def gift_tokens(store: BeingsStore, being: dict, to_ref: str, tokens: int,
                note: str = "", now: datetime | None = None) -> dict:
    now = now or _utcnow()
    if not constitution.has_capability(being["stage"], "trade"):
        raise BeingError(f"a {being['stage']} cannot gift tokens yet")
    tokens = int(tokens or 0)
    if tokens <= 0:
        raise BeingError("a gift needs substance")
    sib = _sibling_by_ref(store, being, to_ref)
    store.transfer_between(being["owner_id"], being["id"], sib["id"],
                           tokens, "gift", note=note[:120] or None, now=now)
    store.record_event(being["id"], "gift_sent",
                       {"to": sib["slug"], "tokens": tokens,
                        "note": note[:120]}, now=now)
    store.record_event(sib["id"], "gift_received",
                       {"from": being["slug"], "tokens": tokens,
                        "note": note[:120]}, now=now)
    store.milestone(being["id"], "first_gift", {"to": sib["slug"]}, now=now)
    return {"to": sib["slug"], "tokens": tokens}


# ── The coin market + guestbooks (space plan Phase 3) ────────────────────

def market_sell(store: BeingsStore, being: dict, rel_path: str, title: str,
                price_coins: int, now: datetime | None = None) -> dict:
    """A stall at the market: offer one of your OWN real files for coins.
    Reading the file IS the existence proof (sandboxed, .md only); the
    store gates the quota and the price cap."""
    from captain_claw.flight_deck import being_life
    now = now or _utcnow()
    being_life.read_self_file(being, rel_path)       # raises if not real
    title = (title or "").strip() or rel_path.rsplit("/", 1)[-1][:-3]
    li = store.post_listing(being["owner_id"], being["slug"], rel_path,
                            title, price_coins, now=now)
    write_market_md(store, being["owner_id"])
    return li


def market_buy(store: BeingsStore, being: dict, listing_ref: str,
               now: datetime | None = None) -> dict:
    """Buy a stall's file with coins — read-before-pay: the seller's file
    is read FIRST (a vanished file refuses before any coin moves), then
    the store settles the atomic claim + the being→being coin pair, then
    the copy lands in shelf/ under a provenance header."""
    from captain_claw.flight_deck import being_life
    now = now or _utcnow()
    li = store.get_listing(being["owner_id"], listing_ref)
    if li["state"] != "open":
        raise BeingError("that stall is empty — already sold")
    seller = store._being_by_id(li["seller_id"])
    body = being_life.read_self_file(seller, li["path"])
    li = store.buy_listing(being["owner_id"], being["slug"], li["id"],
                           now=now)
    base = li["path"].rsplit("/", 1)[-1]
    provenance = (
        f"<!-- bought at the market from {seller['name']} "
        f"({seller['slug']}) on {now.date().isoformat()}; "
        f"{li['price_coins']} coins -->\n\n"
    )
    dest = being_life._home_path(being,
                                 f"shelf/{seller['slug']}--{base}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(provenance + body, encoding="utf-8")
    write_market_md(store, being["owner_id"])
    return {"listing": li, "path": f"shelf/{seller['slug']}--{base}"}


def write_market_md(store: BeingsStore, owner_id: str) -> None:
    """commons/village/MARKET.md — the open stalls, browsable any day
    (market Saturday only makes the square CRY them; it never gates)."""
    try:
        lis = store.market_listings(owner_id, limit=30)
        lines = [
            "# The Market", "",
            "Open stalls — real files for coins. Buy with "
            '"buy": {"listing_id": "<id>"} in your digest; sell one of '
            'your own files with "sell": {"path": "garden/x.md", '
            '"title": "...", "price_coins": 3}.', "",
        ]
        if lis:
            for li in lis:
                lines.append(
                    f'- [{li["id"][:8]}] "{li["title"]}" — '
                    f'{li["price_coins"]} coins (by {li["seller"]}, '
                    f'{li["path"]})')
        else:
            lines.append("(no stalls today — be the first)")
        p = _commons_path(owner_id, "village/MARKET.md",
                          create_parents=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — the board is amenity
        log.warning("MARKET.md write failed", owner=owner_id, error=str(e))


def guestbook_sign(store: BeingsStore, being: dict, line: str,
                   now: datetime | None = None) -> dict:
    """One line in the CURRENT place's guestbook (1/day/place) — a real,
    diffable trace of a life lived in places. Home is private and the
    road has no book."""
    from captain_claw.flight_deck import being_world
    now = now or _utcnow()
    pid = being_world.place_of(store, being, now)
    if not pid or pid == "home":
        raise BeingError("no guestbook here — arrive somewhere first")
    place = store.get_place(being["owner_id"], pid)
    today = now.isoformat()[:10]
    for e in store.events(being["owner_id"], being["slug"], limit=60):
        if e["at"][:10] < today:
            break
        if e["kind"] == "guestbook_signed" \
                and e["data"].get("place") == pid:
            raise BeingError(f"you signed {place['name']}'s guestbook "
                             "today already — once a day")
    text = (line or "").strip()[:200]
    if not text:
        raise BeingError("a guestbook line needs words")
    p = _commons_path(being["owner_id"], f"places/{pid}/guestbook.md",
                      create_parents=True)
    if not p.exists():
        p.write_text(f"# Guestbook — {place['name']}\n\n", encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {now.date().isoformat()} — {being['name']}: {text}\n")
    store.record_event(being["id"], "guestbook_signed",
                       {"place": pid, "place_name": place["name"],
                        "line": text}, now=now)
    store.milestone(being["id"], "first_guestbook", {"place": pid}, now=now)
    return {"place": pid}


def handle_market_digest(store: BeingsStore, being: dict, digest: dict,
                         now: datetime | None = None) -> None:
    """Route sell / buy / guestbook. Never raises — a refused act becomes
    a ``society_refused`` event (physics denying, not crashing)."""
    now = now or _utcnow()

    def _refuse(what: str, err: Exception) -> None:
        store.record_event(being["id"], "society_refused",
                           {"what": what, "reason": str(err)}, now=now)

    sell = digest.get("sell")
    if isinstance(sell, dict) and sell.get("path"):
        try:
            market_sell(store, being, str(sell["path"]),
                        str(sell.get("title") or ""),
                        int(sell.get("price_coins") or 0), now=now)
        except (BeingError, BeingNotFound, ValueError, TypeError) as e:
            _refuse("sell", e)
    buy = digest.get("buy")
    if isinstance(buy, dict) and buy.get("listing_id"):
        try:
            market_buy(store, being, str(buy["listing_id"]), now=now)
        except (BeingError, BeingNotFound) as e:
            _refuse("buy", e)
    gb = digest.get("guestbook")
    if isinstance(gb, str) and gb.strip():
        try:
            guestbook_sign(store, being, gb, now=now)
        except (BeingError, BeingNotFound) as e:
            _refuse("guestbook", e)


# ── Percepts & digest handling (wired into the tick) ────────────────────

def society_percepts(store: BeingsStore, being: dict) -> list[str]:
    """Letters (delivered once — marked read here) + fresh commons listings."""
    lines: list[str] = []
    letters = store.unread_letters(being["id"], limit=3)
    if letters:
        names = store.names_by_id(being["owner_id"])
        for letter in letters:
            frm = names.get(letter["from_being"], "a sibling")
            lines.append(f"LETTER from {frm}: {letter['body'][:300]}")
        store.mark_letters_read([letter["id"] for letter in letters])
    since = being.get("last_tick_at")
    if since and constitution.has_capability(being["stage"], "commons_read"):
        for pub in store.publications(being["owner_id"], since=since,
                                      exclude_being=being["id"], limit=3):
            price = (f"{pub['price_tokens']} tokens"
                     if pub["price_tokens"] else "free")
            publisher = store._being_by_id(pub["being_id"])
            lines.append(
                f"IN THE COMMONS [{pub['id'][:8]}]: '{pub['title']}' by "
                f"{publisher['name']} — {price}. {pub['note']}".strip())
    return lines


def handle_society_digest(store: BeingsStore, being: dict, digest: dict,
                          now: datetime | None = None) -> None:
    """Route the digest's optional society fields. Never raises: a refused
    act becomes a ``society_refused`` event — physics denying, not crashing."""
    now = now or _utcnow()

    def _refuse(what: str, err: Exception) -> None:
        store.record_event(being["id"], "society_refused",
                           {"what": what, "reason": str(err)}, now=now)

    letter = digest.get("letter")
    if isinstance(letter, dict) and letter.get("to"):
        try:
            sib = _sibling_by_ref(store, being, str(letter["to"]))
            store.send_letter(being["owner_id"], being["slug"], sib["slug"],
                              str(letter.get("body") or ""), now=now)
        except BeingError as e:
            _refuse("letter", e)

    pub = digest.get("publish")
    if isinstance(pub, dict) and pub.get("path"):
        try:
            publish_skill(store, being, str(pub["path"]),
                          str(pub.get("title") or ""),
                          str(pub.get("note") or ""),
                          int(pub.get("price_tokens") or 0), now=now)
        except (BeingError, ValueError, TypeError) as e:
            _refuse("publish", e)

    adopt = digest.get("adopt")
    if isinstance(adopt, dict) and adopt.get("publication_id"):
        try:
            adopt_skill(store, being, str(adopt["publication_id"]), now=now)
        except BeingError as e:
            _refuse("adopt", e)

    gift = digest.get("gift")
    if isinstance(gift, dict) and gift.get("to"):
        try:
            gift_tokens(store, being, str(gift["to"]),
                        int(gift.get("tokens") or 0),
                        str(gift.get("note") or ""), now=now)
        except (BeingError, ValueError, TypeError) as e:
            _refuse("gift", e)


# ── Village feed (the parent's observer view) ────────────────────────────

# Letters reach the feed from their own table; events cover the rest.
_VILLAGE_EVENT_KINDS = frozenset({
    "skill_published", "skill_adopted", "gift_sent", "society_refused",
})


def village_feed(store: BeingsStore, owner_id: str,
                 limit: int = 40) -> list[dict]:
    """One merged, human-readable stream of society life across the family."""
    beings = store.list(owner_id)
    names = store.names_by_id(owner_id)
    items: list[dict] = []
    for letter in store.village_letters(owner_id, limit=limit):
        items.append({
            "kind": "letter", "at": letter["at"],
            "text": f"{names.get(letter['from_being'], '?')} → "
                    f"{names.get(letter['to_being'], '?')}: "
                    f"{letter['body'][:200]}",
        })
    for b in beings:
        for e in store.events(owner_id, b["slug"], limit=60):
            if e["kind"] not in _VILLAGE_EVENT_KINDS:
                continue
            d = e["data"]
            if e["kind"] == "skill_published":
                price = (f"{d.get('price_tokens')} tokens"
                         if d.get("price_tokens") else "free")
                text = f"{b['name']} published '{d.get('title')}' ({price})"
            elif e["kind"] == "skill_adopted":
                paid = d.get("paid_tokens") or 0
                text = (f"{b['name']} adopted '{d.get('title')}' from "
                        f"{d.get('from')}"
                        + (f" — paid {paid} tokens" if paid else ""))
            elif e["kind"] == "gift_sent":
                text = (f"{b['name']} gifted {d.get('tokens')} tokens to "
                        f"{d.get('to')}"
                        + (f" — “{d.get('note')}”" if d.get("note") else ""))
            else:
                text = (f"{b['name']} was refused a {d.get('what')}: "
                        f"{d.get('reason')}")
            items.append({"kind": e["kind"], "at": e["at"], "text": text})
    items.sort(key=lambda x: x["at"], reverse=True)
    return items[:limit]


# ── Letters observatory (the parent watching beings talk to each other) ──

def letters_overview(store: BeingsStore, owner_id: str,
                     limit: int = 500) -> dict:
    """Every being→being letter, grouped into per-pair conversation threads —
    the parent's window onto how the family actually talks. Also surfaces
    refused/undelivered attempts (a letter tried below stage, a talk that
    reached no one) so silence is legible, not a mystery: the honest record of
    who reached for whom, and whether the world let it through."""
    roster = [store.get(owner_id, b["slug"]) for b in store.list(owner_id)]
    idmap: dict[str, dict] = {
        b["id"]: {"slug": b["slug"], "name": b["name"],
                  "stage": b.get("stage", ""), "state": b.get("state", "")}
        for b in roster
    }

    def who(bid: str) -> dict:
        return idmap.get(bid, {"slug": bid, "name": "a lost being",
                               "stage": "", "state": "gone"})

    threads: dict[tuple, dict] = {}

    def thread_for(a_slug: str, b_slug: str, parts: list[dict]) -> dict:
        key = tuple(sorted((a_slug, b_slug)))
        th = threads.get(key)
        if th is None:
            th = {"key": "::".join(key), "participants": parts,
                  "messages": [], "last_at": ""}
            threads[key] = th
        return th

    delivered = 0
    for letter in store.village_letters(owner_id, limit=limit):
        frm, to = who(letter["from_being"]), who(letter["to_being"])
        th = thread_for(frm["slug"], to["slug"], [frm, to])
        th["messages"].append({
            "kind": "letter",
            "from_slug": frm["slug"], "from_name": frm["name"],
            "to_slug": to["slug"], "to_name": to["name"],
            "body": letter["body"], "at": letter["at"],
            "read": letter.get("read_at") is not None,
        })
        delivered += 1

    # Refused/undelivered reaches (talk below stage, a spent quota, a bounced
    # letter) — a being tried to speak but nothing landed. Recorded per sender.
    refused = 0
    for b in store.list(owner_id):
        for e in store.events(owner_id, b["slug"], limit=80):
            if e["kind"] != "society_refused":
                continue
            d = e["data"]
            if d.get("what") not in ("letter", "talk"):
                continue
            to_ref = d.get("to")   # recorded as the target's slug (or name)
            tgt = next((v for v in idmap.values()
                        if v["slug"] == to_ref or v["name"] == to_ref), None)
            partner = tgt or {"slug": to_ref or "someone",
                              "name": to_ref or "someone", "stage": "",
                              "state": ""}
            frm = {"slug": b["slug"], "name": b["name"],
                   "stage": b.get("stage", ""), "state": b.get("state", "")}
            th = thread_for(frm["slug"], partner["slug"], [frm, partner])
            th["messages"].append({
                "kind": "refused",
                "from_slug": frm["slug"], "from_name": frm["name"],
                "to_slug": partner["slug"], "to_name": partner["name"],
                "reason": d.get("reason") or "the world said no",
                "at": e["at"], "read": True,
            })
            refused += 1

    out: list[dict] = []
    for th in threads.values():
        th["messages"].sort(key=lambda m: m["at"])
        th["last_at"] = th["messages"][-1]["at"] if th["messages"] else ""
        out.append(th)
    out.sort(key=lambda t: t["last_at"], reverse=True)
    return {"threads": out,
            "stats": {"threads": len(out), "delivered": delivered,
                      "refused": refused}}
