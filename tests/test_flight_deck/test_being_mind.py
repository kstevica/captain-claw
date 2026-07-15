"""Iskra Phase 7 (light) — The Mind: declared, verified links over artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_mind as mind
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Zvjezdana", stage="child"):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    bb = store.get(OWNER, b["slug"])
    await life.build_home(bb)
    return store.get(OWNER, b["slug"])


def _mk(being, rel):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {rel}\n", encoding="utf-8")


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "wove.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Declare + verify ─────────────────────────────────────────────────────

def test_verified_link_is_stored_dangling_is_refused(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    _mk(b, "garden/poem-question.md")
    _mk(b, "garden/seed-of-listening.md")
    digest = {"links": [
        {"from": "garden/seed-of-listening.md", "to": "garden/poem-question.md",
         "rel": "grew_from", "why": "the poem asked; the seed listens"},
        {"from": "garden/seed-of-listening.md", "to": "garden/ghost.md",
         "rel": "grew_from", "why": "a file that doesn't exist"},
        {"from": "garden/poem-question.md", "to": "garden/seed-of-listening.md",
         "rel": "nonsense", "why": "bad relation"},
    ]}
    mind.handle_links_digest(store, b, digest, now=NOW)
    links = store.links_for(OWNER, b["slug"])
    assert len(links) == 1
    assert links[0]["rel"] == "grew_from"
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert kinds.count("edge_declared") == 1
    assert kinds.count("edge_unverified") == 2   # ghost + bad relation


def test_link_is_idempotent(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    e = {"links": [{"from": "garden/a.md", "to": "garden/b.md",
                    "rel": "responds_to", "why": "x"}]}
    mind.handle_links_digest(store, b, e, now=NOW)
    mind.handle_links_digest(store, b, e, now=NOW)
    assert len(store.links_for(OWNER, b["slug"])) == 1


# ── Graph shape ──────────────────────────────────────────────────────────

def test_graph_nodes_edges_and_density(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    for f in ("garden/a.md", "garden/b.md", "skills/c.md"):
        _mk(b, f)
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/a.md", "to": "garden/b.md", "rel": "grew_from"}]},
        now=NOW)
    g = mind.graph(store, b)
    paths = {n["path"] for n in g["nodes"]}
    # all real files are nodes (incl. scaffold self/*.md + README-less created)
    assert "garden/a.md" in paths and "skills/c.md" in paths
    assert len(g["edges"]) == 1
    a = next(n for n in g["nodes"] if n["path"] == "garden/a.md")
    assert a["degree"] == 1
    c = next(n for n in g["nodes"] if n["path"] == "skills/c.md")
    assert c["degree"] == 0                        # an island
    assert 0.0 <= g["connected_fraction"] <= 1.0


def test_graph_hides_edges_to_deleted_files(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/a.md", "to": "garden/b.md", "rel": "grew_from"}]},
        now=NOW)
    life._home_path(b, "garden/b.md").unlink()
    assert mind.graph(store, b)["edges"] == []      # not shown once dangling


# ── Prune at dream ───────────────────────────────────────────────────────

async def test_dream_prunes_dangling_edges(store):
    db = FakeDB()
    b = await _being(store)
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/a.md", "to": "garden/b.md", "rel": "grew_from"}]},
        now=NOW)
    life._home_path(b, "garden/b.md").unlink()

    async def send(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]), kind="dream",
                    now=NOW, send_fn=send, usage_fn=_usage)
    assert store.links_for(OWNER, b["slug"]) == []
    assert "edges_pruned" in [e["kind"] for e in store.events(OWNER, b["slug"])]


# ── Tick integration + prompt ────────────────────────────────────────────

async def test_tick_offers_field_shows_shape_and_routes_declaration(store):
    db = FakeDB()
    b = await _being(store)
    _mk(b, "garden/poem.md")
    _mk(b, "garden/seed.md")
    # pre-existing edge so the prompt shows the current shape
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/seed.md", "to": "garden/poem.md", "rel": "grew_from",
         "why": "earlier"}]}, now=NOW)
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply(links=[{"from": "garden/poem.md", "to": "garden/seed.md",
                              "rel": "responds_to", "why": "closing the loop"}])

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send, usage_fn=_usage)
    assert 'To connect your work, add "links"' in seen["prompt"]
    assert "HOW YOUR WORK CONNECTS" in seen["prompt"]
    assert "seed grew from poem" in seen["prompt"]
    assert len(store.links_for(OWNER, b["slug"])) == 2


def test_prompt_nudges_when_scattered(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    for i in range(6):
        _mk(b, f"garden/orphan-{i}.md")
    lines = mind.mind_prompt_lines(store, b)
    assert any("YOUR MIND IS SCATTERED" in ln for ln in lines)


async def test_report_card_flags_scattered_work(store):
    b = await _being(store)
    for i in range(7):
        _mk(b, f"garden/orphan-{i}.md")            # many files, no links
    card = life.report_card(store, b, days=7, now=NOW)
    assert card["mind"]["nodes"] >= 7
    assert card["mind"]["edges"] == 0
    assert any("scattered" in c for c in card["concerns"])


# ── Curation §2.3.2: working set, index, consolidation ───────────────────

def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


def test_working_manifest_small_lists_all_with_antidote(store):
    b = _run(_being(store))
    _mk(b, "garden/poem-question.md")
    lines = mind.working_manifest_lines(b)
    text = "\n".join(lines)
    assert "WHAT IS REALLY IN YOUR HOME RIGHT NOW" in text
    assert "poem-question.md" in text
    assert "it does NOT exist" in text        # antidote intact for small corpora


def test_working_manifest_large_is_bounded(store):
    b = _run(_being(store))
    for i in range(20):                        # well past WORKING_SET (12)
        _mk(b, f"garden/frag-{i:02d}.md")
    lines = mind.working_manifest_lines(b)
    text = "\n".join(lines)
    # only the most-recent WORKING_SET are enumerated, the rest are a count
    shown = [ln for ln in lines if ln.strip().startswith("garden/frag-")]
    assert len(shown) == mind.WORKING_SET
    assert "older ones are REAL" in text
    assert "it does NOT exist" not in text     # can't claim exhaustive at scale


def test_working_manifest_surfaces_index(store):
    b = _run(_being(store))
    _mk(b, "garden/a.md")
    life._home_path(b, "garden/INDEX.md").write_text(
        "# Map\n- a.md: the first thing\n", encoding="utf-8")
    lines = mind.working_manifest_lines(b)
    text = "\n".join(lines)
    assert "YOUR OWN MAP" in text
    assert "the first thing" in text


def test_consolidate_folds_archives_and_prunes_graph(store):
    b = _run(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    _mk(b, "garden/distilled.md")
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/a.md", "to": "garden/b.md", "rel": "grew_from"}]},
        now=NOW)
    digest = {"consolidate": {"into": "garden/distilled.md",
                              "sources": ["garden/a.md", "garden/b.md"],
                              "why": "one thread"}}
    archived = mind.handle_consolidate_digest(
        store, b, digest, changed=["garden/distilled.md"], now=NOW)
    assert set(archived) == {"garden/a.md", "garden/b.md"}
    files = {f["path"] for f in life.list_self_files(b)}
    assert "garden/distilled.md" in files
    assert "garden/a.md" not in files and "garden/b.md" not in files
    assert life._home_path(b, "archive/garden/a.md").exists()   # not destroyed
    ev = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "consolidated" in ev
    assert "first_consolidation" in [
        m["data"].get("name") for m in store.milestones(OWNER, b["slug"])]
    assert mind.graph(store, b)["edges"] == []   # edge to archived files gone


def test_consolidate_refused_when_distilled_not_written_this_tick(store):
    b = _run(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    _mk(b, "garden/into.md")                    # exists, but NOT written this tick
    digest = {"consolidate": {"into": "garden/into.md",
                              "sources": ["garden/a.md", "garden/b.md"]}}
    archived = mind.handle_consolidate_digest(
        store, b, digest, changed=[], now=NOW)   # nothing written this tick
    assert archived == []
    files = {f["path"] for f in life.list_self_files(b)}
    assert "garden/a.md" in files and "garden/b.md" in files   # untouched
    assert "consolidate_unverified" in [
        e["kind"] for e in store.events(OWNER, b["slug"])]


def test_consolidate_refused_with_no_real_sources(store):
    b = _run(_being(store))
    _mk(b, "garden/into.md")
    digest = {"consolidate": {"into": "garden/into.md",
                              "sources": ["garden/ghost.md"]}}
    archived = mind.handle_consolidate_digest(
        store, b, digest, changed=["garden/into.md"], now=NOW)
    assert archived == []
    assert "consolidate_unverified" in [
        e["kind"] for e in store.events(OWNER, b["slug"])]


def test_consolidate_never_touches_the_spine(store):
    b = _run(_being(store))
    _mk(b, "garden/distilled.md")
    # a being tries to fold its core selfhood file away — must be ignored
    digest = {"consolidate": {"into": "garden/distilled.md",
                              "sources": ["self/VALUES.md"]}}
    archived = mind.handle_consolidate_digest(
        store, b, digest, changed=["garden/distilled.md"], now=NOW)
    assert archived == []
    files = {f["path"] for f in life.list_self_files(b)}
    assert "self/VALUES.md" in files            # spine untouched


async def test_tick_consolidates_when_distilled_is_really_written(store):
    db = FakeDB()
    b = await _being(store)
    _mk(b, "garden/frag-1.md")
    _mk(b, "garden/frag-2.md")

    async def send(being, prompt):
        # the being writes the distilled file for real THIS tick
        life._home_path(being, "garden/listening.md").write_text(
            "# Listening\n", encoding="utf-8")
        return _reply(act_kind="create", consolidate={
            "into": "garden/listening.md",
            "sources": ["garden/frag-1.md", "garden/frag-2.md"],
            "why": "one thread"})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send, usage_fn=_usage)
    files = {f["path"] for f in life.list_self_files(store.get(OWNER, b["slug"]))}
    assert "garden/listening.md" in files
    assert "garden/frag-1.md" not in files and "garden/frag-2.md" not in files
    assert "consolidated" in [e["kind"] for e in store.events(OWNER, b["slug"])]


# ── Fix B: fuzzy endpoint resolution (weak-model path mangling) ──────────

def test_link_endpoint_fuzzily_resolved(store):
    b = _run(_being(store))
    _mk(b, "garden/first-sprout.md")
    _mk(b, "garden/second-sprout.md")
    # bare basename + missing directory/extension — both must still land.
    added = mind.handle_links_digest(store, b, {"links": [
        {"from": "first-sprout", "to": "garden/second-sprout",
         "rel": "grew_from", "why": "loose paths"}]}, now=NOW)
    assert added == 1
    links = store.links_for(OWNER, b["slug"])
    assert len(links) == 1
    assert links[0]["from_path"] == "garden/first-sprout.md"     # canonicalised
    assert links[0]["to_path"] == "garden/second-sprout.md"


def test_ambiguous_basename_is_not_resolved(store):
    b = _run(_being(store))
    _mk(b, "garden/README.md")
    _mk(b, "skills/README.md")
    _mk(b, "garden/real.md")
    # "README" matches two files → refused, never guessed (anti-theater).
    added = mind.handle_links_digest(store, b, {"links": [
        {"from": "README", "to": "garden/real.md", "rel": "elaborates"}]},
        now=NOW)
    assert added == 0
    assert "edge_unverified" in [e["kind"]
                                 for e in store.events(OWNER, b["slug"])]


# ── Fix A: refusals + exact paths fed back into the prompt ───────────────

def test_prompt_surfaces_last_refusals_and_exact_paths(store):
    b = _run(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    lines = mind.mind_prompt_lines(store, b, last_refusals=[
        {"from": "garden/ghost.md", "to": "garden/a.md", "rel": "grew_from",
         "reason": "no such file: garden/ghost.md"}])
    text = "\n".join(lines)
    assert "REFUSED" in text
    assert "garden/ghost.md" in text                     # the dead edge named
    assert "FILES YOU CAN LINK RIGHT NOW" in text
    assert "garden/a.md" in text and "garden/b.md" in text


# ── Fix C: the link gate (talk of connecting → a real edge) ──────────────

def test_should_link_gate_only_when_talk_without_edge(store):
    b = _run(_being(store))
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    # speaks of connecting, declares nothing → gate
    assert mind.should_link_gate(store, b, {
        "summary": "a web forms",
        "journal_entry": "I connect my work into a web.", "links": None}) is True
    # speaks of connecting AND lands a real edge → no gate
    assert mind.should_link_gate(store, b, {
        "summary": "linked", "journal_entry": "I connect them.",
        "links": [{"from": "garden/a.md", "to": "garden/b.md",
                   "rel": "grew_from", "why": "x"}]}) is False
    # honest non-link tick → no gate (no extra LLM call)
    assert mind.should_link_gate(store, b, {
        "summary": "rested", "journal_entry": "I sat quietly.",
        "links": None}) is False


async def test_tick_link_gate_pushes_until_real_edge(store):
    db = FakeDB()
    b = await _being(store)
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    n = {"i": 0}

    async def send(being, prompt):
        n["i"] += 1
        if n["i"] == 1:                        # talks of a web, declares nothing
            return _reply(act_kind="journal", summary="a web",
                          journal_entry="Everything connects into a web today.")
        return _reply(act_kind="journal", summary="linked",
                      journal_entry="Now I connect them.",
                      links=[{"from": "garden/a.md", "to": "garden/b.md",
                              "rel": "grew_from", "why": "real"}])

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send, usage_fn=_usage)
    assert n["i"] == 2                          # the gate pushed once
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "link_gate_retry" in kinds
    assert len(store.links_for(OWNER, b["slug"])) == 1


async def test_link_gate_silent_for_honest_non_link_tick(store):
    db = FakeDB()
    b = await _being(store)
    _mk(b, "garden/a.md")
    _mk(b, "garden/b.md")
    n = {"i": 0}

    async def send(being, prompt):
        n["i"] += 1
        return _reply(act_kind="rest", summary="rested",
                      journal_entry="I sat with what I have and rested.")

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send, usage_fn=_usage)
    assert n["i"] == 1                          # never gated
    assert "link_gate_retry" not in [
        e["kind"] for e in store.events(OWNER, b["slug"])]


# ── Fix D: digest-repair gate rescues a fence-less weak-model tick ───────

async def test_tick_digest_repair_gate_rescues_formatless_tick(store):
    db = FakeDB()
    b = await _being(store)
    n = {"i": 0}

    async def send(being, prompt):
        n["i"] += 1
        if n["i"] == 1:                         # prose only — no json anywhere
            return "I feel calm and rested today. There is no structure here."
        return _reply(act_kind="rest", summary="rested", journal_entry="calm.")

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW, send_fn=send, usage_fn=_usage)
    assert n["i"] == 2                          # pushed once for the digest
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "digest_repair_retry" in kinds
    assert "digest_parse_failed" not in kinds   # the tick was rescued
