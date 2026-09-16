"""Increment 4: reporter pointer-resolution, corrective retry, and smooth mode."""

import types

import pytest

from captain_claw.flight_deck import vatra_routes as vr


def _pure_helpers_setup():
    pass


# ── pure helpers ─────────────────────────────────────────────────────

def test_pointer_paths():
    txt = "The book is complete. File: vfs:proj/fair-measure.md — enjoy."
    assert vr._pointer_paths(txt) == ["vfs:proj/fair-measure.md"]


def test_looks_like_pointer():
    inputs_len = 40000
    assert vr._looks_like_pointer("Fair Measure is assembled and complete. File: vfs:p/x.md", inputs_len) is True
    assert vr._looks_like_pointer("[written to disk: x — use read tool to view]", inputs_len) is True
    # a full 40k deliverable is not a pointer
    assert vr._looks_like_pointer("# Chapter One\n\n" + "x" * 40000, inputs_len) is False
    # empty → pointer (nothing produced)
    assert vr._looks_like_pointer("", inputs_len) is True


def test_resolve_deliverable(tmp_path, monkeypatch):
    (tmp_path / "fair-measure.md").write_text("# Book\n\n" + "y" * 5000, encoding="utf-8")

    def fake_resolve(uid, proj, path):
        name = path.rsplit("/", 1)[-1]
        return tmp_path / name

    monkeypatch.setattr(vr, "_vfs_resolve_under", fake_resolve)
    text, path = vr._resolve_deliverable("u1", "proj", ["vfs:proj/fair-measure.md"])
    assert text.startswith("# Book") and path == "vfs:proj/fair-measure.md"
    # a missing candidate → empty
    text2, _ = vr._resolve_deliverable("u1", "proj", ["vfs:proj/missing.md"])
    assert text2 == ""


# ── _run_reporter with mocked spawn/dispatch ─────────────────────────

@pytest.fixture
def reporter_env(tmp_path, monkeypatch):
    import captain_claw.flight_deck.server as server
    monkeypatch.setattr(server, "DATA_DIR", tmp_path, raising=False)
    ws = tmp_path / "rep" / "data" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vr, "_teardown", lambda slugs: None)
    monkeypatch.setattr(vr, "_track_worker", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)

    async def fake_spawn(*a, **k):
        return {"ok": True, "slug": "rep", "port": 1, "auth": "t", "message": ""}

    monkeypatch.setattr(vr, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(vr, "_capture_generated", lambda *a, **k: ([], ""))
    return tmp_path


_ARCH = {"editor-writer": {"id": "editor-writer", "role": "Editor", "tier": "balanced",
                           "cognitive_mode": "neutra", "tools": ["read", "write", "edit", "glob"]}}
_USABLE = [{"id": "s4", "owner": "editor-writer", "role": "Editor", "title": "Part One",
            "output": "# Chapter One\n\n" + "a" * 3000, "ok": True, "latency_ms": 10}]


@pytest.mark.asyncio
async def test_reporter_pointer_reply_unresolved_falls_back(reporter_env, monkeypatch):
    async def fake_dispatch(port, auth, prompt, timeout, **k):
        return {"output": "Fair Measure is assembled and complete. File: vfs:p/fair-measure.md",
                "ok": True, "latency_ms": 100}

    monkeypatch.setattr(vr, "_dispatch_one", fake_dispatch)
    monkeypatch.setattr(vr, "_vfs_resolve_under", lambda *a: None)  # nothing resolves
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))
    truth, files = await vr._run_reporter(
        stub, {"id": "u1"}, "sid1", "sid1", "tag", "write the book", _USABLE, {}, _ARCH,
        tiers=None, api_key="", env_vars=None, dispatch_timeout=30, input_names=set(),
        dest_dir=reporter_env, seen_gen=set(), resolve_pointer=True, can_retry=True,
        user_id="u1", project="p")
    # unresolved pointer → raw assembly, not the note
    assert "assembled and complete" not in truth
    assert "Chapter One" in truth


@pytest.mark.asyncio
async def test_reporter_pointer_reply_resolves_to_file(reporter_env, monkeypatch):
    big = "# Fair Measure\n\n" + "z" * 50000
    (reporter_env / "fair-measure.md").write_text(big, encoding="utf-8")

    async def fake_dispatch(port, auth, prompt, timeout, **k):
        return {"output": "Done — see vfs:p/fair-measure.md", "ok": True, "latency_ms": 100}

    monkeypatch.setattr(vr, "_dispatch_one", fake_dispatch)
    monkeypatch.setattr(vr, "_vfs_resolve_under",
                        lambda uid, proj, path: reporter_env / path.rsplit("/", 1)[-1])
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))
    truth, files = await vr._run_reporter(
        stub, {"id": "u1"}, "sid1", "sid1", "tag", "write the book", _USABLE, {}, _ARCH,
        tiers=None, api_key="", env_vars=None, dispatch_timeout=30, input_names=set(),
        dest_dir=reporter_env, seen_gen=set(), resolve_pointer=True, can_retry=False,
        user_id="u1", project="p")
    assert truth == big
    assert any(f.get("vfs") == "vfs:p/fair-measure.md" for f in files)


@pytest.mark.asyncio
async def test_reporter_resolve_pointer_off_returns_reply(reporter_env, monkeypatch):
    note = "The report is complete."

    async def fake_dispatch(port, auth, prompt, timeout, **k):
        return {"output": note, "ok": True, "latency_ms": 100}

    monkeypatch.setattr(vr, "_dispatch_one", fake_dispatch)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))
    truth, files = await vr._run_reporter(
        stub, {"id": "u1"}, "sid1", "sid1", "tag", "x", _USABLE, {}, _ARCH,
        tiers=None, api_key="", env_vars=None, dispatch_timeout=30, input_names=set(),
        dest_dir=reporter_env, seen_gen=set(), resolve_pointer=False)
    assert truth == note  # today's behaviour


@pytest.mark.asyncio
async def test_reporter_smooth_small_edit_keeps_file(reporter_env, monkeypatch):
    dfile = reporter_env / "book.md"
    pre = "# Chapter One\n\n" + "a" * 2000 + "\n\n# Chapter Two\n\n" + "b" * 2000
    dfile.write_text(pre, encoding="utf-8")

    async def fake_dispatch(port, auth, prompt, timeout, **k):
        # simulate a tiny in-place edit (same length-ish)
        dfile.write_text(pre + "\n<!-- smoothed -->", encoding="utf-8")
        return {"output": "DONE", "ok": True, "latency_ms": 100}

    monkeypatch.setattr(vr, "_dispatch_one", fake_dispatch)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))
    smooth = {"file": str(dfile), "path": "vfs:p/book.md", "text": pre,
              "findings": [{"kind": "duplicate_chapter", "detail": "check ch2 seam"}],
              "bytes": len(pre.encode())}
    truth, files = await vr._run_reporter(
        stub, {"id": "u1"}, "sid1", "sid1", "tag", "x", _USABLE, {}, _ARCH,
        tiers=None, api_key="", env_vars=None, dispatch_timeout=30, input_names=set(),
        dest_dir=reporter_env, seen_gen=set(), smooth=smooth)
    assert "smoothed" in truth  # the edit was kept


@pytest.mark.asyncio
async def test_reporter_smooth_collapse_restores_assembly(reporter_env, monkeypatch):
    dfile = reporter_env / "book.md"
    pre = "# Chapter One\n\n" + "a" * 3000 + "\n\n# Chapter Two\n\n" + "b" * 3000
    dfile.write_text(pre, encoding="utf-8")

    async def fake_dispatch(port, auth, prompt, timeout, **k):
        dfile.write_text("oops truncated", encoding="utf-8")  # collapsed
        return {"output": "DONE", "ok": True, "latency_ms": 100}

    monkeypatch.setattr(vr, "_dispatch_one", fake_dispatch)
    stub = types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))
    smooth = {"file": str(dfile), "path": "vfs:p/book.md", "text": pre,
              "findings": [], "bytes": len(pre.encode())}
    truth, files = await vr._run_reporter(
        stub, {"id": "u1"}, "sid1", "sid1", "tag", "x", _USABLE, {}, _ARCH,
        tiers=None, api_key="", env_vars=None, dispatch_timeout=30, input_names=set(),
        dest_dir=reporter_env, seen_gen=set(), smooth=smooth)
    assert truth == pre  # collapse detected → restored
    assert dfile.read_text(encoding="utf-8") == pre
