"""Line endings at the write/edit boundary (write_guard eol helpers).

The incident: on Windows the write tool's text mode turned each "\\n" into
"\\r\\n", so a 35-line script sent as 1648 bytes landed as 1683, the readback
called it truncated, and the model re-issued the same write over and over.
"""

from __future__ import annotations

from pathlib import Path

from captain_claw import write_guard
from captain_claw.tools.edit import EditTool
from captain_claw.tools.write import WriteTool


def _kw(tmp_path: Path, session: str = "s1") -> dict:
    return {"_runtime_base_path": tmp_path, "_saved_base_path": tmp_path / "saved",
            "_session_id": session}


# ── helpers ───────────────────────────────────────────────────────────────

def test_eol_of_majority():
    assert write_guard.eol_of(b"a\nb\nc\n") == "\n"
    assert write_guard.eol_of(b"a\r\nb\r\nc\n") == "\r\n"
    assert write_guard.eol_of(b"a\r\nb\nc\n") == "\n"
    assert write_guard.eol_of("a\r\nb\r\n") == "\r\n"
    assert write_guard.eol_of(b"no breaks") is None


def test_to_eol_round_trip():
    assert write_guard.to_eol("a\nb\r\nc", "\r\n") == "a\r\nb\r\nc"
    assert write_guard.to_eol("a\r\nb\nc", "\n") == "a\nb\nc"
    assert write_guard.to_eol("a\nb", None) == "a\nb"


def test_eol_for_write_rules(tmp_path: Path):
    crlf = tmp_path / "old.txt"
    crlf.write_bytes(b"one\r\ntwo\r\n")
    assert write_guard.eol_for_write(crlf, exists=True) == "\r\n"
    assert write_guard.eol_for_write(tmp_path / "run.bat", exists=False) == "\r\n"
    assert write_guard.eol_for_write(tmp_path / "run.CMD", exists=False) == "\r\n"
    assert write_guard.eol_for_write(tmp_path / "render.py", exists=False) is None


# ── verify_readback ───────────────────────────────────────────────────────

def _windows_text_mode(content: str) -> bytes:
    return content.replace("\n", "\r\n").encode("utf-8")


def test_readback_crlf_translation_is_not_truncation(tmp_path: Path):
    body = "abc" + "".join(f"line {i:02d} " + "x" * 38 + "\n" for i in range(35))
    assert len(body.encode()) == 1648 and body.count("\n") == 35
    f = tmp_path / "render_pages.py"
    f.write_bytes(_windows_text_mode(body))
    rb = write_guard.verify_readback(f, body)
    assert rb["ok"] is True
    assert rb["bytes"] == 1683
    assert "line endings" in rb["note"]


def test_readback_real_truncation_still_fails(tmp_path: Path):
    body = "alpha\nbeta\ngamma\n"
    f = tmp_path / "a.txt"
    f.write_bytes(_windows_text_mode(body)[:-4])
    rb = write_guard.verify_readback(f, body)
    assert rb["ok"] is False and "size_mismatch" in rb["reason"]


def test_readback_same_size_different_bytes_fails(tmp_path: Path):
    body = "alpha\nbeta\n"
    f = tmp_path / "a.txt"
    f.write_bytes(b"alphA\nbeta\n")
    rb = write_guard.verify_readback(f, body)
    assert rb["ok"] is False and rb["reason"] == "content_mismatch"


def test_readback_append_crlf_translation_ok(tmp_path: Path):
    f = tmp_path / "log.txt"
    f.write_bytes(b"AAA" + _windows_text_mode("b\nc\n"))
    rb = write_guard.verify_readback(f, "b\nc\n", append=True, prev_size=3)
    assert rb["ok"] is True and rb["note"]


# ── write tool ────────────────────────────────────────────────────────────

async def test_write_lands_exact_bytes(tmp_path: Path):
    body = "import sys\nprint(sys.argv)\n"
    res = await WriteTool().execute(path="tmp/render_pages.py", content=body, **_kw(tmp_path))
    assert res.success, res.content
    target = tmp_path / "saved" / "tmp" / "s1" / "render_pages.py"
    assert target.read_bytes() == body.encode()


async def test_write_keeps_an_existing_crlf_file_crlf(tmp_path: Path):
    target = tmp_path / "saved" / "tmp" / "s1" / "notes.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first\r\nsecond\r\n")
    res = await WriteTool().execute(path="tmp/notes.txt", content="first\nsecond\nthird\n",
                                    overwrite=True, **_kw(tmp_path))
    assert res.success, res.content
    assert target.read_bytes() == b"first\r\nsecond\r\nthird\r\n"


async def test_write_new_batch_file_gets_crlf(tmp_path: Path):
    res = await WriteTool().execute(path="scripts/go.bat", content="@echo off\r\n:top\necho hi\n",
                                    **_kw(tmp_path))
    assert res.success, res.content
    target = tmp_path / "saved" / "scripts" / "s1" / "go.bat"
    assert target.read_bytes() == b"@echo off\r\n:top\r\necho hi\r\n"


# ── edit tool ─────────────────────────────────────────────────────────────

async def test_edit_keeps_crlf(tmp_path: Path):
    target = tmp_path / "config.ini"
    target.write_bytes(b"[a]\r\nkey=1\r\nother=2\r\n")
    res = await EditTool().execute(path=str(target), old_string="key=1\nother=2",
                                   new_string="key=9\nother=2", **_kw(tmp_path))
    assert res.success, res.error
    assert target.read_bytes() == b"[a]\r\nkey=9\r\nother=2\r\n"


async def test_edit_keeps_lf(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_bytes(b"x = 1\ny = 2\n")
    res = await EditTool().execute(path=str(target), old_string="y = 2", new_string="y = 3",
                                   **_kw(tmp_path))
    assert res.success, res.error
    assert target.read_bytes() == b"x = 1\ny = 3\n"


async def test_edit_mixed_file_matches_every_line(tmp_path: Path):
    target = tmp_path / "mixed.txt"
    target.write_bytes(b"a\r\nb\nc\nd\n")
    res = await EditTool().execute(path=str(target), old_string="a\nb", new_string="A\nB",
                                   **_kw(tmp_path))
    assert res.success, res.error
    assert target.read_bytes() == b"A\nB\nc\nd\n"
