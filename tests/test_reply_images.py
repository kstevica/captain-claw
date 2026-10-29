"""Pictures a reply hands the person reach the web chat and the channel.

A plotting script's chart in saved/charts/ (no session folder, no tool
``Path:`` line) used to reach no surface; a reply naming no file at all
neither. Now a picture the reply names is matched among the pictures the turn
wrote under saved/, and Jev picks one when the reply names none.
"""

from __future__ import annotations

import asyncio
import os
import time
import types
from pathlib import Path

import pytest

from captain_claw import jev_client as jc
from captain_claw import reply_images as ri
from captain_claw.config import get_config
from captain_claw.session import Session


@pytest.fixture(autouse=True)
def _fresh_results():
    ri._RESULTS.clear()
    yield
    ri._RESULTS.clear()


@pytest.fixture
def world(tmp_path):
    saved = tmp_path / "workspace" / "saved"
    for sub in ("charts", "showcase/s1", "media/s1", "downloads/s1", "scripts", ".cache"):
        (saved / sub).mkdir(parents=True, exist_ok=True)
    session = Session(id="s1", name="t")
    session.add_message("user", "plot the WACC sensitivity", origin="human", channel="web")
    started = time.time() - 1
    agent = types.SimpleNamespace(
        session=session,
        tools=types.SimpleNamespace(get_saved_base_path=lambda create=False: saved),
        _current_session_slug=lambda: "s1",
    )
    return types.SimpleNamespace(saved=saved.resolve(), session=session, agent=agent,
                                 started=started, tmp=tmp_path)


def _write(path: Path, *, age: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG" + b"x" * 20)
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path.resolve()


def _tool(world, tool: str, content: str) -> None:
    world.session.add_message("tool", content, tool_name=tool, tool_call_id=tool)


async def _find(world, reply: str) -> list[Path]:
    # The turn ran a script (only a turn with tools can have made a picture).
    if not any(m.get("role") == "tool" for m in world.session.messages):
        _tool(world, "shell", "ok")
    return await ri.find(world.agent, reply, 0, world.started)


# ── References in the reply ─────────────────────────────────────────────


def test_references_finds_paths_names_and_markdown():
    reply = (
        "PNG: `saved/showcase/a8a8/wacc-explained.png`\n"
        "Also ![chart](charts/q3.JPG) and plain sales.webp, "
        "`my chart final.png`, file:///tmp/x/y.gif — "
        "see https://example.com/logo.png (not a file)."
    )
    assert ri.references(reply) == [
        "saved/showcase/a8a8/wacc-explained.png",
        "charts/q3.JPG",
        "sales.webp",
        "my chart final.png",
        "/tmp/x/y.gif",
    ]


def test_references_ignores_other_extensions():
    assert ri.references("Saved report.pdf and page.html and image.pngx") == []


# ── Names in the reply ──────────────────────────────────────────────────


async def test_a_script_chart_outside_a_session_folder_is_found(world):
    chart = _write(world.saved / "charts" / "q3.png")
    assert await _find(world, "Here's the chart: saved/charts/q3.png") == [chart]


async def test_a_bare_name_and_an_absolute_path_both_match(world):
    a = _write(world.saved / "charts" / "q3.png")
    b = _write(world.saved / "showcase" / "s1" / "wacc.jpg")
    reply = f"Two pictures: q3.png and {b}"
    assert await _find(world, reply) == [a, b]


async def test_a_name_relative_to_saved_matches(world):
    chart = _write(world.saved / "showcase" / "s1" / "wacc.png")
    assert await _find(world, "PNG: `showcase/s1/wacc.png`") == [chart]


async def test_old_pictures_and_working_folders_are_never_sent(world):
    _write(world.saved / "charts" / "old.png", age=3600)
    _write(world.saved / "downloads" / "s1" / "input.png")
    _write(world.saved / "scripts" / "plot.png")
    _write(world.saved / ".cache" / "thumb.png")
    reply = "old.png, input.png, plot.png, thumb.png"
    assert await _find(world, reply) == []


async def test_a_file_outside_saved_is_never_sent(world):
    outside = world.tmp / "secret.png"
    outside.write_bytes(b"\x89PNG")
    _write(world.saved / "charts" / "unrelated.png")  # so the walk finds something
    assert await _find(world, f"Look at {outside}") == []


async def test_pictures_a_tool_reported_are_left_to_that_tool(world):
    shown = _write(world.saved / "showcase" / "s1" / "wacc.png")
    _tool(world, "visualize", f"Path: {shown}\nHTML: {shown.with_suffix('.html')}")
    assert await _find(world, f"PNG: {shown}") == []


async def test_a_picture_the_person_attached_is_not_sent_back(world):
    photo = _write(world.saved / "media" / "s1" / "IMG_1.jpg")
    world.session.messages[0]["content"] = f"what is this?\n[Attached image: {photo}]"
    assert await _find(world, "Your photo IMG_1.jpg shows a cat.") == []


async def test_a_turn_without_tools_never_walks_saved(world, monkeypatch):
    _write(world.saved / "charts" / "q3.png")  # made by something else meanwhile

    def boom(*_a, **_k):
        raise AssertionError("saved/ must not be walked for a plain chat turn")

    monkeypatch.setattr(ri, "_new_pictures", boom)
    assert await ri.find(world.agent, "q3.png", 0, world.started) == []


async def test_named_pictures_never_ask_jev(world, monkeypatch):
    chart = _write(world.saved / "charts" / "q3.png")

    async def boom(*_a, **_k):
        raise AssertionError("Jev must not be asked when the reply names the file")

    monkeypatch.setattr(ri, "jev_pick", boom)
    assert await _find(world, "q3.png") == [chart]


# ── Jev, when the reply names no file ───────────────────────────────────


def _noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def _choice(name: str, p: float) -> dict:
    return {"type": "choice", "choice": name, "probabilities": {name: p}, "confidence": p}


@pytest.fixture
def jev(monkeypatch):
    """Standalone with a key; JevAPI.ask scripted."""
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    monkeypatch.setenv("JEV_API_KEY", "jev-test")
    monkeypatch.setattr(get_config().tools.jev, "api_key", "")
    monkeypatch.setattr(get_config().tools.jev, "find_images", True)
    state = types.SimpleNamespace(calls=[], answer={"answers": {ri.SHOWS: _noul(0.9)}})

    async def ask(self, st, questions, model=None):
        state.calls.append((st, questions))
        jc.build_request(st, questions, model)  # what would leave must validate
        if isinstance(state.answer, BaseException):
            raise state.answer
        return state.answer

    monkeypatch.setattr(jc.JevAPI, "ask", ask)
    return state


async def test_one_new_picture_and_a_confident_yes_sends_it(world, jev):
    chart = _write(world.saved / "charts" / "fig_1.png")
    assert await _find(world, "The chart is ready — WACC falls as debt rises.") == [chart]
    state, questions = jev.calls[0]
    assert set(questions) == {ri.SHOWS}
    assert state["new_pictures"] == ["charts/fig_1.png"]
    assert "plot the WACC" in state["request"]


async def test_an_unsure_yes_sends_nothing(world, jev):
    _write(world.saved / "charts" / "fig_1.png")
    jev.answer = {"answers": {ri.SHOWS: _noul(0.5)}}
    assert await _find(world, "I looked at the page.") == []


async def test_jev_picks_among_several_new_pictures(world, jev):
    _write(world.saved / "media" / "s1" / "browser_shot.png", age=0.5)
    chart = _write(world.saved / "charts" / "wacc_curve.png")
    jev.answer = {"answers": {ri.SHOWS: _noul(0.92), ri.PICK: _choice("charts/wacc_curve.png", 0.8)}}
    assert await _find(world, "Here is the WACC curve.") == [chart]
    _state, questions = jev.calls[0]
    assert set(questions[ri.PICK]["criteria"]) == {
        "charts/wacc_curve.png", "media/s1/browser_shot.png", ri._NONE}


async def test_a_pick_of_none_or_a_weak_pick_sends_nothing(world, jev):
    _write(world.saved / "charts" / "a.png")
    _write(world.saved / "charts" / "b.png")
    jev.answer = {"answers": {ri.SHOWS: _noul(0.9), ri.PICK: _choice(ri._NONE, 0.9)}}
    assert await _find(world, "Done.") == []
    ri._RESULTS.clear()
    jev.answer = {"answers": {ri.SHOWS: _noul(0.9), ri.PICK: _choice("charts/a.png", 0.3)}}
    assert await _find(world, "Done.") == []


async def test_a_jev_error_sends_nothing(world, jev):
    _write(world.saved / "charts" / "fig_1.png")
    jev.answer = jc.JevError(503, "down")
    assert await _find(world, "The chart is ready.") == []


async def test_no_new_pictures_means_no_jev_call(world, jev):
    assert await _find(world, "The chart is ready.") == []
    assert jev.calls == []


async def test_switched_off_or_a_members_turn_asks_nothing(world, jev, monkeypatch):
    _write(world.saved / "charts" / "fig_1.png")
    monkeypatch.setattr(get_config().tools.jev, "find_images", False)
    assert await _find(world, "The chart is ready.") == []
    monkeypatch.setattr(get_config().tools.jev, "find_images", True)
    ri._RESULTS.clear()
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: True))
    assert await _find(world, "The chart is ready.") == []
    assert jev.calls == []


async def test_surfaces_asking_about_the_same_turn_share_one_jev_call(world, jev):
    chart = _write(world.saved / "charts" / "fig_1.png")
    reply = "The chart is ready."
    a, b = await asyncio.gather(_find(world, reply), _find(world, reply))
    assert a == b == [chart]
    assert len(jev.calls) == 1


# ── Surfaces ────────────────────────────────────────────────────────────


async def test_whatsapp_and_devices_get_a_script_chart_the_reply_names(world):
    from captain_claw.tools.whatsapp_send_file import collect_turn_media

    _tool(world, "shell", "ok")
    chart = _write(world.saved / "charts" / "q3.png")
    got = await collect_turn_media(world.agent, 0, "Here's q3.png", world.started)
    assert got == [chart]


async def test_a_device_that_takes_no_gif_gets_none(world):
    from captain_claw.tools.whatsapp_send_file import collect_turn_media

    _tool(world, "shell", "ok")
    _write(world.saved / "charts" / "anim.gif")
    got = await collect_turn_media(world.agent, 0, "Here's anim.gif", world.started)
    assert got == []  # WhatsApp's default set has no .gif


async def test_with_reply_images_adds_only_what_the_list_misses(world):
    from captain_claw.platform_adapter import with_reply_images

    _tool(world, "shell", "ok")
    a = _write(world.saved / "charts" / "a.png")
    b = _write(world.saved / "charts" / "b.png")
    got = await with_reply_images(world.agent, [a], "a.png and b.png", 0, world.started)
    assert got == [a, b]


async def test_the_web_chat_gets_an_image_frame_for_a_named_picture(world):
    from captain_claw.web.chat_handler import _send_reply_images

    chart = _write(world.saved / "charts" / "q3.png")
    shot = _write(world.saved / "media" / "s1" / "shot.png")
    _tool(world, "browser", f"Screenshot saved.\nPath: {shot}")
    frames: list[dict] = []
    await _send_reply_images(frames.append, world.agent, 0, "q3.png, and the page: shot.png",
                             world.started)
    assert [(f["role"], f["content"]) for f in frames] == [("image", str(chart))]
