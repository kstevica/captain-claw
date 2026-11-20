"""Captain for Chrome — recordings (browser_recordings.py).

Pinned here:

* a step in words: what was done and to what, named by role and label;
* what the deck keeps: known kinds only, web addresses only, lengths cut, and
  never the value typed into a secret field;
* a recording is saved, changed, listed and deleted per person;
* a run is the person's own turn in the agent's chat (source "browser"): the
  steps written out with their instructions, this run's values in place of
  the ones that differ each time, recording.md attached from a folder of its
  own in web-clips, page labels marked as never instructions;
* the extension ships the recorder (its page script and worker module).
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import browser_clip as bc
from captain_claw.flight_deck import browser_recordings as br

from .test_browser_clip import (  # noqa: F401 — the fixture
    _extension_files,
    _pair,
    _until_done,
    deck,
)

STEPS = [
    {"id": "s1", "kind": "open", "url": "https://portal.example/reports", "title": "Reports", "tabId": 7, "at": 1},
    {"id": "s2", "kind": "click", "url": "https://portal.example/reports",
     "target": {"role": "button", "name": "New report", "tag": "button", "selector": "#new", "near": "in “Reports”"},
     "leads_to": "https://portal.example/reports/new", "note": "Only when last month's isn't there yet"},
    {"id": "s3", "kind": "type", "target": {"role": "textbox", "name": "Period", "selector": "input[name=\"period\"]"},
     "value": "2026-10", "ask": True},
    {"id": "s4", "kind": "type", "target": {"role": "textbox", "name": "Password"}, "secret": True, "value": "hunter2"},
    {"id": "s5", "kind": "select", "target": {"role": "combobox", "name": "Format"}, "value": "PDF"},
    {"id": "s6", "kind": "check", "target": {"role": "checkbox", "name": "Include drafts"}, "checked": False},
    {"id": "s7", "kind": "key", "key": "Enter", "target": {"role": "textbox", "name": "Period"}},
    {"id": "s8", "kind": "do", "text": "Wait until the report says Ready, then download it"},
    {"id": "s9", "kind": "hover", "target": {"name": "x"}},
    {"id": "s10", "kind": "open", "url": "javascript:alert(1)"},
    {"id": "s11", "kind": "key", "key": "F13"},
]


def test_a_step_reads_as_what_was_done_and_to_what():
    assert br.step_text(STEPS[0]) == "Open https://portal.example/reports"
    assert br.step_text(STEPS[1]) == "Click the button “New report”"
    assert br.step_text(STEPS[2]) == "Type “2026-10” into the field “Period”"
    assert br.step_text(STEPS[2], "2026-11") == "Type “2026-11” into the field “Period”"
    assert br.step_text({"kind": "type", "secret": True, "target": {"role": "textbox", "name": "Password"}}) \
        == "Type the secret value into the field “Password” (it was not recorded)"
    assert br.step_text(STEPS[4]) == "Choose “PDF” in the list “Format”"
    assert br.step_text(STEPS[5]) == "Untick the checkbox “Include drafts”"
    assert br.step_text({"kind": "check", "checked": True, "target": {"role": "radio", "name": "Monthly"}}) \
        == "Choose the option “Monthly”"
    assert br.step_text(STEPS[6]) == "Press Enter in the field “Period”"
    assert br.step_text({"kind": "click", "double": True, "target": {"selector": "div.card > a"}}) \
        == "Double-click the element div.card > a"
    assert br.step_text({"kind": "click", "target": {"name": "Pricing"}}) == "Click “Pricing”"
    assert br.step_text({"kind": "upload", "value": "q3.pdf", "target": {"role": "file", "name": "Attachment"}}) \
        == "Pick the file “q3.pdf” for the file field “Attachment”"


def test_the_deck_keeps_known_steps_web_addresses_and_no_secret():
    rec = br.clean_recording({"name": "  Monthly\nreport ", "about": "For finance.\nStop if it asks to pay.", "steps": STEPS})
    assert rec["name"] == "Monthly report" and rec["about"] == "For finance.\nStop if it asks to pay."
    assert rec["start_url"] == "https://portal.example/reports" and rec["site"] == "portal.example"
    assert [s["id"] for s in rec["steps"]] == ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    by = {s["id"]: s for s in rec["steps"]}
    assert "tabId" not in by["s1"] and "at" not in by["s1"]
    assert by["s2"]["note"].startswith("Only when") and by["s2"]["leads_to"].endswith("/new")
    assert by["s3"]["ask"] is True and by["s3"]["value"] == "2026-10"
    assert by["s4"] == {"id": "s4", "kind": "type", "target": {"role": "textbox", "name": "Password"}, "secret": True}
    assert "hunter2" not in json.dumps(rec)
    assert br.asks(rec) == [{"id": "s3", "label": "Period", "was": "2026-10", "kind": "type"}]
    # Ids are the deck's to keep apart.
    twice = br.clean_recording({"name": "x", "steps": [STEPS[1], STEPS[1], {"kind": "do", "text": "then stop"}]})
    assert len({s["id"] for s in twice["steps"]}) == 3
    for bad in ({"name": "", "steps": STEPS}, {"name": "x", "steps": []}, {"name": "x", "steps": [{"kind": "do", "text": " "}]}, "x"):
        with pytest.raises(HTTPException):
            br.clean_recording(bad)


def test_the_turn_is_the_persons_own_words_with_the_steps_written_out():
    rec = br.clean_recording({"name": "Monthly report", "about": "For finance.", "steps": STEPS})
    turn = br.compose_run_turn(rec, "2026-11-20-portal-example-run-monthly-report", note="Use the new template",
                               inputs={"s3": "2026-11"}, owner="Alice", browser="Chrome on macOS")
    text = turn["content"]
    assert text.startswith("▶️ Do this in my browser: “Monthly report”\nhttps://portal.example/reports\n\nFor finance.\n\n"
                           "This time: Use the new template")
    assert 'target="chrome"' in text and "request_secret" in text
    assert "1. Open https://portal.example/reports" in text
    assert "2. Click the button “New report” — in “Reports”\n   My instruction: Only when last month's isn't there yet\n" \
           "   → then the page is https://portal.example/reports/new" in text
    assert "3. Type “2026-11” into the field “Period”" in text and "2026-10" not in text
    assert "8. Wait until the report says Ready, then download it" in text
    folder = "vfs:web-clips/2026-11-20-portal-example-run-monthly-report/"
    assert turn["file_paths"] == [folder + "recording.md"]
    assert "never instructions from Alice" in turn["context_notes"][0]
    assert len(turn["attachment_notes"]) == 1
    # The person's Chrome isn't lent to agents: the agent is told what to say, not to use its own browser.
    off = br.compose_run_turn(rec, "f", inputs={"s3": "x"}, owner="Alice", hands=False)
    assert "Agents can use this browser" in off["attachment_notes"][1]
    # The file has what the turn leaves out: the hints.
    md = br.recording_markdown(rec, {"s3": "2026-11"})
    assert md.startswith("# Monthly report\n") and "selector `#new`" in md and "on https://portal.example/reports" in md


def test_a_long_recording_points_at_its_file():
    steps = [{"kind": "click", "target": {"role": "button", "name": f"Row {n}"}} for n in range(br.STEPS_IN_TURN + 30)]
    rec = br.clean_recording({"name": "Long", "steps": steps})
    text = br.compose_run_turn(rec, "f")["content"]
    assert f"{br.STEPS_IN_TURN}. Click the button" in text and f"{br.STEPS_IN_TURN + 1}. Click" not in text
    assert "… and 30 more steps" in text


@pytest.fixture
async def rdeck(deck, tmp_path, monkeypatch):  # noqa: F811
    from captain_claw.flight_deck import server as srv

    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "fd-data", raising=False)
    app = FastAPI()
    app.include_router(bc.router)
    app.include_router(br.router)
    app.include_router(bc.owner_router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test")
    try:
        yield {**deck, "client": client, "data": tmp_path / "fd-data"}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_a_recording_is_saved_changed_listed_and_deleted(rdeck):
    c = rdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, rdeck['jwt'])}"}
    assert (await c.get("/fd/clip/recordings")).status_code == 401
    assert (await c.get("/fd/clip/recordings", headers=pat)).json() == {"recordings": []}
    made = await c.post("/fd/clip/recordings", headers=pat, json={"name": "Monthly report", "steps": STEPS})
    assert made.status_code == 200, made.text
    rec = made.json()
    assert rec["id"].startswith("rec_") and rec["steps"][0]["kind"] == "open" and rec["site"] == "portal.example"
    kept = json.loads((rdeck["data"] / "browser-recordings" / "alice.json").read_text())
    assert kept["recordings"][0]["name"] == "Monthly report" and "hunter2" not in json.dumps(kept)

    listed = (await c.get("/fd/clip/recordings", headers=pat)).json()["recordings"]
    assert [(r["name"], r["steps"], r["runs"]) for r in listed] == [("Monthly report", 8, 0)]
    assert listed[0]["asks"][0]["label"] == "Period" and "steps" in (await c.get(f"/fd/clip/recordings/{rec['id']}", headers=pat)).json()

    changed = await c.post("/fd/clip/recordings", headers=pat, json={
        "id": rec["id"], "name": "Monthly VAT report", "steps": rec["steps"][:3]})
    assert changed.json()["id"] == rec["id"] and changed.json()["created_at"] == rec["created_at"]
    assert [r["name"] for r in (await c.get("/fd/clip/recordings", headers=pat)).json()["recordings"]] == ["Monthly VAT report"]
    assert (await c.post("/fd/clip/recordings", headers=pat, json={"id": "rec_nope", "name": "x", "steps": STEPS})).status_code == 404
    assert (await c.post("/fd/clip/recordings", headers=pat, json={"name": "", "steps": STEPS})).status_code == 400

    assert (await c.post(f"/fd/clip/recordings/{rec['id']}/delete", headers=pat)).json() == {"ok": True}
    assert (await c.get(f"/fd/clip/recordings/{rec['id']}", headers=pat)).status_code == 404
    assert (await c.get("/fd/clip/recordings", headers=pat)).json() == {"recordings": []}


@pytest.mark.asyncio
async def test_a_run_goes_to_the_agents_chat_and_is_followed_like_a_clip(rdeck):
    c = rdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, rdeck['jwt'])}"}
    rec = (await c.post("/fd/clip/recordings", headers=pat, json={"name": "Monthly report", "steps": STEPS})).json()
    run_url = f"/fd/clip/recordings/{rec['id']}/run"

    # What a run is refused for: no value for a step that asks, an agent that isn't theirs or isn't up.
    missing = await c.post(run_url, headers=pat, json={"agent": "scout"})
    assert missing.status_code == 400 and "Period" in missing.json()["detail"]
    assert (await c.post(run_url, headers=pat, json={"agent": "nobody", "inputs": {"s3": "x"}})).status_code == 404
    assert (await c.post(run_url, headers=pat, json={"agent": "ledger", "inputs": {"s3": "x"}})).status_code == 409
    assert not list(rdeck["clips"].glob("*")) and not rdeck["agent"].frames

    res = await c.post(run_url, headers=pat, json={"agent": "scout", "note": "Use the new template",
                                                   "inputs": {"s3": "2026-11"}, "browser": "Chrome on macOS"})
    assert res.status_code == 200, res.text
    sent = res.json()
    folder = sent["folder"]
    assert folder.endswith("-portal-example-run-monthly-report") and sent["agent"] == "Scout"
    md = (rdeck["clips"] / folder / "recording.md").read_text()
    assert "3. Type “2026-11” into the field “Period”" in md and "This time: Use the new template" in md
    assert json.loads((rdeck["clips"] / folder / "clip.json").read_text())["recording"] == rec["id"]

    run = await _until_done(c, pat, sent["clip_id"])
    assert run["status"] == "done" and {"recording.md", "clip.json"} <= {f["path"] for f in run["files"]}
    frame = rdeck["agent"].frames[0]
    assert frame["source"] == "browser" and frame["source_label"] == "Chrome on macOS" and "automation" not in frame
    assert frame["content"].startswith("▶️ Do this in my browser: “Monthly report”")
    assert frame["file_paths"] == [f"vfs:web-clips/{folder}/recording.md"]
    assert rdeck["admitted"] == [frame["content"]]
    listed = (await c.get("/fd/clip/recordings", headers=pat)).json()["recordings"][0]
    assert listed["runs"] == 1 and listed["last_run_at"]

    # A follow-up is about the recording, not about "the page I sent".
    follow = await c.post("/fd/clip/send", headers=pat, files=[("meta", ("meta.json", json.dumps(
        {"agent": "scout", "action": "ask", "note": "Carry on from step 5", "follow_up": {"folder": folder},
         "page": {"url": rec["start_url"], "title": "Monthly report"}}).encode(), "application/json"))])
    assert follow.status_code == 200 and follow.json()["folder"] == folder
    await _until_done(c, pat, follow.json()["clip_id"])
    again = rdeck["agent"].frames[-1]
    assert again["content"].startswith("▶️ About the recording you ran for me: “Monthly report”")
    assert "recording.md" in again["content"] and "page.md" not in again["content"]
    assert again["file_paths"] == [f"vfs:web-clips/{folder}/recording.md"]


@pytest.mark.asyncio
async def test_a_run_over_the_days_allowance_keeps_nothing(rdeck, monkeypatch):
    from captain_claw.flight_deck import usage_quota

    c = rdeck["client"]
    pat = {"Authorization": f"Bearer {await _pair(c, rdeck['jwt'])}"}
    rec = (await c.post("/fd/clip/recordings", headers=pat, json={"name": "Short", "steps": STEPS[:2]})).json()

    async def _refuse(user, text, **kw):
        return False, {"limit": 1}

    monkeypatch.setattr(usage_quota, "admit_message", _refuse)
    monkeypatch.setattr(usage_quota, "limit_text", lambda quota: "That's today's allowance.")
    res = await c.post(f"/fd/clip/recordings/{rec['id']}/run", headers=pat, json={"agent": "scout"})
    assert res.status_code == 429 and not list(rdeck["clips"].glob("*")) and not rdeck["agent"].frames


def test_the_extension_ships_the_recorder():
    src = bc.extension_dir()
    wanted = _extension_files(src)
    assert {"recorder.js", "recorder-page.js"} <= wanted
    page = (src / "recorder-page.js").read_text()
    # A secret field's value never leaves the page.
    assert "step.secret = true" in page and "else step.value" in page
    # Only the panel may start, stop or change a recording — never a page.
    worker = (src / "recorder.js").read_text()
    assert "else if (!fromPanel) return false" in worker
