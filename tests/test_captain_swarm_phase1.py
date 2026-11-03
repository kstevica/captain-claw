"""Captain Swarm, Phase 1: the scorers, the Mrav harness, inbox-triage practice
sets and measuring an agent on them (docs/captain-swarm-plan.md)."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from captain_claw.flight_deck import captain_swarm_measure as measure
from captain_claw.flight_deck import captain_swarm_practice as practice
from captain_claw.flight_deck import captain_swarm_score as score

ROOT = Path(__file__).resolve().parent.parent
DAY = 86_400_000


def run(coro):
    return asyncio.run(coro)


# ── the noise rule ───────────────────────────────────────────────────


def test_stats_and_margin():
    s = score.stats([0.5, 0.6, 0.7, None, float("nan")])
    assert s.n == 3 and round(s.mean, 3) == 0.6 and round(s.stdev, 3) == 0.1
    assert s.low == 0.5 and s.high == 0.7
    assert score.stats([]).n == 0
    assert score.stats([0.4]).stdev == 0.0
    assert score.margin(s, 0.05) == pytest.approx(0.2)
    assert score.margin(score.stats([0.5, 0.5]), 0.05) == 0.05


def test_beats_needs_more_than_noise():
    champion = score.stats([0.60, 0.62, 0.61])
    assert score.beats(score.stats([0.70, 0.71, 0.69]), champion, min_gain=0.05)
    assert not score.beats(score.stats([0.63, 0.64, 0.62]), champion, min_gain=0.05)
    # A candidate's own spread counts too: a lucky high run isn't a win.
    assert not score.beats(score.stats([0.9, 0.5, 0.7]), champion, min_gain=0.05)
    # Lower is better when asked.
    assert score.beats(score.stats([10, 10.2]), score.stats([20, 20.4]), min_gain=1, direction="min")
    assert not score.beats(score.stats([]), champion)


# ── command ──────────────────────────────────────────────────────────


def test_last_json_and_read_number():
    assert score.last_json('progress\n{"a": {"b": 0.5}}\n') == {"a": {"b": 0.5}}
    assert score.last_json('{"x": 1}') == {"x": 1}
    assert score.last_json("nothing here") is None
    assert score.read_number({"a": {"b": 0.5}}, "a.b") == 0.5
    assert score.read_number({"a": True}, "a") is None
    assert score.read_number({"a": "1"}, "a") is None
    assert score.read_number({}, "a.b") is None


def test_score_command_reads_the_number(tmp_path):
    cmd = [sys.executable, "-c", "import json,sys; print('working', file=sys.stderr); "
           "print('x'); print(json.dumps({'summary': {'pass_rate': 0.75}})); sys.exit(1)"]
    got = run(score.score_command(cmd, tmp_path, read="summary.pass_rate"))
    assert got["value"] == 0.75 and got["exit_code"] == 1 and got["error"] == ""


def test_score_command_without_a_number(tmp_path):
    got = run(score.score_command([sys.executable, "-c", "print('{\"other\": 1}')"], tmp_path, read="value"))
    assert got["value"] is None and "no number" in got["error"]
    got = run(score.score_command([sys.executable, "-c", "print('hi')"], tmp_path, read="value"))
    assert got["value"] is None and "no JSON" in got["error"]
    got = run(score.score_command([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path,
                                  read="value", timeout=0.5))
    assert got["value"] is None and "no result" in got["error"]


# ── predicates ───────────────────────────────────────────────────────


def test_predicates(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    nested = tmp_path / "saved" / "tmp" / "s1"
    nested.mkdir(parents=True)
    (nested / "sum.txt").write_text("15\n")
    checks = [
        {"type": "file_exists", "path": "a.txt"},
        {"type": "file_contains", "path": "a.txt", "pattern": "^beta$"},
        {"type": "file_absent", "path": "gone.txt"},
        {"type": "no_pattern", "glob": "*.txt", "pattern": "gamma"},
        {"type": "reply_matches", "pattern": "\\b15\\b"},
        {"type": "reply_matches", "pattern": "FIFTEEN", "ignore_case": True},
        {"type": "reply_any", "texts": ["could not", "Fifteen"]},
    ]
    got = score.check_predicates(tmp_path, "The sum is 15 (fifteen).", checks)
    assert all(c["ok"] for c in got), got
    # Without `anywhere`, a file under saved/ isn't found; with it, it is.
    assert not score.check_predicates(tmp_path, "", [{"type": "file_exists", "path": "sum.txt"}])[0]["ok"]
    assert score.check_predicates(tmp_path, "", [{"type": "file_exists", "path": "sum.txt"}], anywhere=True)[0]["ok"]


def test_predicates_fail_closed(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    bad = [
        {"type": "file_contains", "path": "a.txt", "pattern": "("},
        {"type": "reply_matches", "pattern": "["},
        {"type": "no_pattern", "pattern": "("},
        {"type": "file_exists", "path": "../etc/passwd"},
        {"type": "file_exists", "path": "/etc/passwd"},
        {"type": "file_contains", "path": "missing.txt", "pattern": "x"},
        {"type": "mystery"},
    ]
    got = score.check_predicates(tmp_path, "anything", bad)
    assert not any(c["ok"] for c in got)
    assert all(c["detail"] for c in got)


# ── agreement and owner ──────────────────────────────────────────────


def test_agreement_counts_no_answer_against():
    got = score.agreement([(True, True), (False, False), (True, False), (False, True), (None, True), (None, False)])
    assert (got["tp"], got["tn"], got["fp"], got["fn"], got["unanswered"]) == (1, 1, 2, 2, 2)
    assert got["precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["accuracy"] == pytest.approx(2 / 6, abs=1e-4)
    assert got["base_rate"] == 0.5
    assert score.agreement([])["f1"] == 0.0


def test_owner_score():
    assert score.owner_score([True, 1, False, None]) == {"n": 3, "up": 2, "down": 1, "rate": 0.6667}
    assert score.owner_score([None])["rate"] is None


# ── the Mrav harness ─────────────────────────────────────────────────


def _harness():
    spec = importlib.util.spec_from_file_location("mrav_eval_under_test", ROOT / "scripts" / "mrav_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mrav_eval_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dev_set_loads_and_every_task_is_checked():
    h = _harness()
    tasks = h.load_tasks(ROOT / "evals" / "mrav" / "dev.json")
    assert len(tasks) == 20
    assert {"write-read", "count-files", "grep-find", "multi-step", "rename", "honesty"} <= {t.id for t in tasks}
    assert all(t.checks for t in tasks)


def test_task_file_refuses_unchecked_and_duplicate_tasks(tmp_path):
    h = _harness()
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"tasks": [{"id": "a", "prompt": "p"}]}))
    with pytest.raises(SystemExit):
        h.load_tasks(p)
    p.write_text(json.dumps({"tasks": [{"id": "a", "prompt": "p", "checks": [{"type": "reply_any", "texts": ["x"]}]}] * 2}))
    with pytest.raises(SystemExit):
        h.load_tasks(p)


def test_seed_and_judge(tmp_path):
    h = _harness()
    task = h.Task(id="t", prompt="", files={"old.txt": "x\n", "sub/a.txt": "y\n"}, dirs=["reports"],
                  checks=[{"type": "file_exists", "path": "new.txt"}, {"type": "file_absent", "path": "old.txt"}])
    h.seed(task, tmp_path)
    assert (tmp_path / "sub" / "a.txt").read_text() == "y\n" and (tmp_path / "reports").is_dir()
    assert [c["ok"] for c in h.judge(task, tmp_path, "")] == [False, False]
    (tmp_path / "old.txt").rename(tmp_path / "new.txt")
    assert [c["ok"] for c in h.judge(task, tmp_path, "")] == [True, True]


def test_summarize_gives_rate_spread_and_guardrails():
    h = _harness()
    rows = []
    for k, outcomes in enumerate([("pass", "pass"), ("pass", "FAIL"), ("pass", "TIMEOUT")]):
        for tid, outcome in zip(("a", "b"), outcomes):
            rows.append({"task": tid, "repeat": k, "outcome": outcome, "prompt_tokens": 100,
                         "completion_tokens": 20, "seconds": 2.0})
    s = h.summarize(rows, 3)
    assert s["passed"] == 4 and s["total"] == 6 and s["pass_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert s["per_repeat"] == [1.0, 0.5, 0.5]
    assert s["stdev"] == pytest.approx(0.2887, abs=1e-4)
    assert s["per_task"] == {"a": 1.0, "b": pytest.approx(0.3333, abs=1e-4)}
    assert s["tokens_per_task"] == 120.0 and s["seconds_per_task"] == 2.0
    assert s["outcomes"] == {"pass": 4, "FAIL": 1, "TIMEOUT": 1}


def test_isolated_child_must_import_the_code_under_test(tmp_path):
    """A child that imports captain_claw from anywhere but --code-root is an
    error, never a measurement of the wrong code."""
    h = _harness()
    fake_root = tmp_path / "variant"
    pkg = fake_root / "captain_claw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "config.py").write_text(
        "class _C:\n    def __getattr__(self, k):\n        return self\n    def __setattr__(self, k, v):\n        pass\n"
        "def get_config():\n    return _C()\n")
    (pkg / "agent.py").write_text(
        "class Agent:\n    def __init__(self, **kw):\n        self.last_usage = {'prompt_tokens': 7}\n"
        "    async def complete(self, prompt):\n        return 'echo: ' + prompt\n")
    task = h.Task(id="t", prompt="say hi", checks=[{"type": "reply_matches", "pattern": "echo: say hi"}])
    work = tmp_path / "w"
    work.mkdir()
    spec = {"provider": "x", "model": "y", "base_url": ""}
    got = run(h.run_isolated(task, spec, work, 60, fake_root))
    assert got["error"] == "" and got["reply"] == "echo: say hi" and got["usage"] == {"prompt_tokens": 7}
    # A package that turns out to live somewhere else is refused.
    (pkg / "__init__.py").write_text("__file__ = '/elsewhere/captain_claw/__init__.py'\n")
    got = run(h.run_isolated(task, spec, work, 60, fake_root))
    assert "imported captain_claw from /elsewhere" in got["error"]


# ── inbox triage: labelling ──────────────────────────────────────────


OWN = ["me@example.com", "me.alias@example.com"]


def _m(mid, t, sender="ann@client.example", labels=("INBOX",)):
    return {"id": mid, "internal_ms": t, "labels": list(labels), "from": sender}


def test_label_messages():
    now = 100 * DAY
    rows = [
        _m("a", 10 * DAY),
        _m("r1", 11 * DAY, sender="me@example.com", labels=("SENT",)),      # 1 day after a
        _m("b", 20 * DAY),
        _m("r2", 25 * DAY, sender="Me.Alias+x@example.com", labels=()),     # 5 days after b: too late
        _m("c", 30 * DAY, labels=("INBOX", "SPAM")),
        _m("d", 99 * DAY),                                                   # too new to know
    ]
    got = practice.label_messages(rows, OWN, reply_ms=3 * DAY, now_ms=now)
    assert set(got) == {"a", "b", "d"}
    assert got["a"]["replied"] and got["a"]["mature"] and not got["a"]["owner_wrote_before"]
    assert not got["b"]["replied"] and got["b"]["owner_wrote_before"] and got["b"]["position"] == 2
    assert not got["d"]["mature"]


def test_sample_and_split():
    items = list(range(10))
    assert practice.sample_evenly(items, 5) == [0, 2, 4, 6, 8]
    assert practice.sample_evenly(items, 20) == items
    assert practice.sample_evenly(items, 0) == []
    cases = [{"id": str(i), "received_ms": i * DAY} for i in (5, 1, 9, 3)]
    open_, hold = practice.split_by_time(cases, 5 * DAY)
    assert [c["id"] for c in open_] == ["1", "3"] and [c["id"] for c in hold] == ["5", "9"]


def _b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _full(mid, tid, t, *, sender="Ann <ann@client.example>", subject="Contract", body="Can you sign by Friday?",
          labels=("INBOX", "CATEGORY_PERSONAL", "STARRED", "Label_7"), extra=()):
    headers = [{"name": "From", "value": sender}, {"name": "To", "value": "me@example.com"},
               {"name": "Subject", "value": subject}, *extra]
    return {"id": mid, "threadId": tid, "labelIds": list(labels), "internalDate": str(t),
            "payload": {"mimeType": "text/plain", "headers": headers, "body": {"data": _b64(body)}}}


def test_case_of_keeps_what_a_person_sees_and_nothing_set_later():
    from captain_claw import gmail_inbound
    msg = _full("m1", "t1", 5 * DAY,
                body="Can you sign by Friday?\n\nOn Mon, 6 Oct 2026 at 09:12, Ann <ann@client.example> wrote:\n> old text",
                extra=[{"name": "List-Unsubscribe", "value": "<mailto:x@y>"}])
    case = practice.case_of(gmail_inbound.parse_message(msg), {"replied": True, "position": 2,
                                                                "owner_wrote_before": True}, OWN)
    assert case["label"] is True and case["position"] == 2 and case["owner_wrote_before"]
    assert case["body"] == "Can you sign by Friday?"
    assert case["categories"] == ["personal"]         # STARRED / user labels are left out
    assert case["bulk"] is True
    assert case["from"] == "Ann <ann@client.example>" and case["to"] == ["me@example.com"]


class FakeGmail:
    """Enough of the Gmail API for the builder: a profile, aliases, a paged
    list, threads (metadata) and messages (full)."""

    def __init__(self, threads, page=2):
        self.threads = threads          # tid → [full messages]
        self.page = page
        self.calls: list[str] = []

    async def __call__(self, path, params):
        self.calls.append(path)
        if path == "/profile":
            return 200, {"emailAddress": "me@example.com", "historyId": "1"}
        if path == "/settings/sendAs":
            return 200, {"sendAs": [{"sendAsEmail": "me.alias@example.com", "verificationStatus": "accepted"},
                                    {"sendAsEmail": "unverified@example.com", "verificationStatus": "pending"}]}
        if path == "/messages":
            assert "-in:sent" in params["q"]
            received = sorted((m for ms in self.threads.values() for m in ms if "SENT" not in m["labelIds"]),
                              key=lambda m: -int(m["internalDate"]))
            start = int(params.get("pageToken") or 0)
            chunk = received[start:start + self.page]
            out = {"messages": [{"id": m["id"], "threadId": m["threadId"]} for m in chunk]}
            if start + self.page < len(received):
                out["nextPageToken"] = str(start + self.page)
            return 200, out
        if path.startswith("/threads/"):
            msgs = self.threads[path.rsplit("/", 1)[1]]
            return 200, {"messages": [{**m, "payload": {"headers": m["payload"]["headers"]}} for m in msgs]}
        if path.startswith("/messages/"):
            mid = path.rsplit("/", 1)[1]
            for ms in self.threads.values():
                for m in ms:
                    if m["id"] == mid:
                        return 200, m
            return 404, {}
        return 404, {}


def _mailbox(now):
    return {
        "t1": [_full("a1", "t1", now - 40 * DAY),
               _full("a1r", "t1", now - 39 * DAY, sender="me@example.com", labels=("SENT",))],
        "t2": [_full("b1", "t2", now - 30 * DAY, sender="News <news@shop.example>", subject="Sale")],
        "t3": [_full("c1", "t3", now - 10 * DAY),
               _full("c1r", "t3", now - 9 * DAY, sender="Me <me.alias@example.com>", labels=())],
        "t4": [_full("d1", "t4", now - 5 * DAY, subject="later")],
    }


def test_build_mail_triage_from_a_fake_mailbox():
    now = 400 * DAY
    gmail = FakeGmail(_mailbox(now))
    seen = []
    built = run(practice.build_mail_triage(gmail, now_ms=now, window_days=60, reply_days=3, holdout_days=14,
                                           max_cases=50, progress=lambda *a: seen.append(a)))
    assert built["own"] == ["me@example.com", "me.alias@example.com"]
    # c1r (from an alias, unlabelled) is the owner's; nothing the owner sent is a case.
    ids = {c["id"]: c["label"] for c in built["open"] + built["holdout"]}
    assert ids == {"a1": True, "b1": False, "c1": True, "d1": False}
    assert [c["id"] for c in built["holdout"]] == ["c1", "d1"]
    assert [c["id"] for c in built["open"]] == ["a1", "b1"]
    assert any(s[0] == "reading" for s in seen)
    assert sum(1 for p in gmail.calls if p == "/messages") >= 2        # it followed the pages


def test_build_leaves_out_unsettled_mail():
    now = 400 * DAY
    box = _mailbox(now)
    box["t5"] = [_full("e1", "t5", now - 2 * DAY)]   # inside reply_days: the list window excludes it
    gmail = FakeGmail(box, page=10)
    built = run(practice.build_mail_triage(gmail, now_ms=now, window_days=60, reply_days=3, holdout_days=14))
    assert "e1" not in {c["id"] for c in built["open"] + built["holdout"]}


# ── the store ────────────────────────────────────────────────────────


@pytest.fixture()
def sealed(tmp_path, monkeypatch):
    root = tmp_path / "captain_swarm" / "sealed"
    monkeypatch.setattr(practice, "sealed_root", lambda: root)
    return root


def test_save_list_load_delete_and_permissions(sealed):
    built = {"own": ["me@example.com"], "skipped": {"own": 1},
             "window": {"from": "a", "to": "b", "holdout_from": "c"},
             "open": [{"id": "a", "received_ms": 1, "received": "x", "label": True}],
             "holdout": [{"id": "b", "received_ms": 2, "received": "y", "label": False}]}
    meta = practice.save_set("user-1", "mt-1", built, {"reply_days": 3})
    assert meta["open"]["cases"] == 1 and meta["open"]["base_rate"] == 1.0 and meta["holdout"]["replied"] == 0
    d = practice.set_dir("user-1", "mt-1")
    assert stat.S_IMODE(os.stat(d / "open.jsonl").st_mode) == 0o600
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(sealed).st_mode) == 0o700
    assert [m["id"] for m in practice.list_sets("user-1")] == ["mt-1"]
    assert practice.list_sets("someone-else") == []
    assert [c["id"] for c in practice.load_cases("user-1", "mt-1", "holdout")] == ["b"]
    with pytest.raises(ValueError):
        practice.load_cases("user-1", "mt-1", "secret")
    with pytest.raises(ValueError):
        practice.set_dir("user-1", "../x")
    assert practice.delete_set("user-1", "mt-1") and not d.exists()
    assert not practice.delete_set("user-1", "mt-1")


# ── measuring ────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer,expected", [
    ("YES\nIt asks for a signature.", True),
    ("**No** — a newsletter.", False),
    ("no", False),
    ("Yes, because", True),
    ("I think this needs a reply. YES", True),
    ("Maybe.", None),
    ("", None),
    ("YES or NO? Hard to say.", None),
])
def test_parse_yes_no(answer, expected):
    assert measure.parse_yes_no(answer) is expected


def test_triage_question_never_shows_the_label():
    case = {"id": "x", "from": "Ann <ann@client.example>", "to": ["me@example.com"], "subject": "Contract",
            "received": "2026-10-01T09:00+00:00", "body": "Can you sign?", "categories": ["personal"],
            "bulk": False, "position": 2, "owner_wrote_before": True, "label": True,
            "attachments": ["contract.pdf"]}
    q = measure.triage_question(case, "Stevica")
    assert "Stevica's own reply" in q and "Contract" in q and "contract.pdf" in q and "already written" in q
    assert "label" not in q.lower()


def test_run_triage_scores_repeats_and_counts_failures_against():
    cases = [{"id": "a", "label": True, "body": "A"}, {"id": "b", "label": False, "body": "B"},
             {"id": "c", "label": True, "body": "C"}]
    calls = []

    async def ask(question):
        calls.append(question)
        if "\n\nC\n\n" in question:
            return {"error": "No answer in 5 minutes."}
        return {"answer": "YES" if "\n\nA\n\n" in question else "NO", "seconds": 2, "tokens": 50,
                "lookups": [{"tool": "knowledge"}]}

    got = run(measure.run_triage(cases, ask=ask, repeat=2))
    assert len(calls) == 6 and got["asked"] == 6 and not got["stopped"]
    assert [r["f1"] for r in got["runs"]] == [pytest.approx(0.6667, abs=1e-4)] * 2
    assert got["f1"]["stdev"] == 0.0 and got["value"] == pytest.approx(0.6667, abs=1e-4)
    assert got["guardrails"]["errors"] == 2 and got["guardrails"]["intents_per_case"] == 0.0
    row_c = next(r for r in got["cases"] if r["id"] == "c")
    assert row_c["predicted"] == [None, None]
    assert "body" not in json.dumps(got["cases"])


def test_run_triage_stops_and_keeps_finished_runs():
    cases = [{"id": str(i), "label": i % 2 == 0, "body": str(i)} for i in range(4)]
    asked = []

    async def ask(question):
        asked.append(1)
        return {"answer": "YES", "seconds": 1}

    got = run(measure.run_triage(cases, ask=ask, repeat=3, should_stop=lambda: len(asked) >= 6))
    assert got["stopped"] and got["asked"] == 6 and len(got["runs"]) == 1


# ── the routes ───────────────────────────────────────────────────────


@pytest.fixture()
def client(sealed, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import agent_preset as ap
    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1", "display_name": "Ana", "role": "admin"}
    monkeypatch.setattr(ap, "_own_process", lambda slug, owner: {"name": "Helper"} if slug == "helper"
                        else (_ for _ in ()).throw(__import__("fastapi").HTTPException(404, "no")))
    monkeypatch.setattr(ap, "_owner", lambda user: user["id"])

    async def no_quota(_owner):
        return False
    monkeypatch.setattr(ap, "_quota_spent", no_quota)
    with TestClient(app) as c:
        yield c, routes


def _ready_set(owner="user-1", set_id="mt-1"):
    built = {"own": ["me@example.com"], "skipped": {}, "window": {"from": "", "to": "", "holdout_from": ""},
             "open": [{"id": f"o{i}", "received_ms": i, "received": "", "label": i % 2 == 0, "body": f"o{i}"}
                      for i in range(6)],
             "holdout": [{"id": "h1", "received_ms": 99, "received": "", "label": True, "body": "h1"}]}
    return practice.save_set(owner, set_id, built, {"reply_days": 3})


def test_routes_build_list_and_delete(client, monkeypatch):
    c, routes = client

    async def fake_token(_uid):
        return "tok"

    async def fake_build(fetch, **kw):
        return {"own": ["me@example.com"], "skipped": {}, "window": {"from": "", "to": "", "holdout_from": ""},
                "open": [{"id": "a", "received_ms": 1, "received": "x", "label": True}],
                "holdout": [{"id": "b", "received_ms": 2, "received": "y", "label": False}]}
    import captain_claw.flight_deck.google_oauth_routes as g
    monkeypatch.setattr(g, "get_valid_google_access_token", fake_token)
    monkeypatch.setattr(practice, "build_mail_triage", fake_build)
    r = c.post("/fd/captain-swarm/practice/mail-triage", json={"window_days": 60})
    assert r.status_code == 200 and r.json()["state"] == "building"
    set_id = r.json()["id"]
    for _ in range(50):
        if c.get(f"/fd/captain-swarm/practice/{set_id}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    got = c.get(f"/fd/captain-swarm/practice/{set_id}").json()
    assert got["state"] == "ready" and got["open"]["cases"] == 1
    assert [s["id"] for s in c.get("/fd/captain-swarm/practice").json()["sets"]] == [set_id]
    assert c.delete(f"/fd/captain-swarm/practice/{set_id}").json() == {"deleted": True}
    assert c.get(f"/fd/captain-swarm/practice/{set_id}").status_code == 404


def test_routes_build_without_google_says_so(client, monkeypatch):
    c, _ = client

    async def no_token(_uid):
        return None
    import captain_claw.flight_deck.google_oauth_routes as g
    monkeypatch.setattr(g, "get_valid_google_access_token", no_token)
    set_id = c.post("/fd/captain-swarm/practice/mail-triage", json={}).json()["id"]
    for _ in range(50):
        got = c.get(f"/fd/captain-swarm/practice/{set_id}").json()
        if got["state"] != "building":
            break
        time.sleep(0.02)
    assert got["state"] == "error" and "Connect Google" in got["error"]


def test_routes_measure_an_agent(client, monkeypatch):
    c, routes = client
    _ready_set()

    async def fake_ask_agent(owner, slug, name, mid):
        async def ask(question):
            return {"answer": "YES" if "o0" in question or "o2" in question or "o4" in question else "NO",
                    "seconds": 1, "tokens": 10}
        return ask
    monkeypatch.setattr(routes, "_ask_agent", fake_ask_agent)
    assert c.post("/fd/captain-swarm/practice/mt-1/measure", json={"agent": "someone-elses"}).status_code == 404
    r = c.post("/fd/captain-swarm/practice/mt-1/measure", json={"agent": "helper", "cases": 6, "repeat": 2})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    for _ in range(100):
        m = c.get(f"/fd/captain-swarm/measurements/mt-1/{mid}").json()
        if m["state"] != "running":
            break
        time.sleep(0.02)
    assert m["state"] == "done", m
    assert m["result"]["value"] == 1.0 and m["result"]["f1"]["n"] == 2
    assert len(m["result"]["cases"]) == 6
    listed = c.get("/fd/captain-swarm/practice/mt-1/measurements").json()["measurements"]
    assert listed[0]["id"] == mid and "cases" not in listed[0]["result"]


def test_routes_refuse_a_set_that_is_not_ready(client):
    c, _ = client
    practice.mark("user-1", "mt-2", state="building")
    assert c.post("/fd/captain-swarm/practice/mt-2/measure", json={"agent": "helper"}).status_code == 409
    assert c.get("/fd/captain-swarm/practice/nope").status_code == 404


def test_routes_refuse_to_delete_a_set_being_built(client):
    """The build's last write runs in a thread a cancel can't stop, so a delete
    then could leave the mail copy behind."""
    c, routes = client
    practice.mark("user-1", "mt-3", state="building")
    routes._TASKS["build:user-1/mt-3"] = object()
    try:
        r = c.delete("/fd/captain-swarm/practice/mt-3")
        assert r.status_code == 409 and "still being built" in r.json()["detail"]
        assert practice.get_meta("user-1", "mt-3") is not None
    finally:
        routes._TASKS.pop("build:user-1/mt-3", None)
