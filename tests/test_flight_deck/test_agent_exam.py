"""The onboarding exam's pieces that need no deck (flight_deck/agent_exam.py).

* sections worth a question are sampled, never the whole of the documents:
  every document gets a share (a long one more, not all), each from the parts
  a setup read and the parts it didn't, spread over the document; the most
  instructive become situations; the same exam id samples the same;
* the writer's and the grader's replies are cleaned — a section or item out
  of range, given twice or left empty is dropped;
* a wrong answer's cause comes from what the agent did: it didn't look, it
  looked and never reached the section, it reached it and still got it
  wrong, or it made things up;
* the score counts right as 1 and partly as ½, overall, by document, by kind
  and by whether the setup read that part.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck import agent_exam as ex


def _handbook(chapters: int = 6) -> str:
    out = ["# Handbook\n\nAn introduction, long enough to be more than a heading line for the reader.\n"]
    for c in range(1, chapters + 1):
        out.append(f"## Chapter {c} story\n\n" + f"Long ago chapter {c} began in a small office by the sea. " * 12 + "\n")
        out.append(f"## Chapter {c} procedure\n\n" + "".join(
            f"{n}. Always check item {c}.{n} before you refund more than {n * 10} EUR.\n" for n in range(1, 9)) + "\n")
    return "\n".join(out)


def _src(file: str = "handbook.md", text: str | None = None, read: set[str] | None = None, **kw) -> ex.Source:
    return ex.Source(file=file, title=file.rsplit(".", 1)[0].title(), text=text or _handbook(),
                     read=read or set(), known=read is not None, **kw)


# ── Sampling ──────────────────────────────────────────────────────────


def test_shares_give_every_document_some_and_a_long_one_more():
    got = ex.shares([100, 4, 0], 10)
    assert sum(got) == 10 and got[2] == 0 and got[1] >= 1 and got[0] > got[1]
    assert ex.shares([2, 1], 10) == [2, 1]                     # never more than there is
    assert ex.shares([5, 5, 5], 2).count(0) == 1               # too few to go round


def test_candidates_leave_out_short_and_boilerplate_sections():
    text = ("# Guide\n\nShort.\n\n## Changelog\n\n" + "Always update the version before release. " * 20
            + "\n\n## Returns\n\n" + "You must refund within 14 days of delivery. " * 20 + "\n")
    cands = ex.candidates(_src(text=text, read={"Guide › Returns"}))
    assert [c["title"] for c in cands] == ["Returns"]
    assert cands[0]["read"] is True and cands[0]["path"] == ["Guide"] and cands[0]["score"] > 0
    assert ex.candidates(_src(text=text))[0]["read"] is None   # read tracking not known: None


def test_sample_spreads_over_documents_and_read_and_unread_parts():
    read = {f"Handbook › Chapter {c} {k}" for c in (1, 2, 3) for k in ("story", "procedure")}
    picks = ex.sample([_src(read=read), _src("notes.md", text=_handbook(3).replace("Handbook", "Notes"))],
                      n=10, seed=7)
    assert len(picks) == 12                                    # 20% spare: the writer leaves some out
    assert {p["doc"] for p in picks} == {"handbook.md", "notes.md"}
    hand = [p for p in picks if p["doc"] == "handbook.md"]
    assert any(p["read"] is True for p in hand) and any(p["read"] is False for p in hand)
    keys = [p["key"] for p in picks]
    assert len(keys) == len(set(keys))
    # Situations only where a rule decides: the procedures, never the stories.
    assert any(p["kind"] == "situation" for p in picks)
    assert all("procedure" in p["title"] for p in picks if p["kind"] == "situation")
    assert ex.sample([_src(read=read)], n=10, seed=7) == ex.sample([_src(read=read)], n=10, seed=7)


def test_a_focus_leans_the_sample(monkeypatch):
    def near_three(texts, focus):
        return [1.0 if "chapter 3" in t.lower() else 0.0 for t in texts]

    monkeypatch.setattr(ex.psel, "focus_scores", near_three)
    hits = [sum("Chapter 3" in p["title"] for p in ex.sample([_src()], n=3, focus="chapter three", seed=s))
            for s in range(20)]
    plain = [sum("Chapter 3" in p["title"] for p in ex.sample([_src()], n=3, seed=s)) for s in range(20)]
    assert sum(hits) > sum(plain)


# ── The writer's and the grader's replies ─────────────────────────────


def _batch() -> list[dict]:
    return [c | {"kind": "situation" if "procedure" in c["title"] else "ask"}
            for c in ex.candidates(_src(read=set()))[:2]]


def test_the_writers_questions_are_cleaned():
    batch = _batch()
    reply = json.dumps({"questions": [
        {"section": 1, "kind": "ask", "question": "When did chapter one begin?", "answer": "Long ago.",
         "points": ["long ago", "", 7]},
        {"section": 1, "question": "Again for one?", "answer": "No."},                # given twice
        {"section": 5, "question": "Out of range?", "answer": "No."},
        {"section": 2, "kind": "riddle", "question": "A customer wants 30 EUR back; what first?",
         "answer": "Check items 1.1 to 1.3."},
        {"section": 3, "question": "", "answer": "x"},
    ]})
    got = ex.parse_questions(f"Here you go:\n```json\n{reply}\n```", batch)
    assert [q["question"] for q in got] == ["When did chapter one begin?", "A customer wants 30 EUR back; what first?"]
    assert got[0]["points"] == ["long ago", "7"] and got[0]["doc"] == "handbook.md" and got[0]["read"] is False
    assert got[1]["kind"] == "situation" and got[1]["points"] == ["Check items 1.1 to 1.3."]
    assert ex.parse_questions("no json", batch) == []


def test_the_graders_verdicts_are_cleaned():
    batch = [{}, {}, {}]
    reply = json.dumps({"grades": [
        {"item": 1, "verdict": "Right", "why": "All there."},
        {"item": 2, "verdict": "No answer", "made_up": True, "missing": ["14 days"]},
        {"item": 3, "verdict": "maybe"},
        {"item": 9, "verdict": "wrong"},
    ]})
    got = ex.parse_grades(reply, batch)
    assert set(got) == {1, 2}
    assert got[1] == {"verdict": "right", "made_up": False, "missing": [], "why": "All there."}
    assert got[2]["verdict"] == "no_answer" and got[2]["made_up"] is True and got[2]["missing"] == ["14 days"]


# ── Why it went wrong ─────────────────────────────────────────────────


def _q(**kw) -> dict:
    return {"doc": "handbook.md", "doc_title": "Handbook", "title": "Chapter 2 procedure", "path": ["Handbook"],
            "key": "Handbook › Chapter 2 procedure", "points": ["x"], **kw}


def test_what_the_agent_looked_up_says_whether_it_reached_the_section():
    body = "1. Always check item 2.1 before you refund more than 10 EUR.\n2. Always check item 2.2 first."
    search = {"tool": "knowledge", "args": {"action": "search", "query": "refund"},
              "result": "[1] Handbook › Handbook › Chapter 2 procedure\n    knowledge/handbook.md, lines 40–48\n…"}
    opened = {"tool": "knowledge", "args": {"action": "section", "doc": "handbook", "section": "chapter 2 procedure"},
              "result": "(the section)"}
    quoted = {"tool": "grep", "args": {"pattern": "refund", "path": "knowledge/"},
              "result": "knowledge/handbook.md:41: 1. Always check item 2.1 before you refund more than 10 EUR."}
    elsewhere = {"tool": "knowledge", "args": {"action": "search", "query": "story"},
                 "result": "[1] Handbook › Handbook › Chapter 5 story\nLong ago chapter 5 began…"}
    web = {"tool": "web_search", "args": {"query": "refund policy"}, "result": "Some site."}
    assert ex.looked([search]) and ex.looked([quoted]) and not ex.looked([web]) and not ex.looked([])
    for found in (search, opened, quoted):
        assert ex.reached(_q(), [found], body), found["tool"]
    assert not ex.reached(_q(), [elsewhere, web], body)


def test_a_wrong_answer_gets_its_cause():
    assert ex.cause(_q(verdict="right", looked=False)) == ""
    assert ex.cause(_q(verdict="wrong", looked=False, found=False)) == "didnt_look"
    assert ex.cause(_q(verdict="no_answer", looked=True, found=False)) == "wrong_place"
    assert ex.cause(_q(verdict="partly", looked=True, found=True)) == "got_wrong"
    assert ex.cause(_q(verdict="wrong", looked=True, found=True, made_up=True)) == "made_up"


def test_the_score_counts_partly_as_half():
    qs = [_q(verdict="right", kind="ask", read=True), _q(verdict="partly", kind="situation", read=False,
                                                         cause="got_wrong"),
          _q(verdict="wrong", kind="ask", read=False, cause="didnt_look", made_up=True),
          _q(verdict="right", kind="situation", read=None, doc="notes.md", doc_title="Notes"), _q()]
    s = ex.summary(qs)
    assert (s["questions"], s["points"], s["pct"], s["right"]) == (4, 2.5, 62, 2)
    assert s["kinds"]["ask"]["pct"] == 50 and s["kinds"]["situation"]["pct"] == 75
    assert s["read"]["read"]["pct"] == 100 and s["read"]["unread"]["questions"] == 2
    assert s["causes"] == {"didnt_look": 1, "wrong_place": 0, "got_wrong": 1, "made_up": 0} and s["made_up"] == 1
    assert [(d["title"], d["questions"]) for d in s["docs"]] == [("Handbook", 3), ("Notes", 1)]
    assert ex.summary([])["pct"] is None


# ── Asking the agent (its exam lane) ──────────────────────────────────


class _FakeSocket:
    """The agent's /ws: its welcome and replay, then *frames* once the question is in."""

    def __init__(self, frames: list[dict], hang: bool = False):
        self.before = [{"type": "welcome"}, {"type": "replay_done"}]
        self.frames = frames
        self.hang = hang
        self.sent: list[dict] = []
        self.url = ""

    async def recv(self):
        import asyncio

        if self.before:
            return json.dumps(self.before.pop(0))
        while not self.sent or (self.hang and not any(f.get("type") == "cancel" for f in self.sent)) \
                or not self.frames:
            await asyncio.sleep(0.01)
        return json.dumps(self.frames.pop(0))

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _agent_socket(monkeypatch, sock: _FakeSocket) -> None:
    import websockets

    from captain_claw.flight_deck import server as srv

    def connect(url, **kw):
        sock.url = url
        return sock

    async def up(reader, job):
        return True

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(ex.ap, "_agent_up", up)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: {"helpdesk": {"web_port": 4321, "web_auth": "tok"}})


async def test_a_question_goes_to_the_exam_lane_and_its_reply_comes_back(monkeypatch):
    sock = _FakeSocket([
        {"type": "status", "status": "thinking"},
        {"type": "chat_message", "role": "user", "content": "echo"},
        {"type": "chat_message", "role": "assistant", "content": "14 days.", "exam": {
            "lookups": [{"tool": "knowledge", "args": {"action": "search", "query": "refund"}, "ok": True,
                         "result": "[1] Handbook › Refunds"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 100}}}])
    _agent_socket(monkeypatch, sock)
    job = ex.ExamJob(id="abc123def4567890", agent="helpdesk", agent_name="Helpdesk")
    got = await ex.ask(job, {"id": "q3", "question": "What is our refund window?"})
    assert sock.url == "ws://127.0.0.1:4321/ws?lane=EXAM&token=tok"
    (frame,) = sock.sent
    assert frame["content"] == "What is our refund window?" and frame["no_flow"] and frame["no_next_steps"]
    assert frame["automation"]["kind"] == "exam" and frame["automation"]["mail_write"] == "deny"
    assert frame["automation"]["task"] == {"id": "exam-abc123def4567890-q3", "name": "Onboarding exam", "mirror": False}
    assert got["answer"] == "14 days." and got["tokens"] == 1000 and "error" not in got
    assert got["lookups"][0]["args"] == {"action": "search", "query": "refund"}


async def test_a_question_left_unanswered_is_stopped_and_waited_out(monkeypatch):
    sock = _FakeSocket([{"type": "status", "status": "ready"}], hang=True)
    _agent_socket(monkeypatch, sock)
    monkeypatch.setattr(ex, "_ASK_TIMEOUT", 0.05)
    got = await ex.ask(ex.ExamJob(id="abc", agent="helpdesk"), {"id": "q1", "question": "Hm?"})
    assert got["answer"] == "" and "No answer" in got["error"]
    assert sock.sent[-1] == {"type": "cancel"} and sock.frames == []      # its turn's end was read

    sock = _FakeSocket([{"type": "error", "message": "Lane EXAM is busy processing another request."}])
    _agent_socket(monkeypatch, sock)
    got = await ex.ask(ex.ExamJob(id="abc", agent="helpdesk"), {"id": "q2", "question": "Hm?"})
    assert got["answer"] == "" and "busy" in got["error"]


async def test_only_an_agent_that_knows_exams_takes_one(monkeypatch):
    import httpx

    from captain_claw import exam_turn
    from captain_claw.flight_deck import server as srv

    _agent_socket(monkeypatch, _FakeSocket([]))
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: True)
    asked: list = []
    caps: list = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"version": "x", "capabilities": list(caps)}

    async def fake_get(self, url, params=None, **kw):
        asked.append((url, params))
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    job = ex.ExamJob(id="abc", agent="helpdesk", agent_name="Helpdesk")
    caps[:] = ["context_packs"]
    assert await ex.can_take_exams(job) is False                   # started before it could
    caps[:] = ["context_packs", exam_turn.CAPABILITY]
    assert await ex.can_take_exams(job) is True
    assert asked[-1] == ("http://127.0.0.1:4321/api/version", {"token": "tok"})
