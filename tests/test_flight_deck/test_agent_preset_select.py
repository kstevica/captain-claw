"""A budgeted read — what the model reads of material bigger than the
person's budget, and how what it found becomes one setup:

* a document is cut into its sections, covering all of it, each with where
  it sits; a long one, or one without headings, in pieces;
* a section scores by what it tells an agent to do — a procedure over
  definitions over prose; licenses and changelogs count for little;
* the best sections are read within the budget, every document getting a
  share, in document order; a focus pulls in the sections closest to it;
* material over the budget is read by those sections, the model told so
  and given each document's chapters once; within the budget it is read
  whole;
* what the steps found is folded without the model — the same rule in other
  words kept once, in its most specific wording, first — and one model call
  merges it however many steps there were; too big for one, the fold stands.
"""

from __future__ import annotations

import json
import re

import numpy as np

from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_outline as ol
from captain_claw.flight_deck import agent_preset_select as sel

_PROCEDURE = ("1. Check the order date in the shop admin.\n"
              "2. A refund is allowed only if the order is no more than 14 days old.\n"
              "3. Never refund more than 200 EUR without asking the owner first.\n"
              "4. Always send the refund confirmation template.\n")
_PROSE = ("We started in a small garage. Over the years we moved to a bigger office and hired more people. "
          "The founders still remember the early days fondly and tell stories about them at company dinners. ")


def _handbook(name: str, chapters: int = 4) -> str:
    out = [f"# {name}\n\nAn introduction to {name} that runs on for a while about nothing in particular, "
           "with enough words to be more than a heading.\n"]
    for c in range(1, chapters + 1):
        out.append(f"## Chapter {c} history\n\n{_PROSE * 3}\n")
        out.append(f"## Chapter {c} procedure\n\n{_PROCEDURE}\n{_PROCEDURE}\n")
    return "\n".join(out)


def _offline(texts):
    raise RuntimeError("no local model here")


def test_units_cover_the_document_and_say_where_each_sits():
    text = "Before any heading, a preamble.\n\n# Guide\n\nIntro.\n\n## Refunds\n\nRefund text.\n\n### EU\n\nEU text.\n"
    units = sel.units(text)
    assert "".join(text[u["start"]:u["end"]] for u in units) == text
    assert [(u["title"], u["path"]) for u in units] == [
        ("Its opening", []), ("Guide", []), ("Refunds", ["Guide"]), ("EU", ["Guide", "Refunds"])]
    long = "# Long\n\n" + "\n\n".join(f"Paragraph {n} " + "x" * 400 for n in range(40))
    pieces = sel.units(long, max_chars=2000)
    assert len(pieces) > 5 and all(u["end"] - u["start"] <= 2000 for u in pieces)
    assert pieces[0]["title"].startswith("Long (1/") and "".join(long[u["start"]:u["end"]] for u in pieces) == long
    plain = sel.units("no headings at all. " * 300, max_chars=1500)
    assert len(plain) > 3 and plain[0]["title"] == f"Part 1 (1/{len(plain)})" and plain[0]["key"] == "Part 1"


def test_a_procedure_scores_above_definitions_and_prose():
    definitions = ("**RMA**: return merchandise authorisation, the number a return gets.\n"
                   "**Store credit** means an amount the customer can spend in the shop instead of a refund.\n")
    procedure = sel.instructive(_PROCEDURE * 2)
    assert procedure > sel.instructive(definitions * 2) > sel.instructive(_PROSE * 3) == 0
    assert sel.instructive(_PROCEDURE * 2, "MIT License") < procedure / 4
    assert sel.instructive(_PROCEDURE * 2, "[1.4.2] - 2024-03-01") < procedure / 4
    code = "```python\n" + "x = must_never_always()\n" * 40 + "```\n"
    assert sel.instructive(code) == 0                                   # code isn't prose


def test_the_best_sections_are_read_within_the_budget_every_document_some():
    docs = [{"name": f"h{n}.md", "text": _handbook(f"Handbook {n}"), "usage": "auto"} for n in range(3)]
    budget = 3000
    picked, summary = sel.select(docs, budget)
    assert sum(u["end"] - u["start"] for u in picked) <= budget
    # Each handbook: 4 history and 4 procedure sections (its two-line intro has nothing to read).
    assert all(s["sections"] >= 1 for s in summary) and all(s["of"] == 8 for s in summary)
    assert picked == sorted(picked, key=lambda u: (u["doc"], u["start"]))
    assert {u["title"].split()[-1] for u in picked} == {"procedure"}           # never the history
    everything, _ = sel.select(docs, 10**9)
    assert len(everything) == 24


def test_a_focus_pulls_in_the_sections_closest_to_it(monkeypatch):
    text = (f"# Manual\n\n## Refund steps\n\n{_PROCEDURE * 2}\n## Discount rates\n\n"
            + "The discount rate reflects the risk of the cash flows and the cost of capital. " * 6 + "\n")
    vocab = ["discount", "rate", "risk", "capital", "refund", "order"]

    def fake(texts):
        v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype="float32") + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(ol, "_encode", fake)
    docs = [{"name": "m.md", "text": text, "usage": "auto"}]
    room = len(_PROCEDURE * 2) + 200
    assert [u["title"] for u in sel.select(docs, room)[0]] == ["Refund steps"]
    assert [u["title"] for u in sel.select(docs, room, focus="discount rates and risk")[0]] == ["Discount rates"]
    monkeypatch.setattr(ol, "_encode", _offline)                        # by its words, then
    assert [u["title"] for u in sel.select(docs, room, focus="discount rates")[0]] == ["Discount rates"]


def _doc(name: str, text: str) -> dict:
    return {"name": name, "text": text, "read": True, "description": "", "usage": "auto"}


def test_material_over_the_budget_is_read_by_its_best_sections(monkeypatch):
    monkeypatch.setattr(sel, "BUDGETS", {**sel.BUDGETS, "quick": 1000})       # 4,000 characters
    docs = [_doc("a.md", _handbook("Alpha", 6)), _doc("b.md", _handbook("Beta", 6))]
    steps, notes = ap.plan_steps("Be brief.", docs, step_tok=600, budget="quick")
    pieces = [p for s in steps for p in s["pieces"]]
    assert pieces and all(p.get("selected") for p in pieces)
    assert sum(len(p["text"]) for p in pieces) <= 4000
    assert {p["doc"]["name"] for p in pieces} == {"a.md", "b.md"}
    assert any(re.match(r"a\.md: read \d+ of its 12 sections, about 1k tokens — the ones with the most rules", n)
               for n in notes)
    assert re.match(r"a\.md, Chapter \d procedure( and \d more sections?)?", steps[0]["label"])
    prompts = [ap.step_prompt(None, "Be brief.", s, n, len(steps)) for n, s in enumerate(steps, 1)]
    assert "## Selected sections" in prompts[0] and "### a.md — selected sections" in prompts[0]
    assert "(In: Alpha)" in prompts[0]
    # Each document's chapters once, where it first comes up — not its title above them.
    assert sum(p.count("Its chapters (the whole document): Chapter 1 history; Chapter 1 procedure") for p in prompts) == 2
    whole, notes = ap.plan_steps("", docs, step_tok=600, budget="deep")
    assert notes == [] and not any(p.get("selected") for s in whole for p in s["pieces"])


def test_the_focus_is_named_in_what_the_person_is_told(monkeypatch):
    monkeypatch.setattr(sel, "BUDGETS", {**sel.BUDGETS, "quick": 500})
    monkeypatch.setattr(ol, "_encode", _offline)
    _, notes = ap.plan_steps("", [_doc("a.md", _handbook("Alpha", 6))], step_tok=600, budget="quick",
                             focus="refund steps")
    assert "closest to “refund steps”" in notes[0]


def test_the_rest_of_a_plan_is_cut_again_for_a_smaller_reader(monkeypatch):
    monkeypatch.setattr(sel, "BUDGETS", {**sel.BUDGETS, "quick": 3000})
    docs = [_doc("a.md", _handbook("Alpha", 12))]
    steps, _ = ap.plan_steps("", docs, step_tok=2000, budget="quick")
    again = ap.refit(steps, 2, 400)                                       # 1,600 bytes a step from part 2
    assert again[0] is steps[0] and len(again) > len(steps)
    assert all(sum(len(p["text"].encode()) for p in s["pieces"]) <= 1600 for s in again[1:])
    text = lambda ss: "".join(p["text"] for s in ss for p in s["pieces"])   # noqa: E731
    assert text(again[1:]) == text(steps[1:])                             # nothing lost, nothing twice


# ── Merging ───────────────────────────────────────────────────────────


def _part(**groups) -> dict:
    return ap.normalise(groups)


def test_the_fold_keeps_one_rule_said_twice_in_its_most_specific_words():
    parts = [_part(rules=[{"text": "Answer customers in Croatian."},
                          {"text": "Refunds are allowed for orders up to 14 days old."}]),
             _part(rules=[{"text": "Refunds allowed for orders up to 14 days old, from the order date."},
                          {"text": "Refunds are allowed for orders up to 30 days old."}]),
             _part(rules=[{"text": "answer customers in croatian."}])]
    rules = [r["text"] for r in ap.fold(parts)["rules"]]
    assert rules == ["Answer customers in Croatian.",
                     "Refunds allowed for orders up to 14 days old, from the order date.",
                     "Refunds are allowed for orders up to 30 days old."]


def test_the_fold_joins_skills_by_name_and_every_documents_summaries():
    parts = [_part(intents=[{"name": "Refund", "when": "money back", "steps": ["Check."], "examples": ["a"]}],
                   knowledge=[{"doc": "h.md", "title": "Handbook", "summary": "Quotes."}]),
             _part(intents=[{"name": "refund", "when": "money back", "steps": ["Check.", "Pay."], "examples": ["b"]}],
                   knowledge=[{"doc": "h.md", "title": "", "summary": "Close."}])]
    folded = ap.fold(parts)
    assert [(i["steps"], i["examples"]) for i in folded["intents"]] == [(["Check.", "Pay."], ["a", "b"])]
    assert folded["knowledge"] == [{"doc": "h.md", "title": "Handbook", "summaries": ["Quotes.", "Close."],
                                    "summary": "Quotes. Close."}]


async def test_many_parts_take_one_merge_call(monkeypatch):
    calls: list[dict] = []
    merged = {"rules": [{"text": "Merged."}], "knowledge": [{"doc": "h.md", "title": "Handbook", "summary": "All."}]}

    async def fake_call(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        calls.append({"system": system, "user": user})
        return json.dumps(merged), 50

    monkeypatch.setattr(ap, "call_model", fake_call)
    reader = {"provider": "ollama", "model": "m", "base_url": "http://127.0.0.1:1", "input_ctx": 40_000}
    parts = [_part(rules=[{"text": f"Rule number {n} of the handbook."}], templates=[{"name": "Reply", "body": "Hi"}],
                   knowledge=[{"doc": "h.md", "title": "Handbook", "summary": f"Part {n}."}]) for n in range(12)]
    job = ap.ReadJob(id="j", owner="o")
    got = await ap.merge_parts(reader, parts, "", job)
    assert len(calls) == 1 and "Merging an agent setup" in calls[0]["system"]
    sent = json.loads(calls[0]["user"].split("\n", 1)[1])
    assert len(sent["rules"]) == 12 and "templates" not in sent and sent["knowledge"][0]["summary"].startswith("Part 0.")
    assert [r["text"] for r in got["rules"]] == ["Merged."]
    assert [t["name"] for t in got["templates"]] == ["Reply"]           # the fold's, kept
    calls.clear()
    small = {**reader, "input_ctx": 0, "provider": "anthropic"}
    monkeypatch.setattr(ap, "_merge_payload", lambda folded, room: None)
    got = await ap.merge_parts(small, parts, "", job)
    assert calls == [] and len(got["rules"]) == 12                     # too big for a call: the fold stands
