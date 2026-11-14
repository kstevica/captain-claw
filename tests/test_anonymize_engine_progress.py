"""The anonymizer's code pass on big files: it says how far it is (round, pass,
n of N pieces of text), it can be stopped from outside its worker thread, and
the caller's terms (names the trusted model found) cost the same whether there
are ten or ten thousand. Made-up names."""

from __future__ import annotations

import re
import threading
import time

import pytest

from captain_claw.anonymize import engine as E


def _old_extra_spans(text, extra):
    """The finder as it was: one regex per term."""
    out = []
    for term, kind in extra:
        term = term.strip()
        if len(term) < 2:
            continue
        for m in re.finditer(rf"(?<![\w@.\-])(?i:{re.escape(term)})(?![\w@\-]|\.\w)", text):
            out.append((m.start(), m.end(), kind))
    return sorted(out)


@pytest.mark.parametrize("text", [
    "Ana Kovač i ANA KOVAČ te ana kovač",            # any case
    "pišite na ana.kovac@primjer.hr ili ana@x.hr",    # never a piece of an email
    "marko.k i Marko K. i Marko-Polo i Marko_x",     # dotted, hyphenated, underscore
    "Sokol d.o.o. i Sokol d.o.o.x i Sokol",          # a term with dots in it
    "Ana Kovač d.o.o. — Kovač d.o.o.",               # overlapping terms
    "(Ana) [Kovač] 2Ana Ana2",
])
def test_the_indexed_finder_finds_what_the_regex_did(text):
    extra = (("Ana Kovač", "PERSON"), ("Marko", "PERSON"), ("Sokol d.o.o.", "ORG"), ("Kovač d.o.o.", "ORG"),
             ("Ana", "PERSON"), ("ana", "PERSON"), ("x", "PERSON"))
    new = sorted((s.start, s.end, s.kind) for s in E._extra_finder(extra)(text))
    assert new == sorted(set(_old_extra_spans(text, extra)))


def test_many_names_cost_little():
    names = tuple((f"Ime{i} Prezime{i}", "PERSON") for i in range(2000))
    cells = [E.Segment(f"Plaća za Ime{i % 2000} Prezime{i % 2000}, iznos {i}") for i in range(20000)]
    t = time.time()
    result = E.run(cells, E.VaultState(), E.Options(extra=names))
    assert time.time() - t < 20  # was minutes: one regex per name per cell, past Python's 512-pattern cache
    assert all("⟦PERSON_" in o for o in result.outputs)


def test_the_run_says_how_far_it_is():
    events = []
    token = E.PROGRESS.set(events.append)
    try:
        E.run([E.Segment(f"Sastanak s Ana Kovač, stavka {i}.") for i in range(30)], E.VaultState(),
              E.Options(extra=(("Ana Kovač", "PERSON"),)))
    finally:
        E.PROGRESS.reset(token)
    passes = [(e["round"], e["pass"]) for e in events]
    assert (1, "looking") in passes and (1, "matching") in passes and (1, "replacing") in passes
    looking = [e for e in events if e["pass"] == "looking" and e["round"] == 1]
    assert looking[0]["done"] == 0 and looking[0]["of"] == 30
    replacing = [e for e in events if e["pass"] == "replacing" and e["round"] == 1]
    assert replacing[-1]["done"] == replacing[-1]["of"] == 30
    assert all(set(e) == {"round", "pass", "done", "of"} for e in events)  # counts only, never text


def test_a_run_stops_when_asked():
    stop = threading.Event()
    seen = []

    def hook(e):
        seen.append(e)
        if e["pass"] == "looking" and e["done"] == 0:
            stop.set()  # as a stopped room would, from another thread

    p, c = E.PROGRESS.set(hook), E.CANCEL.set(stop)
    try:
        with pytest.raises(E.RunCancelled):
            E.run([E.Segment(f"Stavka {i}") for i in range(100)], E.VaultState())
    finally:
        E.PROGRESS.reset(p)
        E.CANCEL.reset(c)
    assert len(seen) == 1
