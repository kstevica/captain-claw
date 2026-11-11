"""The name-recognition pack: its bare-name set, its label choice, and — where
a model is downloaded — how the default model does on the set."""

from __future__ import annotations

import pytest

from captain_claw.anonymize import detectors as d
from captain_claw.anonymize import name_eval
from captain_claw.flight_deck import anonymize_ner


def test_the_name_set_is_well_formed():
    cases = name_eval.load("name_cases.json")
    assert len({c["id"] for c in cases}) == len(cases) >= 80
    for c in cases:
        assert c["lang"] in ("hr", "en")
        for key in ("people", "orgs", "keep", "either"):
            for s in c[key]:
                assert name_eval.present(s, c["text"]), (c["id"], s)
    assert sum(len(c["keep"]) for c in cases) >= 100  # the traps are most of it


def test_each_model_takes_only_its_labels():
    def pipe(text):
        return [{"start": 0, "end": 5, "entity_group": "PER", "score": 0.9},
                {"start": 10, "end": 19, "entity_group": "ORG", "score": 0.9},
                {"start": 23, "end": 29, "entity_group": "LOC", "score": 0.9}]

    text = "Marko and Northwind in Zagreb."
    bertic = anonymize_ner._make_backend(pipe, "classla/bcms-bertic-ner")
    xlmr = anonymize_ner._make_backend(pipe, "Davlan/xlm-roberta-base-wikiann-ner")
    assert [f[2] for f in bertic(text)] == ["PER", "ORG"]
    assert [f[2] for f in xlmr(text)] == ["PER"]  # its company labels take days and holidays
    assert anonymize_ner.DEFAULT_MODEL == "classla/bcms-bertic-ner"
    assert set(anonymize_ner.MODEL_LABELS) == set(anonymize_ner.MODELS)


_READY = anonymize_ner.runtime_available() and anonymize_ner.downloaded(anonymize_ner.DEFAULT_MODEL)


@pytest.mark.skipif(not _READY, reason="needs transformers + torch and the default name model downloaded")
def test_the_default_model_on_the_name_set():
    model = anonymize_ner.DEFAULT_MODEL
    d.set_ner_backend(anonymize_ner._make_backend(anonymize_ner._load_pipeline(model), model))
    try:
        r = name_eval.score(name_eval.load("name_cases.json"))
        shared = name_eval.shared_broken(name_eval.load("cases.json"))
    finally:
        d.set_ner_backend(None)
    # Measured 2026-11-12: people 88/97, companies 9/9, traps kept 142/144.
    assert r["people_hit"] / r["people"] >= 0.85, r["missed"]
    assert r["orgs_hit"] / r["orgs"] >= 0.8, r["missed"]
    assert r["keep_ok"] / r["keep"] >= 0.97, r["broken"]
    assert r["verify_failed"] == [] and shared == []
