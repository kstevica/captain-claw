"""One person, one token number, however each file writes the name: a
register's "ŠIMUNIĆ DARKO", a payroll's "Darko Šimunić", a ledger's "Šimunić,
Darko", no diacritics, an initial. Made-up names."""

from __future__ import annotations

from captain_claw.anonymize import engine

E = engine.Entity


def _run(state: engine.VaultState, text: str, model_names: tuple[str, ...] = ()) -> str:
    """As a data room runs a file with a trusted model: its names come in as
    terms, and code takes no name alone."""
    opts = engine.Options(extra=tuple((n, "PERSON") for n in model_names), one_word_names=False)
    return engine.anonymize_text(text, state, opts)[0]


def test_a_name_either_way_round_is_one_person():
    state = engine.VaultState()
    assert _run(state, "| 1 | ŠIMUNIĆ DARKO | voditelj |", ("ŠIMUNIĆ DARKO",)) == "| 1 | ⟦PERSON_1⟧ | voditelj |"
    outs = [
        _run(state, "Plaća za Darko Šimunić: 1.200 EUR", ("Darko Šimunić",)),
        _run(state, "Šimunić Darko, bruto 1.200", ("Šimunić Darko",)),
        _run(state, "SIMUNIC DARKO 1.200", ("SIMUNIC DARKO",)),
        _run(state, "Akontacija Šimunić, Darko 300", ("Šimunić, Darko",)),
        _run(state, "Odobrio D. Šimunić", ("D. Šimunić",)),
        _run(state, "Isplata: Darko Šimunić"),  # code alone
    ]
    assert all("⟦PERSON_1" in o for o in outs), outs
    assert [e.token for e in state.entities] == ["⟦PERSON_1⟧"]
    back = engine.restore(" | ".join(outs), state)[0]
    # Each spelling comes back as written (one that differs only in case, in the vault's own form).
    assert back == ("Plaća za Darko Šimunić: 1.200 EUR | ŠIMUNIĆ DARKO, bruto 1.200 | SIMUNIC DARKO 1.200 | "
                    "Akontacija Šimunić, Darko 300 | Odobrio D. Šimunić | Isplata: Darko Šimunić")


def test_a_vault_that_split_a_person_heals_on_the_next_run():
    """Before this, "Darko Šimunić" after "ŠIMUNIĆ DARKO" became a person of its
    own; the first token wins when a file runs again."""
    state = engine.VaultState([E("⟦PERSON_1⟧", "PERSON", "šimunić darko", "ŠIMUNIĆ DARKO", count=1),
                               E("⟦PERSON_2⟧", "PERSON", "darko šimunić", "Darko Šimunić", count=1)])
    assert _run(state, "Plaća za Darko Šimunić", ("Darko Šimunić",)) == "Plaća za ⟦PERSON_1:2⟧"
    assert _run(state, "Plaća za Darko Šimunić") == "Plaća za ⟦PERSON_1:2⟧"
    assert engine.restore("⟦PERSON_1:2⟧", state)[0] == "Darko Šimunić"


def test_other_people_stay_apart():
    state = engine.VaultState()
    _run(state, "ŠIMUNIĆ DARKO", ("ŠIMUNIĆ DARKO",))
    assert _run(state, "Darko Horvat", ("Darko Horvat",)) == "⟦PERSON_2⟧"
    assert _run(state, "Dino Šimunić", ("Dino Šimunić",)) == "⟦PERSON_3⟧"
    # "D. Šimunić" fits Darko and Dino: nobody's in particular.
    assert _run(state, "D. Šimunić", ("D. Šimunić",)) == "⟦PERSON_4⟧"
    # A middle name more is the one Darko Šimunić known — the same person.
    assert _run(state, "Darko Ivan Šimunić", ("Darko Ivan Šimunić",)) == "⟦PERSON_1:2⟧"
