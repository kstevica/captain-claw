"""A known company written with a word put in or punctuated otherwise — a logo
read by OCR, a letterhead — is that company too: its words in order, at most
one short word between two of them. The text's own spelling comes back
exactly. The data room's leak check sees the same variants. Made-up names."""

from __future__ import annotations

import pytest

from captain_claw.anonymize import engine as E
from captain_claw.flight_deck import rooms_job

OPTS = E.Options(extra=(("Plava Rijeka Capital d.o.o.", "ORG"), ("Kvadrix", "ORG")))


def _state() -> E.VaultState:
    state = E.VaultState()
    E.anonymize_text("Ugovor potpisuju Plava Rijeka Capital d.o.o. i Kvadrix.", state, OPTS)
    return state


@pytest.mark.parametrize("text", [
    "Logo — Text in the picture: PLAVA RIJEKA PRC CAPITAL",   # a logo read by OCR, a word put in
    "Plava Rijeka – Capital",                                 # punctuated otherwise
    "PLAVA RIJEKA\nCAPITAL",                                  # over two lines (a scan)
    "PLAVA RIJEKA PRC CAPITAL FUND I",                        # words after it stay
])
def test_a_known_company_with_a_word_put_in_is_found(text):
    state = _state()
    out, _ = E.anonymize_text(text, state)
    assert "plava" not in out.casefold() and "rijeka" not in out.casefold(), out
    assert "⟦ORG_1" in out
    assert E.restore(out, state)[0] == text  # the text's own spelling, exactly


def test_a_full_legal_name_with_a_word_put_in_is_hidden_either_way():
    """With its legal form it reads as a company's full name — perhaps another
    one (a group company): it may get a number of its own. Nothing leaks."""
    state = _state()
    text = "Plava Rijeka Group Capital d.o.o. is the manager."
    out, _ = E.anonymize_text(text, state)
    assert "plava" not in out.casefold() and out.endswith("⟧ is the manager.")
    assert E.restore(out, state)[0] == text


def test_the_variant_keeps_the_companys_number_and_its_own_spelling():
    state = _state()
    out, _ = E.anonymize_text("Text in the picture: PLAVA RIJEKA PRC CAPITAL", state)
    assert out == "Text in the picture: ⟦ORG_1:2⟧"
    assert E.restore("⟦ORG_1⟧", state)[0] == "Plava Rijeka Capital d.o.o."  # the name itself is unchanged


@pytest.mark.parametrize("text", [
    "Plava Rijeka nova velika Capital",          # two words in one gap
    "Plava Rijeka Investment Capital",           # a long word in between: another name
    "plava rijeka prc capital",                  # written in lower case where the name isn't
    "Plava Rijeka is a river; capital is money",  # far apart
])
def test_other_text_is_left_alone(text):
    state = _state()
    assert E.anonymize_text(text, state)[0] == text


def test_a_one_word_company_isnt_matched_loosely():
    state = _state()
    assert E.anonymize_text("KVADRIX PRC", state)[0] == "⟦ORG_2⟧ PRC"
    assert E.loose_name_words("Kvadrix d.o.o.") == ()


def test_the_leak_check_sees_the_same_variants():
    state = E.VaultState([E.Entity("⟦ORG_1⟧", "ORG", "plava rijeka capital", "Plava Rijeka Capital d.o.o.")])
    needles = rooms_job.needles(state)
    assert rooms_job.leaks("Text in the picture: PLAVA RIJEKA PRC CAPITAL", needles) == ["⟦ORG_1⟧"]
    assert rooms_job.leaks("Plava Rijeka – Capital", needles) == ["⟦ORG_1⟧"]
    assert rooms_job.leaks("Plava Rijeka Investment Capital", needles) == []
    assert rooms_job.leaks("Text in the picture: ⟦ORG_1:2⟧", needles) == []
