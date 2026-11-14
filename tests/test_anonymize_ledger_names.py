"""What a real register, payroll and ledger did to names, addresses and
codes — and what they get now. Made-up names and values throughout.

* a name in lower case had no words to tell it by, so every one became the
  same person;
* a register's name column in capitals, and a payroll's "Ime radnika", weren't
  read as names by code;
* a carer's register name with a middle name ("MARIBEL SANTOS REYES") and the
  payroll's "Reyes Maribel" got two tokens, as did "ANAMARIJA" / "Ana Marija";
* a ledger glues names to the word before ("PRISTOJBAMARIBEL …"), and those
  stayed in clear;
* one address written seven ways got seven tokens;
* invoice numbers read as phone numbers, a tax office as a company, a
  document code as a company.
"""

from __future__ import annotations

import io

import openpyxl

from captain_claw.anonymize import detectors, engine, office


def _model(state: engine.VaultState, text: str, names: tuple[str, ...] = ()) -> str:
    """As a data room runs a file with a trusted model: its names come in as terms."""
    opts = engine.Options(extra=tuple((n, "PERSON") for n in names), one_word_names=False)
    return engine.anonymize_text(text, state, opts)[0]


# ── names in lower case ─────────────────────────────────────────────


def test_names_in_lower_case_are_people_of_their_own():
    state = engine.VaultState()
    out = _model(state, "ivan horvat | petra kovačić | Ivan Horvat", ("ivan horvat", "petra kovačić"))
    assert out == "⟦PERSON_1⟧ | ⟦PERSON_2⟧ | ⟦PERSON_1⟧"
    assert sorted(e.original for e in state.entities) == ["Ivan Horvat", "petra kovačić"]  # the capitalized form kept


def test_a_lower_case_first_name_is_part_of_the_name():
    state = engine.VaultState()
    assert _model(state, "marta ŽUPAN", ("marta ŽUPAN",)) == "⟦PERSON_1⟧"
    assert _model(state, "MARTA ŽUPAN", ("MARTA ŽUPAN",)) == "⟦PERSON_1⟧"
    assert _model(state, "Ana Župan", ("Ana Župan",)) == "⟦PERSON_2⟧"


def test_name_words_leave_out_particles_titles_and_stop_words():
    assert engine._name_words("Ivan Horvat, dipl. oec.") == ["Ivan", "Horvat"]
    assert engine._name_words("Ludwig van Beethoven") == ["Ludwig", "Beethoven"]
    assert engine.normalize("PERSON", "ivan horvat") == "ivan horvat"


# ── name columns ────────────────────────────────────────────────────


def test_name_columns_are_read_in_any_case():
    assert detectors.hint_for("Ime radnika") == "PERSON"
    assert detectors.hint_for("Prezime i ime") == "PERSON"
    assert detectors.hint_for("Ime tvrtke") == ""
    for value, want in [("HORVAT IVAN", "HORVAT IVAN"), ("ivan horvat", "ivan horvat"),
                        ("Horvat, Ivan", "Horvat, Ivan"), ("KOVAČ ANA novi tekući račun", "KOVAČ ANA"),
                        ("I. Horvat", "I. Horvat"), ("ANA KOS NA 4 SATA", "ANA KOS"), ("4 SATA", None),
                        ("Ana Kos (bolovanje)", None)]:
        sp = detectors.hint_span(value, "PERSON")
        assert (value[sp.start:sp.end] if sp else None) == want, value


def _book(rows: list[list]) -> bytes:
    wb = openpyxl.Workbook()
    for r in rows:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _cells(blob: bytes) -> list[list]:
    wb = openpyxl.load_workbook(io.BytesIO(blob))
    return [[c.value for c in row] for row in wb.active.iter_rows()]


def test_a_register_in_capitals_and_a_payroll_are_one_vault_by_code_alone():
    state = engine.VaultState()
    register = _book([["rbr", "IME I PREZIME", "RADNO MJESTO", "ADRESA"],
                      [1, "HORVAT IVAN", "kuhar", "KOVAČKA C.12"],
                      [2, "MARIBEL SANTOS REYES", "njegovatelj", "KOVAČKA CESTA 12"]])
    payroll = _book([["Rbr", "Ime radnika", "Radno mjesto", "Bruto"],
                     [1, "Horvat Ivan", "kuhar", 1200],
                     [2, "Reyes Maribel", "njegovatelj", 1100]])
    outs = []
    for blob in (register, payroll):
        prep = office.prepare(blob, "xlsx")
        result = engine.run(prep.segments, state, engine.Options())
        outs.append(_cells(prep.rebuild(result.edits)))
    reg, pay = outs
    assert reg[1][1] == "⟦PERSON_1⟧" and reg[2][1] == "⟦PERSON_2⟧"
    assert pay[1][1].startswith("⟦PERSON_1") and pay[2][1].startswith("⟦PERSON_2")
    assert reg[1][3] == "⟦ADDRESS_1⟧" and reg[2][3] == "⟦ADDRESS_1:2⟧"  # one place, two spellings
    assert pay[1][2] == "kuhar" and pay[1][3] == 1200


# ── one person, written otherwise ───────────────────────────────────


def test_a_middle_name_more_or_less_is_the_one_person_it_fits():
    state = engine.VaultState()
    assert _model(state, "MARIBEL SANTOS REYES", ("MARIBEL SANTOS REYES",)) == "⟦PERSON_1⟧"
    assert _model(state, "Reyes Maribel", ("Reyes Maribel",)) == "⟦PERSON_1:2⟧"
    assert _model(state, "Plaća Reyes Maribel bruto") == "Plaća ⟦PERSON_1:2⟧ bruto"  # found again by code
    back = engine.restore("⟦PERSON_1⟧ / ⟦PERSON_1:2⟧", state)[0]
    assert back == "MARIBEL SANTOS REYES / Reyes Maribel"


def test_a_middle_name_that_fits_two_people_is_nobody_s():
    state = engine.VaultState()
    _model(state, "Marko Ivan Kos", ("Marko Ivan Kos",))
    _model(state, "Marko Petar Kos", ("Marko Petar Kos",))
    assert _model(state, "Marko Kos", ("Marko Kos",)) == "⟦PERSON_3⟧"


def test_a_two_part_first_name_joined_or_split_is_one_person():
    state = engine.VaultState()
    assert _model(state, "ANAMARIJA PERIĆ", ("ANAMARIJA PERIĆ",)) == "⟦PERSON_1⟧"
    assert _model(state, "Perić Ana Marija", ("Perić Ana Marija",)) == "⟦PERSON_1:2⟧"
    assert _model(state, "Ana-Marija Perić", ("Ana-Marija Perić",)) == "⟦PERSON_1:3⟧"


# ── names glued to the word before ──────────────────────────────────


def test_names_glued_to_the_word_before_are_found():
    state = engine.VaultState()
    _model(state, "MARIBEL SANTOS REYES", ("MARIBEL SANTOS REYES",))
    _model(state, "Reyes Maribel", ("Reyes Maribel",))
    assert _model(state, "UPRAVNA PRISTOJBAMARIBEL SANTOS REYES") == "UPRAVNA PRISTOJBA⟦PERSON_1⟧"
    assert _model(state, "izrada dozvole zaMARIBEL SANTOS REYES") == "izrada dozvole za⟦PERSON_1⟧"
    out = _model(state, "tečaj jezikareyes maribel santos")
    assert out.startswith("tečaj jezika⟦PERSON_1:") and "santos" not in out and "reyes" not in out
    assert engine.restore(out, state)[0] == "tečaj jezikareyes maribel santos"
    assert _model(state, "isplata maribel reyes").startswith("isplata ⟦PERSON_1:")  # lower case, other order


def test_a_glued_ending_alone_is_not_a_name():
    state = engine.VaultState()
    _model(state, "Ana Kos", ("Ana Kos",))
    assert _model(state, "banana kosilica") == "banana kosilica"
    assert _model(state, "Ana Kosić") == "Ana Kosić"


# ── addresses ───────────────────────────────────────────────────────


def test_one_address_written_many_ways_is_one_token():
    cells = ["KOVAČKA C.12", "KOVAČKA C. 12", "KOVAČKA CESTA 12", "KOVAĆKA CESTA 12",
             "Kovačka cesta 12, Gornje Selo", "KOV.CESTA 12", "KOV CESTA 12"]
    state = engine.VaultState()
    r = engine.run([engine.Segment(c, "ADDRESS") for c in cells], state, engine.Options())
    assert {o.split(":")[0].rstrip("⟧") for o in r.outputs} == {"⟦ADDRESS_1"}
    assert engine.restore(" | ".join(r.outputs), state)[0] == " | ".join(cells)


def test_other_places_stay_apart():
    cells = ["Kovačka cesta 12", "Kovačka cesta 14", "Kovačka ulica 12", "Ilica 5, Zagreb", "Ilica 5, Split"]
    state = engine.VaultState()
    r = engine.run([engine.Segment(c, "ADDRESS") for c in cells], state, engine.Options())
    assert r.outputs == [f"⟦ADDRESS_{i}⟧" for i in range(1, 6)]


# ── what isn't anyone's ─────────────────────────────────────────────


def test_invoice_numbers_and_periods_are_not_phone_numbers():
    def phones(t: str) -> list[str]:
        return [t[s.start:s.end] for s in detectors.detect(t, ner=False) if s.kind == "PHONE"]

    assert phones("Račun 001234/56/78") == [] and phones("br. 0012/345/678") == []
    assert phones("Plaća 03-09 2025 bonus") == []
    assert "0049 89 1234567" in phones("Tel 0049 89 1234567")  # a real "00" number still is one
    assert phones("+385 91 234 5678") == ["+385 91 234 5678"]
    assert phones("zovi 091 234 5678") == ["091 234 5678"]


def test_public_bodies_stay_as_they_are():
    assert detectors.public_body("HZZO") and detectors.public_body("HZZO-u")
    assert detectors.public_body("Porezna uprava, Područni ured Zagreb")
    assert not detectors.public_body("Kos d.o.o.")
    state = engine.VaultState()
    out = engine.anonymize_text("Uplata HZZO-u i Porezna uprava za Sokol d.o.o.", state, engine.Options(
        extra=(("HZZO", "ORG"), ("Porezna uprava", "ORG"), ("Sokol d.o.o.", "ORG")), one_word_names=False))[0]
    assert out == "Uplata HZZO-u i Porezna uprava za ⟦ORG_1⟧"


def test_a_code_alone_in_a_column_of_codes_is_kept():
    assert detectors.hint_for("Oznaka knjiženja") == "CODE"
    ledger = _book([["Oznaka isprave", "Šifra konta", "Opis"],
                    ["UFAK", "4100", "Plaća za Ivana Horvata"],
                    ["UFAK", "4100", "UFAK ispravak"]])
    state = engine.VaultState([engine.Entity("⟦PERSON_1⟧", "PERSON", "ivan horvat", "Ivan Horvat")])
    prep = office.prepare(ledger, "xlsx")
    result = engine.run(prep.segments, state, engine.Options(
        extra=(("UFAK", "ORG"), ("Ivan Horvat", "PERSON")), one_word_names=False))
    rows = _cells(prep.rebuild(result.edits))
    assert rows[1][0] == "UFAK" and rows[2][0] == "UFAK"
    assert rows[1][2] == "Plaća za ⟦PERSON_1:2⟧"
    assert rows[2][2] == "⟦ORG_1⟧ ispravak"  # outside the code column the model's call stands


def test_one_stray_code_word_in_a_ledger_line_is_no_header():
    """A data row saying "kod" mustn't turn its column into codes (names there
    would stay): a code column comes only from a row that names two columns."""
    ledger = _book([["Datum", "Opis", "Iznos"],
                    ["01.02.", "plaćeno kod dobavljača", "x"],
                    ["02.02.", "Isplata Ivan Horvat", "y"]])
    prep = office.prepare(ledger, "xlsx")
    assert "CODE" not in {s.hint for s in prep.segments}
