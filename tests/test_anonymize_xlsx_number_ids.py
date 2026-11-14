"""Identifiers kept as numbers in a workbook (an OIB, a JMBG, a phone number
that lost its "+") get tokens; amounts, counts and codes stay numbers Claude
can add up. Made-up values (valid check digits are computed here)."""

from __future__ import annotations

import io

import openpyxl
import pytest

from captain_claw.anonymize import detectors, engine, office


def oib(first10: str) -> str:
    a = 10
    for ch in first10:
        a = (a + int(ch)) % 10 or 10
        a = (a * 2) % 11
    return first10 + str((11 - a) % 10)


def jmbg(first12: str) -> str:
    a = [int(c) for c in first12]
    s = sum(w * (a[i] + a[i + 6]) for i, w in enumerate((7, 6, 5, 4, 3, 2)))
    m = 11 - s % 11
    return first12 + str(0 if m > 9 else m)


@pytest.mark.parametrize("header,kind", [
    ("OIB", "OIB"), ("OIB zaposlenika", "OIB"), ("JMBG", "JMBG"), ("Telefon", "PHONE"),
    ("Kontakt telefon", "PHONE"), ("Broj mobitela", "PHONE"), ("Tel.", "PHONE"), ("IBAN račun", "IBAN"),
    ("Hotel", ""), ("Iznos", ""), ("Šifra", "CODE"), ("Broj računa", "IBAN"),
])
def test_headers_that_name_an_identifier(header, kind):
    assert detectors.hint_for(header) == kind


def _book() -> bytes:
    assert detectors.oib_ok(oib("1234567890")) and detectors.jmbg_ok(jmbg("150398533001"))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Popis zaposlenika 2025"])  # a title above the header row
    ws.append(["Ime i prezime", "OIB", "JMBG", "Kontakt telefon", "Iznos", "Šifra", "Napomena", "Sati"])
    ws.append(["Ana Kovač", int(oib("1234567890")), int(jmbg("150398533001")), 385911234567, 1500000,
               1234567, 385981112223, 168])
    ws.append(["Ivo Babić", float(oib("2345678901")), int(jmbg("120397933004")), 911234567, 125000.5,
               7654321, "bez napomene", 160])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_identifiers_in_number_cells_get_tokens_and_amounts_stay_numbers():
    prep = office.prepare(_book(), "xlsx")
    state = engine.VaultState()
    result = engine.run(prep.segments, state, engine.Options(one_word_names=False))
    out = openpyxl.load_workbook(io.BytesIO(prep.rebuild(result.edits))).active
    row3 = [c.value for c in out[3]]
    row4 = [c.value for c in out[4]]
    # Names (header in row 2, below a title), OIB, JMBG, phones — tokens.
    assert [str(v)[:2] for v in row3[:4]] == ["⟦P", "⟦O", "⟦J", "⟦P"]
    assert str(row4[1]).startswith("⟦OIB_") and str(row4[3]).startswith("⟦PHONE_")  # "….0" and no "+385"
    assert str(row3[6]).startswith("⟦PHONE_")  # 385… with no header saying so
    # Amounts, codes and hours: still numbers, as they were.
    assert row3[4:6] == [1500000, 1234567] and row4[4:6] == [125000.5, 7654321]
    assert row3[7] == 168 and isinstance(row3[4], int) and isinstance(row4[4], float)
    assert {e.kind for e in state.entities} >= {"OIB", "JMBG", "PHONE", "PERSON"}


def test_a_row_of_data_is_not_a_header():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Stavka", "Iznos"])
    ws.append(["Telefon", 4500000])  # a cost line called "Telefon", not a header
    ws.append(["Struja", 1250000])
    buf = io.BytesIO()
    wb.save(buf)
    prep = office.prepare(buf.getvalue(), "xlsx")
    result = engine.run(prep.segments, engine.VaultState(), engine.Options(one_word_names=False))
    out = openpyxl.load_workbook(io.BytesIO(prep.rebuild(result.edits))).active
    assert [c.value for c in out[3]] == ["Struja", 1250000]
