"""Numbers in a workbook's text read as Excel shows them: dates as dates,
times as times, no float noise. Made-up data."""

from __future__ import annotations

import datetime as dt

import openpyxl
import pytest

from captain_claw.tools import document_extract as dx


@pytest.mark.parametrize("code,kind", [
    ("dd.mm.yyyy", "date"), ('[$-41A]d. mmmm yyyy."', "date"), ("mmm-yy", "date"), ("d/m/yy h:mm", "datetime"),
    ("hh:mm", "time"), ("mm:ss", "time"), ("[h]:mm:ss", "time"), ("[h]:mm", "time"),
    ("#,##0.00", ""), ('#,##0.00\\ "kn"', ""), ('0.00 "dana"', ""), ("General", ""), ("@", ""),
    ("[Red]#,##0;[Blue]-#,##0", ""),
])
def test_format_kinds(code, kind):
    assert dx._xlsx_format_kind(code) == kind


def test_numbers_as_excel_shows_them():
    assert dx._xlsx_number("1234.5599999999999") == "1234.56"
    assert dx._xlsx_number("0.30000000000000004") == "0.3"
    assert dx._xlsx_number("45000") == "45000"
    assert dx._xlsx_number("12345678903") == "12345678903"  # an ID number stored as a number
    assert dx._xlsx_number("1.2345678901234567E+20") == "1.2345678901234567E+20"
    assert dx._xlsx_number("45000", "date") == "2023-03-15"
    assert dx._xlsx_number("45000.5", "datetime") == "2023-03-15 12:00"
    assert dx._xlsx_number("0.3541666666666667", "time") == "8:30"
    assert dx._xlsx_number("1.5", "time") == "36:00"  # an elapsed time
    assert dx._xlsx_number("0", "date", date1904=True) == "1904-01-01"
    assert dx._xlsx_number("n/a", "date") == "n/a"


def test_a_workbook_reads_its_dates_and_amounts(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Ime", "Iznos", "Dolazak", "Vrijeme"])
    ws.append(["Ana", 1234.56, dt.date(2023, 3, 15), dt.time(8, 30)])
    ws["C2"].number_format = "dd.mm.yyyy"
    ws["D2"].number_format = "hh:mm"
    ws.append(["Ivo", 0.1 + 0.2, 45000, None])  # a plain number that only looks like a date
    path = tmp_path / "plaće.xlsx"
    wb.save(path)
    text = dx._extract_xlsx_markdown(path, 100)
    assert "| Ana | 1234.56 | 2023-03-15 | 8:30 |" in text
    assert "| Ivo | 0.3 | 45000 |  |" in text
