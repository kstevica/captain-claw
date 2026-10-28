"""The datastore's own xlsx reading (``DatastoreManager._parse_xlsx``, behind
``import_xlsx`` and the datastore tool's ``sheet``): every sheet by name,
whether the workbook points at its sheets from ``xl/`` (Excel) or from the
package root (openpyxl), with shared, inline or rich-text strings, and
cells that leave out their reference. A sheet asked for by name is that
sheet or an error, never another one."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from captain_claw.datastore import DatastoreManager

_MAIN_NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _write_xlsx(path: Path, sheets: dict[str, str], absolute: bool, shared: list[str] = (),
                missing: str = "") -> Path:
    """A workbook of the given ``<sheetData>`` bodies, its relationships
    pointing at them from ``xl/`` or (``absolute``) the package root; the
    ``missing`` sheet's part isn't in it."""
    names = list(sheets)
    workbook = (f'<?xml version="1.0"?><workbook {_MAIN_NS} xmlns:r="{_REL_NS}"><sheets>'
                + "".join(f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1))
                + "</sheets></workbook>")
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="{_REL_NS}/worksheet" '
                      f'Target="{"/xl/" if absolute else ""}worksheets/sheet{i}.xml"/>' for i in range(1, len(names) + 1))
            + "</Relationships>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        if shared:
            archive.writestr("xl/sharedStrings.xml", f'<?xml version="1.0"?><sst {_MAIN_NS}>'
                             + "".join(f"<si><t>{t}</t></si>" for t in shared) + "</sst>")
        for i, name in enumerate(names, 1):
            if name == missing:
                continue
            archive.writestr(f"xl/worksheets/sheet{i}.xml",
                             f'<?xml version="1.0"?><worksheet {_MAIN_NS}><sheetData>{sheets[name]}</sheetData></worksheet>')
    return path


def _inline(text: str) -> str:
    return f'<c t="inlineStr"><is><t>{text}</t></is></c>'


_PLANS_SHARED = ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                 '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>12.5</v></c></row>')
_ADDONS_INLINE = (f'<row>{_inline("Code")}{_inline("Name")}</row>'
                  '<row><c t="inlineStr"><is><r><t>B-</t></r><r><t>2</t></r></is></c>'
                  f'{_inline("Backup")}</row>')


@pytest.mark.parametrize("absolute", [False, True], ids=["excel", "package-root"])
def test_every_sheet_reads_by_name(tmp_path, absolute):
    path = _write_xlsx(tmp_path / "prices.xlsx", {"Plans": _PLANS_SHARED, "Add-ons": _ADDONS_INLINE},
                       absolute, shared=["SKU", "Price", "A-1"])
    assert DatastoreManager._parse_xlsx(path) == (["SKU", "Price"], [["A-1", "12.5"]])
    assert DatastoreManager._parse_xlsx(path, "add-ons") == (["Code", "Name"], [["B-2", "Backup"]])
    with pytest.raises(ValueError, match="Sheet not found"):
        DatastoreManager._parse_xlsx(path, "Nope")


def test_a_named_sheet_that_cant_be_found_in_the_package_is_an_error(tmp_path):
    path = _write_xlsx(tmp_path / "broken.xlsx", {"Plans": _PLANS_SHARED, "Add-ons": _ADDONS_INLINE},
                       False, shared=["SKU", "Price", "A-1"], missing="Add-ons")
    with pytest.raises(ValueError, match="Sheet not readable: Add-ons"):
        DatastoreManager._parse_xlsx(path, "Add-ons")


async def test_import_xlsx_takes_a_sheet_of_a_workbook_openpyxl_wrote(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Plans"
    wb.active.append(["SKU", "Price"])
    wb.active.append(["00123", 12.5])
    extra = wb.create_sheet("Add-ons")
    extra.append(["Code", "Name"])
    extra.append(["B-2", "Backup"])
    path = tmp_path / "prices.xlsx"
    wb.save(path)

    dm = DatastoreManager(db_path=tmp_path / "store.db")
    try:
        assert (await dm.import_xlsx(path, "addons", "Add-ons"))["rows_imported"] == 1
        assert (await dm.query("addons", columns=["code", "name"]))["rows"] == [["B-2", "Backup"]]
        await dm.import_xlsx(path, "plans")
        assert (await dm.query("plans", columns=["price"]))["rows"] == [[12.5]]
    finally:
        await dm.close()
