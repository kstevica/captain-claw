"""The anonymizer's file path with pictures and scans: pictures="strip" for
Office files, scanned PDF pages and picture files read with local OCR.
The OCR tests run only where an engine is installed. Made-up data."""

from __future__ import annotations

import io
import zipfile

import docx
import pytest
from docx.shared import Inches
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont

from captain_claw.anonymize import scans
from captain_claw.flight_deck import (
    agent_secret,
    anonymize_routes,
    anonymize_vault,
    auth,
    fd_crypto,
)
from captain_claw.flight_deck.db import FlightDeckDB

ALICE = "user-alice"
EMAIL = "ana.kovac@primjer-firma.hr"
ocr_only = pytest.mark.skipif(not scans.available(), reason="no local OCR engine installed")


@pytest.fixture()
async def store(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    s = anonymize_vault.VaultStore(tmp_path / "anonymize.db")
    anonymize_vault.set_store(s)
    try:
        yield s
    finally:
        anonymize_vault.set_store(None)
        await s.close()
        fd_crypto.reset_for_tests()
        auth.set_auth_db(prev_db)
        await d.close()


def _picture(text: str, size=(1200, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((60, size[1] // 2 - 30), text, fill="black", font=ImageFont.load_default(size=44))
    return img


def _bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _letter_with_a_scanned_card() -> bytes:
    d = docx.Document()
    d.add_paragraph("Vizitka u prilogu.")
    d.add_picture(io.BytesIO(_bytes(_picture(f"Kontakt: {EMAIL}"), "PNG")), width=Inches(4))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


async def _run(name: str, blob: bytes, **data):
    return await anonymize_routes.anonymize_blob(
        ALICE, name, blob, {"vault": "Room test", "lane": "deterministic", **data}, actor=ALICE, source="test")


async def test_pictures_mode_is_checked(store):
    with pytest.raises(HTTPException) as exc:
        await _run("a.docx", _letter_with_a_scanned_card(), pictures="blur")
    assert exc.value.status_code == 422


async def test_strip_replaces_the_picture(store, monkeypatch):
    monkeypatch.setattr(scans, "picture_reader", lambda: None)  # no engine: replaced, not read
    out, info = await _run("a.docx", _letter_with_a_scanned_card(), pictures="strip")
    z = zipfile.ZipFile(io.BytesIO(out))
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    original = zipfile.ZipFile(io.BytesIO(_letter_with_a_scanned_card()))
    assert media and all(z.read(n) != original.read(n) for n in media)
    joined = " ".join(info["warnings"])
    assert "picture(s) replaced" in joined and "No OCR engine" in joined


@ocr_only
async def test_strip_reads_the_picture_into_anonymized_alt_text(store):
    out, info = await _run("a.docx", _letter_with_a_scanned_card(), pictures="strip")
    xml = zipfile.ZipFile(io.BytesIO(out)).read("word/document.xml").decode()
    assert EMAIL not in xml and "Text in the picture: Kontakt: ⟦EMAIL_" in xml
    assert "the text read from 1 of them" in " ".join(info["warnings"])


@ocr_only
async def test_a_scanned_pdf_is_read_with_ocr(store):
    page = Image.new("RGB", (1240, 1754), "white")
    ImageDraw.Draw(page).text((120, 300), f"Kontakt: {EMAIL}", fill="black", font=ImageFont.load_default(size=44))
    out, info = await _run("scan.pdf", _bytes(page, "PDF", resolution=150))
    text = out.decode()
    assert info["filename"].endswith(".md") and EMAIL not in text and "⟦EMAIL_" in text
    assert "read with OCR" in " ".join(info["warnings"])


@ocr_only
async def test_a_picture_file_is_read_with_ocr(store):
    out, info = await _run("card.png", _bytes(_picture(f"Kontakt: {EMAIL}"), "PNG"))
    assert info["filename"].endswith(".md")
    assert EMAIL not in out.decode() and "⟦EMAIL_" in out.decode()


async def test_pictures_are_known_file_kinds():
    assert ".png" in anonymize_routes.FILE_KINDS and ".tif" in anonymize_routes.FILE_KINDS
