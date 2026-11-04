"""Local OCR for scanned PDFs and pictures (``ocr_pdf`` / ``ocr_image``).

No language model reads anything here: a local engine (Tesseract or RapidOCR)
does, by fixed rules, and every result says which engine, version, language
and settings produced it. See :mod:`captain_claw.ocr.document` for the
pipeline and :mod:`captain_claw.ocr.engines` for the engines.
"""

from captain_claw.ocr.engines import (  # noqa: F401
    INSTALL_HELP,
    EngineError,
    get_engine,
    pick_engine,
    reset_engines,
)
