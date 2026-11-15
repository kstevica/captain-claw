"""httpx logs every request URL at INFO, and Flight Deck prints INFO: the
Telegram bot token (in the URL path) and ``?token=`` query values are blanked
before the line is written. The token here is made up."""

from __future__ import annotations

import logging

from captain_claw.flight_deck.context_packs import _RedactHttpxTokenFilter

_FAKE_TOKEN = "1234567890:AAFakeTokenValue_for-tests-0123456789"


def _filtered(msg: str, *args) -> str:
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, msg, args, None)
    assert _RedactHttpxTokenFilter().filter(record)
    return record.getMessage()


def test_the_bot_token_in_a_telegram_url_is_blanked():
    line = _filtered('HTTP Request: %s %s "%s"', "GET",
                     f"https://api.telegram.org/bot{_FAKE_TOKEN}/getUpdates?timeout=25", "HTTP/1.1 200 OK")
    assert "AAFakeTokenValue" not in line
    assert "api.telegram.org/bot1234567890:…/getUpdates?timeout=25" in line


def test_file_downloads_and_query_tokens_are_blanked_too():
    line = _filtered(f"HTTP Request: GET https://api.telegram.org/file/bot{_FAKE_TOKEN}/photos/a.jpg")
    assert "AAFakeTokenValue" not in line
    line = _filtered("HTTP Request: GET http://localhost:24080/api/capabilities?token=s3cret&x=1")
    assert "s3cret" not in line and "x=1" in line


def test_other_lines_are_left_alone():
    line = "HTTP Request: GET https://example.com/robots.txt"
    assert _filtered(line) == line
