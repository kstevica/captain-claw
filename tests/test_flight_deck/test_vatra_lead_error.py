"""The Lead-decompose failure message must never be blank — a bare TimeoutError
stringifies to '' (the old mysterious 'Vatra Lead failed:' with no detail)."""

from __future__ import annotations

import asyncio

from captain_claw.flight_deck import vatra_routes as v


def test_timeout_gives_an_actionable_message():
    msg = v._lead_error_msg(asyncio.TimeoutError())
    assert msg and "timed out" in msg
    assert str(v._DECOMPOSE_TIMEOUT) in msg  # names the actual limit


def test_error_with_text_passes_through():
    assert v._lead_error_msg(ValueError("invalid JSON from the Lead")) == "invalid JSON from the Lead"


def test_error_without_text_falls_back_to_type_name():
    assert v._lead_error_msg(RuntimeError()) == "RuntimeError"


def test_decompose_timeout_is_generous_for_local_models():
    assert v._DECOMPOSE_TIMEOUT >= 300
