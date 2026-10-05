"""Shared-agent speaker handshake — the agent side of contract part 0 §5.

Flight Deck opens a member's upstream socket with ONE extra header,
``X-FD-Speaker: v1.<payload_b64>.<sig_b64>``, signed with a key derived from
the agent's own web auth token. The agent verifies it in constant time,
bounds its lifetime, refuses replays and answers with ``speaker_ack``.
Both FD and agent tests assert the same vector.
"""

from __future__ import annotations

import base64
import json

import pytest

from captain_claw import speaker
from captain_claw.speaker import (
    Principal,
    SpeakerAuthError,
    sign_assertion,
    speaker_ack_for,
    speaker_signing_key,
    verify_assertion,
)

WEB_AUTH = "test-web-auth"
PAYLOAD = {
    "v": 1, "sub": "u-member", "name": "Ana", "owner": "u-owner", "owner_name": "Olga",
    "ref": "process:helper:0123456789abcdef", "lane": "A", "conn": "c0ffee00c0ffee00",
    "iat": 1760000000, "exp": 1760000060, "nonce": "00112233445566aa",
}
KEY_HEX = "43ef71203472b43d8204b4b75e823784c90669104c1ce40e80adfc07682850c2"
HEADER = (
    "v1.eyJjb25uIjoiYzBmZmVlMDBjMGZmZWUwMCIsImV4cCI6MTc2MDAwMDA2MCwiaWF0IjoxNzYwMDAwMDAwLCJsYW5lIjoi"
    "QSIsIm5hbWUiOiJBbmEiLCJub25jZSI6IjAwMTEyMjMzNDQ1NTY2YWEiLCJvd25lciI6InUtb3duZXIiLCJvd25lcl9uYW1l"
    "IjoiT2xnYSIsInJlZiI6InByb2Nlc3M6aGVscGVyOjAxMjM0NTY3ODlhYmNkZWYiLCJzdWIiOiJ1LW1lbWJlciIsInYiOjF9"
    ".ylLbFYmtrikkdnr6BjLD0BEIO2RxWIW8A5X7zpnivHc"
)
ACK = "941895124a6e13f5"
NOW = 1760000010


@pytest.fixture(autouse=True)
def fresh_nonces():
    speaker._NONCES.clear()
    yield
    speaker._NONCES.clear()


def _payload(**over) -> dict:
    p = dict(PAYLOAD)
    p.update(over)
    return p


# ── the shared test vector ───────────────────────────────────────────


def test_vector_key():
    assert speaker_signing_key(WEB_AUTH).hex() == KEY_HEX


def test_vector_header_is_reproduced_exactly():
    assert sign_assertion(PAYLOAD, WEB_AUTH) == HEADER


def test_vector_verifies():
    p = verify_assertion(HEADER, WEB_AUTH, now=NOW)
    assert p == Principal(
        speaker_id="u-member", display_name="Ana", owner_name="Olga",
        lane="A", agent_ref="process:helper:0123456789abcdef",
    )


def test_vector_ack():
    assert speaker_ack_for(HEADER) == ACK


# ── forged / malformed ───────────────────────────────────────────────


def _flip(text: str, idx: int) -> str:
    ch = text[idx]
    return text[:idx] + ("B" if ch != "B" else "C") + text[idx + 1:]


def test_tampered_payload_is_refused():
    head, payload_b64, sig = HEADER.split(".")
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"{head}.{_flip(payload_b64, 10)}.{sig}", WEB_AUTH, now=NOW)


def test_payload_swapped_under_a_valid_signature_is_refused():
    other = sign_assertion(_payload(sub="u-admin", nonce="ffffffffffffffff"), WEB_AUTH)
    _, other_payload, _ = other.split(".")
    head, _, sig = HEADER.split(".")
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"{head}.{other_payload}.{sig}", WEB_AUTH, now=NOW)


def test_tampered_signature_is_refused():
    head, payload_b64, sig = HEADER.split(".")
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"{head}.{payload_b64}.{_flip(sig, 3)}", WEB_AUTH, now=NOW)


def test_wrong_web_auth_is_refused():
    with pytest.raises(SpeakerAuthError):
        verify_assertion(HEADER, "another-agents-token", now=NOW)


@pytest.mark.parametrize("web_auth", ["", None])
def test_empty_web_auth_is_refused(web_auth):
    # An agent without a token can't verify anything — and must not accept
    # an assertion signed with the empty key.
    forged = sign_assertion(PAYLOAD, "")
    with pytest.raises(SpeakerAuthError):
        verify_assertion(forged, web_auth, now=NOW)


@pytest.mark.parametrize("bad", [
    "", "v1", "v1.abc", "v1..sig", "v2." + HEADER[3:], HEADER + ".extra",
    "v1.e30.éé", "x" * 9000,
])
def test_malformed_headers_are_refused(bad):
    with pytest.raises(SpeakerAuthError):
        verify_assertion(bad, WEB_AUTH, now=NOW)


def test_unsigned_json_garbage_is_refused():
    body = base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode()
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"v1.{body}.AAAA", WEB_AUTH, now=NOW)


@pytest.mark.parametrize("field,value", [
    ("v", 2), ("v", True), ("v", "1"),
    ("lane", "D"), ("lane", "a"), ("lane", ""),
    ("sub", ""), ("sub", 7), ("nonce", ""),
    ("iat", "1760000000"), ("exp", None),
])
def test_bad_fields_are_refused_even_when_signed(field, value):
    header = sign_assertion(_payload(**{field: value}), WEB_AUTH)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(header, WEB_AUTH, now=NOW)


def test_signed_non_object_payload_is_refused():
    body = json.dumps([1, 2, 3], separators=(",", ":")).encode()
    payload_b64 = speaker._b64url(body)
    import hashlib
    import hmac

    sig = hmac.new(speaker_signing_key(WEB_AUTH), f"v1.{payload_b64}".encode(), hashlib.sha256)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"v1.{payload_b64}.{speaker._b64url(sig.digest())}", WEB_AUTH, now=NOW)


# ── lifetime ─────────────────────────────────────────────────────────


def test_expired_assertion_is_refused():
    with pytest.raises(SpeakerAuthError):
        verify_assertion(HEADER, WEB_AUTH, now=PAYLOAD["exp"] + 1)


def test_assertion_is_valid_up_to_exp():
    verify_assertion(HEADER, WEB_AUTH, now=PAYLOAD["exp"])


def test_future_iat_beyond_skew_is_refused():
    with pytest.raises(SpeakerAuthError):
        verify_assertion(HEADER, WEB_AUTH, now=PAYLOAD["iat"] - 31)


def test_iat_skew_of_30s_is_tolerated():
    verify_assertion(HEADER, WEB_AUTH, now=PAYLOAD["iat"] - 30)


def test_lifetime_over_120s_is_refused():
    header = sign_assertion(_payload(exp=PAYLOAD["iat"] + 121), WEB_AUTH)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(header, WEB_AUTH, now=NOW)


def test_lifetime_of_120s_is_accepted():
    header = sign_assertion(_payload(exp=PAYLOAD["iat"] + 120), WEB_AUTH)
    verify_assertion(header, WEB_AUTH, now=NOW)


def test_exp_before_iat_is_refused():
    header = sign_assertion(_payload(exp=PAYLOAD["iat"] - 1), WEB_AUTH)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(header, WEB_AUTH, now=PAYLOAD["iat"] - 5)


# ── replay ───────────────────────────────────────────────────────────


def test_replayed_nonce_is_refused():
    verify_assertion(HEADER, WEB_AUTH, now=NOW)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(HEADER, WEB_AUTH, now=NOW + 1)


def test_same_nonce_in_a_freshly_signed_assertion_is_refused():
    verify_assertion(HEADER, WEB_AUTH, now=NOW)
    again = sign_assertion(_payload(conn="1111111111111111"), WEB_AUTH)
    with pytest.raises(SpeakerAuthError):
        verify_assertion(again, WEB_AUTH, now=NOW + 1)


def test_distinct_nonces_both_pass():
    verify_assertion(HEADER, WEB_AUTH, now=NOW)
    verify_assertion(sign_assertion(_payload(nonce="00112233445566ab"), WEB_AUTH), WEB_AUTH, now=NOW)


def test_failed_signature_does_not_burn_the_nonce():
    head, payload_b64, sig = HEADER.split(".")
    with pytest.raises(SpeakerAuthError):
        verify_assertion(f"{head}.{payload_b64}.{_flip(sig, 3)}", WEB_AUTH, now=NOW)
    verify_assertion(HEADER, WEB_AUTH, now=NOW)       # the genuine one still works


def test_nonce_cache_is_purged_after_180s():
    verify_assertion(HEADER, WEB_AUTH, now=NOW)
    assert PAYLOAD["nonce"] in speaker._NONCES
    later = NOW + 181
    fresh = sign_assertion(_payload(nonce="aaaaaaaaaaaaaaaa", iat=later, exp=later + 60), WEB_AUTH)
    verify_assertion(fresh, WEB_AUTH, now=later)
    assert PAYLOAD["nonce"] not in speaker._NONCES
    # …and the old header is still dead: it expired long before.
    with pytest.raises(SpeakerAuthError):
        verify_assertion(HEADER, WEB_AUTH, now=later)


def test_errors_never_carry_the_header():
    head, payload_b64, sig = HEADER.split(".")
    with pytest.raises(SpeakerAuthError) as info:
        verify_assertion(f"{head}.{payload_b64}.{_flip(sig, 3)}", WEB_AUTH, now=NOW)
    assert payload_b64 not in str(info.value) and sig not in str(info.value)


def test_display_name_falls_back_when_empty():
    p = verify_assertion(sign_assertion(_payload(name="  "), WEB_AUTH), WEB_AUTH, now=NOW)
    assert p.display_name == "Member"
