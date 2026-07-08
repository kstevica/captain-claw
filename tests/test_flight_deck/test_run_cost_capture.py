"""The run-cost accumulator: the contextvar recorder + the provider wrapper that
close the undercount (auxiliary _provider_call spend + Deep-mode rollouts)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from captain_claw.flight_deck import basna_routes as b


@dataclass
class _FakeResp:
    model: str = ""
    usage: dict = field(default_factory=dict)


class _FakeProvider:
    """Records the args it was called with; returns a fixed usage."""
    def __init__(self, resp):
        self.resp = resp
        self.calls = 0
        self.temperature = 0.5  # an arbitrary passthrough attribute

    async def complete(self, *a, **kw):
        self.calls += 1
        return self.resp


def _reset(sid):
    b._run_sid.set(sid)
    b._RUN_USAGE[sid] = []


def test_recorder_appends_only_token_fields_under_active_sid():
    _reset("s1")
    b._record_run_usage("claude-opus-4-8",
                        {"prompt_tokens": 1000, "completion_tokens": 200, "model": "junk"})
    assert len(b._RUN_USAGE["s1"]) == 1
    entry = b._RUN_USAGE["s1"][0]
    assert entry["model"] == "claude-opus-4-8"
    assert "model" not in entry["usage"]        # non-token keys stripped
    assert entry["usage"]["prompt_tokens"] == 1000
    b._RUN_USAGE.pop("s1", None)


def test_recorder_noop_without_active_sid():
    b._run_sid.set("")            # no run bound
    before = dict(b._RUN_USAGE)
    b._record_run_usage("claude-opus-4-8", {"prompt_tokens": 500})
    assert b._RUN_USAGE == before  # nothing recorded


def test_recorder_skips_empty_usage():
    _reset("s2")
    b._record_run_usage("m", {"prompt_tokens": 0, "completion_tokens": 0})
    assert b._RUN_USAGE["s2"] == []
    b._RUN_USAGE.pop("s2", None)


def test_wrapped_provider_records_and_is_transparent():
    _reset("s3")
    inner = _FakeProvider(_FakeResp(model="claude-haiku-4-5",
                                    usage={"prompt_tokens": 300, "completion_tokens": 50}))
    wrapped = b._UsageRecordingProvider(inner)
    # Transparent passthrough for non-complete attributes.
    assert wrapped.temperature == 0.5
    resp = asyncio.run(wrapped.complete("x"))
    assert resp is inner.resp and inner.calls == 1
    assert len(b._RUN_USAGE["s3"]) == 1
    assert b._RUN_USAGE["s3"][0]["usage"]["prompt_tokens"] == 300
    b._RUN_USAGE.pop("s3", None)


def test_provider_call_wraps_only_inside_a_run():
    creds = {"provider": "anthropic", "model": "claude-haiku-4-5"}
    b._run_sid.set("")  # outside a run
    prov_out, _ = b._provider_call(creds, temperature=0.0, default_max=100, cap=200)
    assert not isinstance(prov_out, b._UsageRecordingProvider)
    b._run_sid.set("s4")  # inside a run
    prov_in, _ = b._provider_call(creds, temperature=0.0, default_max=100, cap=200)
    assert isinstance(prov_in, b._UsageRecordingProvider)
    b._run_sid.set("")
