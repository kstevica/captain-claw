"""Write-as-you-go for ensemble workers' scale loops.

A reply-sunk scale-loop item lives only in the worker's context, which dies
wholesale when the dispatch budget runs out before the final answer (the Vatra
deep-researcher failure: 11 annexes extracted, nothing on disk, timeout → zero
output). `_persist_scale_item_for_team` makes each processed item durable in the
run's shared VFS folder — for Basna/Vatra workers only; a standalone agent's
reply loop keeps today's behaviour.
"""

from __future__ import annotations

from captain_claw.agent_scale_loop_mixin import AgentScaleLoopMixin


class _Result:
    def __init__(self, success: bool = True, error: str = ""):
        self.success = success
        self.error = error


class _StubAgent:
    """Duck-typed `self` for the unbound mixin method."""

    def __init__(self, write_ok: bool = True, write_raises: bool = False):
        self.writes: list[dict] = []
        self.emitted: list[dict] = []
        self._write_ok = write_ok
        self._write_raises = write_raises

    async def _execute_tool_with_guard(self, *, name, arguments, **kw):
        if self._write_raises:
            raise RuntimeError("disk on fire")
        self.writes.append({"name": name, **arguments})
        return _Result(self._write_ok, "" if self._write_ok else "denied")

    def _emit_tool_output(self, tool, meta, text):
        self.emitted.append({"tool": tool, **meta})


async def _persist(stub, monkeypatch, *, project="vatra-abc123", worker="CLAW_VATRA_WORKER",
                   item_num=3, label="Annex IV — Application Form", content="the extraction"):
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
    monkeypatch.delenv("CLAW_BASNA_WORKER", raising=False)
    monkeypatch.delenv("CLAW_VATRA_WORKER", raising=False)
    if project:
        monkeypatch.setenv("CLAW_VFS_PROJECT", project)
    if worker:
        monkeypatch.setenv(worker, "1")
    await AgentScaleLoopMixin._persist_scale_item_for_team(
        stub, item_num, label, content, {})


async def test_worker_item_lands_in_the_shared_extracts_folder(monkeypatch):
    stub = _StubAgent()
    await _persist(stub, monkeypatch)
    assert len(stub.writes) == 1
    w = stub.writes[0]
    assert w["name"] == "write"
    assert w["path"] == "vfs:vatra-abc123/extracts/03-annex-iv-application-form.md"
    assert w["append"] is False
    assert w["content"].startswith("# Annex IV — Application Form\n")
    assert "the extraction" in w["content"]
    # The persist is visible in the run log as a scale_micro_loop step.
    assert stub.emitted and stub.emitted[0]["step"] == "persist"


async def test_basna_worker_marker_also_qualifies(monkeypatch):
    stub = _StubAgent()
    await _persist(stub, monkeypatch, worker="CLAW_BASNA_WORKER")
    assert len(stub.writes) == 1


async def test_standalone_agent_keeps_todays_behaviour(monkeypatch):
    # A bound project WITHOUT a worker marker (standalone agent) → no write.
    stub = _StubAgent()
    await _persist(stub, monkeypatch, worker=None)
    assert stub.writes == []


async def test_no_shared_folder_or_empty_content_is_a_noop(monkeypatch):
    stub = _StubAgent()
    await _persist(stub, monkeypatch, project=None)
    assert stub.writes == []
    await _persist(stub, monkeypatch, content="   ")
    assert stub.writes == []


async def test_failed_or_raising_write_never_fails_the_item(monkeypatch):
    # Best-effort: neither a refused write nor an exception may propagate.
    await _persist(_StubAgent(write_ok=False), monkeypatch)
    await _persist(_StubAgent(write_raises=True), monkeypatch)
