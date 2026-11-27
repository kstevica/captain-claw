"""A real Agent turn: a request to publish reaches the model with the
safety_guard preflight findings, before its first step, and without the
secret. Separate from test_safety_guard.py because an Agent leaves aiosqlite
worker threads behind, which keep pytest from exiting (run it on its own)."""

from __future__ import annotations

import subprocess

import pytest

FAKE_ANTHROPIC = "sk-ant-api03-" + "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4z" * 2


def _git_repo(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


@pytest.mark.asyncio
async def test_a_publish_request_reaches_the_model_with_the_findings_and_no_secret(monkeypatch, tmp_path):
    from captain_claw.agent import Agent
    from captain_claw.llm import LLMProvider, LLMResponse
    from captain_claw.session import Session

    seen: list[str] = []

    class _Provider(LLMProvider):
        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            seen.append("\n".join(str(getattr(m, "content", "") or "") for m in messages))
            return LLMResponse(content="I found a key in agent.py and did not push.")

        async def complete_streaming(self, *a, **kw):
            if False:
                yield ""

        def count_tokens(self, text: str) -> int:
            return len(text)

    class _NoSave:
        async def save_session(self, session):
            return None

    repo = _git_repo(tmp_path / "mm_agents")
    (repo / "agent.py").write_text(f'KEY = "{FAKE_ANTHROPIC}"\n')
    agent = Agent(provider=_Provider())
    agent._initialized = True
    agent.session = Session(id="sg1", name="default")
    agent.session_manager = _NoSave()
    agent.workspace_base_path = tmp_path
    agent.tools = __import__("captain_claw.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry()
    await agent.complete("Please push mm_agents to the main branch of the remote repository.")
    assert seen, "the model was never called"
    assert "Safety check before publishing" in seen[0] and "mm_agents/agent.py:1" in seen[0]
    assert FAKE_ANTHROPIC not in seen[0]
