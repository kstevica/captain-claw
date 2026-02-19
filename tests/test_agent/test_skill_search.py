from pathlib import Path

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.skills import SkillCatalogEntry, parse_skill_catalog_entries


class SearchRankingProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content='{"selected_ids":[1,2]}')

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


class NonJsonSearchProvider(LLMProvider):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="not-json")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


def test_parse_skill_catalog_entries_extracts_markdown_bullets():
    markdown = """
- [Skill Alpha](https://example.com/alpha) - Summarize docs and release notes.
- [Skill Beta](skills/beta) - Search issue tracker and classify bugs.
- [Not a Skill](#section)
"""
    entries = parse_skill_catalog_entries(markdown, source_base_url="https://github.com/VoltAgent/awesome-openclaw-skills")

    assert len(entries) == 2
    assert entries[0].name == "Skill Alpha"
    assert entries[0].url == "https://example.com/alpha"
    assert "Summarize docs" in entries[0].description
    assert entries[1].name == "Skill Beta"
    assert entries[1].url == "https://github.com/VoltAgent/awesome-openclaw-skills/skills/beta"


@pytest.mark.asyncio
async def test_search_skill_catalog_uses_llm_ranked_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    cfg.skills.search_source_url = "https://github.com/VoltAgent/awesome-openclaw-skills"
    cfg.skills.search_limit = 2
    set_config(cfg)
    try:
        entries = [
            SkillCatalogEntry(name="Docs Writer", url="https://example.com/docs", description="Write docs"),
            SkillCatalogEntry(name="Bug Triage", url="https://example.com/bugs", description="Classify bugs"),
            SkillCatalogEntry(name="Code Refactor", url="https://example.com/refactor", description="Refactor code"),
        ]

        def _fake_loader(source_url: str, timeout_seconds: int = 20, max_candidates: int = 300):
            return source_url, entries

        monkeypatch.setattr("captain_claw.agent_skills_mixin.load_skill_catalog_entries", _fake_loader)
        agent = Agent(provider=SearchRankingProvider())

        result = await agent.search_skill_catalog("bug triage")

        assert result["ok"] is True
        assert result["mode"] == "search"
        assert [item["name"] for item in result["results"]] == ["Bug Triage", "Docs Writer"]
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_search_skill_catalog_falls_back_when_llm_response_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    cfg.skills.search_source_url = "https://github.com/VoltAgent/awesome-openclaw-skills"
    cfg.skills.search_limit = 2
    set_config(cfg)
    try:
        entries = [
            SkillCatalogEntry(name="Docs Writer", url="https://example.com/docs", description="Write docs"),
            SkillCatalogEntry(name="Bug Triage", url="https://example.com/bugs", description="Classify bugs"),
        ]

        def _fake_loader(source_url: str, timeout_seconds: int = 20, max_candidates: int = 300):
            return source_url, entries

        monkeypatch.setattr("captain_claw.agent_skills_mixin.load_skill_catalog_entries", _fake_loader)
        agent = Agent(provider=NonJsonSearchProvider())

        result = await agent.search_skill_catalog("classify bugs")

        assert result["ok"] is True
        assert [item["name"] for item in result["results"]][:1] == ["Bug Triage"]
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_search_skill_catalog_includes_late_catalog_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.workspace.path = str(tmp_path)
    cfg.skills.search_source_url = "https://github.com/VoltAgent/awesome-openclaw-skills"
    cfg.skills.search_limit = 10
    set_config(cfg)
    try:
        entries = [
            SkillCatalogEntry(name=f"Skill {index}", url=f"https://example.com/{index}", description="misc")
            for index in range(700)
        ]
        entries[600] = SkillCatalogEntry(
            name="launch-strategy",
            url="https://example.com/launch-strategy",
            description="When the user wants to plan",
        )

        def _fake_loader(source_url: str, timeout_seconds: int = 20, max_candidates: int = 300):
            return source_url, entries[:max_candidates]

        monkeypatch.setattr("captain_claw.agent_skills_mixin.load_skill_catalog_entries", _fake_loader)
        agent = Agent(provider=NonJsonSearchProvider())

        result = await agent.search_skill_catalog("when the user wants to plan")

        assert result["ok"] is True
        assert any(item["name"] == "launch-strategy" for item in result["results"])
    finally:
        set_config(old_cfg)
