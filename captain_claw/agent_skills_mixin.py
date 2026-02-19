"""Skills snapshot, prompt wiring, and manual skill invocation helpers for Agent."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message
from captain_claw.logging import get_logger
from captain_claw.skills import (
    SkillCatalogEntry,
    SkillCommandSpec,
    SkillSnapshot,
    apply_skill_env_overrides,
    build_workspace_skill_snapshot,
    compute_skills_snapshot_version,
    find_skill_command,
    install_skill_dependencies as install_skill_dependencies_for_entry,
    install_skill_from_github_url,
    load_skill_catalog_entries,
    rank_skill_catalog_entries,
    snapshot_from_dict,
    snapshot_to_dict,
)

log = get_logger(__name__)


class AgentSkillsMixin:
    """Skill loading and invocation helpers."""

    @staticmethod
    def _extract_skill_search_ids(payload: Any) -> list[int]:
        """Extract selected candidate IDs from LLM payload."""
        if not isinstance(payload, dict):
            return []
        raw_candidates = [
            payload.get("selected_ids"),
            payload.get("top_ids"),
            payload.get("ids"),
            payload.get("selected"),
            payload.get("results"),
        ]
        values: list[Any] = []
        for candidate in raw_candidates:
            if isinstance(candidate, list):
                values.extend(candidate)

        selected_ids: list[int] = []
        for item in values:
            value: Any = item
            if isinstance(item, dict):
                value = (
                    item.get("id")
                    or item.get("candidate_id")
                    or item.get("candidateId")
                    or item.get("index")
                )
            try:
                parsed = int(str(value).strip())
            except Exception:
                continue
            if parsed <= 0:
                continue
            if parsed not in selected_ids:
                selected_ids.append(parsed)
        return selected_ids

    async def _rank_skill_search_entries_with_llm(
        self,
        query: str,
        entries: list[SkillCatalogEntry],
        limit: int,
    ) -> list[SkillCatalogEntry]:
        """Use LLM to rank catalog entries for query relevance."""
        if not entries or limit <= 0:
            return []
        provider = getattr(self, "provider", None)
        if provider is None:
            return []
        candidate_payload = [
            {
                "id": idx + 1,
                "name": entry.name,
                "description": entry.description,
                "url": entry.url,
            }
            for idx, entry in enumerate(entries)
        ]
        system_prompt = (
            "You rank OpenClaw skill candidates by semantic relevance to the user query. "
            "You may handle typos in the query and map them to intended terms. "
            "Only choose candidates that are genuinely relevant. "
            "Return strict JSON only."
        )
        user_prompt = (
            f"Task: select up to {limit} best skills for the query.\n"
            f"Query: {query.strip()}\n\n"
            "Return JSON exactly as:\n"
            '{"selected_ids":[1,2,3]}\n\n'
            "Rules:\n"
            "- Do not select irrelevant candidates.\n"
            "- Prefer direct topic alignment over generic utility skills.\n"
            "- If fewer than requested are relevant, return fewer IDs.\n\n"
            "Candidates:\n"
            f"{json.dumps(candidate_payload, ensure_ascii=True)}"
        )
        try:
            response = await provider.complete(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                tools=None,
                temperature=0.1,
                max_tokens=900,
            )
        except Exception as exc:
            log.warning("Skill search LLM ranking failed", error=str(exc))
            return []

        payload = None
        try:
            parsed = json.loads(str(response.content or "").strip())
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = None
        if payload is None:
            extract_fn = getattr(self, "_extract_json_object", None)
            if callable(extract_fn):
                try:
                    payload = extract_fn(str(response.content or ""))
                except Exception:
                    payload = None
        selected_ids = self._extract_skill_search_ids(payload)
        if not selected_ids:
            return []
        selected: list[SkillCatalogEntry] = []
        for selected_id in selected_ids:
            idx = selected_id - 1
            if idx < 0 or idx >= len(entries):
                continue
            entry = entries[idx]
            if entry in selected:
                continue
            selected.append(entry)
            if len(selected) >= limit:
                break
        return selected

    def _resolve_skills_snapshot(self, force_refresh: bool = False) -> SkillSnapshot:
        """Resolve current workspace skills snapshot with session-aware caching."""
        cfg = get_config()
        watch_enabled = bool(getattr(cfg.skills.load, "watch", True))
        cached = getattr(self, "_skills_snapshot_cache", None)
        version = ""
        if watch_enabled:
            debounce_ms = max(0, int(getattr(cfg.skills.load, "watch_debounce_ms", 250) or 0))
            last_checked = getattr(self, "_skills_snapshot_last_checked_at", None)
            if (
                not force_refresh
                and debounce_ms > 0
                and isinstance(cached, SkillSnapshot)
                and isinstance(last_checked, datetime)
            ):
                elapsed_ms = (datetime.now(UTC) - last_checked).total_seconds() * 1000.0
                if elapsed_ms < float(debounce_ms):
                    return cached
            version = compute_skills_snapshot_version(self.workspace_base_path, cfg)
            self._skills_snapshot_last_checked_at = datetime.now(UTC)
            if (
                not force_refresh
                and isinstance(cached, SkillSnapshot)
                and cached.version
                and cached.version == version
            ):
                return cached
        elif not force_refresh and isinstance(cached, SkillSnapshot):
            return cached

        session_payload = None
        if self.session and isinstance(self.session.metadata, dict):
            skills_meta = self.session.metadata.get("skills")
            if isinstance(skills_meta, dict):
                snapshot_payload = skills_meta.get("snapshot")
                if isinstance(snapshot_payload, dict):
                    session_payload = snapshot_payload
        if not force_refresh and isinstance(session_payload, dict):
            restored = snapshot_from_dict(session_payload)
            if restored and (not watch_enabled or restored.version == version):
                self._skills_snapshot_cache = restored
                return restored

        if not version:
            version = compute_skills_snapshot_version(self.workspace_base_path, cfg)
        snapshot = build_workspace_skill_snapshot(
            self.workspace_base_path,
            cfg,
            version=version,
        )
        self._skills_snapshot_cache = snapshot
        if self.session and isinstance(self.session.metadata, dict):
            skills_meta = self.session.metadata.setdefault("skills", {})
            if isinstance(skills_meta, dict):
                skills_meta["snapshot"] = snapshot_to_dict(snapshot)
                skills_meta["updated_at"] = datetime.now(UTC).isoformat()
        return snapshot

    def _build_skills_system_prompt_section(self) -> str:
        """Build OpenClaw-style skills section appended to system prompt."""
        snapshot = self._resolve_skills_snapshot(force_refresh=False)
        prompt_block = (snapshot.prompt or "").strip()
        if not prompt_block:
            return ""
        lines = [
            "## Skills (mandatory)",
            "Before replying: scan <available_skills> <description> entries.",
            '- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.',
            "- If multiple could apply: choose the most specific one, then read/follow it.",
            "- If none clearly apply: do not read any SKILL.md.",
            "Constraints: never read more than one skill up front; only read after selecting.",
            prompt_block,
        ]
        return "\n".join(lines).strip()

    def list_user_invocable_skills(self) -> list[SkillCommandSpec]:
        """List currently available user-invocable skill command specs."""
        snapshot = self._resolve_skills_snapshot(force_refresh=False)
        return list(snapshot.commands)

    def rewrite_prompt_for_skill(self, command: SkillCommandSpec, args: str | None = None) -> str:
        """Rewrite request so the model explicitly uses a selected skill."""
        raw_args = (args or "").strip()
        parts = [f'Use the "{command.skill_name}" skill for this request.']
        if raw_args:
            parts.append(f"User input:\n{raw_args}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _resolve_script_dispatch_command(command: SkillCommandSpec, raw_args: str) -> tuple[str, str]:
        dispatch = command.dispatch
        if dispatch is None or dispatch.kind != "script":
            raise ValueError("Command is not configured for script dispatch.")
        script_path_raw = str(dispatch.script_path or "").strip()
        if not script_path_raw:
            raise ValueError("Skill script path is missing.")
        script_path = Path(script_path_raw).expanduser().resolve()
        if not script_path.exists() or not script_path.is_file():
            raise FileNotFoundError(f"Skill script not found: {script_path}")

        interpreter = str(dispatch.script_interpreter or "").strip()
        if interpreter:
            try:
                command_parts = shlex.split(interpreter)
            except Exception:
                command_parts = [interpreter]
            command_parts.append(str(script_path))
        else:
            suffix = script_path.suffix.lower()
            if suffix == ".py":
                command_parts = [sys.executable, str(script_path)]
            elif suffix == ".sh":
                command_parts = ["bash", str(script_path)]
            else:
                command_parts = [str(script_path)]

        full_parts = list(command_parts)
        if raw_args:
            try:
                full_parts.extend(shlex.split(raw_args))
            except ValueError as exc:
                raise ValueError(f"Invalid skill command arguments: {str(exc)}") from exc
        shell_command = " ".join(shlex.quote(part) for part in full_parts)
        return shell_command, str(script_path)

    async def invoke_skill_command(
        self,
        name: str,
        args: str | None = None,
        turn_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Resolve and run manual `/skill` invocation."""
        snapshot = self._resolve_skills_snapshot(force_refresh=True)
        command = find_skill_command(snapshot.commands, name)
        if not command:
            return {
                "ok": False,
                "mode": "error",
                "error": f"Skill not found or not user-invocable: {name}",
            }

        raw_args = (args or "").strip()
        if command.dispatch and command.dispatch.kind == "tool":
            tool_name = command.dispatch.tool_name
            tool_args = {
                "command": raw_args,
                "commandName": command.name,
                "skillName": command.skill_name,
            }
            try:
                result = await self._execute_tool_with_guard(
                    name=tool_name,
                    arguments=tool_args,
                    interaction_label=f"skill_dispatch:{command.name}",
                    turn_usage=turn_usage,
                )
                output = result.content if result.success else f"Error: {result.error}"
                success = bool(result.success)
            except Exception as exc:
                output = f"Error: {str(exc)}"
                success = False

            self._add_session_message(
                role="tool",
                content=output,
                tool_name=tool_name,
                tool_arguments=tool_args,
            )
            self._emit_tool_output(tool_name, tool_args, output)
            if self.session:
                await self.session_manager.save_session(self.session)
            return {
                "ok": success,
                "mode": "dispatch",
                "text": output,
                "command": command,
            }
        if command.dispatch and command.dispatch.kind == "script":
            shell_command = ""
            try:
                shell_command, script_path = self._resolve_script_dispatch_command(command, raw_args)
                result = await self._execute_tool_with_guard(
                    name="shell",
                    arguments={"command": shell_command},
                    interaction_label=f"skill_dispatch:{command.name}",
                    turn_usage=turn_usage,
                )
                output = result.content if result.success else f"Error: {result.error}"
                success = bool(result.success)
            except Exception as exc:
                output = f"Error: {str(exc)}"
                success = False
                script_path = str(command.dispatch.script_path or "").strip()

            tool_args = {
                "command": shell_command,
                "commandName": command.name,
                "skillName": command.skill_name,
                "scriptPath": script_path,
            }
            self._add_session_message(
                role="tool",
                content=output,
                tool_name="shell",
                tool_arguments=tool_args,
            )
            self._emit_tool_output("shell", tool_args, output)
            if self.session:
                await self.session_manager.save_session(self.session)
            return {
                "ok": success,
                "mode": "dispatch",
                "text": output,
                "command": command,
            }

        rewritten = self.rewrite_prompt_for_skill(command, args=raw_args)
        return {
            "ok": True,
            "mode": "rewrite",
            "prompt": rewritten,
            "command": command,
        }

    async def install_skill_from_github(self, github_url: str) -> dict[str, Any]:
        """Install a skill from a GitHub URL into managed skills directory."""
        raw_url = str(github_url or "").strip()
        if not raw_url:
            return {
                "ok": False,
                "mode": "error",
                "error": "Usage: /skill install <github-url>",
            }
        try:
            installed = install_skill_from_github_url(raw_url, get_config())
            snapshot = self._resolve_skills_snapshot(force_refresh=True)
            aliases = [
                command.name for command in snapshot.commands if command.skill_name == installed.skill_name
            ]
        except Exception as exc:
            return {
                "ok": False,
                "mode": "error",
                "error": f"Skill install failed: {str(exc)}",
            }
        return {
            "ok": True,
            "mode": "install",
            "skill_name": installed.skill_name,
            "destination": installed.destination,
            "repo": installed.repo,
            "ref": installed.ref,
            "skill_path": installed.skill_path,
            "source_url": installed.source_url,
            "aliases": aliases,
        }

    async def install_skill_dependencies(
        self,
        skill_name: str,
        install_id: str | None = None,
    ) -> dict[str, Any]:
        """Install dependencies for a known skill from metadata.install definitions."""
        raw_name = str(skill_name or "").strip()
        raw_install_id = str(install_id or "").strip() or None
        if not raw_name:
            return {
                "ok": False,
                "mode": "error",
                "error": "Usage: /skill install <skill-name> [install-id]",
            }
        try:
            result = install_skill_dependencies_for_entry(
                skill_name=raw_name,
                workspace_dir=self.workspace_base_path,
                cfg=get_config(),
                install_id=raw_install_id,
            )
            if not result.ok:
                return {
                    "ok": False,
                    "mode": "error",
                    "error": result.message,
                    "skill_name": result.skill_name,
                    "install_id": result.install_id,
                    "kind": result.kind,
                    "command": result.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "code": result.code,
                }
            return {
                "ok": True,
                "mode": "install-deps",
                "message": result.message,
                "skill_name": result.skill_name,
                "install_id": result.install_id,
                "kind": result.kind,
                "command": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "code": result.code,
            }
        except Exception as exc:
            return {
                "ok": False,
                "mode": "error",
                "error": f"Skill dependency install failed: {str(exc)}",
            }

    async def search_skill_catalog(self, criteria: str) -> dict[str, Any]:
        """Search configured skill catalog and return top candidates."""
        query = re.sub(r"\s+", " ", str(criteria or "").strip())
        if not query:
            return {
                "ok": False,
                "mode": "error",
                "error": "Usage: /skill search <search criteria>",
            }
        cfg = get_config()
        source_url = str(cfg.skills.search_source_url or "").strip()
        if not source_url:
            return {
                "ok": False,
                "mode": "error",
                "error": "Skill search source URL is not configured (skills.search_source_url).",
            }
        limit = max(1, int(getattr(cfg.skills, "search_limit", 10) or 10))
        max_candidates = max(limit, int(getattr(cfg.skills, "search_max_candidates", 5000) or 5000))
        timeout_seconds = max(1, int(getattr(cfg.skills, "search_http_timeout_seconds", 20) or 20))
        try:
            display_source, entries = await asyncio.to_thread(
                load_skill_catalog_entries,
                source_url,
                timeout_seconds,
                max_candidates,
            )
        except Exception as exc:
            return {
                "ok": False,
                "mode": "error",
                "error": f"Skill search failed: {str(exc)}",
            }
        if not entries:
            return {
                "ok": False,
                "mode": "error",
                "error": "Skill search catalog returned no parseable skill entries.",
            }

        lexical_pool_size = max(limit * 20, 120)
        lexical_ranked = rank_skill_catalog_entries(query, entries, limit=lexical_pool_size)
        llm_pool: list[SkillCatalogEntry] = list(lexical_ranked)
        if len(llm_pool) < lexical_pool_size:
            for entry in entries:
                if entry in llm_pool:
                    continue
                llm_pool.append(entry)
                if len(llm_pool) >= lexical_pool_size:
                    break

        llm_ranked = await self._rank_skill_search_entries_with_llm(query, llm_pool, limit=limit)
        fallback_ranked = rank_skill_catalog_entries(query, entries, limit=limit * 3)
        ranked: list[SkillCatalogEntry] = []
        for entry in llm_ranked + fallback_ranked:
            if entry in ranked:
                continue
            ranked.append(entry)
            if len(ranked) >= limit:
                break

        return {
            "ok": True,
            "mode": "search",
            "query": query,
            "source": display_source,
            "limit": limit,
            "results": [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "url": entry.url,
                }
                for entry in ranked
            ],
        }

    def _apply_skill_env_overrides_for_run(self) -> Callable[[], None]:
        """Apply skill env/apiKey overrides and return restore callback."""
        try:
            snapshot = self._resolve_skills_snapshot(force_refresh=False)
        except Exception as exc:
            log.warning("Skills snapshot resolution failed; skipping env overrides", error=str(exc))
            return lambda: None
        try:
            return apply_skill_env_overrides(snapshot, get_config())
        except Exception as exc:
            log.warning("Skill env override application failed; skipping", error=str(exc))
            return lambda: None
