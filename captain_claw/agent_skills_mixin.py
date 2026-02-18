"""Skills snapshot, prompt wiring, and manual skill invocation helpers for Agent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.skills import (
    SkillCommandSpec,
    SkillSnapshot,
    apply_skill_env_overrides,
    build_workspace_skill_snapshot,
    compute_skills_snapshot_version,
    find_skill_command,
    snapshot_from_dict,
    snapshot_to_dict,
)

log = get_logger(__name__)


class AgentSkillsMixin:
    """Skill loading and invocation helpers."""

    def _resolve_skills_snapshot(self, force_refresh: bool = False) -> SkillSnapshot:
        """Resolve current workspace skills snapshot with session-aware caching."""
        cfg = get_config()
        version = compute_skills_snapshot_version(self.workspace_base_path, cfg)
        cached = getattr(self, "_skills_snapshot_cache", None)
        if (
            not force_refresh
            and isinstance(cached, SkillSnapshot)
            and cached.version
            and cached.version == version
        ):
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
            if restored and restored.version == version:
                self._skills_snapshot_cache = restored
                return restored

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

        rewritten = self.rewrite_prompt_for_skill(command, args=raw_args)
        return {
            "ok": True,
            "mode": "rewrite",
            "prompt": rewritten,
            "command": command,
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

