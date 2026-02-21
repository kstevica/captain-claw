"""Shared mutable runtime state for the interactive session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from captain_claw.agent import Agent
from captain_claw.cli import TerminalUI
from captain_claw.execution_queue import CommandQueueManager, FollowupQueueManager


@dataclass
class PlatformState:
    """Per-platform bridge + user management state."""

    name: str = ""
    bridge: Any = None  # TelegramBridge | SlackBridge | DiscordBridge | None
    poll_task: asyncio.Task[None] | None = None
    enabled: bool = False
    approved_users: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_pairings: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_key_approved: str = ""
    state_key_pending: str = ""
    offsets: Any = None  # int | None for telegram, dict[str, str] for slack/discord
    config: Any = None  # TelegramConfig | SlackConfig | DiscordConfig


@dataclass
class RuntimeContext:
    """Shared mutable state for the interactive runtime.

    Replaces the closure-captured variables in the old monolithic
    ``run_interactive()`` function.  Every extracted module receives
    a reference to this single context object.
    """

    agent: Agent
    ui: TerminalUI
    command_queue: CommandQueueManager = field(default_factory=CommandQueueManager)
    followup_queue: FollowupQueueManager = field(default_factory=FollowupQueueManager)
    cron_running_job_ids: set[str] = field(default_factory=set)
    cron_poll_seconds: float = 2.0
    last_exec_seconds: float | None = None
    last_completed_at: datetime | None = None

    telegram: PlatformState = field(default_factory=lambda: PlatformState(
        name="telegram",
        state_key_approved="telegram_approved_users",
        state_key_pending="telegram_pending_pairings",
    ))
    slack: PlatformState = field(default_factory=lambda: PlatformState(
        name="slack",
        state_key_approved="slack_approved_users",
        state_key_pending="slack_pending_pairings",
    ))
    discord: PlatformState = field(default_factory=lambda: PlatformState(
        name="discord",
        state_key_approved="discord_approved_users",
        state_key_pending="discord_pending_pairings",
    ))

    # Telegram command specs (shared by both registration and help text)
    telegram_command_specs: list[tuple[str, str]] = field(default_factory=lambda: [
        ("start", "Start Captain Claw in Telegram"),
        ("help", "Show Telegram command guide"),
        ("new", "Create a new session"),
        ("sessions", "List recent sessions"),
        ("session", "Inspect/switch session"),
        ("models", "List allowed models"),
        ("config", "Show active configuration"),
        ("history", "Show recent message history"),
        ("clear", "Clear active session messages"),
        ("compact", "Compact active session memory"),
        ("pipeline", "Set pipeline mode (loop/contracts)"),
        ("planning", "Legacy alias for pipeline"),
        ("skills", "List available skills"),
        (
            "skill",
            "Run/search/install skill: /skill <name> [args] | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]",
        ),
        ("cron", "Run one-off cron prompt"),
        ("todo", "Manage to-do items"),
        ("contacts", "Manage address book contacts"),
    ])

    def get_platform_state(self, platform: str) -> PlatformState:
        """Get platform state by name."""
        if platform == "telegram":
            return self.telegram
        if platform == "slack":
            return self.slack
        if platform == "discord":
            return self.discord
        raise ValueError(f"Unknown platform: {platform}")

    def platform_names(self) -> list[str]:
        return ["telegram", "slack", "discord"]
