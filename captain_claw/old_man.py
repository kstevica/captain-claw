"""Old Man — lightweight desktop supervisor agent.

Runs as a persistent session that triages user requests and delegates
complex tasks to the orchestrator or peer agents in Flight Deck.
Designed for desktop use (hotkey enabled by default) but works equally
well from a git-clone/pip-install server start.

Integration:
    Called from ``web_server._run_server()`` and ``main.main()`` to apply
    Old Man overrides before the agent is initialized.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from captain_claw.config import get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent
    from captain_claw.config import Config

log = get_logger(__name__)

# Session metadata key so we can recognize Old Man sessions on reload.
OLD_MAN_SESSION_TAG = "old_man_supervisor"


def is_old_man_enabled(cfg: "Config | None" = None) -> bool:
    """Return True when Old Man mode is active."""
    if cfg is None:
        cfg = get_config()
    return bool(cfg.old_man.enabled)


def _detect_fd_url() -> str:
    """Return the Flight Deck URL if we're running inside FD, else ''."""
    return os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")


def apply_old_man_config_overrides(cfg: "Config") -> None:
    """Mutate *cfg* in-place to apply Old Man defaults.

    Called early in startup — before the agent or hotkey daemon are
    initialized — so that downstream code sees the overrides without
    any special-casing.
    """
    om = cfg.old_man
    if not om.enabled:
        return

    # Force hotkey on (the defining Old Man feature).
    cfg.tools.screen_capture.hotkey_enabled = om.hotkey_enabled
    if om.hotkey_trigger_key:
        cfg.tools.screen_capture.hotkey_trigger_key = om.hotkey_trigger_key

    # Ensure orchestrator limits align with Old Man's delegation cap.
    if om.max_delegated_sessions:
        cfg.orchestrator.max_parallel = max(
            cfg.orchestrator.max_parallel,
            om.max_delegated_sessions,
        )

    # Ensure flight_deck tool is enabled when FD is available.
    fd_url = _detect_fd_url()
    if fd_url and "flight_deck" not in cfg.tools.enabled:
        cfg.tools.enabled.append("flight_deck")

    log.info(
        "old_man_config_applied",
        hotkey=om.hotkey_enabled,
        trigger_key=cfg.tools.screen_capture.hotkey_trigger_key,
        auto_orchestrate=om.auto_orchestrate,
        voice_response=om.voice_response,
        fd_url=fd_url or "(not in Flight Deck)",
    )


def build_old_man_session_instructions(cfg: "Config | None" = None) -> str:
    """Return the session-level instructions block injected for Old Man.

    This is stored in ``session.metadata["session_instructions"]`` so
    that the existing system-prompt builder picks it up automatically —
    no changes to ``_build_system_prompt`` required.
    """
    if cfg is None:
        cfg = get_config()
    om = cfg.old_man

    loader = InstructionLoader()
    try:
        base = loader.load("old_man_system_prompt.md")
    except FileNotFoundError:
        base = (
            "You are running in Old Man mode — a desktop supervisor. "
            "Triage requests: answer simple ones directly, delegate "
            "complex multi-step tasks to the orchestrator."
        )

    # Fill conditional voice block.
    if om.voice_response:
        voice_block = (
            "When responding to voice-activated requests, use the pocket_tts "
            "tool to speak your answer aloud. Keep spoken responses brief and "
            "conversational."
        )
    else:
        voice_block = ""

    base = base.replace("{voice_response_block}", voice_block)
    return base


async def setup_old_man_session(agent: "Agent") -> None:
    """Tag the agent's session as an Old Man supervisor session.

    Called once after ``agent.initialize()`` to inject the Old Man
    instructions into the session metadata. Idempotent — safe to call
    on every startup (only writes if the tag is missing).
    """
    cfg = get_config()
    if not cfg.old_man.enabled:
        return

    session = agent.session
    if session is None:
        return

    meta = session.metadata or {}
    if meta.get("old_man") == OLD_MAN_SESSION_TAG:
        # Already tagged — skip to avoid overwriting user edits.
        return

    meta["old_man"] = OLD_MAN_SESSION_TAG
    meta["session_display_name"] = meta.get("session_display_name") or "Old Man"
    meta["session_description"] = (
        meta.get("session_description")
        or "Desktop supervisor — triages and delegates tasks."
    )
    meta["session_instructions"] = build_old_man_session_instructions(cfg)

    # Store FD URL so the flight_deck tool can find it from session metadata.
    fd_url = _detect_fd_url()
    if fd_url:
        meta["fd_url"] = fd_url

    session.metadata = meta

    # Persist so it survives restarts.
    try:
        from captain_claw.session import get_session_manager
        await get_session_manager().save_session(session)
    except Exception as exc:
        log.warning("old_man_session_save_failed", error=str(exc))

    log.info(
        "old_man_session_tagged",
        session_id=session.id,
        fd_url=fd_url or "(standalone)",
    )

    # Optionally greet the user on startup.
    if cfg.old_man.idle_greeting and not session.messages:
        greeting = (
            "Old Man supervisor is active. "
            "I'm listening — use the hotkey to talk, or type here."
        )
        if fd_url:
            greeting += (
                " I'm connected to Flight Deck and can spawn or delegate "
                "to other agents in the fleet."
            )
        agent._add_session_message(role="assistant", content=greeting)


def print_old_man_banner(cfg: "Config") -> None:
    """Print a startup banner when Old Man mode is active."""
    if not cfg.old_man.enabled:
        return
    om = cfg.old_man
    trigger = cfg.tools.screen_capture.hotkey_trigger_key
    fd_url = _detect_fd_url()
    lines = [
        "  Old Man supervisor active",
        f"    hotkey: {trigger} (2x=voice, 3x=screen+voice)",
        f"    auto-orchestrate: {'yes' if om.auto_orchestrate else 'no'}",
        f"    voice response: {'yes' if om.voice_response else 'no'}",
    ]
    if fd_url:
        lines.append(f"    flight deck: {fd_url}")
    if om.triage_model:
        lines.append(f"    triage model: {om.triage_model}")
    print("\n".join(lines))
