"""Interactive onboarding workflow for first-time Captain Claw setup."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from captain_claw.config import DEFAULT_CONFIG_PATH, LOCAL_CONFIG_FILENAME, Config

_ONBOARDING_STATE_FILENAME = "onboarding_state.json"

_PROVIDER_ORDER = ("openai", "anthropic", "gemini", "ollama")
_PROVIDER_LABELS = {
    "openai": "OpenAI / ChatGPT",
    "anthropic": "Anthropic / Claude",
    "gemini": "Google / Gemini",
    "ollama": "Ollama (local/self-hosted)",
}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
}
_PROVIDER_ALIASES = {
    "chatgpt": "openai",
    "claude": "anthropic",
    "google": "gemini",
}


def _resolve_state_path(state_path: Path | str | None = None) -> Path:
    if state_path is not None:
        return Path(state_path).expanduser()
    return Path("~/.captain-claw").expanduser() / _ONBOARDING_STATE_FILENAME


def _resolve_global_config_path(global_config_path: Path | str | None = None) -> Path:
    if global_config_path is not None:
        return Path(global_config_path).expanduser()
    return DEFAULT_CONFIG_PATH.expanduser()


def is_onboarding_completed(state_path: Path | str | None = None) -> bool:
    """Return True when onboarding has previously completed."""
    path = _resolve_state_path(state_path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("completed"))


def mark_onboarding_completed(state_path: Path | str | None = None) -> None:
    """Persist onboarding completion marker."""
    path = _resolve_state_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": True,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def has_existing_configuration(
    cwd: Path | str | None = None,
    global_config_path: Path | str | None = None,
) -> bool:
    """Return True if local/global configuration already exists."""
    base = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    local_path = base / LOCAL_CONFIG_FILENAME
    global_path = _resolve_global_config_path(global_config_path)
    return local_path.exists() or global_path.exists()


def should_run_onboarding(
    force: bool = False,
    state_path: Path | str | None = None,
    config_path: Path | str | None = None,
    cwd: Path | str | None = None,
    global_config_path: Path | str | None = None,
) -> bool:
    """Decide whether onboarding should run for this launch."""
    if force:
        return True
    if is_onboarding_completed(state_path=state_path):
        return False
    if config_path:
        explicit_path = Path(config_path).expanduser()
        if explicit_path.exists():
            return False
    if has_existing_configuration(cwd=cwd, global_config_path=global_config_path):
        return False
    return True


def can_run_onboarding_interactively() -> bool:
    """Return True when stdin/stdout are interactive terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)
    if normalized in _PROVIDER_ORDER:
        return normalized
    return "openai"


def _select_config_path(console: Console, config_path: Path | str | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()

    local_path = (Path.cwd() / LOCAL_CONFIG_FILENAME).resolve()
    global_path = _resolve_global_config_path(None)
    default_choice = "2" if local_path.exists() and not global_path.exists() else "1"

    table = Table(title="Config Destination", show_header=True, header_style="bold cyan")
    table.add_column("Option", justify="center")
    table.add_column("Path", overflow="fold")
    table.add_row("1", str(global_path))
    table.add_row("2", str(local_path))
    console.print(table)

    choice = Prompt.ask(
        "Where should configuration be stored?",
        choices=["1", "2"],
        default=default_choice,
    )
    return global_path if choice == "1" else local_path


def _select_provider(console: Console, current_provider: str) -> str:
    table = Table(title="Model Provider", show_header=True, header_style="bold cyan")
    table.add_column("Option", justify="center")
    table.add_column("Provider")
    table.add_column("Suggested model")

    normalized_current = _normalize_provider(current_provider)
    default_index = 1
    for idx, key in enumerate(_PROVIDER_ORDER, start=1):
        if key == normalized_current:
            default_index = idx
        table.add_row(str(idx), _PROVIDER_LABELS[key], _PROVIDER_DEFAULT_MODELS[key])
    console.print(table)

    choice = Prompt.ask(
        "Choose your default model provider",
        choices=["1", "2", "3", "4"],
        default=str(default_index),
    )
    return _PROVIDER_ORDER[int(choice) - 1]


def run_onboarding_wizard(
    config_path: Path | str | None = None,
    state_path: Path | str | None = None,
    require_interactive: bool = False,
) -> Path | None:
    """Run interactive onboarding and persist selected configuration."""
    console = Console()
    if not can_run_onboarding_interactively():
        message = "Onboarding wizard requires an interactive terminal."
        if require_interactive:
            raise RuntimeError(message)
        console.print(f"[yellow]{message} Skipping automatic onboarding.[/yellow]")
        return None

    intro = Panel(
        "[bold cyan]Captain Claw Onboarding[/bold cyan]\n"
        "This wizard configures your default model provider, workspace, safety settings, and Telegram integration.",
        border_style="cyan",
    )
    console.print(intro)

    target_config_path = _select_config_path(console, config_path=config_path)
    cfg = Config.from_yaml(target_config_path)

    provider = _select_provider(console, cfg.model.provider)
    default_model = (
        cfg.model.model.strip()
        if _normalize_provider(cfg.model.provider) == provider and cfg.model.model.strip()
        else _PROVIDER_DEFAULT_MODELS[provider]
    )
    model_name = Prompt.ask("Default model id/name", default=default_model).strip() or default_model

    existing_api_key = cfg.model.api_key.strip()
    api_key = ""
    if provider != "ollama":
        store_api_key = Confirm.ask(
            "Store provider API key in config file? (you can use env vars instead)",
            default=bool(existing_api_key),
        )
        if store_api_key:
            api_key = (
                Prompt.ask(
                    "API key",
                    default=existing_api_key,
                    password=True,
                ).strip()
                or existing_api_key
            )

    if provider == "ollama":
        default_base_url = cfg.model.base_url.strip() or "http://127.0.0.1:11434"
        base_url = Prompt.ask("Ollama base URL", default=default_base_url).strip()
    else:
        use_custom_base = Confirm.ask(
            "Set a custom provider base URL?",
            default=bool(cfg.model.base_url.strip()),
        )
        if use_custom_base:
            base_url = Prompt.ask("Provider base URL", default=cfg.model.base_url.strip()).strip()
        else:
            base_url = ""

    workspace_path = Prompt.ask("Workspace path", default=cfg.workspace.path).strip() or "./workspace"

    guards_default = bool(
        cfg.guards.input.enabled or cfg.guards.output.enabled or cfg.guards.script_tool.enabled
    )
    enable_guards = Confirm.ask(
        "Enable safety guards (approval mode) for input/output/script checks?",
        default=guards_default,
    )

    existing_telegram_token = cfg.telegram.bot_token.strip()
    telegram_enabled_default = bool(cfg.telegram.enabled or existing_telegram_token)
    configure_telegram = Confirm.ask(
        "Configure Telegram bot integration now?",
        default=telegram_enabled_default,
    )
    telegram_enabled = cfg.telegram.enabled
    telegram_token = existing_telegram_token
    if configure_telegram:
        telegram_enabled = Confirm.ask(
            "Enable Telegram integration now?",
            default=True if not telegram_enabled_default else telegram_enabled_default,
        )
        store_telegram_token = Confirm.ask(
            "Store Telegram bot token in config file? (or use TELEGRAM_BOT_TOKEN env var)",
            default=bool(existing_telegram_token),
        )
        if store_telegram_token:
            telegram_token = (
                Prompt.ask(
                    "Telegram bot token",
                    default=existing_telegram_token,
                    password=True,
                ).strip()
                or existing_telegram_token
            )
        else:
            telegram_token = ""

    summary = Table(title="Onboarding Summary", show_header=False, box=None)
    summary.add_column("Setting", style="bold")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Config file", str(target_config_path))
    summary.add_row("Provider", provider)
    summary.add_row("Model", model_name)
    summary.add_row("API key stored", "yes" if bool(api_key) else "no (env var recommended)")
    summary.add_row("Base URL", base_url or "(provider default)")
    summary.add_row("Workspace", workspace_path)
    summary.add_row(
        "Safety guards",
        "enabled (ask_for_approval)" if enable_guards else "disabled",
    )
    summary.add_row(
        "Telegram integration",
        "enabled" if telegram_enabled else "disabled",
    )
    summary.add_row(
        "Telegram token stored",
        "yes" if bool(telegram_token) else "no (env var recommended)",
    )
    console.print(summary)

    if not Confirm.ask("Save configuration and finish onboarding?", default=True):
        console.print("[yellow]Onboarding skipped. No changes were saved.[/yellow]")
        return None

    cfg.model.provider = provider
    cfg.model.model = model_name
    cfg.model.api_key = api_key
    cfg.model.base_url = base_url
    cfg.workspace.path = workspace_path
    if configure_telegram:
        cfg.telegram.enabled = telegram_enabled
        cfg.telegram.bot_token = telegram_token

    if enable_guards:
        cfg.guards.input.enabled = True
        cfg.guards.input.level = "ask_for_approval"
        cfg.guards.output.enabled = True
        cfg.guards.output.level = "ask_for_approval"
        cfg.guards.script_tool.enabled = True
        cfg.guards.script_tool.level = "ask_for_approval"
    else:
        cfg.guards.input.enabled = False
        cfg.guards.output.enabled = False
        cfg.guards.script_tool.enabled = False

    cfg.save(target_config_path)
    mark_onboarding_completed(state_path=state_path)

    console.print(
        Panel(
            "[bold green]Setup complete.[/bold green]\n"
            f"Configuration saved to [cyan]{target_config_path}[/cyan]",
            border_style="green",
        )
    )
    return target_config_path
