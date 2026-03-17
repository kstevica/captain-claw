"""Interactive onboarding workflow for first-time Captain Claw setup."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.config import DEFAULT_CONFIG_PATH, LOCAL_CONFIG_FILENAME, Config

log = logging.getLogger(__name__)

_ONBOARDING_STATE_FILENAME = "onboarding_state.json"

_PROVIDER_ORDER = ("openai", "anthropic", "gemini", "xai", "ollama")
_PROVIDER_LABELS = {
    "openai": "OpenAI / ChatGPT",
    "anthropic": "Anthropic / Claude",
    "gemini": "Google / Gemini",
    "xai": "xAI / Grok",
    "ollama": "Ollama (local/self-hosted)",
}
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-3-flash-preview",
    "xai": "grok-3-mini",
    "ollama": "llama3.2",
}
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY or GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "ollama": "(none required)",
}
_PROVIDER_ALIASES = {
    "chatgpt": "openai",
    "claude": "anthropic",
    "google": "gemini",
    "grok": "xai",
}

# Default allowed models seeded during onboarding so every fresh install
# gets a useful multi-model configuration out of the box.
_DEFAULT_ALLOWED_MODELS: list[dict[str, Any]] = [
    {
        "id": "gpt-5-mini",
        "provider": "openai",
        "model": "gpt-5-mini",
        "reasoning_level": "high",
        "description": "Good for everyday tasks, light coding, reasoning",
    },
    {
        "id": "gpt-5-nano",
        "provider": "openai",
        "model": "gpt-5-nano",
        "description": "good for classification of data, making lists",
    },
    {
        "id": "gpt-5.3-codex",
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "reasoning_level": "high",
        "description": "extremely good for coding and deep reasoning",
    },
    {
        "id": "claude-sonnet",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    },
    {
        "id": "gemini-flash",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
    },
    {
        "id": "claude-opus",
        "provider": "anthropic",
        "model": "claude-opus-4-6",
    },
    {
        "id": "claude-haiku",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    },
    {
        "id": "gemini-flash-lite",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite-preview",
        "temperature": 0,
        "description": "simple and fast model",
    },
    {
        "id": "gemini-pro",
        "provider": "gemini",
        "model": "gemini-2.5-pro",
        "temperature": 0,
        "description": "the best model, can handle really complex tasks, but it's expensive",
    },
    {
        "id": "image",
        "provider": "gemini",
        "model": "imagen-4.0-fast-generate-001",
        "temperature": 0,
        "description": "image generation",
        "model_type": "image",
    },
    {
        "id": "gemini-ocr",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "temperature": 0,
        "model_type": "ocr",
    },
    {
        "id": "gemini-vision",
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "temperature": 0,
        "model_type": "vision",
    },
]


# ── Path resolution helpers ──────────────────────────────────────────


def _resolve_state_path(state_path: Path | str | None = None) -> Path:
    if state_path is not None:
        return Path(state_path).expanduser()
    return Path("~/.captain-claw").expanduser() / _ONBOARDING_STATE_FILENAME


def _resolve_global_config_path(global_config_path: Path | str | None = None) -> Path:
    if global_config_path is not None:
        return Path(global_config_path).expanduser()
    return DEFAULT_CONFIG_PATH.expanduser()


# ── State checks ─────────────────────────────────────────────────────


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


# ── Shared validation helper ─────────────────────────────────────────


async def validate_provider_connection(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
) -> tuple[bool, str | None]:
    """Test an LLM provider connection with a lightweight request.

    For Ollama, hits ``GET {base_url}/api/tags`` instead of a completion.
    Returns ``(ok, error_message | None)``.
    """
    normalized = _normalize_provider(provider)

    if normalized == "ollama":
        return await _validate_ollama(base_url or "http://127.0.0.1:11434")

    try:
        from captain_claw.llm import Message, create_provider

        llm = create_provider(
            provider=normalized,
            model=model,
            api_key=api_key or None,
            base_url=base_url or None,
            temperature=0.0,
            max_tokens=5,
        )
        resp = await llm.complete(
            messages=[Message(role="user", content="Say OK")],
            max_tokens=5,
        )
        if resp and resp.content:
            return True, None
        return False, "Empty response from provider."
    except Exception as exc:
        return False, str(exc)


async def _validate_ollama(base_url: str) -> tuple[bool, str | None]:
    """Validate Ollama connectivity by hitting /api/tags."""
    import httpx

    url = base_url.rstrip("/") + "/api/tags"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return True, None
            return False, f"Ollama returned status {resp.status_code}"
    except Exception as exc:
        return False, f"Cannot reach Ollama at {base_url}: {exc}"


def validate_provider_connection_sync(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
) -> tuple[bool, str | None]:
    """Synchronous wrapper around :func:`validate_provider_connection`."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                validate_provider_connection(provider, model, api_key, base_url),
            ).result(timeout=30)
    return asyncio.run(
        validate_provider_connection(provider, model, api_key, base_url)
    )


# ── Shared save helper ───────────────────────────────────────────────


def save_onboarding_config(
    values: dict[str, Any],
    config_path: Path | str | None = None,
    state_path: Path | str | None = None,
) -> Path:
    """Build a Config from onboarding values, save to YAML, mark completed.

    ``values`` keys:
        provider, model, api_key, base_url, workspace_path,
        enable_guards, allowed_models (list of dicts with
        provider/model/api_key fields), telegram_enabled,
        telegram_token, provider_keys (dict of provider→key),
        openai_headers (list of header strings).

    Returns the config file path.
    """
    target = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH.expanduser()
    cfg = Config.from_yaml(target)

    cfg.model.provider = values.get("provider", cfg.model.provider)
    cfg.model.model = values.get("model", cfg.model.model)
    cfg.model.api_key = values.get("api_key", "")
    cfg.model.base_url = values.get("base_url", "")

    workspace = values.get("workspace_path", "").strip()
    if workspace:
        cfg.workspace.path = workspace

    enable_guards = values.get("enable_guards", False)
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

    allowed = values.get("allowed_models") or []
    if allowed:
        from captain_claw.config import Config as _Cfg

        AllowedModel = _Cfg.ModelConfig.AllowedModelConfig if hasattr(_Cfg, "ModelConfig") else type(cfg.model).AllowedModelConfig
        new_entries = []
        for idx, entry in enumerate(allowed):
            am = AllowedModel(
                id=entry.get("id") or f"{entry['provider']}-{idx + 1}",
                provider=entry["provider"],
                model=entry["model"],
            )
            new_entries.append(am)
        cfg.model.allowed = list(cfg.model.allowed) + new_entries

    # Seed default allowed models when the config has none yet.
    if not cfg.model.allowed:
        AllowedModel = type(cfg.model).AllowedModelConfig
        cfg.model.allowed = [
            AllowedModel(**entry) for entry in _DEFAULT_ALLOWED_MODELS
        ]

    # Provider keys (from the new API Keys onboarding step)
    pk = values.get("provider_keys") or {}
    if pk.get("openai"):
        cfg.provider_keys.openai = pk["openai"]
    if pk.get("anthropic"):
        cfg.provider_keys.anthropic = pk["anthropic"]
    if pk.get("gemini"):
        cfg.provider_keys.gemini = pk["gemini"]
    if pk.get("xai"):
        cfg.provider_keys.xai = pk["xai"]
    if pk.get("brave"):
        cfg.provider_keys.brave = pk["brave"]

    # OpenAI OAuth headers (from Codex CLI import)
    openai_headers = values.get("openai_headers") or []
    if openai_headers:
        cfg.provider_keys.openai_headers = list(openai_headers)

    # Telegram
    if "telegram_enabled" in values:
        cfg.telegram.enabled = bool(values["telegram_enabled"])
    telegram_token = values.get("telegram_token", "").strip()
    if telegram_token:
        cfg.telegram.bot_token = telegram_token

    cfg.save(target)
    mark_onboarding_completed(state_path=state_path)
    return target


# ── TUI wizard helpers ───────────────────────────────────────────────


def _select_config_path(console, config_path: Path | str | None = None) -> Path:  # type: ignore[type-arg]
    from rich.prompt import Prompt
    from rich.table import Table

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


def _select_provider(console, current_provider: str) -> str:  # type: ignore[type-arg]
    from rich.prompt import Prompt
    from rich.table import Table

    table = Table(title="Model Provider", show_header=True, header_style="bold cyan")
    table.add_column("Option", justify="center")
    table.add_column("Provider")
    table.add_column("Suggested model")
    table.add_column("Env var")

    normalized_current = _normalize_provider(current_provider)
    default_index = 1
    for idx, key in enumerate(_PROVIDER_ORDER, start=1):
        if key == normalized_current:
            default_index = idx
        table.add_row(
            str(idx),
            _PROVIDER_LABELS[key],
            _PROVIDER_DEFAULT_MODELS[key],
            _PROVIDER_ENV_VARS[key],
        )
    console.print(table)

    choice = Prompt.ask(
        "Choose your default model provider",
        choices=["1", "2", "3", "4", "5"],
        default=str(default_index),
    )
    return _PROVIDER_ORDER[int(choice) - 1]


def _run_validation_spinner(console, provider: str, model: str, api_key: str, base_url: str) -> tuple[bool, str | None]:  # type: ignore[type-arg]
    """Run provider validation with a Rich spinner."""
    from rich.live import Live
    from rich.spinner import Spinner

    result: tuple[bool, str | None] = (False, "Validation did not complete")

    with Live(Spinner("dots", text="Validating connection..."), console=console, refresh_per_second=10):
        result = validate_provider_connection_sync(provider, model, api_key, base_url)

    return result


def _collect_additional_models(console) -> list[dict[str, str]]:  # type: ignore[type-arg]
    """Prompt for up to 3 additional models."""
    from rich.prompt import Confirm, Prompt

    models: list[dict[str, str]] = []
    for i in range(3):
        if not Confirm.ask(
            f"Add {'another' if i > 0 else 'an additional'} model?",
            default=False,
        ):
            break

        provider = _select_provider(console, "openai")
        default_model = _PROVIDER_DEFAULT_MODELS[provider]
        model = Prompt.ask("Model id/name", default=default_model).strip() or default_model

        api_key = ""
        if provider != "ollama":
            store_key = Confirm.ask("Store API key for this model?", default=False)
            if store_key:
                api_key = Prompt.ask("API key", password=True).strip()

        console.print("[dim]Validating...[/dim]")
        ok, error = validate_provider_connection_sync(provider, model, api_key, "")
        if ok:
            console.print(f"[green]Connected to {provider}/{model}.[/green]")
        else:
            console.print(f"[yellow]Validation failed: {error}[/yellow]")
            if not Confirm.ask("Add this model anyway?", default=False):
                continue

        models.append({"provider": provider, "model": model, "api_key": api_key})

    return models


# ── Main TUI wizard ──────────────────────────────────────────────────


def run_onboarding_wizard(
    config_path: Path | str | None = None,
    state_path: Path | str | None = None,
    require_interactive: bool = False,
) -> Path | None:
    """Run interactive onboarding and persist selected configuration."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    console = Console()
    if not can_run_onboarding_interactively():
        message = "Onboarding wizard requires an interactive terminal."
        if require_interactive:
            raise RuntimeError(message)
        console.print(f"[yellow]{message} Skipping automatic onboarding.[/yellow]")
        return None

    # Step 1 — Welcome
    intro = Panel(
        "[bold cyan]Captain Claw Onboarding[/bold cyan]\n"
        "Let's get Captain Claw ready. This takes about a minute.\n"
        "The only requirement is one working model — everything else is optional.",
        border_style="cyan",
    )
    console.print(intro)

    # Step 2 — Config location
    target_config_path = _select_config_path(console, config_path=config_path)
    cfg = Config.from_yaml(target_config_path)

    # Step 3 — Provider
    provider = _select_provider(console, cfg.model.provider)

    # Step 4 — Model name
    default_model = (
        cfg.model.model.strip()
        if _normalize_provider(cfg.model.provider) == provider and cfg.model.model.strip()
        else _PROVIDER_DEFAULT_MODELS[provider]
    )
    model_name = Prompt.ask("Default model id/name", default=default_model).strip() or default_model

    # Step 5 — API key / base URL
    existing_api_key = cfg.model.api_key.strip()
    api_key = ""
    base_url = ""

    if provider == "ollama":
        default_base_url = cfg.model.base_url.strip() or "http://127.0.0.1:11434"
        base_url = Prompt.ask("Ollama base URL", default=default_base_url).strip()
    else:
        env_var = _PROVIDER_ENV_VARS.get(provider, "")
        console.print(f"[dim]Tip: You can also set {env_var} as an environment variable.[/dim]")
        store_api_key = Confirm.ask(
            "Store provider API key in config file?",
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

    # Step 6 — Validate connection
    console.print()
    ok, error = _run_validation_spinner(console, provider, model_name, api_key, base_url)
    if ok:
        console.print("[bold green]Connection successful.[/bold green]")
    else:
        console.print(f"[bold red]Connection failed:[/bold red] {error}")
        retry = Confirm.ask("Re-enter API key?", default=True)
        if retry and provider != "ollama":
            api_key = Prompt.ask("API key", password=True).strip()
            ok, error = _run_validation_spinner(console, provider, model_name, api_key, base_url)
            if ok:
                console.print("[bold green]Connection successful.[/bold green]")
            else:
                console.print(f"[yellow]Still failing: {error}. Continuing anyway.[/yellow]")
        elif retry and provider == "ollama":
            base_url = Prompt.ask("Ollama base URL", default=base_url).strip()
            ok, error = _run_validation_spinner(console, provider, model_name, api_key, base_url)
            if ok:
                console.print("[bold green]Connection successful.[/bold green]")
            else:
                console.print(f"[yellow]Still failing: {error}. Continuing anyway.[/yellow]")
        else:
            console.print("[yellow]Skipping validation. You can fix this later in /settings.[/yellow]")

    # Step 7 — Additional models (optional)
    console.print()
    console.print(
        f"[dim]Captain Claw comes pre-configured with {len(_DEFAULT_ALLOWED_MODELS)} additional models "
        f"(OpenAI, Anthropic, Gemini + image/OCR/vision). You can manage them later in /settings.[/dim]"
    )
    additional_models: list[dict[str, str]] = []
    if Confirm.ask("Add extra models beyond the defaults?", default=False):
        additional_models = _collect_additional_models(console)

    # Step 8 — Safety guards
    console.print()
    guards_default = bool(
        cfg.guards.input.enabled or cfg.guards.output.enabled or cfg.guards.script_tool.enabled
    )
    enable_guards = Confirm.ask(
        "Enable safety guards? (recommended for shared environments)",
        default=guards_default,
    )

    # Step 9 — Summary
    console.print()
    summary = Table(title="Onboarding Summary", show_header=False, box=None)
    summary.add_column("Setting", style="bold")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Config file", str(target_config_path))
    summary.add_row("Provider", provider)
    summary.add_row("Model", model_name)
    summary.add_row("API key stored", "yes" if bool(api_key) else "no (env var)")
    summary.add_row("Base URL", base_url or "(provider default)")
    summary.add_row(
        "Safety guards",
        "enabled (ask_for_approval)" if enable_guards else "disabled",
    )
    summary.add_row(
        "Pre-configured models",
        f"{len(_DEFAULT_ALLOWED_MODELS)} (OpenAI, Anthropic, Gemini + image/OCR/vision)",
    )
    if additional_models:
        for idx, am in enumerate(additional_models, start=1):
            summary.add_row(f"Extra model #{idx}", f"{am['provider']} / {am['model']}")
    console.print(summary)

    if not Confirm.ask("Save configuration and finish onboarding?", default=True):
        console.print("[yellow]Onboarding skipped. No changes were saved.[/yellow]")
        return None

    # Save
    result = save_onboarding_config(
        values={
            "provider": provider,
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "workspace_path": cfg.workspace.path,
            "enable_guards": enable_guards,
            "allowed_models": additional_models,
        },
        config_path=target_config_path,
        state_path=state_path,
    )

    console.print(
        Panel(
            "[bold green]Setup complete.[/bold green]\n"
            f"Configuration saved to [cyan]{result}[/cyan]\n\n"
            "[dim]Change settings anytime: edit config.yaml or open /settings in the web UI.[/dim]",
            border_style="green",
        )
    )
    return result
