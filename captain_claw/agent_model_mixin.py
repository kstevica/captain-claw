"""Runtime model selection helpers for Agent."""

from datetime import UTC, datetime
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import create_provider, set_provider


class AgentModelMixin:
    """Runtime model resolution and session model selection."""
    @staticmethod
    def _normalize_provider_key(provider: str) -> str:
        raw = str(provider or "").strip().lower()
        aliases = {
            "chatgpt": "openai",
            "claude": "anthropic",
            "google": "gemini",
            "googleai": "gemini",
        }
        return aliases.get(raw, raw)

    def _refresh_runtime_model_details(
        self,
        source: str = "config",
        model_id: str = "",
    ) -> None:
        """Refresh current runtime model details from active provider/config."""
        cfg = get_config()
        provider_name = str(getattr(self.provider, "provider", cfg.model.provider or "")).strip()
        model_name = str(getattr(self.provider, "model", cfg.model.model or "")).strip()
        temperature = getattr(self.provider, "temperature", cfg.model.temperature)
        max_tokens = getattr(self.provider, "max_tokens", cfg.model.max_tokens)
        base_url = str(getattr(self.provider, "base_url", cfg.model.base_url or "") or "").strip()
        self._runtime_model_details = {
            "provider": provider_name or str(cfg.model.provider),
            "model": model_name or str(cfg.model.model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
            "source": source,
            "id": model_id,
        }

    def get_runtime_model_details(self) -> dict[str, Any]:
        """Return active runtime model details for UI/status display."""
        if not self._runtime_model_details:
            self._refresh_runtime_model_details(source="config")
        return dict(self._runtime_model_details)

    def get_allowed_models(self) -> list[dict[str, Any]]:
        """Return allowed model selections from config.

        Falls back to current config model if explicit allowlist is empty.
        """
        cfg = get_config()
        options: list[dict[str, Any]] = []
        
        def _pick(entry: Any, key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        for idx, item in enumerate(cfg.model.allowed, start=1):
            model_id = str(_pick(item, "id", "")).strip() or f"model-{idx}"
            provider = str(_pick(item, "provider", "")).strip()
            model = str(_pick(item, "model", "")).strip()
            base_url = str(_pick(item, "base_url", "") or "").strip()
            temperature = _pick(item, "temperature", None)
            max_tokens = _pick(item, "max_tokens", None)
            if not provider or not model:
                continue
            options.append({
                "id": model_id,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })

        if options:
            return options

        return [{
            "id": "default",
            "provider": str(cfg.model.provider),
            "model": str(cfg.model.model),
            "base_url": str(cfg.model.base_url or ""),
            "temperature": cfg.model.temperature,
            "max_tokens": cfg.model.max_tokens,
        }]

    def _resolve_allowed_model(self, selector: str) -> dict[str, Any] | None:
        """Resolve user selector to an allowed model entry."""
        key = (selector or "").strip()
        if not key:
            return None
        options = self.get_allowed_models()

        lowered = key.lower()
        for option in options:
            if str(option.get("id", "")).strip().lower() == lowered:
                return option

        index_text = key[1:] if key.startswith("#") else key
        if index_text.isdigit():
            index = int(index_text)
            if 1 <= index <= len(options):
                return options[index - 1]

        for option in options:
            provider = str(option.get("provider", "")).strip().lower()
            model = str(option.get("model", "")).strip().lower()
            if lowered in {f"{provider}:{model}", f"{provider}/{model}", model}:
                return option
        return None

    def _apply_model_option(
        self,
        option: dict[str, Any],
        source: str,
        model_id: str = "",
    ) -> None:
        """Instantiate provider for chosen model and switch runtime."""
        provider = str(option.get("provider", "")).strip()
        model = str(option.get("model", "")).strip()
        base_url_raw = str(option.get("base_url", "") or "").strip()
        base_url = base_url_raw or None
        temperature = option.get("temperature")
        max_tokens = option.get("max_tokens")
        cfg = get_config()
        normalized_provider = self._normalize_provider_key(provider)
        normalized_cfg_provider = self._normalize_provider_key(str(cfg.model.provider))
        cfg_base_url = str(cfg.model.base_url or "").strip()

        # Prevent default-provider base_url (e.g. Ollama localhost) from leaking
        # into a different selected provider via persisted session metadata.
        if base_url and normalized_provider != normalized_cfg_provider and cfg_base_url and base_url == cfg_base_url:
            base_url = None

        fallback_base_url = cfg_base_url if normalized_provider == normalized_cfg_provider else ""
        resolved_temperature = float(cfg.model.temperature if temperature is None else temperature)
        resolved_max_tokens = int(cfg.model.max_tokens if max_tokens is None else max_tokens)

        self.provider = create_provider(
            provider=provider,
            model=model,
            api_key=cfg.model.api_key or None,
            base_url=base_url or fallback_base_url or None,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            num_ctx=cfg.context.max_tokens,
        )
        set_provider(self.provider)
        self._refresh_runtime_model_details(source=source, model_id=model_id)

    def _session_model_selection(self) -> dict[str, Any] | None:
        """Return model-selection metadata for active session, if present."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return None
        raw = self.session.metadata.get("model_selection")
        if not isinstance(raw, dict):
            return None
        provider = str(raw.get("provider", "")).strip()
        model = str(raw.get("model", "")).strip()
        if not provider or not model:
            return None
        return {
            "id": str(raw.get("id", "")).strip(),
            "provider": provider,
            "model": model,
            "base_url": str(raw.get("base_url", "") or "").strip(),
            "temperature": raw.get("temperature"),
            "max_tokens": raw.get("max_tokens"),
        }

    def _apply_default_config_model_if_needed(self) -> None:
        """Apply config default model unless agent is using external provider override."""
        cfg = get_config()
        if self._provider_override and self.provider is not None:
            self._refresh_runtime_model_details(source="override")
            return
        self._apply_model_option(
            {
                "provider": str(cfg.model.provider),
                "model": str(cfg.model.model),
                "base_url": str(cfg.model.base_url or ""),
                "temperature": cfg.model.temperature,
                "max_tokens": cfg.model.max_tokens,
            },
            source="config",
            model_id="default",
        )

    async def set_session_model(self, selector: str, persist: bool = True) -> tuple[bool, str]:
        """Select runtime model for active session from allowed model list."""
        if not self.session:
            return False, "No active session"
        key = (selector or "").strip()
        if not key:
            return False, "Usage: /session model <id|#index|provider:model|default>"

        lowered = key.lower()
        if lowered in {"default", "config"}:
            self.session.metadata.pop("model_selection", None)
            self._apply_default_config_model_if_needed()
            if persist:
                await self.session_manager.save_session(self.session)
            return True, "Session model reset to default config"

        option = self._resolve_allowed_model(key)
        if not option:
            return False, f"Model not found in allowlist: {selector}"

        try:
            self._apply_model_option(
                option,
                source="session",
                model_id=str(option.get("id", "")).strip(),
            )
        except Exception as e:
            return False, f"Failed to activate model: {e}"
        details = self.get_runtime_model_details()
        self.session.metadata["model_selection"] = {
            "id": str(option.get("id", "")).strip(),
            "provider": details.get("provider"),
            "model": details.get("model"),
            "base_url": details.get("base_url", ""),
            "temperature": details.get("temperature"),
            "max_tokens": details.get("max_tokens"),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if persist:
            await self.session_manager.save_session(self.session)
        return True, f"Session model set to {details.get('provider')}/{details.get('model')}"
