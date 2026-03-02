"""Routing logic for matching concerns to CC instances."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from botport.config import get_config
from botport.models import Concern, InstanceInfo
from botport.registry import Registry

log = logging.getLogger(__name__)


class Router:
    """Routes concerns to the best available CC instance."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry
        self._http_client: httpx.AsyncClient | None = None

    async def route(
        self,
        concern: Concern,
        exclude_instance: str | None = None,
    ) -> InstanceInfo | None:
        """Find the best instance to handle a concern.

        Strategy chain:
        1. Tag matching against persona expertise
        2. LLM-assisted classification (if enabled and tag match fails)
        3. Least-loaded fallback
        """
        # Use the concern's originating instance as exclusion.
        exclude = exclude_instance or concern.from_instance

        # Strategy 1: Tag matching.
        if concern.expertise_tags:
            matches = self._registry.find_by_expertise(
                concern.expertise_tags, exclude_instance=exclude,
            )
            if matches:
                best, score = matches[0]
                log.info(
                    "Tag match: concern %s -> %s (score=%d)",
                    concern.id[:8], best.name, score,
                )
                return best

        # Strategy 2: LLM-assisted routing.
        cfg = get_config()
        if cfg.llm.enabled and cfg.routing.strategy == "llm_assisted":
            result = await self._llm_route(concern, exclude)
            if result:
                return result

        # Strategy 3: Least-loaded fallback (only if there are available instances).
        fallback = self._registry.get_least_loaded(exclude_instance=exclude)
        if fallback:
            log.info(
                "Fallback routing: concern %s -> %s (least loaded)",
                concern.id[:8], fallback.name,
            )
            return fallback

        log.warning("No available instance for concern %s", concern.id[:8])
        return None

    async def _llm_route(
        self,
        concern: Concern,
        exclude_instance: str,
    ) -> InstanceInfo | None:
        """Use LLM to classify the concern and pick the best instance."""
        cfg = get_config()
        available = self._registry._connections.list_available(exclude=exclude_instance)
        if not available:
            return None

        # Build instance descriptions for the prompt.
        instance_descriptions: list[str] = []
        for inst in available:
            personas_desc = ", ".join(
                f"{p.name} ({', '.join(p.expertise_tags)})" for p in inst.personas
            )
            instance_descriptions.append(
                f"- {inst.name}: personas=[{personas_desc}], "
                f"tools=[{', '.join(inst.tools[:10])}], "
                f"load={inst.active_concerns}/{inst.max_concurrent}"
            )

        prompt = (
            "You are a routing assistant. Given a task and available agent instances, "
            "pick the single best instance to handle the task.\n\n"
            f"Task: {concern.task}\n\n"
            f"Available instances:\n" + "\n".join(instance_descriptions) + "\n\n"
            "Respond with ONLY the instance name (nothing else)."
        )

        try:
            chosen_name = await self._call_llm(prompt, cfg)
            chosen_name = chosen_name.strip().strip('"').strip("'")

            for inst in available:
                if inst.name.lower() == chosen_name.lower():
                    log.info(
                        "LLM routing: concern %s -> %s",
                        concern.id[:8], inst.name,
                    )
                    return inst

            log.warning("LLM returned unknown instance name: %r", chosen_name)
        except Exception as exc:
            log.warning("LLM routing failed: %s", exc)

        return None

    async def _call_llm(self, prompt: str, cfg: Any) -> str:
        """Make a single LLM completion call for routing."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        llm_cfg = cfg.llm
        provider = llm_cfg.provider.lower().strip()
        base_url = llm_cfg.base_url.rstrip("/")

        if provider == "ollama":
            url = f"{base_url}/api/chat"
            body = {
                "model": llm_cfg.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": llm_cfg.temperature,
                    "num_predict": llm_cfg.max_tokens,
                },
            }
            resp = await self._http_client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return str((data.get("message") or {}).get("content", ""))

        # OpenAI-compatible (openai, anthropic via litellm proxy, etc.)
        url = f"{base_url}/v1/chat/completions"
        headers: dict[str, str] = {}
        if llm_cfg.api_key:
            headers["Authorization"] = f"Bearer {llm_cfg.api_key}"

        body = {
            "model": llm_cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
        }
        resp = await self._http_client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            return str((choices[0].get("message") or {}).get("content", ""))
        return ""

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
