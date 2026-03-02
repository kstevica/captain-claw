"""Routing logic for matching concerns to CC instances."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from botport.config import get_config
from botport.models import Concern, InstanceInfo
from botport.registry import Registry

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Result of routing a concern — includes both target instance and persona."""

    instance: InstanceInfo
    persona_name: str = ""
    reason: str = ""


class Router:
    """Routes concerns to the best available CC instance."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    async def route(
        self,
        concern: Concern,
        exclude_instance: str | None = None,
    ) -> RouteResult | None:
        """Find the best instance to handle a concern.

        Strategy chain:
        1. Strong tag matching (≥50% tag overlap) against persona expertise
        2. LLM-assisted routing (if enabled)
        3. Least-loaded fallback
        """
        exclude = exclude_instance or concern.from_instance

        # Strategy 1: Strong tag matching.
        if concern.expertise_tags:
            result = self._tag_route(concern, exclude)
            if result:
                return result

        # Strategy 2: LLM-assisted routing.
        cfg = get_config()
        if cfg.llm.enabled:
            result = await self._llm_route(concern, exclude)
            if result:
                return result

        # Strategy 3: Least-loaded fallback.
        fallback = self._registry.get_least_loaded(exclude_instance=exclude)
        if fallback:
            persona_name = self._pick_best_persona(fallback, concern.expertise_tags)
            log.debug(
                "Fallback routing: concern %s -> %s (least loaded)",
                concern.id[:8], fallback.name,
            )
            return RouteResult(
                instance=fallback,
                persona_name=persona_name,
                reason="least_loaded_fallback",
            )

        log.warning("No available instance for concern %s", concern.id[:8])
        return None

    def _tag_route(
        self,
        concern: Concern,
        exclude_instance: str,
    ) -> RouteResult | None:
        """Strong tag matching — only if ≥50% of requested tags match."""
        matches = self._registry.find_by_expertise(
            concern.expertise_tags, exclude_instance=exclude_instance,
        )
        if not matches:
            return None

        best, score = matches[0]
        required = len(concern.expertise_tags)

        # Require at least 50% tag overlap for a "strong" match.
        if required > 0 and score / required < 0.5:
            log.debug(
                "Tag match too weak for concern %s: %d/%d (%.0f%%)",
                concern.id[:8], score, required, score / required * 100,
            )
            return None

        persona_name = self._pick_best_persona(best, concern.expertise_tags)
        log.debug(
            "Tag match: concern %s -> %s (score=%d/%d, persona=%s)",
            concern.id[:8], best.name, score, required, persona_name,
        )
        return RouteResult(
            instance=best,
            persona_name=persona_name,
            reason=f"tag_match ({score}/{required})",
        )

    async def _llm_route(
        self,
        concern: Concern,
        exclude_instance: str,
    ) -> RouteResult | None:
        """Use LLM to classify the concern and pick the best instance."""
        available = self._registry._connections.list_available(exclude=exclude_instance)
        if not available:
            return None

        # Build the rich prompt.
        system_msg = (
            "You are a routing assistant for a multi-agent system. Select the best agent "
            "instance AND persona to handle a task based on expertise, background, and workload.\n"
            "- Pick the instance/persona whose expertise best matches the task.\n"
            "- Prefer lower-load instances when expertise is similar.\n"
            "- If NO instance is a good fit, respond with NONE.\n"
            "- Respond ONLY with valid JSON, no markdown fences."
        )

        # Build instance descriptions with full persona details.
        instance_blocks: list[str] = []
        for inst in available:
            load_status = "available" if inst.active_concerns < inst.max_concurrent else "full"
            block = f"Instance: {inst.name}\n"
            block += f"  Load: {inst.active_concerns}/{inst.max_concurrent} ({load_status})\n"

            if inst.personas:
                block += "  Personas:\n"
                for p in inst.personas:
                    block += f"    - Persona: {p.name}\n"
                    if p.description:
                        block += f"      Description: {p.description}\n"
                    if p.background:
                        block += f"      Background: {p.background}\n"
                    if p.expertise_tags:
                        block += f"      Expertise: {', '.join(p.expertise_tags)}\n"
            else:
                block += "  Personas: (none defined)\n"

            instance_blocks.append(block)

        user_msg = (
            f"Task: {concern.task}\n\n"
            f"Available agents:\n\n"
            + "\n".join(instance_blocks) + "\n"
            'Respond with JSON: {"instance_name": "<name>", "persona_name": "<name>", "reason": "<brief>"}\n'
            'Or if no agent fits: {"instance_name": "NONE", "persona_name": "", "reason": "<why>"}'
        )

        try:
            raw = await self._call_llm(system_msg, user_msg)
            return self._parse_llm_response(raw, available)
        except Exception as exc:
            log.warning("LLM routing failed: %s", exc)
            return None

    async def _call_llm(self, system_msg: str, user_msg: str) -> str:
        """Make a single LLM completion call via litellm."""
        from litellm import acompletion

        cfg = get_config().llm

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "timeout": cfg.timeout,
        }

        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["api_base"] = cfg.base_url

        response = await acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()

    def _parse_llm_response(
        self,
        raw: str,
        available: list[InstanceInfo],
    ) -> RouteResult | None:
        """Parse LLM JSON response and validate against available instances."""
        # Strip markdown code fences if present.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("LLM returned non-JSON: %s", raw[:200])
            return None

        instance_name = str(data.get("instance_name", "")).strip()
        persona_name = str(data.get("persona_name", "")).strip()
        reason = str(data.get("reason", "")).strip()

        # Check for explicit NONE response.
        if instance_name.upper() == "NONE":
            log.info("LLM says no suitable agent: %s", reason)
            return None

        # Find matching instance.
        for inst in available:
            if inst.name.lower() == instance_name.lower():
                # Validate persona exists if specified.
                if persona_name:
                    persona_found = any(
                        p.name.lower() == persona_name.lower()
                        for p in inst.personas
                    )
                    if not persona_found:
                        # LLM hallucinated a persona — fall back to best match.
                        log.debug(
                            "LLM picked non-existent persona %r, using best match",
                            persona_name,
                        )
                        persona_name = self._pick_best_persona(inst, [])

                log.info(
                    "LLM routing: concern -> %s (persona=%s, reason=%s)",
                    inst.name, persona_name or "auto", reason,
                )
                return RouteResult(
                    instance=inst,
                    persona_name=persona_name,
                    reason=f"llm: {reason}",
                )

        log.warning("LLM returned unknown instance name: %r", instance_name)
        return None

    def _pick_best_persona(
        self,
        instance: InstanceInfo,
        expertise_tags: list[str],
    ) -> str:
        """Pick the best persona name from instance based on expertise overlap."""
        if not instance.personas:
            return ""

        if not expertise_tags:
            return instance.personas[0].name

        query_tags = {t.lower() for t in expertise_tags}
        best_name = ""
        best_score = 0

        for persona in instance.personas:
            persona_tags = {t.lower() for t in persona.expertise_tags}
            overlap = len(query_tags & persona_tags)
            if overlap > best_score:
                best_score = overlap
                best_name = persona.name

        return best_name or instance.personas[0].name

    async def close(self) -> None:
        """No-op — litellm manages its own connections."""
        pass
