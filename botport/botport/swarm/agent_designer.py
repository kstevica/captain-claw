"""LLM-based agent designer for Swarm orchestration.

Analyzes decomposed tasks and generates optimal agent specs
(persona + model selection) for each task or task group.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from botport.swarm.decomposer import (
    TaskDecomposer,
    _build_instructions_context,
    _build_tools_context,
    get_cc_models,
)
from botport.swarm.models import SwarmTask

log = logging.getLogger(__name__)


# ── Model tier classification ────────────────────────────────

_MODEL_TIER_HINTS = {
    # Keywords in model IDs that hint at tier.
    "haiku": "fast",
    "flash": "fast",
    "mini": "fast",
    "nano": "fast",
    "gpt-4o-mini": "fast",
    "sonnet": "mid",
    "gpt-4o": "mid",
    "gpt-4.1": "mid",
    "pro": "mid",
    "opus": "premium",
    "o1": "premium",
    "o3": "premium",
    "deepthink": "premium",
}


def _classify_model_tier(model_id: str) -> str:
    """Classify a model ID into fast/mid/premium tier."""
    lower = model_id.lower()
    for keyword, tier in _MODEL_TIER_HINTS.items():
        if keyword in lower:
            return tier
    return "mid"  # default


def _build_models_context() -> str:
    """Build a description of available models for the LLM prompt."""
    models = get_cc_models()
    if not models:
        return "No models available."

    lines = []
    for m in models:
        mid = m.get("id", f"{m.get('provider', '')}/{m.get('model', '')}")
        desc = m.get("description", "")
        mtype = m.get("model_type", "llm")
        tier = _classify_model_tier(mid)
        parts = [f"  - {mid}"]
        if desc:
            parts.append(f"({desc})")
        parts.append(f"[type={mtype}, tier={tier}]")
        lines.append(" ".join(parts))

    return (
        "Available models (use model IDs exactly as listed):\n"
        + "\n".join(lines)
        + "\n\n"
        "Model selection guidelines:\n"
        "- **fast** tier (haiku, flash, mini): simple tasks — web fetching, data extraction, "
        "classification, file operations, format conversion.\n"
        "- **mid** tier (sonnet, gpt-4o, pro): analysis, synthesis, report writing, "
        "code generation, multi-step reasoning.\n"
        "- **premium** tier (opus, o1, o3): complex creative writing, deep analysis, "
        "novel problem solving, critical decisions.\n"
        "- Use model_type='multimodal' for image analysis/description tasks.\n"
        "- Use model_type='image' for image generation tasks.\n"
        "- Prefer cheaper (fast) models when quality requirements are low.\n"
        "- Use mid/premium only when the task genuinely requires reasoning or quality."
    )


# ── Agent Designer ───────────────────────────────────────────


class AgentDesigner(TaskDecomposer):
    """Generates task-optimized agent specs using LLM.

    Inherits ``_call_llm``, ``_strip_fences``, and all config/key
    resolution infrastructure from TaskDecomposer.
    """

    async def design_agents(
        self,
        tasks: list[SwarmTask],
        model: str = "",
    ) -> dict[str, dict[str, Any]]:
        """For each task, generate an optimal agent spec.

        Returns ``{task_id: agent_spec_dict}`` where each spec contains:
        - persona_name, persona_description, persona_expertise (list),
          persona_instructions
        - model_id (from CC allowed models)
        """
        if not tasks:
            return {}

        models_ctx = _build_models_context()
        tools_ctx = _build_tools_context()
        instructions_ctx = _build_instructions_context()

        # Build task descriptions for the LLM.
        task_lines = []
        for t in tasks:
            complexity = t.metadata.get("estimated_complexity", "medium")
            task_lines.append(
                f'  - id="{t.id}", name="{t.name}", '
                f"complexity={complexity}\n"
                f"    Description: {t.description[:400]}"
            )
        task_block = "\n".join(task_lines)

        system_msg = (
            "You are an expert agent architect for a multi-agent orchestration system.\n\n"
            "Each task in a swarm will be executed by a temporary agent with a specific "
            "persona (name, expertise, instructions) and an assigned LLM model.\n\n"
            f"{models_ctx}\n\n"
            f"{tools_ctx}\n\n"
            f"{instructions_ctx}\n\n"
            "Your job is to design the optimal agent persona and model for each task.\n\n"
            "Rules:\n"
            "1. Group similar tasks under the SAME persona name — e.g., if 3 tasks all "
            "fetch web pages and extract content, they should share one persona like "
            '"Web Researcher".\n'
            "2. Each persona needs a clear name, short description, list of expertise tags, "
            "and specific instructions for how this persona should approach its tasks.\n"
            "3. Select the most cost-effective model for each task — fast models for simple "
            "tasks, premium models only when genuinely needed.\n"
            "4. Persona instructions should be 2-4 sentences guiding the agent's approach, "
            "tone, and priorities. Do NOT repeat the task description.\n"
            "5. The persona_name should be a professional role title "
            '(e.g., "Web Researcher", "Data Analyst", "Editorial Writer").\n\n'
            "Respond ONLY with valid JSON (no markdown fences):\n"
            "{\n"
            '  "agents": {\n'
            '    "<task_id>": {\n'
            '      "persona_name": "Role Title",\n'
            '      "persona_description": "One sentence describing this persona",\n'
            '      "persona_expertise": ["tag1", "tag2"],\n'
            '      "persona_instructions": "Specific instructions for how this agent should work",\n'
            '      "model_id": "provider/model-name from the available list"\n'
            "    }\n"
            "  }\n"
            "}"
        )

        user_msg = f"Design optimal agents for these tasks:\n\n{task_block}"

        # Retry with increasing token limits — agent design for many tasks
        # can exceed 2k tokens easily.
        for tokens in (4000, 6000):
            raw = await self._call_llm(
                system_msg, user_msg,
                max_tokens=tokens, model_id=model,
            )
            result = self._parse_agent_specs(raw, tasks)
            if result:
                return result
        return self._fallback_specs(tasks)

    def _parse_agent_specs(
        self, raw: str, tasks: list[SwarmTask],
    ) -> dict[str, dict[str, Any]]:
        """Parse LLM output into validated agent specs."""
        text = self._strip_fences(raw)

        data = None
        for attempt_text in (text, self._repair_json(raw)):
            try:
                data = json.loads(attempt_text)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            log.warning("Agent design returned non-JSON: %s", raw[:300])
            return {}

        agents = data.get("agents", data)  # Handle both {agents:{...}} and flat dict
        if not isinstance(agents, dict):
            log.warning("Agent design returned unexpected format")
            return self._fallback_specs(tasks)

        # Validate model IDs against available models.
        available_ids = set()
        for m in get_cc_models():
            mid = m.get("id", f"{m.get('provider', '')}/{m.get('model', '')}")
            available_ids.add(mid)
            available_ids.add(m.get("model", ""))

        result: dict[str, dict[str, Any]] = {}
        task_ids = {t.id for t in tasks}

        for task_id, spec in agents.items():
            if task_id not in task_ids:
                continue
            if not isinstance(spec, dict):
                continue

            # Validate model_id — fall back to empty (= use default) if invalid.
            model_id = str(spec.get("model_id", ""))
            if model_id and model_id not in available_ids:
                # Try partial match.
                matched = False
                for avail in available_ids:
                    if model_id in avail or avail in model_id:
                        model_id = avail
                        matched = True
                        break
                if not matched:
                    log.warning(
                        "Agent design picked unknown model %s for task %s, using default",
                        model_id, task_id[:8],
                    )
                    model_id = ""

            result[task_id] = {
                "persona_name": str(spec.get("persona_name", "Agent")),
                "persona_description": str(spec.get("persona_description", "")),
                "persona_expertise": list(spec.get("persona_expertise", [])),
                "persona_instructions": str(spec.get("persona_instructions", "")),
                "model_id": model_id,
            }

        # Fill in any tasks that the LLM missed.
        for t in tasks:
            if t.id not in result:
                result[t.id] = self._default_spec(t)

        log.info(
            "Designed %d agent specs (%d unique personas)",
            len(result),
            len({s["persona_name"] for s in result.values()}),
        )
        return result

    def _default_spec(self, task: SwarmTask) -> dict[str, Any]:
        """Generate a sensible default spec for a task."""
        complexity = task.metadata.get("estimated_complexity", "medium")
        # Pick a model tier based on complexity.
        models = get_cc_models()
        model_id = ""
        if models:
            target_tier = {"low": "fast", "medium": "mid", "high": "premium"}.get(
                complexity, "mid"
            )
            for m in models:
                mid = m.get("id", f"{m.get('provider', '')}/{m.get('model', '')}")
                if _classify_model_tier(mid) == target_tier:
                    model_id = mid
                    break
            if not model_id:
                model_id = models[0].get(
                    "id", f"{models[0].get('provider', '')}/{models[0].get('model', '')}"
                )

        return {
            "persona_name": "Task Agent",
            "persona_description": "General-purpose task execution agent",
            "persona_expertise": ["task execution"],
            "persona_instructions": (
                "Execute the task efficiently and deliver a complete result. "
                "Use the most appropriate tools for the job."
            ),
            "model_id": model_id,
        }

    def _fallback_specs(self, tasks: list[SwarmTask]) -> dict[str, dict[str, Any]]:
        """Generate default specs for all tasks when LLM output is unparseable."""
        return {t.id: self._default_spec(t) for t in tasks}
