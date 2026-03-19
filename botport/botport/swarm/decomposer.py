"""LLM-based task rephrasing and decomposition for Swarm orchestration.

Uses Captain Claw's config for model selection and provider keys,
matching the same LLM infrastructure (litellm, provider resolution).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import yaml

from botport.swarm.models import SwarmEdge, SwarmTask, _utcnow_iso

log = logging.getLogger(__name__)

# ── CC config helpers ─────────────────────────────────────────

_cc_config_cache: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base (base values win for non-empty leaves).

    Special handling for ``model.allowed`` lists: entries from both configs
    are combined (deduplicated by ``id``) so models defined in either the
    local or home config are all available.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        elif (
            k == "allowed"
            and isinstance(result.get(k), list)
            and isinstance(v, list)
        ):
            # Merge allowed model lists — combine entries, dedup by id.
            seen_ids: set[str] = set()
            merged_list: list[dict] = []
            for entry in result[k]:
                eid = entry.get("id", "") if isinstance(entry, dict) else ""
                if eid:
                    seen_ids.add(eid)
                merged_list.append(entry)
            for entry in v:
                eid = entry.get("id", "") if isinstance(entry, dict) else ""
                if eid and eid not in seen_ids:
                    merged_list.append(entry)
                    seen_ids.add(eid)
            result[k] = merged_list
        elif k not in result or not result[k]:
            # Only fill in keys that are missing or empty in base.
            result[k] = v
    return result


def _load_cc_config() -> dict[str, Any]:
    """Load Captain Claw's config.yaml (cached).

    Merges local ./config.yaml with ~/.captain-claw/config.yaml so that
    provider_keys (secrets) from the home dir config are available even
    when the local config is used for model/allowed settings.
    """
    global _cc_config_cache
    if _cc_config_cache is not None:
        return _cc_config_cache

    local_path = Path("config.yaml")
    try:
        home_path = Path("~/.captain-claw/config.yaml").expanduser()
    except RuntimeError:
        home_path = Path("/tmp/.captain-claw/config.yaml")

    local_data: dict[str, Any] = {}
    home_data: dict[str, Any] = {}

    for p, target in [(local_path, "local"), (home_path, "home")]:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if target == "local":
                    local_data = data
                else:
                    home_data = data
                log.info("Loaded CC config (%s) from %s", target, p)
            except Exception as exc:
                log.warning("Failed to read CC config %s: %s", p, exc)

    # Merge: local config is primary, home config fills in missing values
    # (especially provider_keys which typically live in home config).
    merged = _deep_merge(local_data, home_data) if local_data else home_data

    _cc_config_cache = merged
    if not merged:
        log.warning("No Captain Claw config.yaml found")
    return _cc_config_cache


def reload_cc_config() -> dict[str, Any]:
    """Force reload of CC config (e.g. after settings change)."""
    global _cc_config_cache
    _cc_config_cache = None
    return _load_cc_config()


def get_cc_models() -> list[dict[str, Any]]:
    """Return the allowed models list from CC config.

    Returns list of dicts with: id, provider, model, description,
    temperature, max_tokens, model_type, etc.
    """
    cfg = _load_cc_config()
    model_cfg = cfg.get("model", {})
    allowed = model_cfg.get("allowed", [])

    if not allowed:
        # Fallback: build a single entry from the default model.
        provider = model_cfg.get("provider", "ollama")
        model = model_cfg.get("model", "llama3.2")
        return [{
            "id": f"{provider}/{model}",
            "provider": provider,
            "model": model,
            "description": "Default model",
            "temperature": model_cfg.get("temperature", 0.7),
            "max_tokens": model_cfg.get("max_tokens", 32000),
            "model_type": "llm",
        }]

    return allowed


def get_cc_default_model() -> dict[str, Any]:
    """Return the default model info from CC config."""
    cfg = _load_cc_config()
    model_cfg = cfg.get("model", {})
    return {
        "provider": model_cfg.get("provider", "ollama"),
        "model": model_cfg.get("model", "llama3.2"),
        "temperature": model_cfg.get("temperature", 0.7),
        "max_tokens": model_cfg.get("max_tokens", 32000),
        "api_key": model_cfg.get("api_key", ""),
        "base_url": model_cfg.get("base_url", ""),
    }


_PROVIDER_ALIASES = {
    "chatgpt": "openai",
    "claude": "anthropic",
    "google": "gemini",
    "grok": "xai",
}


def _normalize_provider(name: str) -> str:
    return _PROVIDER_ALIASES.get(name.lower().strip(), name.lower().strip())


def _resolve_api_key(provider: str) -> str:
    """Resolve API key for a provider: env → .env → CC provider_keys → CC model.api_key."""
    normalized = _normalize_provider(provider)

    # 1. Environment variables.
    env_keys: dict[str, list[str]] = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "xai": ["XAI_API_KEY"],
    }
    for k in env_keys.get(normalized, []):
        val = os.getenv(k, "").strip()
        if val:
            return val

    # 2. CC provider_keys from config.
    cfg = _load_cc_config()
    pk = cfg.get("provider_keys", {})
    val = str(pk.get(normalized, "") or "").strip()
    if val:
        return val

    # 3. Fallback to CC model.api_key (may be for a different provider).
    model_cfg = cfg.get("model", {})
    model_provider = _normalize_provider(model_cfg.get("provider", ""))
    if normalized == model_provider:
        val = str(model_cfg.get("api_key", "") or "").strip()
        if val:
            return val

    return ""


def _resolve_extra_headers(provider: str) -> dict[str, str] | None:
    """Resolve extra headers for a provider from CC config."""
    normalized = _normalize_provider(provider)
    cfg = _load_cc_config()
    pk = cfg.get("provider_keys", {})
    raw = pk.get(f"{normalized}_headers", []) or []
    if not raw:
        return None
    result: dict[str, str] = {}
    for tag in raw:
        if isinstance(tag, str) and ":" in tag:
            k, v = tag.split(":", 1)
            result[k.strip()] = v.strip()
    return result or None


def _resolve_model_params(model_id: str) -> dict[str, Any]:
    """Resolve full LLM call parameters for a model ID from CC allowed list.

    Returns dict with: provider, model, api_key, base_url, temperature,
    max_tokens, extra_headers.
    """
    cc_default = get_cc_default_model()

    if not model_id:
        # Use CC default model.
        provider = cc_default["provider"]
        normalized = _normalize_provider(provider)
        headers = _resolve_extra_headers(provider)
        return {
            "provider": provider,
            "model": cc_default["model"],
            "api_key": "" if headers else _resolve_api_key(provider),
            "base_url": cc_default.get("base_url", ""),
            "temperature": cc_default["temperature"],
            "max_tokens": cc_default["max_tokens"],
            "extra_headers": headers,
        }

    # Look up in allowed list.
    allowed = get_cc_models()
    entry: dict[str, Any] | None = None
    for m in allowed:
        if m.get("id") == model_id:
            entry = m
            break

    if entry is None:
        # Try matching by provider/model string.
        for m in allowed:
            full = f"{m.get('provider', '')}/{m.get('model', '')}"
            if full == model_id or m.get("model") == model_id:
                entry = m
                break

    if entry is None:
        # Not in allowed list — try parsing as provider/model directly.
        if "/" in model_id:
            provider, model = model_id.split("/", 1)
        else:
            provider = cc_default["provider"]
            model = model_id
        headers = _resolve_extra_headers(provider)
        return {
            "provider": provider,
            "model": model,
            "api_key": "" if headers else _resolve_api_key(provider),
            "base_url": "",
            "temperature": cc_default["temperature"],
            "max_tokens": cc_default["max_tokens"],
            "extra_headers": headers,
        }

    provider = entry.get("provider", cc_default["provider"])
    model = entry.get("model", cc_default["model"])
    temperature = entry.get("temperature")
    if temperature is None:
        temperature = cc_default["temperature"]
    max_tokens = entry.get("max_tokens")
    if max_tokens is None:
        max_tokens = cc_default["max_tokens"]
    base_url = entry.get("base_url", "")
    headers = _resolve_extra_headers(provider)

    return {
        "provider": provider,
        "model": model,
        "api_key": "" if headers else _resolve_api_key(provider),
        "base_url": base_url,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "extra_headers": headers,
    }


# ── CC tools & instructions context ───────────────────────────

# Tool descriptions matching CC's agent_context_mixin._TOOL_PROMPT_DESCRIPTIONS
_CC_TOOL_DESCRIPTIONS: dict[str, str] = {
    "shell": "Execute shell commands in the terminal",
    "read": "Read file contents from the filesystem",
    "write": "Write content to files",
    "edit": "Modify existing files (find-and-replace)",
    "glob": "Find files by pattern",
    "web_fetch": "Fetch a URL and return clean readable TEXT (never raw HTML)",
    "web_get": "Fetch a URL and return raw HTML source (only for scraping/DOM inspection)",
    "web_search": "Search the web for up-to-date information",
    "pdf_extract": "Extract a .pdf file into markdown",
    "docx_extract": "Extract a .docx file into markdown",
    "xlsx_extract": "Extract a .xlsx file into markdown tables",
    "pptx_extract": "Extract a .pptx file into markdown",
    "pocket_tts": "Convert text to speech audio (MP3)",
    "image_gen": "Generate images from text prompts",
    "image_ocr": "OCR — extract text from images",
    "image_vision": "Vision analysis on images (describe, analyze)",
    "send_mail": "Send emails via SMTP (to, cc, bcc, subject, body, attachments)",
    "gws": "Google Workspace: Drive, Docs, Calendar, Gmail",
    "datastore": "Persistent relational data tables (create, query, insert, update)",
    "personality": "Read or update the agent personality profile",
    "browser": "Headless browser for web app interaction (login, forms, dynamic pages)",
    "direct_api": "Register and execute HTTP API endpoints",
    "termux": "Android device: photo, battery, GPS, torch",
    "summarize_files": "Summarize multiple files/documents in a folder (handles PDF/DOCX/XLSX/PPTX)",
    "desktop_action": "Control the desktop: click, type, scroll, keys, open apps",
    "screen_capture": "Capture screenshots of the desktop",
    "clipboard": "Read or write the system clipboard",
    "todo": "Manage to-do items",
    "contacts": "Manage contacts/address book",
    "scripts": "Manage saved scripts",
    "apis": "Manage saved API definitions",
    "playbooks": "Manage playbooks (multi-step workflows)",
}


def get_cc_enabled_tools() -> list[str]:
    """Return the list of enabled tool names from CC config."""
    cfg = _load_cc_config()
    tools_cfg = cfg.get("tools", {})
    return tools_cfg.get("enabled", list(_CC_TOOL_DESCRIPTIONS.keys()))


def _build_tools_context() -> str:
    """Build a tools description block for the decomposer LLM context."""
    enabled = get_cc_enabled_tools()
    lines = []
    for name in enabled:
        desc = _CC_TOOL_DESCRIPTIONS.get(name, "")
        if desc:
            lines.append(f"  - {name}: {desc}")
    if not lines:
        return ""
    return (
        "The executing agents (Captain Claw instances) have these tools available:\n"
        + "\n".join(lines)
    )


def _build_instructions_context() -> str:
    """Build agent instructions context for task planning.

    These are CRITICAL rules that the decomposer must understand so it
    creates tasks that CC agents can actually execute correctly.
    """
    return (
        "IMPORTANT — Agent execution rules the tasks MUST respect:\n"
        "1. Agents respond with plain text or markdown. They NEVER produce raw HTML "
        "in their responses. If a task needs HTML output, instruct the agent to use "
        "the `write` tool to save HTML to a file — the response itself must remain text.\n"
        "2. For web content retrieval, agents use `web_fetch` (clean text) or `web_search` "
        "(search). They must NOT write Python scripts to fetch web pages.\n"
        "3. For file operations, agents use `read`, `write`, `edit`, `glob` tools directly.\n"
        "4. For document processing (PDF, DOCX, XLSX, PPTX), agents use the corresponding "
        "extract tool. For multiple files in a folder, use `summarize_files`.\n"
        "5. Agents can execute shell commands via `shell`, but prefer direct tool calls.\n"
        "6. For charts/visualizations, agents generate self-contained HTML files "
        "(Chart.js, D3.js, Plotly.js) saved to disk — NOT Python matplotlib.\n"
        "7. When a task requires data from a predecessor task, the agent receives it "
        "as input context automatically — no need to instruct file passing.\n"
        "8. Each task description should be a clear instruction the agent can act on "
        "immediately using its available tools. Be specific about what tool to use "
        "when it matters (e.g., 'use web_fetch to retrieve...' or 'use web_search to find...').\n"
        "9. Agents can browse interactive web apps via the `browser` tool (login, forms, "
        "dynamic content). Use this when web_fetch is insufficient.\n"
        "10. NEVER ask the agent to return HTML in its response. All markup must be "
        "written to files using the `write` tool.\n"
        "11. This is an AUTOMATED workflow — agents execute autonomously without human "
        "interaction. Task descriptions must NOT require the agent to ask questions, "
        "present options (Option A / Option B), or seek confirmation. Each task must "
        "be self-contained with enough detail for the agent to make all decisions "
        "independently and deliver a complete result.\n"
        "12. Do NOT instruct agents to run tests, write unit tests, optimize, refactor, "
        "or do second/review passes. Focus only on producing the core deliverables.\n"
        "13. CRITICAL FILE PATHS: When a task specifies files to create, the agent MUST use "
        "the EXACT relative paths from the project file tree (e.g., `tetris-game/css/variables.css`, "
        "NOT just `css/variables.css` or `variables.css`). The full path including the project "
        "root folder is essential for proper file organization. Instruct agents to use the "
        "`write` tool with the complete relative path.\n"
        "14. Agents must NOT create helper scripts, test runners, or temporary processing "
        "files. Focus only on the deliverable files specified in the task."
    )


# ── Decomposer ────────────────────────────────────────────────


class DecompositionResult:
    """Result of decomposing a task into subtasks."""

    def __init__(
        self,
        tasks: list[SwarmTask],
        edges: list[SwarmEdge],
        reasoning: str = "",
        file_tree: str = "",
    ) -> None:
        self.tasks = tasks
        self.edges = edges
        self.reasoning = reasoning
        self.file_tree = file_tree


class TaskDecomposer:
    """Handles task rephrasing and decomposition via LLM.

    Uses Captain Claw's config for model selection and API keys.
    """

    async def _call_llm(
        self, system_msg: str, user_msg: str,
        max_tokens: int = 2000, model_id: str = "",
    ) -> str:
        """Make a single LLM completion call via litellm.

        Args:
            model_id: Model ID from CC allowed list, or provider/model string.
                      Empty = use CC default model.
        """
        from litellm import acompletion

        params = _resolve_model_params(model_id)
        provider = _normalize_provider(params["provider"])
        model = params["model"]

        # Build litellm model string (provider/model format).
        if provider == "ollama":
            litellm_model = f"ollama/{model}"
        elif provider in {"openai", "anthropic", "gemini", "xai"}:
            litellm_model = f"{provider}/{model}"
        else:
            litellm_model = f"{provider}/{model}"

        # Normalize temperature for model constraints (e.g. GPT-5 only accepts 1).
        temperature = params["temperature"]
        base_model = model.split("/")[-1].lower() if "/" in model else model.lower()
        if provider == "openai" and base_model.startswith("gpt-5"):
            temperature = 1.0

        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 120,
        }

        api_key = params.get("api_key")
        if api_key:
            kwargs["api_key"] = api_key

        base_url = params.get("base_url")
        if base_url:
            kwargs["api_base"] = base_url

        extra_headers = params.get("extra_headers")
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        log.info("LLM call: model=%s provider=%s", litellm_model, provider)
        response = await acompletion(**kwargs)
        content = response.choices[0].message.content or ""
        return content.strip()

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences from LLM output."""
        import re
        text = text.strip()
        # Remove opening ```json / ```JSON / ``` and closing ```.
        text = re.sub(r"^```\w*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    # ── Rephrase ──────────────────────────────────────────────

    async def rephrase(self, task: str, model: str = "") -> str:
        """Rephrase a complex task for clarity and parallel agent execution.

        Returns the rephrased task text.
        """
        tools_ctx = _build_tools_context()
        instructions_ctx = _build_instructions_context()

        system_msg = (
            "You are an expert task architect for a multi-agent orchestration system. "
            "Each agent is a Captain Claw instance with specific tools and capabilities.\n\n"
            f"{tools_ctx}\n\n"
            f"{instructions_ctx}\n\n"
            "Your job is to rephrase complex tasks so they are:\n"
            "1. Clear and unambiguous — no room for misinterpretation.\n"
            "2. Structured for parallel processing — identify parts that can run concurrently.\n"
            "3. Explicit about inputs, outputs, and success criteria for each logical step.\n"
            "4. Free of implicit assumptions — make all requirements explicit.\n"
            "5. Aware of which tools the agents should use for each step.\n"
            "6. All agent responses must be plain text/markdown — NEVER request HTML output "
            "in responses. If HTML is needed, instruct the agent to write it to a file.\n\n"
            "IMPORTANT constraints:\n"
            "- Do NOT include instructions to run tests, write unit tests, or create test suites.\n"
            "- Do NOT include optimization passes, performance tuning, or refactoring steps.\n"
            "- Do NOT include second passes, review passes, or iterative improvement steps.\n"
            "- Focus ONLY on the core deliverable — producing the requested output files.\n"
            "- If testing/optimization is needed later, it will be handled separately.\n\n"
            "CRITICAL — File Planning:\n"
            "As the LAST section of your rephrased output, include a complete file manifest "
            "listing ALL files that will need to be created, organized by folder structure. "
            "Use this exact format:\n\n"
            "## Files to Create\n"
            "```\n"
            "project-root/\n"
            "├── folder1/\n"
            "│   ├── file1.ext\n"
            "│   └── file2.ext\n"
            "├── folder2/\n"
            "│   └── file3.ext\n"
            "└── file4.ext\n"
            "```\n\n"
            "This file manifest is critical for downstream task decomposition and agent coordination.\n\n"
            "Rephrase the task into a well-structured description. Keep the same intent "
            "but make it suitable for decomposition into independent subtasks that agents "
            "can execute using their available tools. Output ONLY the rephrased task text "
            "(including the file manifest), nothing else."
        )

        result = await self._call_llm(system_msg, task, max_tokens=8000, model_id=model)
        log.info("Task rephrased (%d chars -> %d chars)", len(task), len(result))
        return result

    # ── Decompose ─────────────────────────────────────────────

    async def decompose(
        self,
        task: str,
        available_agents: list[dict] | None = None,
        swarm_id: str = "",
        model: str = "",
    ) -> DecompositionResult:
        """Decompose a task into subtasks with dependencies.

        Args:
            task: The (preferably rephrased) task description.
            available_agents: List of agent dicts with name, personas, expertise info.
            swarm_id: The swarm these tasks belong to.

        Returns:
            DecompositionResult with tasks, edges, and reasoning.
        """
        agent_context = ""
        if available_agents:
            agent_lines = []
            for agent in available_agents:
                personas = agent.get("personas", [])
                persona_info = ", ".join(
                    f"{p.get('name', '?')} ({', '.join(p.get('expertise_tags', []))})"
                    for p in personas
                ) if personas else "no personas"
                agent_lines.append(f"- {agent.get('name', '?')}: {persona_info}")
            agent_context = (
                "\n\nAvailable agents and their personas:\n"
                + "\n".join(agent_lines)
                + "\n\nWhen suggesting a persona for each task, pick from the above list. "
                "If no persona is a good fit, leave suggested_persona empty."
            )

        tools_ctx = _build_tools_context()
        instructions_ctx = _build_instructions_context()

        system_msg = (
            "You are a task decomposition engine for a multi-agent orchestration system. "
            "Each executing agent is a Captain Claw instance with specific tools. "
            "Break down a complex task into smaller, independent subtasks that form a DAG "
            "(Directed Acyclic Graph).\n\n"
            f"{tools_ctx}\n\n"
            f"{instructions_ctx}\n\n"
            "Decomposition rules:\n"
            "1. Each subtask must be self-contained and actionable by a single agent using the tools above.\n"
            "2. Identify dependencies: which tasks must complete before others can start.\n"
            "3. Maximize parallelism: tasks without dependencies should be independent.\n"
            "4. Each task needs a clear name and a concise description (instruction for the agent).\n"
            "5. If a task prepares data for another, mark the dependency explicitly.\n"
            "6. In task descriptions, mention which tools the agent should use when relevant.\n"
            "7. NEVER instruct an agent to produce HTML in its response. If HTML output is needed, "
            "tell the agent to write it to a file using the `write` tool.\n"
            "8. Suggest the best persona for each task based on required expertise.\n"
            "9. Create 3-10 subtasks. Merge related work into single tasks rather than splitting too finely.\n"
            "10. Keep descriptions SHORT (2-3 sentences max). The agent will figure out details.\n"
            "11. Do NOT create tasks for running tests, writing unit tests, or creating test suites.\n"
            "12. Do NOT create optimization, performance tuning, or refactoring tasks.\n"
            "13. Do NOT create second-pass, review, or iterative improvement tasks.\n"
            "14. Focus ONLY on tasks that directly produce the requested deliverables.\n\n"
            "CRITICAL: Respond ONLY with valid JSON. No markdown fences, no ```json wrapper. "
            "Keep the total response compact.\n\n"
            "Exact format:\n"
            "{\n"
            '  "reasoning": "Brief explanation of decomposition strategy",\n'
            '  "file_tree": "Complete folder/file tree of ALL files to be created across all tasks '
            '(use tree format with ├── └── │ characters, one file per line)",\n'
            '  "tasks": [\n'
            "    {\n"
            '      "id": "t1",\n'
            '      "name": "Short task name",\n'
            '      "description": "Detailed agent instruction — what to do, which tools to use, expected output format",\n'
            '      "files": ["list of files this task will create (relative paths)"],\n'
            '      "depends_on": [],\n'
            '      "suggested_persona": "persona name or empty",\n'
            '      "priority": 0,\n'
            '      "is_periodic": false,\n'
            '      "estimated_complexity": "low|medium|high"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "The file_tree MUST list every file that any task will create. Each task's files array "
            "MUST list the specific files that task is responsible for creating. Include the file_tree "
            "in every task description so agents know the full project structure."
            + agent_context
        )

        # Retry with increasing token limits on JSON parse failure.
        # Decomposition of complex tasks can exceed 3k tokens easily.
        last_err: Exception | None = None
        for attempt, tokens in enumerate((8000, 16000), start=1):
            raw = await self._call_llm(system_msg, task, max_tokens=tokens, model_id=model)
            try:
                return self._parse_decomposition(raw, swarm_id)
            except ValueError as exc:
                last_err = exc
                log.warning("Decomposition parse failed (attempt %d): %s", attempt, exc)
        raise ValueError(f"Decomposition failed: {last_err}")

    @staticmethod
    def _repair_json(text: str) -> str:
        """Best-effort repair of common LLM JSON issues."""
        import re
        # Find the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        # Remove trailing commas before } or ].
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Remove control characters (except \n, \t) that break JSON.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text

    def _parse_decomposition(self, raw: str, swarm_id: str) -> DecompositionResult:
        """Parse LLM decomposition JSON into SwarmTasks and SwarmEdges."""
        text = self._strip_fences(raw)

        data = None
        attempts = [("stripped", text), ("repaired", self._repair_json(raw))]
        for label, attempt_text in attempts:
            try:
                data = json.loads(attempt_text)
                log.info("Decomposition JSON parsed OK via %s", label)
                break
            except json.JSONDecodeError as exc:
                log.warning("JSON parse failed (%s): %s — text starts: %s", label, exc, attempt_text[:120])
                continue
        if data is None:
            log.warning("Decomposition returned non-JSON: %s", raw[:300])
            raise ValueError("LLM returned invalid JSON for decomposition")

        reasoning = str(data.get("reasoning", ""))
        file_tree = str(data.get("file_tree", ""))
        raw_tasks = data.get("tasks", [])

        if not raw_tasks:
            raise ValueError("Decomposition produced no tasks")

        # Map temporary IDs to real UUIDs.
        id_map: dict[str, str] = {}
        now = _utcnow_iso()
        tasks: list[SwarmTask] = []

        for rt in raw_tasks:
            temp_id = str(rt.get("id", ""))
            real_id = str(uuid.uuid4())
            id_map[temp_id] = real_id

            complexity = str(rt.get("estimated_complexity", "medium")).lower()
            timeout = {"low": 300, "medium": 600, "high": 1200}.get(complexity, 600)

            # Extract per-task files list.
            task_files = rt.get("files", [])
            if not isinstance(task_files, list):
                task_files = []
            task_files = [str(f).strip() for f in task_files if str(f).strip()]

            # Append file tree context to each task description so agents know the full structure.
            description = str(rt.get("description", ""))
            if file_tree:
                description += f"\n\nFull project file structure:\n{file_tree}"
            if task_files:
                description += (
                    f"\n\nFILES YOU MUST CREATE (use these EXACT paths with the write tool):\n"
                    + "\n".join(f"  - {f}" for f in task_files)
                    + "\n\nYou MUST create ALL files listed above. Use the complete relative "
                    "path exactly as shown (including the project root folder). "
                    "Do NOT create any other files."
                )

            meta: dict[str, Any] = {"estimated_complexity": complexity}
            if task_files:
                meta["files"] = task_files

            task = SwarmTask(
                id=real_id,
                swarm_id=swarm_id,
                name=str(rt.get("name", "")),
                description=description,
                priority=int(rt.get("priority", 0)),
                assigned_persona=str(rt.get("suggested_persona", "")),
                timeout_seconds=timeout,
                is_periodic=bool(rt.get("is_periodic", False)),
                created_at=now,
                updated_at=now,
                metadata=meta,
            )
            tasks.append(task)

        # Build edges from depends_on references.
        edges: list[SwarmEdge] = []
        for rt in raw_tasks:
            temp_id = str(rt.get("id", ""))
            to_id = id_map.get(temp_id, "")
            if not to_id:
                continue

            for dep in (rt.get("depends_on") or []):
                dep_str = str(dep)
                from_id = id_map.get(dep_str, "")
                if from_id:
                    edges.append(SwarmEdge(
                        swarm_id=swarm_id,
                        from_task_id=from_id,
                        to_task_id=to_id,
                    ))
                else:
                    log.warning(
                        "Decomposition: task %s depends on unknown task %s",
                        temp_id, dep_str,
                    )

        log.info(
            "Decomposition: %d tasks, %d edges, file_tree=%d chars, reasoning: %s",
            len(tasks), len(edges), len(file_tree), reasoning[:100],
        )
        return DecompositionResult(tasks=tasks, edges=edges, reasoning=reasoning, file_tree=file_tree)

    # ── Agent selection ───────────────────────────────────────

    async def select_agents(
        self,
        tasks: list[SwarmTask],
        available_agents: list[dict],
        model: str = "",
    ) -> dict[str, str]:
        """For each task, pick the best agent+persona from available pool.

        Returns {task_id: persona_name}.
        """
        if not tasks or not available_agents:
            return {}

        task_descriptions = "\n".join(
            f"- Task '{t.name}' (id={t.id}): {t.description[:200]}"
            for t in tasks
        )

        agent_lines = []
        for agent in available_agents:
            for p in agent.get("personas", []):
                tags = ", ".join(p.get("expertise_tags", []))
                desc = p.get("description", "")
                agent_lines.append(
                    f"- {p.get('name', '?')} on {agent.get('name', '?')}: "
                    f"{desc[:100]} (expertise: {tags})"
                )

        system_msg = (
            "You are an agent assignment engine. For each task, select the best "
            "persona from the available list based on expertise match.\n\n"
            "Respond ONLY with valid JSON mapping task IDs to persona names:\n"
            '{"task_id_1": "persona_name", "task_id_2": "persona_name", ...}\n'
            "If no persona fits a task, map it to an empty string."
        )

        user_msg = f"Tasks:\n{task_descriptions}\n\nAvailable personas:\n" + "\n".join(agent_lines)

        raw = await self._call_llm(system_msg, user_msg, max_tokens=1000, model_id=model)
        text = self._strip_fences(raw)

        try:
            result = json.loads(text)
            return {str(k): str(v) for k, v in result.items()}
        except (json.JSONDecodeError, AttributeError):
            log.warning("Agent selection returned non-JSON: %s", raw[:200])
            return {}
