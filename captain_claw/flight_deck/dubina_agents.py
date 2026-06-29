"""Dubina run-targets — dispatch a step to an archetype or a live agent.

The engine's `Generator` is model-agnostic: it asks a *provider* (anything with an
async ``complete(messages) -> response.content``) for text. So a "run-target" is
just a provider-shaped **adapter** whose ``complete`` dispatches the prompt to a
spawned archetype agent or an already-running fleet agent and returns its reply.
That means the whole reasoning stack (generator, self-consistency, critics) runs
**unchanged** over agents — Dubina becomes a general intent runner over the fleet.

Two adapters, both with injectable spawn/send seams so they unit-test with stubs:

* ``make_agent_factory``  — dispatch to a live agent at (port, token). The tier is
  ignored (a running agent has a fixed model); escalation reduces to the sampling +
  fix axes. Lowest cost — no spawn.
* ``ArchetypeRunner``     — spawn a fresh archetype agent at the rung's tier (so the
  ladder's tier climb re-spawns a stronger instance), cache per tier, dispose at end.

Both produce a ``provider_for_tier`` the executor hands to the engine.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from captain_claw.llm import LLMResponse, Message
from captain_claw.logging import get_logger

log = get_logger(__name__)

_DEFAULT_TIMEOUT = 300.0

# A dispatch closure: ``async (prompt) -> reply_text``.
Dispatch = Callable[[str], Awaitable[str]]


def _prompt_from_messages(messages: list[Message]) -> str:
    """Flatten the generator's system+user messages into one prompt for an agent.

    Agents carry their own system prompt, so we fold our instruction (e.g. "end with
    'Answer:'") into the message body rather than relying on a system role.
    """
    return "\n\n".join(m.content for m in messages if m.content).strip()


class DispatchProvider:
    """Provider-shaped adapter: ``complete`` dispatches and wraps the reply text."""

    def __init__(self, dispatch: Dispatch):
        self._dispatch = dispatch

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None) -> LLMResponse:
        text = await self._dispatch(_prompt_from_messages(messages))
        return LLMResponse(content=text or "", finish_reason="stop")


# ── Live agent target ────────────────────────────────────────────────

async def _real_send(port: int, token: str, prompt: str, timeout: float,
                     fleet_instructions: str = "", agent_name: str = "",
                     on_action=None, on_usage=None) -> str:
    from captain_claw.flight_deck.basna_routes import _send_chat_and_collect
    reply, _actions = await _send_chat_and_collect(
        port, token, prompt, timeout,
        fleet_instructions=fleet_instructions, agent_name=agent_name,
        on_action=on_action, on_usage=on_usage,
    )
    return reply


def make_agent_factory(
    port: int, token: str, *, send=_real_send, timeout: float = _DEFAULT_TIMEOUT,
    fleet_instructions: str = "", agent_name: str = "", on_action=None, on_usage=None,
):
    """Build a ``provider_for_tier`` that dispatches to one live agent (tier ignored).

    ``on_action(act)`` / ``on_usage(pt, ct, tt)`` stream the agent's tool calls,
    narration and token usage live (the same monitor events Basna surfaces).
    """
    async def dispatch(prompt: str) -> str:
        return await send(port, token, prompt, timeout,
                          fleet_instructions=fleet_instructions, agent_name=agent_name,
                          on_action=on_action, on_usage=on_usage)

    def factory(tier: str) -> DispatchProvider:
        return DispatchProvider(dispatch)

    return factory


def resolve_agent_port_token(agent_id: str) -> tuple[int, str]:
    """Look up a running agent's (port, token) by slug/name via the fleet registry."""
    from captain_claw.flight_deck.server import _load_process_registry, _resolve_agent_auth
    reg = _load_process_registry()
    entry = reg.get(agent_id)
    if entry is None:
        entry = next((e for s, e in reg.items() if e.get("name") == agent_id), None)
    if not entry or not entry.get("web_port"):
        raise ValueError(f"agent {agent_id!r} not found or has no port")
    port = int(entry["web_port"])
    return port, entry.get("web_auth", "") or _resolve_agent_auth(port)


# ── Archetype target (spawn per tier) ────────────────────────────────

async def _real_spawn(archetype: dict, tier: str, tcfg: dict, request, user,
                      name_suffix: str = "", env_vars: list | None = None) -> tuple[int, str, str]:
    from captain_claw.flight_deck.server import _load_process_registry, spawn_process
    cfg = _build_agent_config(archetype, tier, tcfg, name_suffix=name_suffix, env_vars=env_vars)
    res = await spawn_process(cfg, request, user)
    slug = getattr(res, "slug", "")
    entry = _load_process_registry().get(slug) or {}
    if not getattr(res, "ok", False) or not entry.get("web_port"):
        raise RuntimeError(f"spawn failed: {getattr(res, 'message', 'no port')}")
    return int(entry["web_port"]), entry.get("web_auth", ""), slug


async def _real_stop(slug: str) -> None:
    from captain_claw.flight_deck.server import _do_stop_process
    await _do_stop_process(slug)


def _build_agent_config(archetype: dict, tier: str, tcfg: dict, name_suffix: str = "",
                        env_vars: list | None = None):
    """An ephemeral AgentConfig for an archetype, modeled on Basna's spawn path.

    ``name_suffix`` keeps each run's agent name/slug unique so concurrent or
    back-to-back runs of the same archetype+tier don't collide on spawn.
    ``env_vars`` are the owner's configured tool keys/env (e.g. BRAVE_API_KEY,
    TAVILY_API_KEY) — without them the spawned agent's web_search/etc. tools have
    no credentials. Written into the child's .env by ``_build_env``.
    """
    from captain_claw.flight_deck.server import AgentConfig
    suffix = f"-{name_suffix}" if name_suffix else ""
    base = dict(
        name=f"dubina-{archetype.get('id', 'arch')}-{tier}{suffix}",
        description=f"Dubina ephemeral · {archetype.get('role', '')}",
        cognitive_mode=archetype.get("cognitive_mode", "neutra"),
        tools=archetype.get("tools") or AgentConfig().tools,
        env_vars=list(env_vars or []),
        web_enabled=True, web_port=0,
    )
    model = tcfg.get("model")
    if model:
        return AgentConfig(
            **base, tier="", provider=tcfg.get("provider", ""), model=model,
            provider_api_key=tcfg.get("api_key", "") or "",
            base_url=tcfg.get("base_url", "") or "",
            max_tokens=int(tcfg.get("output_ctx") or 0) or 32768,
            max_context=int(tcfg.get("input_ctx") or 0),
        )
    return AgentConfig(**base, tier=tier)


# ── Shared spawn/dispose seam (used by Dubina AND the Flow engine) ────
#
# The Flow runner's ``archetype:<id>`` selector spawns an ephemeral archetype
# agent for the duration of a run, then disposes it — exactly the lifecycle
# ``ArchetypeRunner`` already implements per tier. Rather than duplicate the
# spawn path, both go through these thin public wrappers over ``_real_spawn`` /
# ``_real_stop`` so the agent config (tools, cognitive_mode, tier→model) stays
# defined in one place.

async def spawn_archetype_agent(
    archetype: dict, tier: str, tcfg: dict, request, user, name_suffix: str = "",
    env_vars: list | None = None,
) -> tuple[int, str, str]:
    """Spawn one ephemeral archetype agent; return ``(port, token, slug)``.

    ``tcfg`` is the resolved tier config (provider/model/key) for ``tier`` — pass
    ``{}`` to let the archetype's own tier resolve via ``AgentConfig(tier=...)``.
    ``env_vars`` are the owner's tool keys/env (BRAVE_API_KEY, …) so the spawned
    agent's tools have credentials.
    """
    return await _real_spawn(archetype, tier, tcfg, request, user,
                             name_suffix=name_suffix, env_vars=env_vars)


async def stop_archetype_agent(slug: str) -> None:
    """Dispose a previously spawned archetype agent (best-effort)."""
    await _real_stop(slug)


class ArchetypeRunner:
    """Spawns a fresh archetype agent per tier, dispatches steps to it, disposes."""

    def __init__(
        self, archetype: dict, request, user, tiers_map: dict | None = None,
        *, spawn=_real_spawn, send=_real_send, stop=_real_stop, timeout: float = _DEFAULT_TIMEOUT,
        on_action=None, on_usage=None, name_suffix: str = "",
    ):
        self.archetype = archetype
        self.request = request
        self.user = user
        self.tiers_map = tiers_map or {}
        self._spawn, self._send, self._stop = spawn, send, stop
        self._timeout = timeout
        self._on_action, self._on_usage = on_action, on_usage
        self._suffix = name_suffix or uuid.uuid4().hex[:6]  # unique per runner → per run
        self._agents: dict[str, tuple[int, str]] = {}  # tier -> (port, token)
        self._slugs: list[str] = []

    async def _ensure(self, tier: str) -> tuple[int, str]:
        if tier not in self._agents:
            port, token, slug = await self._spawn(
                self.archetype, tier, self.tiers_map.get(tier, {}), self.request, self.user,
                name_suffix=self._suffix,
            )
            self._agents[tier] = (port, token)
            if slug:
                self._slugs.append(slug)
        return self._agents[tier]

    def provider_for_tier(self):
        def factory(tier: str) -> DispatchProvider:
            async def dispatch(prompt: str) -> str:
                port, token = await self._ensure(tier)
                return await self._send(
                    port, token, prompt, self._timeout,
                    fleet_instructions=self.archetype.get("fleet_instructions", ""),
                    agent_name=self.archetype.get("role", ""),
                    on_action=self._on_action, on_usage=self._on_usage,
                )
            return DispatchProvider(dispatch)
        return factory

    def agent_slugs(self) -> list[str]:
        """Slugs of the agents spawned so far — to capture their generated files."""
        return list(self._slugs)

    async def dispose(self) -> None:
        for slug in self._slugs:
            try:
                await self._stop(slug)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning("dubina archetype dispose failed", slug=slug)
        self._slugs.clear()
        self._agents.clear()
