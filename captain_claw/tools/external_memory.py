"""external_memory — the person's external memory (Membase, …), shared by all of their agents.

It lives in Flight Deck (Connections → External memory): Flight Deck holds the
credentials and talks to the service; this tool asks
``/fd/external-memory/agent/*`` (:mod:`captain_claw.external_memory_client`).

Each provider has its own actions and rules, so this tool has no fixed schema:
its ``action`` enum, its parameters and its description (with the rules) are
built from the manifest Flight Deck sends for the connected provider. Every
provider offers ``remember`` and ``search``; Membase adds ``profile``,
``add_document``, ``forget`` (with confirm) and more, as far as the person's key
allows.

Insights, intentions, reflections, intuitions and lessons are copied in
automatically (mirroring) — the tool is for what the agent decides to save or
look up itself.

The registry hides the tool (metadata ``requires_external_memory``) until the
owner has a memory connected; owner-only, like Jev.
"""

from __future__ import annotations

from typing import Any

from captain_claw import external_memory_client as emc
from captain_claw.tools.registry import Tool, ToolResult

_BASE_DESCRIPTION = (
    "The person's external memory, shared by ALL of their agents: save facts worth keeping "
    "(remember) and look things up (search) — about the person, their work, decisions, "
    "preferences, people."
)


def _provider_actions() -> list[dict[str, Any]]:
    actions = emc.manifest().get("actions")
    return [a for a in actions if isinstance(a, dict) and a.get("name")] if isinstance(actions, list) else []


class ExternalMemoryTool(Tool):
    name = "external_memory"
    # Search can take up to a minute on the provider's side after a quiet spell.
    timeout_seconds = 160.0

    @property  # type: ignore[override]
    def description(self) -> str:
        m = emc.manifest()
        if not m.get("connected"):
            return _BASE_DESCRIPTION
        label = str(m.get("label") or m.get("provider") or "external memory")
        lines = [f"{_BASE_DESCRIPTION} Connected: {label}."]
        reaches = [str(r) for r in m.get("reaches") or []]
        if reaches:
            lines.append("It reaches: " + ", ".join(reaches[:12]) + ".")
        lines.append("Actions:")
        for a in _provider_actions():
            params = ", ".join(
                f"{p['name']}{'*' if p.get('required') else ''}"
                for p in a.get("params") or [] if isinstance(p, dict) and p.get("name")
            )
            lines.append(f"- {a['name']}({params}): {a.get('description', '')}")
        rules = [str(r) for r in m.get("rules") or [] if str(r).strip()]
        if rules:
            lines.append("Rules:")
            lines.extend(f"- {r}" for r in rules)
        return "\n".join(lines)

    @property  # type: ignore[override]
    def parameters(self) -> dict[str, Any]:
        actions = _provider_actions()
        names = [str(a["name"]) for a in actions] or ["remember", "search"]
        props: dict[str, Any] = {
            "action": {
                "type": "string",
                "enum": names,
                "description": "What to do (see the tool description for each action).",
            },
        }
        # One flat object: the union of every action's parameters. A name two
        # actions share gets both descriptions. Never a parameter named
        # ``timeout`` — the registry reads that as the call's own timeout.
        for a in actions:
            for p in a.get("params") or []:
                if not isinstance(p, dict) or not p.get("name") or p["name"] in ("action", "timeout"):
                    continue
                name = str(p["name"])
                desc = f"{a['name']}: {p.get('description', '')}".strip()
                if name in props:
                    props[name]["description"] = (props[name].get("description", "") + " | " + desc)[:600]
                    continue
                spec: dict[str, Any] = {"type": p.get("type") or "string", "description": desc}
                if p.get("enum"):
                    spec["enum"] = list(p["enum"])
                props[name] = spec
        if len(props) == 1:  # not connected: the two every provider has
            props.update({
                "content": {"type": "string", "description": "remember: the fact to save."},
                "query": {"type": "string", "description": "search: what to look for."},
            })
        return {"type": "object", "properties": props, "required": ["action"]}

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        for k in [k for k in kwargs if str(k).startswith("_")]:
            kwargs.pop(k)
        action = str(action or "").strip().lower()
        if emc._member_call():
            return ToolResult(success=False, error=emc.MEMBER_REFUSED)
        if not emc.under_flight_deck():
            return ToolResult(success=False, error=emc.NOT_UNDER_FD)
        if not emc.is_connected_cached():
            await emc.refresh(force=True)
        if not emc.manifest().get("connected"):
            return ToolResult(
                success=False,
                error="No external memory is connected — the person connects one in Flight Deck → "
                "Connections → External memory.",
            )
        spec = next((a for a in _provider_actions() if a["name"] == action), None)
        if spec is None:
            offered = ", ".join(a["name"] for a in _provider_actions())
            return ToolResult(success=False, error=f"Unknown action '{action}' — this memory offers: {offered}.")
        wanted = {str(p.get("name")) for p in spec.get("params") or [] if isinstance(p, dict)}
        args = {k: v for k, v in kwargs.items() if k in wanted and v is not None and v != ""}
        missing = [
            str(p["name"]) for p in spec.get("params") or []
            if isinstance(p, dict) and p.get("required") and p.get("name") not in args
        ]
        if missing:
            return ToolResult(success=False, error=f"{action} needs: {', '.join(missing)}")
        try:
            out = await emc.run(action, args)
        except emc.ExternalMemoryError as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, content=str(out.get("text") or "Done."))
