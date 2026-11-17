"""Tool for discovering and consulting peer agents via Flight Deck fleet API.

Provides live peer discovery by querying the Flight Deck /fd/fleet endpoint,
replacing the static peer list pushed at WebSocket connect time.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from captain_claw import mail_authority
from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


def _norm_fd_url(url: str | None) -> str:
    return str(url or "").strip().rstrip("/")


def _is_pinned_fd_url(url: str | None) -> bool:
    """See ``fd_client.is_pinned_flight_deck_url`` (local-host aliases fold)."""
    from captain_claw.fd_client import is_pinned_flight_deck_url

    return is_pinned_flight_deck_url(url)


_UNPINNED_LOGGED: set[str] = set()


def _log_unpinned_fd_url_once(url: str | None, tool: str) -> None:
    n = _norm_fd_url(url)
    if n in _UNPINNED_LOGGED:
        return
    _UNPINNED_LOGGED.add(n)
    log.warning(
        "Not sending this agent's Flight Deck identity to an unpinned URL",
        tool=tool, fd_url=n,
    )


def _fd_agent_headers(fd_url: str) -> dict[str, str]:
    """This agent's credentials for Flight Deck's agent-facing routes at
    ``fd_url`` — none unless that is a pinned FD URL (`_is_pinned_fd_url`);
    built by ``fd_client.agent_identity_headers``.

    * ``X-Agent-Auth`` — the agent's own web_auth token. FD maps it to the
      owner it recorded at spawn, which is how a child spawned by this agent
      belongs to the same user (an unauthenticated spawn is refused otherwise).
    * ``X-Agent-Secret`` — the per-deck agent secret; mandatory under
      FD_LOCKDOWN, where loopback alone no longer authorizes. The configured
      ``google_oauth.flight_deck_secret`` first (as google_oauth_manager
      sends), else the deck secret (FD_AGENT_SHARED_SECRET env, then the
      per-deck file — as tools.basna sends).

    Together they pass FD's agent gates and fetch the owner's Google token, so
    they never go to a URL a websocket client chose (``peer_agents.fd_url``
    lands in session metadata from any socket, public-run ones included).

    Best-effort: a header that can't be resolved is simply left out.
    """
    if not _is_pinned_fd_url(fd_url):
        _log_unpinned_fd_url_once(fd_url, "flight_deck")
        return {}
    from captain_claw.fd_client import agent_identity_headers

    return agent_identity_headers(fd_url)


def _resolve_local_file(kwargs: dict[str, Any], path_arg: str):
    """Resolve a sender-local file path (saved/ workspace or absolute)."""
    from pathlib import Path

    if not path_arg:
        return None
    registry = kwargs.get("_file_registry")
    if registry is not None:
        try:
            physical = registry.resolve(path_arg)
        except Exception:
            physical = None
        if physical and Path(physical).is_file():
            return Path(physical)
    cand = Path(path_arg).expanduser()
    if cand.is_absolute() and cand.is_file():
        return cand
    stripped = path_arg[len("saved/"):] if path_arg.startswith("saved/") else path_arg
    for base in (kwargs.get("_saved_base_path"), kwargs.get("_runtime_base_path")):
        if base:
            for rel in (path_arg, stripped):
                p = (Path(base) / rel).resolve()
                if p.is_file():
                    return p
    return None


class FlightDeckTool(Tool):
    name = "flight_deck"
    description = (
        "Discover, communicate with, and spawn peer agents in the Flight Deck environment. "
        "Actions: 'list_agents' to discover peers, 'consult' for quick synchronous Q&A, "
        "'delegate' for tasks the peer should do independently, "
        "'spawn_agent' to create a new agent in the fleet (it becomes your child), "
        "'report' to send the agent you sit under (your parent) a report, "
        "'ask' / 'task' to ask, or give a task to, several agents at once by how they're related to you "
        "(to='parent', 'children', 'siblings', 'family', 'group:<name>') or one by name. "
        "Agents on OTHER Flight Decks (connected through BotPort) appear in list_agents as "
        "'Agent@deck' — consult/delegate them by that full name, exactly like local peers. "
        "IMPORTANT: When the user says 'delegate', or the task is large/long-running "
        "(scraping, research, analysis, file creation), ALWAYS use action='delegate'. "
        "Only use 'consult' for quick questions where you need the answer immediately to continue."
    )
    timeout_seconds = 600.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_agents", "consult", "delegate", "spawn_agent", "report", "ask", "task"],
                "description": (
                    "'list_agents' — get all agents currently in the fleet. "
                    "'consult' — ONLY for quick questions where you need the answer right now to continue your work (synchronous, blocks until response). "
                    "'delegate' — for ANY task the peer should handle: research, scraping, analysis, summarization, file creation, etc. "
                    "You send the task and immediately free yourself. The peer works independently and delivers results back to you when done. "
                    "'spawn_agent' — create a new agent in the fleet. Provide agent_name (required), and optionally "
                    "an `archetype` (e.g. 'fact-checker' or 'fact-checker@reason') to base the new agent on a "
                    "library archetype (its role, tools, cognitive mode, and tier→model). "
                    "You can also pass provider/model/api_key/base_url/description/tools overrides, and an "
                    "`env` map ({\"BRAVE_API_KEY\": \"...\"}) to give the new agent extra credentials, via the "
                    "message field as JSON. Without an archetype it uses your own provider/model/key/base_url as defaults "
                    "(your key only together with your own provider and base_url — for another one pass its api_key). "
                    "'ask' — ask the agents in `to` the question in `message`, all at once, and wait for their "
                    "answers (they come back together). 'task' — give the agents in `to` the task in `message`; "
                    "you're free at once, and their results come back to you together when all are done. "
                    "'report' — send your parent (the agent you sit under, named in your family block) a report "
                    "in `message`: when you finish work it asked for, or it should know something. It reaches "
                    "your parent even when it's asleep. "
                    "RULE: If the user says 'delegate' or the task involves work (not just a question), use 'delegate'."
                ),
            },
            "agent_name": {
                "type": "string",
                "description": (
                    "Name of the peer agent to consult or delegate to "
                    "(required for 'consult' and 'delegate' actions). "
                    "Use list_agents first to see available agents. For an agent on "
                    "another Flight Deck use its full 'Agent@deck' name."
                ),
            },
            "archetype": {
                "type": "string",
                "description": (
                    "Optional, for 'spawn_agent' only: base the new agent on a library "
                    "archetype. Format 'id' or 'id@tier' (e.g. 'fact-checker' or "
                    "'fact-checker@reason'). The archetype supplies the agent's role, "
                    "tools, cognitive mode, and model tier. The tier's model (with its "
                    "own key and endpoint) is used when the owner's tier set defines "
                    "it; otherwise the new agent runs on your own model. Explicit "
                    "tools/description overrides in the message JSON still take precedence."
                ),
            },
            "to": {
                "type": "string",
                "description": (
                    "For 'ask' and 'task': who — 'parent', 'children', 'siblings', 'family' (all of them), "
                    "'group:<name>' (one of your owner's agent groups), or one agent's name. At most 12 agents."
                ),
            },
            "made_for": {
                "type": "string",
                "description": (
                    "Optional, for 'spawn_agent' only: one line on why you are making "
                    "this agent (e.g. 'drafts follow-ups after deal calls'). The new "
                    "agent is YOUR child: its owner sees it under you, with this line."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The message/task to send to the peer agent "
                    "(required for 'consult', 'delegate' and 'report' actions)."
                ),
            },
            "file": {
                "type": "string",
                "description": (
                    "Optional path to a file to send ALONG WITH the message to the "
                    "peer (for 'consult'/'delegate'). Use this to share an image, "
                    "PDF, or document with another agent. Relative paths resolve "
                    "from your saved/ workspace (e.g. 'showcase/<session>/report.pdf'); "
                    "absolute paths also work. Images are delivered for the peer's "
                    "vision; other files as attachments."
                ),
            },
        },
        "required": ["action"],
    }

    def _get_fd_url(self, **kwargs: Any) -> str:
        """Resolve the Flight Deck URL: the pinned one (a session / agent
        ``fd_url`` a websocket client supplied only wins when it names the same
        deck — see ``fd_client.resolve_flight_deck_url``). Only a pinned URL
        gets this agent's identity (see _fd_agent_headers)."""
        from captain_claw.fd_client import resolve_flight_deck_url

        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        return resolve_flight_deck_url(fd_url)

    async def _list_agents(self, fd_url: str, **kwargs: Any) -> ToolResult:
        """Query /fd/fleet for live agent list."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{fd_url}/fd/fleet", params={"include_remote": "1"},
                                        headers=_fd_agent_headers(fd_url))
                if resp.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Flight Deck returned HTTP {resp.status_code}",
                    )
                agents = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Cannot reach Flight Deck: {e}")

        if not agents:
            return ToolResult(success=True, content="No agents are currently running in the fleet.")

        # Filter out self if possible
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        my_name = metadata.get("session_display_name", "")

        lines = []
        remote_lines = []
        for a in agents:
            name = a.get("name", "?")
            kind = a.get("kind", "?")
            status = a.get("status", "?")
            port = a.get("port", 0)
            desc = a.get("description", "")
            if kind == "remote":
                label = a.get("deck_label") or ""
                where = f"'{a.get('deck', '?')}'" + (f" ({label})" if label else "")
                line = f"- **{name}** [remote · Flight Deck {where} via BotPort]"
                if desc:
                    line += f" — {desc}"
                remote_lines.append(line)
                continue
            marker = " (you)" if my_name and name.lower() == my_name.lower() else ""
            line = f"- **{name}**{marker} [{kind}] — status: {status}, port: {port}"
            if desc:
                line += f" — {desc}"
            lines.append(line)

        content = f"Fleet agents ({len(lines)}):\n" + "\n".join(lines)
        if remote_lines:
            content += (
                f"\n\nAgents on other Flight Decks ({len(remote_lines)}) — consult/delegate "
                f"them by their full Agent@deck name:\n" + "\n".join(remote_lines)
            )
        return ToolResult(success=True, content=content)

    @staticmethod
    def _is_self_target(target: dict, **kwargs: Any) -> bool:
        """True if the resolved target is THIS agent (prevents self-delegation)."""
        if target.get("kind") == "remote":
            return False  # another deck's agent is never this one
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fid = metadata.get("fleet_identity") or {}
        own_port = fid.get("port")
        own_name = str(metadata.get("session_display_name") or fid.get("name") or "").strip().lower()
        tport, tname = target.get("port"), str(target.get("name") or "").strip().lower()
        try:
            if own_port and tport and int(own_port) == int(tport):
                return True
        except (TypeError, ValueError):
            pass
        return bool(own_name and tname and own_name == tname)

    async def _consult(self, fd_url: str, agent_name: str, message: str, **kwargs: Any) -> ToolResult:
        """Consult a peer agent via /fd/fleet lookup + /fd/consult-peer."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")
        from captain_claw.fd_client import refusal_detail

        target, agents, error = await self._resolve_target(fd_url, agent_name, **kwargs)
        if error:
            return ToolResult(success=False, error=error)
        if self._is_self_target(target, **kwargs):
            return ToolResult(success=False, error=(
                f"'{target.get('name', agent_name)}' is YOU. Do NOT consult/delegate "
                f"to yourself — handle the task directly (e.g. describe the image you can see)."
            ))

        host = target.get("host", "localhost")
        port = target.get("port", 0)

        # Check approval requirement
        agent = kwargs.get("_agent")
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}

        # Get source agent name
        source_name = metadata.get("session_display_name", "another agent")
        peer_display = target.get("name", agent_name)

        # Broadcast peer activity if available
        broadcast = getattr(agent, "ws_broadcast", None) if agent else None

        def _emit(activity_type: str, detail: str = "") -> None:
            if not callable(broadcast):
                return
            broadcast({
                "type": "peer_activity",
                "peer_name": peer_display,
                "activity_type": activity_type,
                "detail": detail,
            })

        _emit("connecting", f"Connecting to {peer_display}...")

        log.info("Consulting peer via fleet", target=peer_display, host=host, port=port)

        # Attach a file: Flight Deck uploads it to the target (it holds the
        # target's auth token, which the sending agent doesn't have).
        attach = kwargs.get("_attach_file")
        attach_path = str(attach) if attach is not None else ""
        if attach_path:
            _emit("uploading", f"Sending file to {peer_display}...")

        body: dict[str, Any] = {
            "host": host,
            "port": port,
            "auth": "",
            "message": message,
            "source_name": source_name,
            "timeout": 480.0,
            "attach_path": attach_path,
            # The peer may write email only when both this question
            # and the text that started our turn ask for one.
            "mail_intent_text": mail_authority.narrower_intent(
                message, mail_authority.intent_source_text(agent),
            )[:mail_authority.JOB_TEXT_MAX],
        }
        if target.get("kind") == "remote":
            # Another Flight Deck's agent: FD carries it over the BotPort link.
            body.update({"port": 0, "target_deck": target.get("deck", ""),
                         "target_agent": target.get("slug") or target.get("name", "").rpartition("@")[0]})

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST",
                    f"{fd_url}/fd/consult-peer",
                    headers=_fd_agent_headers(fd_url),
                    json=body,
                ) as resp:
                    if resp.status_code != 200:
                        return ToolResult(success=False, error=await refusal_detail(resp))

                    final_response = ""
                    got_final = False
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "done" in event and event.get("ok"):
                            final_response = event.get("response", "(no response)")
                            got_final = True
                            _emit("done", "Consultation complete")
                            break
                        if event.get("ok") is False:
                            _emit("error", event.get("error", "Failed"))
                            return ToolResult(
                                success=False,
                                error=event.get("error", "Peer consultation failed"),
                            )

                        evt_type = event.get("event", "")
                        data = event.get("data", {})
                        if evt_type == "heartbeat":
                            elapsed = data.get("elapsed", 0)
                            timeout = data.get("timeout", 0)
                            _emit("status", f"Still working... ({elapsed}s / {timeout}s)")
                        elif evt_type == "status":
                            _emit("status", data.get("status", ""))
                        elif evt_type == "thinking":
                            tool = data.get("tool", "")
                            text = data.get("text", "")
                            detail = f"Using {tool}" if tool else text
                            _emit("thinking", detail[:200])
                        elif evt_type == "monitor":
                            tool_name = data.get("tool_name", "")
                            _emit("tool", tool_name)

                    if not got_final:
                        # The stream ended without an answer (FD restarted, a
                        # link dropped): that is a failure, not an empty reply.
                        _emit("error", "Connection closed")
                        return ToolResult(
                            success=False,
                            error=f"The connection to Flight Deck closed before {peer_display} answered. "
                                  f"Try again.",
                        )
                    return ToolResult(
                        success=True,
                        content=f"Response from {peer_display}:\n\n{final_response}",
                    )

        except httpx.TimeoutException:
            _emit("error", "Timed out")
            return ToolResult(success=False, error=f"Timed out waiting for '{agent_name}'.")
        except httpx.ConnectError as e:
            _emit("error", "Connection failed")
            return ToolResult(success=False, error=f"Cannot connect to Flight Deck at {fd_url}.")
        except Exception as e:
            log.error("Fleet consultation failed", error=str(e))
            _emit("error", str(e)[:100])
            return ToolResult(success=False, error=str(e))

    async def _resolve_target(self, fd_url: str, agent_name: str, **kwargs: Any):
        """Look up a target agent from the fleet. Returns (target_dict, agents_list) or raises."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{fd_url}/fd/fleet", params={"include_remote": "1"},
                                        headers=_fd_agent_headers(fd_url))
                if resp.status_code != 200:
                    return None, None, f"Fleet query failed: HTTP {resp.status_code}"
                agents = resp.json()
        except Exception as e:
            return None, None, f"Cannot reach Flight Deck: {e}"

        target, ambiguous = self._match_agent(agents, agent_name)
        if ambiguous:
            return None, agents, (
                f"'{agent_name}' is ambiguous — it runs on several Flight Decks. "
                f"Use one of: {', '.join(ambiguous)}"
            )

        if not target:
            available = ", ".join(a.get("name", "?") for a in agents)
            return None, agents, f"Agent '{agent_name}' not found in fleet. Available: {available}"

        # "asleep": Flight Deck stopped it for being idle and wakes it for this.
        if target.get("status") not in ("running", "asleep"):
            return None, agents, f"Agent '{agent_name}' is not running (status: {target.get('status')})."

        if target.get("kind") != "remote" and not target.get("port"):
            return None, agents, f"No port info for agent '{agent_name}'."

        return target, agents, None

    @staticmethod
    def _match_agent(agents: list[dict], agent_name: str) -> tuple[dict | None, list[str]]:
        """``(target, ambiguous_names)`` for ``agent_name``.

        Exact full name first (local names, or a remote ``Agent@deck``). An
        ``x@deck`` reference only ever matches that deck's agent. Fuzzy
        substring matching — the long-standing local behaviour — never picks a
        remote agent; a bare name reaches a remote agent only when exactly one
        deck has it.
        """
        query = agent_name.strip().lower()
        local = [a for a in agents if a.get("kind") != "remote"]
        remote = [a for a in agents if a.get("kind") == "remote"]

        for a in agents:
            if a.get("name", "").lower() == query:
                return a, []

        def _bare(a: dict) -> str:
            return a.get("name", "").rpartition("@")[0].lower()

        if "@" in query:
            wanted, _, deck = query.rpartition("@")
            for a in remote:
                if a.get("deck", "").lower() == deck and wanted in (_bare(a), a.get("slug", "").lower()):
                    return a, []
            return None, []

        for a in local:
            aname = a.get("name", "").lower()
            if query in aname or aname in query:
                return a, []

        hits = [a for a in remote if query in (_bare(a), a.get("slug", "").lower())]
        if len(hits) == 1:
            return hits[0], []
        if len(hits) > 1:
            return None, [a.get("name", "?") for a in hits]
        return None, []

    async def _delegate(self, fd_url: str, agent_name: str, message: str, **kwargs: Any) -> ToolResult:
        """Delegate a task to a peer agent (fire-and-forget). Results are delivered back as a message."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")
        from captain_claw.fd_client import refusal_detail

        target, agents, error = await self._resolve_target(fd_url, agent_name, **kwargs)
        if error:
            return ToolResult(success=False, error=error)
        if self._is_self_target(target, **kwargs):
            return ToolResult(success=False, error=(
                f"'{target.get('name', agent_name)}' is YOU. Do NOT delegate to yourself — "
                f"handle the task directly (e.g. describe the image you can see)."
            ))

        agent = kwargs.get("_agent")
        session = kwargs.get("_session")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        source_name = metadata.get("session_display_name", "")
        peer_display = target.get("name", agent_name)

        # Find source agent's port — try multiple strategies
        source_port = 0
        source_host = "localhost"

        # Strategy 1: match by session_display_name in fleet
        if source_name:
            for a in agents:
                if a.get("name", "").lower() == source_name.lower():
                    source_port = a.get("port", 0)
                    source_host = a.get("host", "localhost")
                    break

        # Strategy 2: match by config web port in fleet
        if not source_port:
            try:
                from captain_claw.config import get_config
                cfg_port = get_config().web.port
                if cfg_port:
                    for a in agents:
                        if a.get("port") == cfg_port:
                            source_port = cfg_port
                            source_host = a.get("host", "localhost")
                            if not source_name:
                                source_name = a.get("name", "this agent")
                            break
                # Strategy 3: use config port directly as last resort
                if not source_port:
                    source_port = cfg_port
                    if not source_name:
                        source_name = "this agent"
            except Exception:
                pass

        if not source_port:
            return ToolResult(
                success=False,
                error="Cannot delegate: could not determine own port. Use 'consult' (synchronous) instead.",
            )

        # Detect originating platform (telegram, web, etc.) so delegate results
        # are delivered back to the correct session/channel.
        origin_platform = "web"
        origin_user_id = ""
        origin_chat_id = 0
        if agent and getattr(agent, "_telegram_chat_id", 0) and (
            hasattr(agent, "_user_id") or hasattr(agent, "_telegram_user_id")
        ):
            origin_platform = "telegram"
            # The main agent answering its owner's Telegram carries the chat's
            # user apart from _user_id (the persona lookup key).
            origin_user_id = str(getattr(agent, "_telegram_user_id", "") or getattr(agent, "_user_id", ""))
            origin_chat_id = int(getattr(agent, "_telegram_chat_id", 0))

        log.info("Delegating task to peer", target=peer_display, source=source_name,
                 target_port=target.get("port"), source_port=source_port,
                 origin_platform=origin_platform)

        # Attach a file: Flight Deck uploads it to the target on our behalf.
        attach = kwargs.get("_attach_file")
        attach_path = str(attach) if attach is not None else ""

        body: dict[str, Any] = {
            "target_host": target.get("host", "localhost"),
            "target_port": target.get("port"),
            "target_name": peer_display,
            "source_host": "localhost",
            "source_port": source_port,
            "source_name": source_name,
            "message": message,
            "timeout": 600.0,
            "origin_platform": origin_platform,
            "origin_user_id": origin_user_id,
            "origin_chat_id": origin_chat_id,
            "attach_path": attach_path,
            # As for consult: the narrower of the task and our turn's text.
            "mail_intent_text": mail_authority.narrower_intent(
                message, mail_authority.intent_source_text(agent),
            )[:mail_authority.JOB_TEXT_MAX],
        }
        if target.get("kind") == "remote":
            body.update({"target_port": 0, "target_deck": target.get("deck", ""),
                         "target_agent": target.get("slug") or target.get("name", "").rpartition("@")[0]})

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{fd_url}/fd/delegate-peer",
                    headers=_fd_agent_headers(fd_url),
                    json=body,
                )
                if resp.status_code != 200:
                    return ToolResult(success=False, error=await refusal_detail(resp))
                result = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to delegate: {e}")

        if isinstance(result, dict) and result.get("ok") is False:
            return ToolResult(success=False, error=str(result.get("message") or "Delegation refused"))

        return ToolResult(
            success=True,
            content=(
                f"Task delegated to **{peer_display}**. The task is now {peer_display}'s responsibility.\n\n"
                f"IMPORTANT: Do NOT attempt to do this task yourself. Do NOT use your own tools "
                f"(browser, web_search, shell, etc.) to work on the delegated task. "
                f"{peer_display} will deliver results back to you as a message when finished. "
                f"Tell the user the task has been delegated and move on."
            ),
        )

    async def _spawn_agent(self, fd_url: str, agent_name: str, message: str = "", **kwargs: Any) -> ToolResult:
        """Spawn a new agent in the Flight Deck fleet.

        Uses the caller's own provider/model/api_key/base_url as defaults so that
        Old Man can spawn agents that inherit its configuration.
        The *message* field can optionally contain a JSON object with overrides:
        ``{"provider": "...", "model": "...", "api_key": "...", "base_url": "...",
        "description": "...", "tools": [...], "env": {"BRAVE_API_KEY": "..."}}``.
        The new agent inherits the Flight Deck server's environment (minus FD's
        own secrets); use ``env`` (or ``env_vars``) to hand it credentials the
        server doesn't already hold. It belongs to this agent's owner — FD
        resolves that from this agent's own identity (see _fd_agent_headers).
        """
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")

        # Resolve caller's own model config as defaults for the new agent.
        agent = kwargs.get("_agent")
        provider_obj = getattr(agent, "provider", None) if agent else None
        default_provider = str(getattr(provider_obj, "provider", "ollama") or "ollama")
        default_model = str(getattr(provider_obj, "model", "") or "")
        default_api_key = str(getattr(provider_obj, "api_key", "") or "")
        # Inherit the caller's endpoint too — without it, a child that inherits
        # provider=openai (or any custom-endpoint provider) would fall back to the
        # default cloud URL and misroute. An explicit override still wins.
        default_base_url = str(getattr(provider_obj, "base_url", "") or "")

        # Parse optional overrides from message.
        overrides: dict[str, Any] = {}
        if message.strip().startswith("{"):
            try:
                overrides = json.loads(message)
            except json.JSONDecodeError:
                pass

        # The child belongs to THIS agent's owner: FD resolves it from our
        # X-Agent-Auth (see _fd_agent_headers). The hint only has to agree with
        # it — FD never takes an owner from the body.
        import os
        owner_hint = os.environ.get("FD_OWNER_ID", "")

        # Optional archetype selector — `id` or `id@tier` (e.g. "fact-checker@reason"),
        # from the top-level param or the JSON overrides. When set, the FD spawn
        # endpoint folds the archetype's cognitive_mode/tools/role/tier→model into
        # the config; we deliberately DON'T send a tools/description baseline in that
        # case so the archetype's own values win (the server only fills defaults).
        archetype = str(kwargs.get("archetype") or overrides.get("archetype") or "").strip()

        # Optional extra env for the child, as `env` (a {KEY: value} dict, the
        # ergonomic form) or `env_vars` (the raw [{"key","value"}] list the config
        # expects). Both normalise to the list shape; the child ALSO inherits the FD
        # server's own environment, so this is only for keys the server lacks.
        env_vars: list[dict[str, str]] = []
        raw_env = overrides.get("env")
        if isinstance(raw_env, dict):
            env_vars.extend({"key": str(k), "value": str(v)} for k, v in raw_env.items() if str(k).strip())
        raw_env_list = overrides.get("env_vars")
        if isinstance(raw_env_list, list):
            for ev in raw_env_list:
                if isinstance(ev, dict) and str(ev.get("key", "")).strip():
                    env_vars.append({"key": str(ev["key"]), "value": str(ev.get("value", ""))})

        # Our key goes with OUR endpoint (provider + base_url): a child that an
        # override moves to another provider must not be routed at our endpoint,
        # and one moved anywhere else must not be handed our key — a blank key
        # lets Flight Deck supply the team's key for that endpoint instead.
        from captain_claw.flight_deck.endpoints import same_endpoint
        from captain_claw.llm import _is_codex_family_model, _normalize_provider_name

        default_provider = _normalize_provider_name(default_provider)
        provider = _normalize_provider_name(str(overrides.get("provider") or default_provider))
        model = str(overrides.get("model") or default_model)
        inherited_key = default_api_key
        if provider != default_provider:
            default_api_key = default_base_url = ""
        elif "base_url" in overrides:
            if same_endpoint(provider, overrides["base_url"], default_provider, default_base_url):
                overrides = {**overrides, "base_url": default_base_url}  # ours, however it was spelled
            else:
                default_api_key = ""
        elif provider == "openai" and _is_codex_family_model(model) != _is_codex_family_model(default_model):
            # GPT-5 / Codex models sign in through ChatGPT, the others with a
            # key: another model across that line is another place too.
            default_api_key = default_base_url = ""
        key_withheld = bool(inherited_key) and not default_api_key and "api_key" not in overrides

        spawn_body: dict[str, Any] = {
            "name": agent_name,
            # Inherit the caller's working provider/model/key/base_url as a baseline
            # so the child always has a usable, correctly-routed model even if an
            # archetype tier doesn't resolve; explicit overrides (and an archetype's
            # tier) may still override these server-side.
            "provider": provider,
            "model": model,
            "provider_api_key": overrides.get("api_key", default_api_key),
            "base_url": overrides.get("base_url", default_base_url),
            "web_enabled": True,
            "web_port": overrides.get("web_port", 0),  # 0 = auto-assign
            "owner_hint": owner_hint,
        }
        if env_vars:
            spawn_body["env_vars"] = env_vars
        # Why we make it: FD records the new agent as our child (agent_family).
        made_for = " ".join(str(kwargs.get("made_for") or overrides.get("made_for") or "").split())[:200]
        if made_for:
            spawn_body["made_for"] = made_for
        if archetype:
            spawn_body["archetype"] = archetype
            if "tools" in overrides:
                spawn_body["tools"] = overrides["tools"]
            if "description" in overrides:
                spawn_body["description"] = overrides["description"]
        else:
            spawn_body["description"] = overrides.get("description", "Agent spawned by Old Man")
            spawn_body["tools"] = overrides.get("tools", [
                "shell", "read", "write", "glob", "edit",
                "web_fetch", "web_search", "browser",
                "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
                "scripts", "playbooks", "personality", "flight_deck",
            ])

        # Try process spawn first (no Docker needed), fall back to docker.
        headers = _fd_agent_headers(fd_url)
        last_error = ""
        for endpoint in ["/fd/spawn-process", "/fd/spawn"]:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(f"{fd_url}{endpoint}", json=spawn_body, headers=headers)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("ok"):
                            arch_line = f"Archetype: {archetype}\n" if archetype else ""
                            key_line = (
                                "Note: your own API key was not passed along — the new agent runs on "
                                "another provider or endpoint than you. It uses the team's key for "
                                "that endpoint if Flight Deck has one, else the key for this provider in "
                                "Flight Deck's own environment; pass \"api_key\" in the message JSON to "
                                "choose the key (any placeholder for a local server that needs none).\n"
                            ) if key_withheld and not archetype else ""
                            return ToolResult(
                                success=True,
                                content=(
                                    f"Agent **{agent_name}** spawned successfully.\n"
                                    f"{arch_line}"
                                    f"Provider: {spawn_body['provider']}, Model: {spawn_body['model']}\n"
                                    f"{key_line}"
                                    f"The agent will be available in the fleet shortly. "
                                    f"Use list_agents to check when it's running."
                                ),
                            )
                        else:
                            last_error = f"{endpoint} returned 200 but ok=false: {result}"
                    # If process spawn fails with 400 (already exists) or 409 (a name
                    # that's taken, or past the family limits: too deep, too many
                    # agents under us), don't retry with docker
                    elif resp.status_code == 400 or (resp.status_code == 409 and endpoint == "/fd/spawn-process"):
                        detail = resp.json().get("detail", "")
                        return ToolResult(success=False, error=f"Spawn failed: {detail}")
                    # FD couldn't attribute this agent (identity / owner), which
                    # the Docker endpoint would refuse the same way.
                    elif resp.status_code in (401, 403) and endpoint == "/fd/spawn-process":
                        try:
                            detail = resp.json().get("detail", "")
                        except Exception:
                            detail = resp.text[:300]
                        return ToolResult(
                            success=False,
                            error=(
                                f"Flight Deck refused the spawn (HTTP {resp.status_code}: {detail}). "
                                "It could not verify which user this agent belongs to."
                            ),
                        )
                    else:
                        last_error = f"{endpoint} returned {resp.status_code}: {resp.text[:300]}"
            except Exception as exc:
                last_error = f"{endpoint} exception: {exc}"
                continue

        return ToolResult(success=False, error=f"Failed to spawn agent via Flight Deck. {last_error}")

    async def execute(self, action: str, agent_name: str = "", message: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(
                success=False,
                error="Flight Deck URL not available. This tool requires Flight Deck.",
            )

        # Resolve an optional file to transfer to the peer (consult/delegate).
        file_arg = str(kwargs.get("file") or "").strip()
        if file_arg and action in ("consult", "delegate"):
            resolved = _resolve_local_file(kwargs, file_arg)
            if resolved is None:
                return ToolResult(
                    success=False,
                    error=f"Could not find file '{file_arg}' to send to the peer.",
                )
            kwargs["_attach_file"] = resolved

        if action == "list_agents":
            return await self._list_agents(fd_url, **kwargs)
        elif action == "consult":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for consult action.")
            if not message:
                return ToolResult(success=False, error="message is required for consult action.")
            return await self._consult(fd_url, agent_name, message, **kwargs)
        elif action == "delegate":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for delegate action.")
            if not message:
                return ToolResult(success=False, error="message is required for delegate action.")
            return await self._delegate(fd_url, agent_name, message, **kwargs)
        elif action == "spawn_agent":
            if not agent_name:
                return ToolResult(success=False, error="agent_name is required for spawn_agent action.")
            return await self._spawn_agent(fd_url, agent_name, message, **kwargs)
        elif action == "report":
            if not message.strip():
                return ToolResult(success=False, error="message is required for report action.")
            return await self._report(fd_url, message)
        elif action in ("ask", "task"):
            to = str(kwargs.get("to") or "").strip()
            if not to:
                return ToolResult(success=False, error=f"to is required for {action}: parent, children, siblings, "
                                                       "family, group:<name>, or an agent's name.")
            if not message.strip():
                return ToolResult(success=False, error=f"message is required for {action} action.")
            return await self._family_send(fd_url, action, to, message)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}. Use 'list_agents', 'consult', 'delegate', 'spawn_agent', 'report', 'ask' or 'task'.")

    async def _family_send(self, fd_url: str, mode: str, to: str, message: str) -> ToolResult:
        """Ask, or give a task to, several agents of this one's family (or a group) at once."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")
        from captain_claw.fd_client import refusal_detail

        # An ask waits for every answer (the deck caps each at 8 minutes); a task returns at once.
        wait = 560.0 if mode == "ask" else 30.0
        try:
            async with httpx.AsyncClient(timeout=wait) as client:
                resp = await client.post(f"{fd_url}/fd/family/agent/send", headers=_fd_agent_headers(fd_url),
                                         json={"to": to, "message": message[:8000], "mode": mode,
                                               "mail_intent_text": message[:4000]})
                if resp.status_code != 200:
                    return ToolResult(success=False, error=await refusal_detail(resp))
                result = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to {mode}: {e}")
        return ToolResult(success=True, content=str(result.get("message") or "Sent."))

    async def _report(self, fd_url: str, message: str) -> ToolResult:
        """Send the agent this one sits under (its parent) a report: Flight Deck
        finds the parent from our own identity and gives it a turn on it."""
        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required")
        from captain_claw.fd_client import refusal_detail

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(f"{fd_url}/fd/family/agent/report",
                                         headers=_fd_agent_headers(fd_url), json={"message": message[:8000]})
                if resp.status_code != 200:
                    return ToolResult(success=False, error=await refusal_detail(resp))
                result = resp.json()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to report: {e}")
        return ToolResult(success=True, content=str(result.get("message") or "Report sent."))
