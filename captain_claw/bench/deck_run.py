"""One task through Flight Deck — the whole system, not a bare agent.

The run goes the way an outside client's task goes on a real deck:

1. a deck: a private one started here (loopback, sign-in off, its own data
   folder and HOME — nothing of the person's decks or ``~/.captain-claw`` is
   read or written), or a running deck given by ``--deck-url``;
2. Flight Deck spawns a process agent (``/fd/spawn-process``) whose workspace
   is the task folder — with the deck's own setup around it: confinement,
   deck services, idle wake, usage, the default tools of a new agent;
3. the task goes in through the deck's inbound MCP (``send_task``) and the
   runner polls ``get_result`` until the agent answers — the same path Claude
   Code uses to hand an agent work;
4. the transcript is read back from the agent's session, the workspace is
   compared before/after, and everything lands in ``result.json``.

The agent is stopped at the end (on a ``--deck-url`` deck it is also taken
off the agent list — its folder stays, it is never purged) and a private deck
is shut down.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Provider → the environment variables its key may be in, first wins.
PROVIDER_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "chatgpt": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "grok": ("XAI_API_KEY",),
}
# Tool keys handed to the agent when the caller has them (web search).
TOOL_KEY_ENVS = ("BRAVE_API_KEY", "TAVILY_API_KEY", "YDC_API_KEY", "KEENABLE_API_KEY")
# The inbound MCP caps one task at 30 minutes.
MCP_TASK_CAP_SECONDS = 1800.0
_POLL_SECONDS = 2.0


def _say(text: str) -> None:
    print(f"[captain] {text}", file=sys.stderr, flush=True)


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def provider_key(provider: str, env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    for name in PROVIDER_KEY_ENVS.get(provider.strip().lower(), ()):
        if env.get(name):
            return env[name]
    return ""


def agent_env_vars(args: argparse.Namespace, env: dict[str, str] | None = None) -> list[dict[str, str]]:
    """The spawned agent's extra environment: direct writes, the step budget,
    and the caller's web-search keys."""
    env = os.environ if env is None else env
    out: list[dict[str, str]] = []
    if not args.sandboxed_writes:
        out.append({"key": "CLAW_WRITE_DIRECT", "value": "1"})
    if args.max_iterations > 0:
        out.append({"key": "CLAW_MAX_ITERATIONS", "value": str(args.max_iterations)})
    for name in TOOL_KEY_ENVS:
        if env.get(name):
            out.append({"key": name, "value": env[name]})
    return out


def agent_config(args: argparse.Namespace, workspace: Path, name: str) -> dict[str, Any]:
    """The ``AgentConfig`` body for ``/fd/spawn-process``: a new agent as the
    deck makes one, pointed at the task folder."""
    body: dict[str, Any] = {
        "name": name,
        "description": "Benchmark run (captain-claw-once)",
        "provider": args.provider,
        "model": args.model,
        "provider_api_key": provider_key(args.provider),
        "workspace_path": str(workspace),
        "env_vars": agent_env_vars(args),
        "web_enabled": True,
        "web_port": 0,
    }
    if args.base_url:
        body["base_url"] = args.base_url
    if args.archetype:
        body["archetype"] = args.archetype
    return body


# ── a private deck ────────────────────────────────────────────────────


class PrivateDeck:
    """Flight Deck on loopback with sign-in off, in its own folder."""

    def __init__(self, run_dir: Path, port: int = 0):
        self.root = run_dir / "deck"
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = run_dir / "deck.log"

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Our own process may carry these from a standalone run; the deck
        # decides them for its agents.
        for key in ("CLAW_WRITE_DIRECT", "CLAW_MAX_ITERATIONS", "CLAW_CONFINE",
                    "CLAW_CONFINE_HOME", "CLAW_CONFINE_WORKSPACE"):
            env.pop(key, None)
        home = self.root / "home"
        home.mkdir(parents=True, exist_ok=True)
        env.update({
            "FD_AUTH_ENABLED": "false",
            "FD_DATA_DIR": str(self.root / "data"),
            "HOME": str(home),
        })
        return env

    async def start(self, timeout: float = 180.0) -> None:
        import httpx

        self.root.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "captain_claw.flight_deck.server",
             "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=log, stderr=subprocess.STDOUT, env=self.env(), cwd=str(self.root),
            start_new_session=True,
        )
        log.close()
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"Flight Deck exited (code {self.proc.returncode}); see {self.log_path}")
                try:
                    if (await client.get(f"{self.url}/fd/health")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        raise RuntimeError(f"Flight Deck did not come up in {timeout:g}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.forget_keys()

    def forget_keys(self) -> None:
        """The deck wrote the agent's model and tool keys into its ``.env``;
        the run folder outlives the run, the keys shouldn't."""
        for env_file in (self.root / "data").glob("*/.env"):
            try:
                env_file.unlink()
            except OSError:
                pass


# ── talking to a deck ─────────────────────────────────────────────────


class DeckClient:
    def __init__(self, url: str, token: str = ""):
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.url = url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.url, headers=headers, timeout=60.0)
        self._rpc_id = 0

    async def close(self) -> None:
        await self.http.aclose()

    async def spawn_agent(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self.http.post("/fd/spawn-process", json=body, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"spawn failed: {data.get('message') or data}")
        return data

    async def process(self, slug: str) -> dict[str, Any] | None:
        resp = await self.http.get("/fd/processes")
        resp.raise_for_status()
        return next((p for p in resp.json() if p.get("slug") == slug), None)

    async def wait_agent(self, slug: str, timeout: float = 180.0) -> dict[str, Any]:
        """The agent's process entry once its web server answers."""
        import httpx

        deadline = time.monotonic() + timeout
        last = "not listed"
        while time.monotonic() < deadline:
            info = await self.process(slug)
            if info and info.get("status") == "running" and info.get("web_port"):
                try:
                    probe = await self.http.get(
                        f"http://127.0.0.1:{info['web_port']}/api/sessions",
                        params={"token": info.get("web_auth") or ""}, timeout=5.0,
                    )
                    if probe.status_code == 200:
                        return info
                    last = f"web server answered {probe.status_code}"
                except httpx.HTTPError as exc:
                    last = f"web server not up ({type(exc).__name__})"
            elif info:
                last = f"status {info.get('status')}"
            await asyncio.sleep(1.0)
        raise RuntimeError(f"agent {slug} not ready in {timeout:g}s ({last})")

    async def set_effort(self, slug: str, level: str) -> None:
        resp = await self.http.put(f"/fd/agent-reasoning/process/{slug}", json={"default": level})
        resp.raise_for_status()

    async def set_drive(self, slug: str, level: str) -> None:
        resp = await self.http.put(f"/fd/agent-drive/process/{slug}", json={"default": level})
        resp.raise_for_status()

    async def set_model(self, slug: str, provider: str, model: str, api_key: str, base_url: str) -> None:
        body: dict[str, Any] = {"provider": provider, "model": model}
        if api_key:
            body["api_key"] = api_key
        if base_url:
            body["base_url"] = base_url
        resp = await self.http.put(f"/fd/agent-model/process/{slug}", json=body)
        resp.raise_for_status()

    async def restart_agent(self, slug: str) -> None:
        resp = await self.http.post(f"/fd/processes/{slug}/restart", timeout=120.0)
        resp.raise_for_status()

    async def store(self) -> list[dict[str, Any]]:
        """The deck's agent store: its archetype catalog (built-in, published,
        the person's own and shared ones)."""
        resp = await self.http.get("/fd/archetypes", params={"light": "true"})
        resp.raise_for_status()
        return list(resp.json().get("archetypes") or [])

    async def mcp(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._rpc_id += 1
        resp = await self.http.post("/fd/mcp-server", json={
            "jsonrpc": "2.0", "id": self._rpc_id, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        })
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"{tool}: {body['error'].get('message')}")
        result = body.get("result") or {}
        text = "".join(part.get("text", "") for part in result.get("content") or [])
        if result.get("isError"):
            raise RuntimeError(f"{tool}: {text}")
        return json.loads(text) if text else {}

    async def stop_agent(self, slug: str) -> None:
        await self.http.post(f"/fd/processes/{slug}/stop")

    async def forget_agent(self, slug: str) -> None:
        """Off the agent list; its folder stays (never purged)."""
        await self.http.delete(f"/fd/processes/{slug}")

    async def usage(self, info: dict[str, Any]) -> dict[str, Any]:
        """Every LLM call the agent recorded (``/api/usage``): the turn and the
        deck's side calls around it. A fresh agent, so all of it is this run."""
        resp = await self.http.get(
            f"http://127.0.0.1:{info['web_port']}/api/usage",
            params={"period": "all", "token": info.get("web_auth") or ""},
        )
        resp.raise_for_status()
        return resp.json()

    async def transcript(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        """The messages of the agent's most recently updated session."""
        base = f"http://127.0.0.1:{info['web_port']}"
        params = {"token": info.get("web_auth") or ""}
        sessions = (await self.http.get(f"{base}/api/sessions", params=params)).json()
        sessions = [s for s in sessions if s.get("message_count")]
        if not sessions:
            return []
        latest = max(sessions, key=lambda s: str(s.get("updated_at") or ""))
        detail = (await self.http.get(f"{base}/api/sessions/{latest['id']}", params=params)).json()
        return list(detail.get("messages") or [])


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") or {}
    out: dict[str, Any] = {"type": event.get("type"), "at": event.get("at")}
    if event.get("type") == "monitor":
        out["tool"] = data.get("tool_name")
        out["arguments"] = data.get("arguments")
        out["output"] = _short(data.get("output") or "", 2000)
    elif event.get("type") == "status":
        out["status"] = data.get("status")
    return out


def usage_of(raw: dict[str, Any] | None) -> dict[str, int]:
    """Token counts from the agent's ``usage`` frame (``total``, else ``last``)."""
    if not isinstance(raw, dict):
        return {}
    picked = raw.get("total") or raw.get("last") or {}
    return {k: int(v) for k, v in picked.items() if isinstance(v, (int, float))}


_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens",
               "cache_creation_input_tokens", "cache_read_input_tokens")


def recorded_usage(
    data: dict[str, Any], provider: str, model: str,
) -> tuple[dict[str, int], float | None, list[str]]:
    """Totals, estimated USD and the models that actually ran, from the
    agent's ``/api/usage`` answer — each call priced at its own model (side
    calls may run on another one)."""
    from captain_claw.bench.run_once import estimate_cost

    totals = data.get("totals") or {}
    usage = {k: int(totals.get(k) or 0) for k in _USAGE_KEYS}
    usage["llm_calls"] = int(totals.get("total_calls") or 0)
    cost, priced, models = 0.0, False, set()
    for rec in data.get("records") or []:
        rec_provider, rec_model = str(rec.get("provider") or provider), str(rec.get("model") or model)
        models.add(f"{rec_provider}/{rec_model}")
        part = estimate_cost(rec_provider, rec_model, {k: int(rec.get(k) or 0) for k in _USAGE_KEYS})
        if part is not None:
            cost, priced = cost + part, True
    return usage, (round(cost, 6) if priced else None), sorted(models)


# The store's Featured row (flight-deck/src/components/store/storeFeatured.ts).
STORE_FEATURED = ("deep-researcher", "inbox-manager", "data-analyst", "editor-writer", "computer-user")


def store_lines(items: list[dict[str, Any]]) -> list[str]:
    """One line per store agent, the Featured row first, then by family."""
    def order(a: dict[str, Any]) -> tuple:
        aid = str(a.get("id") or "")
        featured = STORE_FEATURED.index(aid) if aid in STORE_FEATURED else len(STORE_FEATURED)
        return (featured, str(a.get("family") or ""), aid)

    lines = []
    for a in sorted(items, key=order):
        aid = str(a.get("id") or "")
        mark = "*" if aid in STORE_FEATURED else " "
        role = _short(a.get("role") or a.get("description") or "", 70)
        lines.append(f"{mark} {aid:30} {str(a.get('family') or ''):28} {role}")
    return lines


async def list_store(args: argparse.Namespace, run_dir: Path) -> int:
    """Print the deck's agent store — the ids ``--archetype`` takes."""
    deck: PrivateDeck | None = None
    try:
        if args.deck_url:
            url = args.deck_url
        else:
            deck = PrivateDeck(run_dir, port=args.deck_port)
            await deck.start()
            url = deck.url
        client = DeckClient(url, token=args.deck_token or os.environ.get("CLAW_DECK_TOKEN", ""))
        try:
            items = await client.store()
        finally:
            await client.close()
    finally:
        if deck is not None:
            deck.stop()
    print("\n".join(store_lines(items)))
    print(f"\n{len(items)} agents (* = Featured in the store). Use one with --archetype <id>.")
    return 0


# ── the run ───────────────────────────────────────────────────────────


async def run_on_deck(args: argparse.Namespace, instruction: str, run_dir: Path) -> int:
    from captain_claw.bench.run_once import (
        build_result_payload,
        diff_snapshots,
        report,
        snapshot,
        tool_counts,
        user_path,
        write_outputs,
    )

    workspace = user_path(args.workspace) if args.workspace else Path.cwd().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    result_path = user_path(args.result_json) if args.result_json else run_dir / "result.json"
    timeout = min(args.timeout, MCP_TASK_CAP_SECONDS) if args.timeout > 0 else MCP_TASK_CAP_SECONDS
    if not args.provider or not args.model:
        print("captain-claw-once: --provider and --model are required on a deck", file=sys.stderr)
        return 2

    started = time.monotonic()
    deck: PrivateDeck | None = None
    client: DeckClient | None = None
    slug, info, task_id = "", None, ""
    answer, error, ok = "", None, False
    usage: dict[str, int] = {}
    cost: float | None = None
    models_used: list[str] = []
    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    before = snapshot(workspace)

    stop_requested = asyncio.Event()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop_requested.set)
    except (NotImplementedError, RuntimeError):
        pass

    try:
        if args.deck_url:
            url = args.deck_url
        else:
            deck = PrivateDeck(run_dir, port=args.deck_port)
            if not args.quiet:
                _say(f"starting a private Flight Deck on {deck.url}")
            await deck.start()
            url = deck.url
        client = DeckClient(url, token=args.deck_token or os.environ.get("CLAW_DECK_TOKEN", ""))

        name = f"bench-{uuid.uuid4().hex[:6]}"
        spawned = await client.spawn_agent(agent_config(args, workspace, name))
        slug = spawned["slug"]
        info = await client.wait_agent(slug)
        spawned_on = (str(info.get("provider") or ""), str(info.get("model") or ""))
        if spawned_on != (args.provider, args.model):
            # A store agent's tier can swap in its own model on the same
            # provider; a benchmark runs the model it was asked for.
            events.append({"type": "note", "text": f"spawned on {'/'.join(spawned_on)}, "
                                                   f"set to {args.provider}/{args.model}"})
            await client.set_model(slug, args.provider, args.model, provider_key(args.provider), args.base_url)
            await client.restart_agent(slug)
            info = await client.wait_agent(slug)
        if args.effort:
            await client.set_effort(slug, args.effort)
        if args.drive:
            await client.set_drive(slug, args.drive)
        if not args.quiet:
            store = f"store agent {args.archetype} · " if args.archetype else ""
            knobs = " · ".join(f"{k} {v}" for k, v in (("effort", args.effort), ("drive", args.drive)) if v)
            _say(f"agent {slug} · {store}{args.provider}/{args.model}{' · ' + knobs if knobs else ''} · "
                 f"port {info['web_port']} · in {workspace}")

        sent = await client.mcp("send_task", {"agent": slug, "task": instruction, "timeout_seconds": timeout})
        task_id = sent["task_id"]
        cursor, deadline = 0, time.monotonic() + timeout + 30
        while True:
            if stop_requested.is_set() or time.monotonic() > deadline:
                await client.mcp("cancel_task", {"task_id": task_id})
                error = "stopped" if stop_requested.is_set() else f"timed out after {timeout:g}s"
                break
            got = await client.mcp("get_result", {"task_id": task_id, "since": cursor})
            for event in got.get("events") or []:
                compact = compact_event(event)
                if compact["type"] in ("monitor", "status"):
                    events.append(compact)
                if compact["type"] == "monitor" and not args.quiet:
                    _say(f"tool {compact['tool']} {_short(compact['arguments'])}")
            cursor = int(got.get("next_cursor") or cursor)
            if got.get("done"):
                status = got.get("status")
                if status == "done":
                    answer, ok = str(got.get("result") or ""), True
                    usage = usage_of(got.get("usage"))
                else:
                    error = str(got.get("error") or status)
                break
            await asyncio.sleep(_POLL_SECONDS)

        try:
            usage, cost, models_used = recorded_usage(await client.usage(info), args.provider, args.model)
        except Exception as exc:  # the MCP's usage frame, if it caught one
            events.append({"type": "note", "text": f"usage not read from the agent: {exc}"})
        try:
            messages = await client.transcript(info)
        except Exception as exc:
            events.append({"type": "note", "text": f"transcript not read: {exc}"})
    except Exception as exc:  # the result file says what went wrong
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None and slug:
            try:
                await client.stop_agent(slug)
                if args.deck_url and not args.keep_agent:
                    await client.forget_agent(slug)
            except Exception:
                pass
        if client is not None:
            await client.close()
        if deck is not None:
            deck.stop()

    counts = tool_counts(messages)
    result = build_result_payload(
        ok=ok, error=error, answer=answer, provider=args.provider, model=args.model,
        effort=args.effort, workspace=workspace, run_dir=run_dir, started=started,
        usage=usage, cost=cost, messages=messages, counts=counts,
        files=diff_snapshots(before, snapshot(workspace)),
        extra={
            "mode": "deck",
            "drive": args.drive or None,
            "store_agent": args.archetype or None,
            "models_used": models_used,
            "deck": {"url": args.deck_url or (deck.url if deck else ""), "private": deck is not None,
                     "agent": slug, "task_id": task_id},
            "events": events,
        },
    )
    write_outputs(args, result, result_path)
    report(args, result, result_path)
    return 0 if ok else 1
