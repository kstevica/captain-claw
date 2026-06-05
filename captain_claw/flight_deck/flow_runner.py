"""FlowRunner — executes a Flow inside Flight Deck, dispatching steps to the pool.

Deterministic loop owned by FD; the work happens on pooled agents:
  * ``tool``  step → direct RPC to an agent's ``/api/tool`` (no LLM), or an
                     FD-internal tool (e.g. face_identify) when ``on: fd``.
  * ``agent`` step → ``/fd/consult-peer`` (scoped prompt; reuses busy-retry).
  * ``branch``     → conditional jump (deterministic).
  * ``emit``       → push to a channel (WhatsApp / log).

Dependencies are injected by the FD server to avoid import cycles.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from captain_claw.logging import get_logger
from captain_claw.flight_deck.flows_store import FlowStore

log = get_logger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.\[\]]+)\s*\}\}")
_VISION_HINTS = ("vision", "image", "multimodal", "minimax", "llava", "vl")
_DEFAULT_MAX_STEPS = 20

# Sentinel goto target that ends the flow (used by branch `goto`/`else`).
_STOP_TARGET = "__stop__"


class _RunControl:
    """In-memory control handle for a live run — pause / resume / stop.

    A run registers one of these while executing; the loop checks it at each
    step boundary. ``resume`` is set while running and cleared to pause;
    ``stopped`` ends the run at the next boundary (optionally after delivering
    ``stop_message`` on the originating channel)."""

    __slots__ = ("resume", "stopped", "stop_message")

    def __init__(self) -> None:
        self.resume = asyncio.Event()
        self.resume.set()  # set = running, cleared = paused
        self.stopped = False
        self.stop_message = ""


_RUN_CONTROL: dict[str, _RunControl] = {}


def request_pause(run_id: str) -> bool:
    """Pause a live run at its next step boundary. False if not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c or c.stopped:
        return False
    c.resume.clear()
    return True


def request_resume(run_id: str) -> bool:
    """Resume a paused run. False if the run is not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c:
        return False
    c.resume.set()
    return True


def request_stop(run_id: str, message: str = "") -> bool:
    """Stop a live run at its next boundary, optionally sending *message* first.
    False if the run is not controllable."""
    c = _RUN_CONTROL.get(run_id)
    if not c:
        return False
    c.stopped = True
    c.stop_message = str(message or "")
    c.resume.set()  # unblock a paused run so it can observe the stop
    return True


def run_is_controllable(run_id: str) -> bool:
    """True while a run is registered and can accept pause/resume/stop."""
    return run_id in _RUN_CONTROL


def _dig(ctx: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'steps.analyze.output' against ctx."""
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur


def _render(value: Any, ctx: dict[str, Any]) -> Any:
    """Substitute {{path}} in strings; recurse into dicts/lists."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            got = _dig(ctx, m.group(1))
            return got if isinstance(got, str) else json.dumps(got, default=str)
        return _TEMPLATE_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _render(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, ctx) for v in value]
    return value


class FlowRunner:
    def __init__(
        self,
        store: FlowStore,
        *,
        get_agents: Callable[[], list[dict[str, Any]]],
        resolve_auth: Callable[[int], str],
        fd_self_base: str,
        fd_tools: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] | None = None,
        whatsapp_send: Callable[[str, str], Awaitable[Any]] | None = None,
        transfer_file: Callable[[str, int, str], Awaitable[tuple[list[str], list[str]]]] | None = None,
    ) -> None:
        self.store = store
        self.get_agents = get_agents
        self.resolve_auth = resolve_auth
        self.fd_self_base = fd_self_base.rstrip("/")
        self.fd_tools = fd_tools or {}        # FD-internal tools (e.g. face_identify)
        self.whatsapp_send = whatsapp_send
        # Uploads a file to a target agent, returning (image_paths, file_paths)
        # ON THE TARGET. Lets the runner verify delivery + use the target's path.
        self.transfer_file = transfer_file

    # ── agent pool selection ───────────────────────────────────────────

    def _select_agent(self, selector: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        sel = (selector or "origin").strip()
        agents = [a for a in self.get_agents() if str(a.get("status", "")).lower() in ("running", "")]
        if sel == "origin" or not sel:
            host = payload.get("origin_host") or "localhost"
            port = payload.get("origin_port")
            name = str(payload.get("origin_name") or "").strip()
            # Resolve to the LIVE registry entry so we get the current port AND
            # the matching auth token. Prefer name (the origin may have drifted
            # ports since the message arrived), then port.
            if name:
                for a in agents:
                    if str(a.get("name", "")).lower() == name.lower():
                        return a
            if port:
                for a in agents:
                    if int(a.get("port") or 0) == int(port):
                        return a
                # Fallback: hand-built target with a resolved token.
                return {"name": name or "origin", "host": host, "port": int(port),
                        "auth": self.resolve_auth(int(port)) if self.resolve_auth else ""}
            return agents[0] if agents else None
        if sel.startswith("name:"):
            want = sel.split(":", 1)[1].strip().lower()
            for a in agents:
                if str(a.get("name", "")).lower() == want:
                    return a
            return None
        if sel.startswith("capability:"):
            cap = sel.split(":", 1)[1].strip().lower()
            hints = _VISION_HINTS if cap in ("vision", "image", "multimodal") else (cap,)
            for a in agents:
                blob = f"{a.get('name','')} {a.get('description','')}".lower()
                if any(h in blob for h in hints):
                    return a
            return None
        # "any"
        return agents[0] if agents else None

    # ── step dispatch ──────────────────────────────────────────────────

    async def _run_tool(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str]:
        """Return (output, agent_label)."""
        tool = str(step.get("tool") or "")
        args = _render(step.get("args") or {}, ctx)
        selector = str(step.get("on") or "origin")

        # FD-internal tool (e.g. face_identify) — runs in-process, no agent.
        if selector == "fd" or tool in self.fd_tools:
            fn = self.fd_tools.get(tool)
            if not fn:
                return f"(no FD-internal tool '{tool}')", "fd"
            return (await fn(args)), "fd"

        agent = self._select_agent(selector, payload)
        if not agent:
            return f"(no agent available for selector '{selector}')", ""
        import httpx
        url = f"http://{agent['host']}:{agent['port']}/api/tool"
        token = self.resolve_auth(int(agent["port"]))
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(url, json={"tool": tool, "args": args, "token": token})
            if resp.status_code != 200:
                return f"(tool {tool} failed: HTTP {resp.status_code} {resp.text[:200]})", agent.get("name", "")
            data = resp.json() or {}
            out = data.get("content") if data.get("success") else f"(error: {data.get('error')})"
            return str(out or ""), agent.get("name", "")
        except Exception as exc:
            return f"(tool {tool} dispatch failed: {exc})", agent.get("name", "")

    async def _run_agent_step(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str]:
        guard = step.get("guardrails") or {}
        deny = guard.get("deny") or []
        attach = _render(str(step.get("attach") or ""), ctx)
        selector = str(step.get("on") or "capability:vision")
        agent = self._select_agent(selector, payload)
        if not agent:
            return f"(no agent for selector '{selector}')", ""

        # When a file/image is attached, UPLOAD it to the target FIRST and use
        # the TARGET-local path. `attach`/{{trigger.image_path}} is the ORIGIN
        # agent's path (where the message arrived) — the target can't read that
        # path, so we transfer the bytes and reference the copy on the target.
        # Verify it actually landed, then render the prompt with the TARGET path.
        target_images: list[str] = []
        target_files: list[str] = []
        render_ctx = ctx
        if attach:
            if not self.transfer_file:
                return "(cannot attach: file transfer unavailable)", agent.get("name", "")
            try:
                target_images, target_files = await self.transfer_file(
                    agent["host"], int(agent["port"]), attach,
                )
            except Exception as exc:
                return f"(failed to send the attachment to {agent.get('name','')}: {exc})", agent.get("name", "")
            if not target_images and not target_files:
                return (f"(attachment was NOT received by {agent.get('name','')} — "
                        f"upload failed or source file missing: {attach})"), agent.get("name", "")
            # Make {{trigger.image_path}} / {{attached_image_path}} resolve to the
            # TARGET copy for THIS step's prompt (origin path is meaningless there).
            _tpath = target_images[0] if target_images else target_files[0]
            render_ctx = dict(ctx)
            render_ctx["trigger"] = {**(ctx.get("trigger") or {}), "image_path": _tpath}
            render_ctx["attached_image_path"] = _tpath

        prompt = _render(str(step.get("prompt") or ""), render_ctx)

        # FD-only mitigation (works even if the target runs older code): when a
        # file/image is attached, prepend a hard instruction to ignore memory and
        # describe ONLY the attachment. The deterministic version (suppressing the
        # target's memory injection) needs the target on current code.
        if attach:
            prompt = (
                "A file/image is attached to THIS message. Answer using ONLY what you "
                "actually see in the attachment. IGNORE everything else in your context "
                "— do NOT use 'Persistent insights from memory', remembered facts, fleet "
                "ports, or earlier descriptions. Describe only the attached content.\n\n"
                + prompt
            )

        # Deterministic tool denials for the target. Start with the step's own
        # deny list; when a file/image is attached, also block shell/scripts/read
        # so the model uses the ATTACHED content instead of operating on the path.
        _deny = list(deny)
        if attach:
            for t in ("shell", "scripts", "read"):
                if t not in _deny:
                    _deny.append(t)
        if deny:
            prompt = (
                f"Constraints: do NOT use these tools this turn: {', '.join(deny)}. "
                f"Do not write or run scripts. Answer directly.\n\n{prompt}"
            )
        import httpx
        body = {
            "host": agent["host"], "port": int(agent["port"]),
            "auth": str(agent.get("auth") or ""),  # token from the same entry as the port
            "message": prompt, "source_name": "FlowEngine", "timeout": 480.0,
            "no_flow": True,         # loop guard
            "no_broadcast": True,    # FlowRunner is the sole deliverer (no channel leak)
            "deny_tools": _deny,
            "image_paths": target_images,   # already on the TARGET (verified)
            "file_paths": target_files,
        }
        final, err = "", ""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream("POST", f"{self.fd_self_base}/fd/consult-peer", json=body) as resp:
                    if resp.status_code != 200:
                        return f"(agent step failed: HTTP {resp.status_code})", agent.get("name", "")
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("done") and evt.get("ok"):
                            final = str(evt.get("response") or "")
                            break
                        if evt.get("ok") is False:
                            err = str(evt.get("error") or "agent error")
                            break
        except Exception as exc:
            err = str(exc)
        return (final or f"(no result: {err})"), agent.get("name", "")

    async def _run_vision_step(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str]:
        """Lean image describe: upload the image to a vision agent and call its
        /api/vision (raw model call) — NO agent loop, memory, tools, or history."""
        image_src = _render(str(step.get("attach") or step.get("image") or "{{trigger.image_path}}"), ctx)
        prompt = _render(str(step.get("prompt") or "Describe this image in detail."), ctx)
        selector = str(step.get("on") or "capability:vision")
        agent = self._select_agent(selector, payload)
        if not agent:
            return f"(no vision agent for selector '{selector}')", ""
        if not image_src:
            return "(no image to describe)", agent.get("name", "")
        if not self.transfer_file:
            return "(file transfer unavailable)", agent.get("name", "")
        # Upload to the target and verify it landed; use the TARGET-local path.
        try:
            imgs, files = await self.transfer_file(agent["host"], int(agent["port"]), image_src)
        except Exception as exc:
            return f"(failed to send image to {agent.get('name','')}: {exc})", agent.get("name", "")
        target = imgs[0] if imgs else (files[0] if files else "")
        if not target:
            return f"(image NOT received by {agent.get('name','')}: {image_src})", agent.get("name", "")
        import httpx
        url = f"http://{agent['host']}:{int(agent['port'])}/api/vision"
        token = str(agent.get("auth") or "")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    url, params=({"token": token} if token else {}),
                    json={"image": target, "prompt": prompt},
                )
            if resp.status_code != 200:
                return f"(vision failed: HTTP {resp.status_code} {resp.text[:200]})", agent.get("name", "")
            data = resp.json() or {}
            out = data.get("content") if data.get("success") else f"(vision error: {data.get('error')})"
            return str(out or "(empty vision result)"), agent.get("name", "")
        except Exception as exc:
            return f"(vision dispatch failed: {exc})", agent.get("name", "")

    async def _deliver(self, payload: dict[str, Any], text: str, *, role: str = "assistant") -> bool:
        """Send a message to the user on the originating channel.

        WhatsApp → whatsapp_send. Agent-handled channels (web/glasses) → push
        into the origin agent's chat UI via /api/chat/push (the flow runs in FD,
        so it can't rely on the agent relaying inside a blocking turn)."""
        text = str(text or "")
        if not text:
            return False
        waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
        if waid and self.whatsapp_send:
            try:
                await self.whatsapp_send(waid, text)
                return True
            except Exception as exc:
                log.warning("flow deliver (whatsapp) failed: %s", exc)
                return False
        agent = self._select_agent("origin", payload)
        if not agent:
            return False
        import httpx
        url = f"http://{agent['host']}:{agent['port']}/api/chat/push"
        token = agent.get("auth") or (self.resolve_auth(int(agent["port"])) if self.resolve_auth else "")
        params = {"token": token} if token else {}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, params=params, json={"text": text, "role": role})
            return resp.status_code == 200
        except Exception as exc:
            log.warning("flow deliver (agent push) failed: %s", exc)
            return False

    async def _run_input_step(
        self, step: dict[str, Any], ctx: dict[str, Any],
        payload: dict[str, Any], flow: dict[str, Any], *, dry: bool = False,
    ) -> tuple[str, str]:
        """Pause the run, prompt the user (naming the flow), and resume with
        their reply as this step's output. Returns (user_text, "")."""
        prompt = _render(str(step.get("prompt") or "Please reply with your input."), ctx)
        flow_name = str(flow.get("name") or "Flow")

        if dry:
            return f"(dry-run: would pause and ask: {prompt})", ""

        waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
        channel = str(payload.get("channel") or "")
        origin_port = int(payload.get("origin_port") or 0)
        # Announce that input is needed — the flow name is required so the user
        # knows which automation is asking.
        announce = f"⏳ *{flow_name}* needs your input:\n\n{prompt}"

        if not await self._deliver(payload, announce):
            # No reachable channel to ask on — fail clearly instead of hanging.
            return "(input step: no channel available to prompt the user)", ""

        from captain_claw.flight_deck import flow_router
        timeout = float(step.get("timeout") or 3600.0)
        key = flow_router.input_key(waid=waid, channel=channel, origin_port=origin_port)
        try:
            text = await flow_router.wait_for_input(key, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"timed out waiting for input ({flow_name})") from exc
        except asyncio.CancelledError as exc:
            raise RuntimeError(f"input wait superseded ({flow_name})") from exc
        return text, ""

    async def _emit(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> str:
        channel = str(step.get("channel") or "log")
        body = _render(str(step.get("body") or "{{steps}}"), ctx)
        # 'same'/'whatsapp'/'glasses'/'web' all deliver on the originating
        # channel (whatsapp_send or agent chat-push); 'log' just records.
        if channel in ("whatsapp", "same", "glasses", "web"):
            if await self._deliver(payload, body):
                return f"(emitted to {channel})"
        return body  # 'log' / fallthrough — captured in the run record

    # ── main loop ──────────────────────────────────────────────────────

    async def run(self, flow: dict[str, Any], payload: dict[str, Any] | None = None, *, dry: bool = False, run_id: str | None = None) -> dict[str, Any]:
        payload = payload or {}
        steps = list(flow.get("steps") or [])
        guard = flow.get("guardrails") or {}
        max_steps = int(guard.get("max_steps", _DEFAULT_MAX_STEPS))
        _now = datetime.now(UTC)
        ctx: dict[str, Any] = {
            "trigger": payload,
            "steps": {},
            "system": {
                "now": _now.isoformat(),
                "date": _now.strftime("%Y-%m-%d"),
                "time": _now.strftime("%H:%M"),
                "agent": str(payload.get("origin_name") or ""),
                "channel": str(payload.get("channel") or ""),
            },
        }
        trace: list[dict[str, Any]] = []

        if dry:
            run_id = ""
        elif not run_id:
            run_id = await self.store.start_run(flow["id"], flow.get("name", ""), payload)
        # Register a control handle so the run can be paused/resumed/stopped.
        ctrl: _RunControl | None = None
        if not dry and run_id:
            ctrl = _RunControl()
            _RUN_CONTROL[run_id] = ctrl
        status = "done"
        error = ""
        final_text = ""
        paused_marked = False
        by_id = {s.get("id"): i for i, s in enumerate(steps)}
        i = 0
        executed = 0
        try:
            while i < len(steps):
                # ── pause / stop control (checked at each step boundary) ──
                if ctrl is not None:
                    if not ctrl.stopped and not ctrl.resume.is_set():
                        if not paused_marked:
                            paused_marked = True
                            try:
                                await self.store.set_run_status(run_id, "paused")
                            except Exception:
                                pass
                        await ctrl.resume.wait()
                        if not ctrl.stopped:
                            paused_marked = False
                            try:
                                await self.store.set_run_status(run_id, "running")
                            except Exception:
                                pass
                    if ctrl.stopped:
                        status = "stopped"
                        if ctrl.stop_message:
                            try:
                                await self._deliver(payload, ctrl.stop_message)
                            except Exception as exc:
                                log.warning("flow stop message delivery failed: %s", exc)
                        break
                if executed >= max_steps:
                    raise RuntimeError(f"max_steps ({max_steps}) exceeded")
                step = steps[i]
                sid = str(step.get("id") or f"step{i}")
                stype = str(step.get("type") or "tool")
                executed += 1
                t0 = time.monotonic()

                if stype == "branch":
                    # Multi-case switch: evaluate each case's condition in order;
                    # first true wins. Falls back to the single when/goto for
                    # legacy flows, and to `else`/`default` when nothing matches.
                    cases = step.get("cases")
                    if not cases:
                        cases = [{"when": step.get("when"), "goto": step.get("goto")}]
                    target = None
                    label = "not taken"
                    for c in cases:
                        if _eval_expr(str(c.get("when") or ""), ctx):
                            target = c.get("goto")
                            label = f"taken→{target}"
                            break
                    if target is None:
                        els = step.get("else") or step.get("default")
                        if els:
                            target = els
                            label = f"else→{target}"
                    out = f"branch {label}"
                    _record(trace, sid, stype, "done", "", out, t0)
                    if not dry:
                        await self.store.add_step_result(run_id, sid, executed, type=stype, status="done", output_text=out, ms=_ms(t0))
                    if target and str(target) == _STOP_TARGET:
                        break  # branch ends the flow
                    if target and target in by_id:
                        i = by_id[target]
                        continue
                    i += 1
                    continue

                if stype == "tool":
                    out, agent = await self._run_tool(step, ctx, payload)
                elif stype == "agent":
                    out, agent = await self._run_agent_step(step, ctx, payload)
                elif stype == "vision":
                    out, agent = await self._run_vision_step(step, ctx, payload)
                elif stype == "input":
                    out, agent = await self._run_input_step(step, ctx, payload, flow, dry=dry)
                elif stype == "emit":
                    out, agent = await self._emit(step, ctx, payload), ""
                else:
                    out, agent = f"(unknown step type '{stype}')", ""

                ctx["steps"][sid] = {"output": out}
                # Convenience: expose a trailing 'label' if the tool returned JSON.
                _maybe_attach_fields(ctx["steps"][sid], out)
                _record(trace, sid, stype, "done", agent, out, t0)
                if not dry:
                    await self.store.add_step_result(
                        run_id, sid, executed, type=stype, status="done",
                        agent=agent, output_text=str(out), ms=_ms(t0),
                    )
                # Track the last EXECUTED step's output (correct under goto/stop,
                # unlike steps[-1] which is just the last step in the list).
                final_text = str(out)
                i += 1
                if step.get("stop"):
                    break  # "Stop after this step" flag → end the flow here

            # Deliver final output (the last executed step's output) — only on a
            # clean finish, not when the run was stopped mid-flight.
            output = flow.get("output") or {}
            out_channel = str(output.get("channel") or "log")
            if status == "done" and not dry and out_channel in ("whatsapp", "same", "glasses", "web") and final_text:
                try:
                    await self._deliver(payload, str(final_text))
                except Exception as exc:
                    log.warning("flow output delivery failed: %s", exc)
        except Exception as exc:
            status = "error"
            error = str(exc)
            log.warning("flow run failed", flow=flow.get("name"), error=error)

        if not dry:
            await self.store.finish_run(run_id, status, error)
            _RUN_CONTROL.pop(run_id, None)
        return {"run_id": run_id, "status": status, "error": error, "steps": trace, "output": final_text}


# ── Branch condition evaluator ─────────────────────────────────────────
#
# A small, SAFE boolean expression evaluator (no eval()). Supports:
#   • logicals:    and / or / not   (also && / || / !)
#   • grouping:    ( ... )
#   • comparisons: ==  !=  >  <  >=  <=  contains  matches (~)
#   • operands:    {{path}} (resolved against ctx), "quoted", or bare words
#   • truthiness:  a lone operand is true unless it's empty/false/0/none/no
#
# {{...}} tokens are resolved at eval time (not string-substituted first), so a
# value containing spaces/quotes/operators can't corrupt the parse.

_FALSY = {"", "false", "0", "none", "no", "null"}

_TOKEN_RE = re.compile(
    r"""\{\{.*?\}\}            # {{var}}
      | "[^"]*" | '[^']*'      # quoted string
      | >= | <= | == | !=      # 2-char comparisons
      | && | \|\|              # logical and / or
      | [()<>~!]               # parens & single-char ops
      | [^\s()<>~!&|"'=]+       # bare operand / keyword
    """,
    re.X,
)

_CMP = {"==", "!=", ">", "<", ">=", "<=", "contains", "matches", "~"}


def _val_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _resolve_operand(tok: str, ctx: dict[str, Any]) -> str:
    if tok.startswith("{{") and tok.endswith("}}"):
        return _val_to_str(_dig(ctx, tok[2:-2].strip()))
    if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0]:
        return tok[1:-1]
    return tok


def _apply_cmp(op: str, lhs: str, rhs: str) -> bool:
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    if op == "contains":
        return rhs.lower() in lhs.lower()
    if op in ("matches", "~"):
        try:
            return bool(re.search(rhs, lhs, re.I))
        except re.error:
            return False
    # Numeric comparisons, with lexicographic fallback.
    try:
        a: Any = float(lhs)
        b: Any = float(rhs)
    except ValueError:
        a, b = lhs, rhs
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    return False


def _eval_expr(expr: str, ctx: dict[str, Any]) -> bool:
    """Evaluate a boolean branch condition against the run context. Safe: a
    hand-written recursive-descent parser, never eval(). Returns False on any
    parse/eval error so a malformed condition can't crash the run."""
    toks = [m.group(0) for m in _TOKEN_RE.finditer(expr or "")]
    if not toks:
        return False
    i = 0

    def low() -> str | None:
        return toks[i].lower() if i < len(toks) else None

    def parse_or() -> bool:
        nonlocal i
        v = parse_and()
        while low() in ("or", "||"):
            i += 1
            r = parse_and()
            v = v or r
        return v

    def parse_and() -> bool:
        nonlocal i
        v = parse_not()
        while low() in ("and", "&&"):
            i += 1
            r = parse_not()
            v = v and r
        return v

    def parse_not() -> bool:
        nonlocal i
        if low() in ("not", "!"):
            i += 1
            return not parse_not()
        return parse_atom()

    def parse_atom() -> bool:
        nonlocal i
        if i >= len(toks):
            return False
        if toks[i] == "(":
            i += 1
            v = parse_or()
            if i < len(toks) and toks[i] == ")":
                i += 1
            return v
        lhs = _resolve_operand(toks[i], ctx)
        i += 1
        op = low()
        if op in _CMP:
            i += 1
            rhs = _resolve_operand(toks[i], ctx) if i < len(toks) else ""
            if i < len(toks):
                i += 1
            return _apply_cmp(op, lhs, rhs)
        return lhs.strip().lower() not in _FALSY

    try:
        return bool(parse_or())
    except Exception:
        return False


def _maybe_attach_fields(slot: dict[str, Any], out: Any) -> None:
    """If a tool returned a JSON object string, expose its keys (e.g. .label)."""
    if not isinstance(out, str):
        return
    s = out.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    slot.setdefault(k, v)
        except (ValueError, TypeError):
            pass


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _record(trace: list, sid: str, stype: str, status: str, agent: str, out: Any, t0: float) -> None:
    trace.append({"step_id": sid, "type": stype, "status": status, "agent": agent,
                  "output": str(out)[:4000], "ms": _ms(t0)})
