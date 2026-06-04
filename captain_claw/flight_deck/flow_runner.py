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

    async def _emit(self, step: dict[str, Any], ctx: dict[str, Any], payload: dict[str, Any]) -> str:
        channel = str(step.get("channel") or "log")
        body = _render(str(step.get("body") or "{{steps}}"), ctx)
        if channel in ("whatsapp", "same") and self.whatsapp_send:
            waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
            if waid:
                await self.whatsapp_send(waid, body)
                return f"(emitted to whatsapp {waid})"
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
        status = "done"
        error = ""
        final_text = ""
        by_id = {s.get("id"): i for i, s in enumerate(steps)}
        i = 0
        executed = 0
        try:
            while i < len(steps):
                if executed >= max_steps:
                    raise RuntimeError(f"max_steps ({max_steps}) exceeded")
                step = steps[i]
                sid = str(step.get("id") or f"step{i}")
                stype = str(step.get("type") or "tool")
                executed += 1
                t0 = time.monotonic()

                if stype == "branch":
                    cond = _render(str(step.get("when") or ""), ctx)
                    goto = step.get("goto")
                    taken = bool(_eval_when(cond))
                    out = f"branch {'taken→'+str(goto) if taken else 'not taken'}"
                    _record(trace, sid, stype, "done", "", out, t0)
                    if not dry:
                        await self.store.add_step_result(run_id, sid, executed, type=stype, status="done", output_text=out, ms=_ms(t0))
                    if taken and goto in by_id:
                        i = by_id[goto]
                        continue
                    i += 1
                    continue

                if stype == "tool":
                    out, agent = await self._run_tool(step, ctx, payload)
                elif stype == "agent":
                    out, agent = await self._run_agent_step(step, ctx, payload)
                elif stype == "vision":
                    out, agent = await self._run_vision_step(step, ctx, payload)
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
                i += 1

            # Deliver final output.
            output = flow.get("output") or {}
            out_channel = str(output.get("channel") or "log")
            final_text = ctx["steps"].get(steps[-1]["id"], {}).get("output", "") if steps else ""
            if not dry and out_channel in ("whatsapp", "same", "glasses") and self.whatsapp_send:
                waid = str(payload.get("waid") or payload.get("whatsapp_waid") or "")
                if waid and final_text:
                    try:
                        await self.whatsapp_send(waid, str(final_text))
                    except Exception as exc:
                        log.warning("flow output delivery failed: %s", exc)
        except Exception as exc:
            status = "error"
            error = str(exc)
            log.warning("flow run failed", flow=flow.get("name"), error=error)

        if not dry:
            await self.store.finish_run(run_id, status, error)
        return {"run_id": run_id, "status": status, "error": error, "steps": trace, "output": final_text}


def _eval_when(cond: str) -> bool:
    """Tiny safe evaluator: supports 'A == B', 'A != B', or truthiness of A."""
    cond = (cond or "").strip()
    for op in ("==", "!="):
        if op in cond:
            lhs, rhs = (x.strip().strip('"\'') for x in cond.split(op, 1))
            return (lhs == rhs) if op == "==" else (lhs != rhs)
    return bool(cond) and cond.lower() not in ("none", "false", "0", "")


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
