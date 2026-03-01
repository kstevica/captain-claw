"""Termux API tool for interacting with Android device features via Termux."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class TermuxTool(Tool):
    """Interact with Android device via Termux API commands.

    Supports camera photo capture, battery status, GPS location,
    and torch (flashlight) control.  Each action maps directly to
    the corresponding ``termux-*`` CLI command from the termux-api
    package.
    """

    name = "termux"
    timeout_seconds = 60.0
    description = (
        "Interact with the Android device via Termux API. "
        "Supported actions:\n"
        "  photo   – take a photo with the device camera\n"
        "  battery – report battery status (level, health, temperature)\n"
        "  location – get current GPS/network location\n"
        "  torch   – turn the flashlight on or off"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["photo", "battery", "location", "torch"],
                "description": (
                    "Which Termux API action to perform: "
                    "'photo', 'battery', 'location', or 'torch'."
                ),
            },
            "camera_id": {
                "type": "integer",
                "description": (
                    "Camera ID for 'photo' action. 0 = back camera (default), "
                    "1 = front/selfie camera."
                ),
            },
            "provider": {
                "type": "string",
                "enum": ["gps", "network", "passive"],
                "description": (
                    "Location provider for 'location' action. "
                    "'gps' (default, most accurate, outdoor), "
                    "'network' (WiFi/cell, works indoors), "
                    "'passive' (last cached location, fastest)."
                ),
            },
            "state": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "Torch state for 'torch' action: 'on' or 'off'.",
            },
        },
        "required": ["action"],
    }

    async def _run_command(
        self,
        cmd: str,
        timeout: float = 15.0,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[bool, str, str]:
        """Run a shell command and return (success, stdout, stderr)."""
        env = os.environ.copy()
        env["PATH"] = os.environ.get("PATH", "/data/data/com.termux/files/usr/bin:/usr/local/bin:/usr/bin:/bin")

        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        communicate_task = asyncio.create_task(process.communicate())
        abort_wait_task: asyncio.Task[bool] | None = None
        if isinstance(abort_event, asyncio.Event):
            abort_wait_task = asyncio.create_task(abort_event.wait())
        try:
            wait_tasks: set[asyncio.Task[Any]] = {communicate_task}
            if abort_wait_task is not None:
                wait_tasks.add(abort_wait_task)
            done, _ = await asyncio.wait(
                wait_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if communicate_task in done:
                stdout, stderr = await communicate_task
            elif abort_wait_task is not None and abort_wait_task in done:
                process.kill()
                await process.wait()
                communicate_task.cancel()
                try:
                    await communicate_task
                except asyncio.CancelledError:
                    pass
                return False, "", "Command aborted"
            else:
                process.kill()
                await process.wait()
                communicate_task.cancel()
                try:
                    await communicate_task
                except asyncio.CancelledError:
                    pass
                return False, "", f"Command timed out after {int(timeout)}s"
        finally:
            if abort_wait_task is not None and not abort_wait_task.done():
                abort_wait_task.cancel()
                try:
                    await abort_wait_task
                except asyncio.CancelledError:
                    pass

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        success = process.returncode == 0
        return success, stdout_text, stderr_text

    async def _action_photo(
        self,
        camera_id: int = 0,
        **kwargs: Any,
    ) -> ToolResult:
        """Take a photo using termux-camera-photo."""
        saved_root = kwargs.get("_saved_base_path")
        if saved_root is None:
            runtime_base = kwargs.get("_runtime_base_path", Path.cwd())
            saved_root = Path(runtime_base) / "saved"

        session_id = str(kwargs.get("_session_id", "")).strip() or "default"
        session_id = session_id.replace("/", "_").replace("..", "_")

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        cam_label = "front" if camera_id == 1 else "back"
        filename = f"termux-photo-{cam_label}-{stamp}.jpg"
        output_dir = Path(saved_root) / "media" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

        cmd = f"termux-camera-photo -c {int(camera_id)} {output_file}"
        log.info("Taking photo", camera_id=camera_id, output=str(output_file))

        abort_event = kwargs.get("_abort_event")
        success, stdout, stderr = await self._run_command(
            cmd, timeout=30.0, abort_event=abort_event,
        )

        if not success:
            error_msg = stderr or stdout or "termux-camera-photo failed"
            return ToolResult(success=False, error=error_msg)

        if not output_file.exists():
            return ToolResult(
                success=False,
                error="Photo file was not created. Is termux-api installed and the Termux:API app granted camera permission?",
            )

        file_registry = kwargs.get("_file_registry")
        logical_path = f"media/{session_id}/{filename}"
        if file_registry is not None:
            try:
                file_registry.register(
                    logical_path=logical_path,
                    physical_path=str(output_file),
                    task_id=str(kwargs.get("_task_id", "")),
                )
            except Exception:
                pass

        size_kb = output_file.stat().st_size / 1024.0
        return ToolResult(
            success=True,
            content=(
                f"Photo captured successfully.\n"
                f"Path: {output_file}\n"
                f"Camera: {cam_label} (id={camera_id})\n"
                f"File size: {size_kb:.1f} KB"
            ),
        )

    async def _action_battery(self, **kwargs: Any) -> ToolResult:
        """Get battery status using termux-battery-status."""
        abort_event = kwargs.get("_abort_event")
        success, stdout, stderr = await self._run_command(
            "termux-battery-status", timeout=10.0, abort_event=abort_event,
        )

        if not success:
            error_msg = stderr or stdout or "termux-battery-status failed"
            return ToolResult(success=False, error=error_msg)

        return ToolResult(success=True, content=stdout or "[no output]")

    async def _action_location(
        self,
        provider: str = "gps",
        **kwargs: Any,
    ) -> ToolResult:
        """Get device location using termux-location."""
        provider = str(provider or "gps").strip().lower()
        if provider not in ("gps", "network", "passive"):
            provider = "gps"

        cmd = f"termux-location -p {provider} -r once"
        log.info("Getting location", provider=provider)

        abort_event = kwargs.get("_abort_event")
        success, stdout, stderr = await self._run_command(
            cmd, timeout=15.0, abort_event=abort_event,
        )

        if not success:
            error_msg = stderr or stdout or "termux-location failed"
            return ToolResult(success=False, error=error_msg)

        return ToolResult(success=True, content=stdout or "[no output]")

    async def _action_torch(
        self,
        state: str = "on",
        **kwargs: Any,
    ) -> ToolResult:
        """Toggle flashlight using termux-torch."""
        state = str(state or "on").strip().lower()
        if state not in ("on", "off"):
            return ToolResult(
                success=False,
                error=f"Invalid torch state '{state}'. Use 'on' or 'off'.",
            )

        cmd = f"termux-torch {state}"
        log.info("Toggling torch", state=state)

        abort_event = kwargs.get("_abort_event")
        success, stdout, stderr = await self._run_command(
            cmd, timeout=10.0, abort_event=abort_event,
        )

        if not success:
            error_msg = stderr or stdout or "termux-torch failed"
            return ToolResult(success=False, error=error_msg)

        return ToolResult(
            success=True,
            content=f"Torch turned {state}.",
        )

    async def execute(
        self,
        action: str,
        camera_id: int = 0,
        provider: str = "gps",
        state: str = "on",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a Termux API action."""
        action = str(action or "").strip().lower()

        if action == "photo":
            return await self._action_photo(camera_id=camera_id, **kwargs)
        elif action == "battery":
            return await self._action_battery(**kwargs)
        elif action == "location":
            return await self._action_location(provider=provider, **kwargs)
        elif action == "torch":
            return await self._action_torch(state=state, **kwargs)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown termux action '{action}'. Use: photo, battery, location, torch.",
            )
