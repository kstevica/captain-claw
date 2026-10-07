"""Bat hard floor (Invariant A) — the un-widenable system-harm denylist.

Covers the pure classifier (`screen`), the marker gate (`active`), and the
wiring into the universal tool chokepoint (`ToolRegistry.execute`).
"""

from __future__ import annotations

import pytest

from captain_claw import bat_floor
from captain_claw.exceptions import ToolBlockedError
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


# ── the pure classifier ────────────────────────────────────────────────

CATASTROPHIC = [
    "rm -rf /",
    "rm -fr /",
    "rm -r -f /",
    "rm -Rf /",
    "rm --recursive --force /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf $HOME",
    "rm -rf ${HOME}",
    "rm -rf .",
    "rm -rf *",
    "sudo rm -rf /",
    "cd /tmp && rm -rf /",
    "rm -rf /etc",
    "rm -rf /usr/local",      # system dir, any depth
    "rm -rf /System/Library",
    "rm -rf /Users",          # the users root
    "rm -rf /Users/alice",    # a whole user home
    "rm -rf /Volumes/Backup",  # a mounted drive
    "rm -f /etc/passwd",       # no -r, but a system file
    "mkfs.ext4 /dev/sda1",
    "newfs_apfs disk2",
    "diskutil eraseDisk JHFS+ X disk2",
    "diskutil eraseVolume free n /dev/disk3",
    "diskutil unmountDisk force disk2",
    "umount /Volumes/Backup",
    "dd if=/dev/zero of=/dev/disk0 bs=1m",
    "shutdown -h now",
    "sudo reboot",
    "systemctl poweroff",
    ":(){ :|:& };:",
    "find / -name '*.log' -delete",
    "wipefs -a /dev/sda",
    "python3 -c \"import shutil; shutil.rmtree('/')\"",
]

BENIGN = [
    "ls -la",
    "echo hello",
    "rm -rf build",
    "rm -rf node_modules",
    "rm -rf ./dist",
    "rm -rf src/tmp",
    "rm -rf /Users/alice/Dev/captain-claw/build",  # deep project path
    "rm -rf /tmp/scratch",       # /tmp is not protected
    "rm file.txt",
    "rm -f relative.log",
    "git reset --hard HEAD~1",   # destructive to WORK, not the system → allowed
    "dropdb devtest",            # a dev DB drop → not the floor's concern
    "chmod -R 755 ./build",
    "npm run build && pytest -q",
    "curl -fsSL https://example.com/install.sh | sh",  # installer, allowed
]


@pytest.mark.parametrize("cmd", CATASTROPHIC)
def test_screen_blocks_catastrophic(cmd):
    blocked, reason = bat_floor.screen("shell", {"command": cmd})
    assert blocked, f"should block: {cmd!r}"
    assert reason


@pytest.mark.parametrize("cmd", BENIGN)
def test_screen_allows_benign(cmd):
    blocked, reason = bat_floor.screen("shell", {"command": cmd})
    assert not blocked, f"should allow: {cmd!r} (reason={reason!r})"


def test_recursive_chmod_on_root_blocked_but_not_relative():
    assert bat_floor.screen("shell", {"command": "chmod -R 000 /"})[0]
    assert bat_floor.screen("shell", {"command": "chown -R root /System"})[0]
    assert not bat_floor.screen("shell", {"command": "chmod -R 755 ./dist"})[0]
    # chmod WITHOUT -R on a protected path is not a recursive wipe → allowed
    assert not bat_floor.screen("shell", {"command": "chmod 644 /etc/hosts"})[0]


def test_terminal_tool_fields_are_screened():
    # `terminal` drives the user's real Mac — its run/send payloads must be
    # screened just like shell.
    assert bat_floor.screen("terminal", {"action": "run", "command": "rm -rf /"})[0]
    assert bat_floor.screen("terminal", {"action": "send", "data": "umount /Volumes/X"})[0]
    assert not bat_floor.screen("terminal", {"action": "run", "command": "ls"})[0]


def test_desktop_action_typed_text_screened():
    assert bat_floor.screen("desktop_action", {"action": "type", "text": "sudo rm -rf /"})[0]
    assert not bat_floor.screen("desktop_action", {"action": "type", "text": "hello world"})[0]


def test_non_exec_tools_are_never_screened():
    # A file whose CONTENT mentions rm -rf / is not an execution.
    assert not bat_floor.screen("write", {"path": "x.sh", "content": "rm -rf /"})[0]
    assert not bat_floor.screen("edit", {"content": "diskutil eraseDisk ..."})[0]
    assert not bat_floor.screen("read", {"path": "/etc/passwd"})[0]


def test_active_reads_marker(monkeypatch):
    monkeypatch.setattr(bat_floor, "_ACTIVE", None)
    monkeypatch.delenv(bat_floor.MARKER, raising=False)
    assert bat_floor.active() is False

    monkeypatch.setattr(bat_floor, "_ACTIVE", None)
    monkeypatch.setenv(bat_floor.MARKER, "1")
    assert bat_floor.active() is True


# ── wiring into the universal chokepoint ────────────────────────────────


class _RecordingShell(Tool):
    """A stand-in for the real shell tool: records whether it ever executed."""

    def __init__(self):
        self.name = "shell"
        self.description = "shell"
        self.parameters = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        self.ran: list[str] = []

    async def execute(self, command: str = "", **kwargs):
        self.ran.append(command)
        return ToolResult(success=True, content="ran")


async def test_registry_refuses_destructive_when_floor_active(monkeypatch):
    monkeypatch.setattr(bat_floor, "_ACTIVE", True)
    reg = ToolRegistry()
    shell = _RecordingShell()
    reg.register(shell)

    with pytest.raises(ToolBlockedError):
        await reg.execute("shell", {"command": "rm -rf /"})
    assert shell.ran == [], "the destructive command must never reach execute()"


async def test_registry_allows_benign_when_floor_active(monkeypatch):
    monkeypatch.setattr(bat_floor, "_ACTIVE", True)
    reg = ToolRegistry()
    shell = _RecordingShell()
    reg.register(shell)

    result = await reg.execute("shell", {"command": "echo hi"})
    assert result.success and shell.ran == ["echo hi"]


async def test_registry_no_op_when_floor_inactive(monkeypatch):
    monkeypatch.setattr(bat_floor, "_ACTIVE", False)
    reg = ToolRegistry()
    shell = _RecordingShell()
    reg.register(shell)

    # Floor inactive (not a Bat worker) → it never runs; the dummy shell
    # executes normally. (Non-Bat agents keep the shell tool's own
    # blocked/deny patterns, unchanged by this work.)
    result = await reg.execute("shell", {"command": "echo done"})
    assert result.success and shell.ran == ["echo done"]
