"""Gather real-world system information for injection into agent system prompts."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime


def _get_local_ip() -> str:
    """Return the host's primary LAN IP address."""
    try:
        # UDP connect trick — no traffic is sent, but the OS picks the right interface.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


def _get_public_ip() -> str:
    """Return the host's public IP via a lightweight HTTP call (best-effort)."""
    try:
        import httpx

        resp = httpx.get("https://api.ipify.org", timeout=2, follow_redirects=True)
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception:
        pass
    return ""


def _get_disk_free(path: str = "/") -> str:
    """Return free disk space as a human-readable string."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 1:
            return f"{free_gb:.1f} GB"
        free_mb = usage.free / (1024 ** 2)
        return f"{free_mb:.0f} MB"
    except Exception:
        return ""


def _get_memory_usage() -> tuple[str, str]:
    """Return (used/total human-readable, compact) memory usage strings.

    Reads ``/proc/meminfo`` on Linux or calls ``vm_stat`` + ``sysctl`` on macOS.
    Returns ``("", "")`` on failure.
    """
    total_bytes: int | None = None
    avail_bytes: int | None = None

    # Linux: /proc/meminfo
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    # Values in /proc/meminfo are in kB.
                    info[key] = int(parts[1]) * 1024
        total_bytes = info.get("MemTotal")
        avail_bytes = info.get("MemAvailable")
        if avail_bytes is None and total_bytes is not None:
            # Fallback: free + buffers + cached.
            avail_bytes = (
                info.get("MemFree", 0)
                + info.get("Buffers", 0)
                + info.get("Cached", 0)
            )
    except Exception:
        pass

    # macOS: sysctl + vm_stat
    if total_bytes is None and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ).strip()
            total_bytes = int(out)
        except Exception:
            pass
        try:
            vm = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            page_size = 16384  # Apple Silicon default
            for line in vm.splitlines():
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                    break
            vals: dict[str, int] = {}
            for line in vm.splitlines():
                if ":" in line and "page size" not in line:
                    key, val = line.split(":", 1)
                    vals[key.strip()] = int(val.strip().rstrip(".")) * page_size
            free = vals.get("Pages free", 0)
            inactive = vals.get("Pages inactive", 0)
            purgeable = vals.get("Pages purgeable", 0)
            avail_bytes = free + inactive + purgeable
        except Exception:
            pass

    if total_bytes is None:
        return "", ""

    used_bytes = total_bytes - (avail_bytes or 0) if avail_bytes else None

    def _fmt(b: int) -> str:
        gb = b / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{b / (1024 ** 2):.0f} MB"

    def _fmt_compact(b: int) -> str:
        gb = b / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f}GB"
        return f"{b / (1024 ** 2):.0f}MB"

    total_str = _fmt(total_bytes)
    if used_bytes is not None:
        normal = f"{_fmt(used_bytes)} / {total_str}"
        compact = f"{_fmt_compact(used_bytes)}/{_fmt_compact(total_bytes)}"
    else:
        normal = total_str
        compact = _fmt_compact(total_bytes)

    return normal, compact


def _get_system_load() -> str:
    """Return system load averages (1, 5, 15 min) on Unix, empty on Windows."""
    try:
        load = os.getloadavg()
        return f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"
    except (OSError, AttributeError):
        return ""


def _get_uptime() -> str:
    """Return system uptime as a human-readable string."""
    seconds: float | None = None

    # Linux: read /proc/uptime directly.
    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
    except Exception:
        pass

    # macOS: sysctl kern.boottime.
    if seconds is None and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "kern.boottime"],
                text=True,
                timeout=3,
            ).strip()
            # Format: "{ sec = 1709500000, usec = 0 } ..."
            sec_str = out.split("sec =")[1].split(",")[0].strip()
            boot = int(sec_str)
            seconds = time.time() - boot
        except Exception:
            pass

    if seconds is None:
        return ""

    seconds = max(0, seconds)
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if not parts and minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return ", ".join(parts) if parts else "< 1 minute"


def _get_uptime_compact() -> str:
    """Return a compact uptime string like '5d3h'."""
    seconds: float | None = None

    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
    except Exception:
        pass

    if seconds is None and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "kern.boottime"],
                text=True,
                timeout=3,
            ).strip()
            sec_str = out.split("sec =")[1].split(",")[0].strip()
            boot = int(sec_str)
            seconds = time.time() - boot
        except Exception:
            pass

    if seconds is None:
        return ""

    seconds = max(0, seconds)
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def build_system_info_block(detail_level: str = "normal") -> str:
    """Build a system environment info block for the agent system prompt.

    Args:
        detail_level: ``"normal"`` for full labelled block,
                      ``"micro"`` for a single compact line,
                      ``"nano"`` returns empty string (nano prompts are ultra-minimal).

    Returns:
        Formatted string ready for template injection, or ``""`` if *nano*.
    """
    if detail_level == "nano":
        return ""

    now = datetime.now()
    hostname = socket.gethostname()
    disk_free = _get_disk_free()
    local_ip = _get_local_ip()
    public_ip = _get_public_ip()
    load_avg = _get_system_load()
    mem_normal, mem_compact = _get_memory_usage()

    if detail_level == "micro":
        parts = [f"{now:%a %Y-%m-%d %H:%M}"]
        parts.append(f"host={hostname}")
        if mem_compact:
            parts.append(f"mem={mem_compact}")
        if disk_free:
            parts.append(f"disk={disk_free.replace(' ', '')}")
        if local_ip:
            parts.append(f"ip={local_ip}")
        if public_ip:
            parts.append(f"pub={public_ip}")
        if load_avg:
            parts.append(f"load={load_avg.split()[0]}")
        uptime_compact = _get_uptime_compact()
        if uptime_compact:
            parts.append(f"up={uptime_compact}")
        return "Env: " + " | ".join(parts)

    # Normal (full) format.
    lines = ["System environment:"]
    lines.append(f"- Date: {now:%A, %B %d, %Y %H:%M}")
    lines.append(f"- Hostname: {hostname}")
    if mem_normal:
        lines.append(f"- Memory: {mem_normal}")
    if disk_free:
        lines.append(f"- Disk free: {disk_free}")
    if local_ip:
        lines.append(f"- Local IP: {local_ip}")
    if public_ip:
        lines.append(f"- Public IP: {public_ip}")
    if load_avg:
        lines.append(f"- System load: {load_avg}")
    uptime = _get_uptime()
    if uptime:
        lines.append(f"- Uptime: {uptime}")
    return "\n".join(lines)
