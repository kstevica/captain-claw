"""System / DevOps routes — enumerate and stop OS processes spawned by Flight Deck.

This is the "devops checkup" surface. A logged-in user sees the process trees
rooted at agents *they own* (Flight Deck process-agents + hosted VFS apps) plus
every child process those roots have spawned (bash, python, git, node, …).
Admins additionally see system / unattributed trees — the Flight Deck server
itself and legacy agents that carry no owner.

Enumeration is done by snapshotting the OS process table with ``ps`` (no extra
dependency) and stitching it against the Flight Deck process registry, which is
the source of truth for *ownership* and *slug* (needed for a graceful stop).
``ps`` gives us the descendants + live CPU / memory that the registry can't.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/system", tags=["system"])

# PID of the Flight Deck server process (this module is imported into it). We
# refuse to ever signal this, and we surface it as a system root for admins.
FD_SERVER_PID = os.getpid()

# Ordering weight for the different kinds of root, most interesting first.
_KIND_ORDER = {"agent": 0, "hosted-app": 1, "code-app": 2, "flight-deck": 9}


# ── ps snapshot ──────────────────────────────────────────────────────────────

def _ps_command() -> list[str]:
    fmt = "pid=,ppid=,pcpu=,pmem=,rss=,etime=,command="
    if sys.platform == "darwin":
        # BSD ps: -a all users, -x incl. no-tty, -ww unlimited width.
        return ["ps", "-axww", "-o", fmt]
    # Linux (procps) and other unixes.
    return ["ps", "-eww", "-o", fmt]


def _etime_to_seconds(etime: str) -> int:
    """Parse ps elapsed time ([[DD-]HH:]MM:SS) into whole seconds."""
    etime = (etime or "").strip()
    if not etime:
        return 0
    days = 0
    if "-" in etime:
        d, etime = etime.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            days = 0
    try:
        parts = [int(p) for p in etime.split(":")]
    except ValueError:
        return 0
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return days * 86400 + secs


def _snapshot_processes() -> tuple[dict[int, dict], dict[int, list[int]]]:
    """Return ``(by_pid, children)`` for the whole process table.

    ``by_pid``:  pid -> {pid, ppid, cpu, mem, rss_mb, etime, elapsed_s, command, name}
    ``children``: ppid -> [child pid, …]
    Returns empty maps if ``ps`` is unavailable — callers degrade gracefully.
    """
    try:
        out = subprocess.run(
            _ps_command(), capture_output=True, text=True, timeout=6, check=False
        )
    except Exception:
        return {}, {}

    by_pid: dict[int, dict] = {}
    children: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        pid_s, ppid_s, cpu_s, mem_s, rss_s, etime_s, command = parts
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
        except ValueError:
            continue
        try:
            cpu = float(cpu_s)
        except ValueError:
            cpu = 0.0
        try:
            mem = float(mem_s)
        except ValueError:
            mem = 0.0
        try:
            rss_kb = int(rss_s)
        except ValueError:
            rss_kb = 0
        first = command.split(" ", 1)[0]
        name = os.path.basename(first) if first else command[:24]
        by_pid[pid] = {
            "pid": pid,
            "ppid": ppid,
            "cpu": round(cpu, 1),
            "mem": round(mem, 1),
            "rss_mb": round(rss_kb / 1024, 1),
            "etime": etime_s,
            "elapsed_s": _etime_to_seconds(etime_s),
            "command": command,
            "name": name,
        }
        children.setdefault(ppid, []).append(pid)
    return by_pid, children


# ── Flight Deck roots (owner + slug attribution) ────────────────────────────

def _collect_roots() -> list[dict]:
    """Roots that Flight Deck knows it spawned, with owner + kind + (optional) slug.

    Each root: {pid, owner (str|None), kind, label, slug (str|None), detail}.
    ``owner`` None/"" means system / unattributed (admin-only visibility).
    """
    from captain_claw.flight_deck import server as _srv

    roots: list[dict] = []
    seen: set[int] = set()

    def _add(pid, owner, kind, label, slug, detail):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return
        if pid in seen:
            return
        seen.add(pid)
        roots.append({
            "pid": pid,
            "owner": owner or None,
            "kind": kind,
            "label": label,
            "slug": slug,
            "detail": detail,
        })

    # 1) Process-agent registry — the bulk of managed agents.
    try:
        reg = _srv._load_process_registry()
    except Exception:
        reg = {}
    for slug, entry in reg.items():
        pid = entry.get("pid")
        if not pid:
            continue
        detail = " ".join(x for x in [entry.get("provider", ""), entry.get("model", "")] if x)
        _add(pid, entry.get("owner"), "agent", entry.get("name") or slug, slug, detail)

    # 2) Hosted VFS apps (published sites / servers).
    try:
        from captain_claw.flight_deck import vfs_hosting

        for name, ent in (vfs_hosting.load_registry() or {}).items():
            pid = ent.get("pid")
            if not pid:
                continue
            port = ent.get("port")
            _add(pid, ent.get("owner"), "hosted-app", name, None,
                 f"port {port}" if port else "")
    except Exception:
        pass

    # 3) The Flight Deck server itself — system root, admin-only.
    _add(FD_SERVER_PID, None, "flight-deck", "Flight Deck server", None, "")

    return roots


# ── Forest assembly ──────────────────────────────────────────────────────────

def _build_forest(by_pid: dict[int, dict], children: dict[int, list[int]],
                  roots: list[dict]) -> list[dict]:
    """Assemble a process forest. Each alive root owns its subtree; descendants
    that are themselves roots are excluded (they stand as their own tree), so
    an owned agent never gets swallowed into the Flight Deck-server system tree.
    """
    root_pids = {r["pid"] for r in roots if r["pid"] in by_pid}

    def build_node(pid: int, root_info: dict, is_root: bool, visited: set[int]) -> dict:
        info = by_pid[pid]
        node = {
            "pid": pid,
            "ppid": info["ppid"],
            "kind": root_info["kind"] if is_root else "child",
            "label": root_info["label"] if is_root else info["name"],
            "slug": root_info["slug"] if is_root else None,
            "owner": root_info["owner"],
            "command": info["command"],
            "name": info["name"],
            "cpu": info["cpu"],
            "mem": info["mem"],
            "rss_mb": info["rss_mb"],
            "elapsed": info["etime"],
            "elapsed_s": info["elapsed_s"],
            "detail": root_info.get("detail", "") if is_root else "",
            "is_root": is_root,
            "children": [],
        }
        visited.add(pid)
        for cpid in sorted(children.get(pid, [])):
            if cpid in root_pids or cpid in visited or cpid not in by_pid:
                continue
            node["children"].append(build_node(cpid, root_info, False, visited))
        return node

    def aggregate(node: dict) -> tuple[float, float, int]:
        cpu = node["cpu"]
        mem_mb = node["rss_mb"]
        count = 0
        for ch in node["children"]:
            c_cpu, c_mem, c_cnt = aggregate(ch)
            cpu += c_cpu
            mem_mb += c_mem
            count += 1 + c_cnt
        node["agg_cpu"] = round(cpu, 1)
        node["agg_mem_mb"] = round(mem_mb, 1)
        node["descendant_count"] = count
        return cpu, mem_mb, count

    forest: list[dict] = []
    for r in roots:
        if r["pid"] not in by_pid:
            continue  # registry entry whose process is dead → not a live tree
        node = build_node(r["pid"], r, True, set())
        aggregate(node)
        forest.append(node)

    forest.sort(key=lambda n: (_KIND_ORDER.get(n["kind"], 5), -n["agg_cpu"]))
    return forest


def _stamp_emails(node: dict, emails: dict[str, str]) -> None:
    if node["owner"]:
        node["owner_email"] = emails.get(node["owner"], node["owner"][:8])
    elif node["kind"] == "flight-deck":
        node["owner_email"] = "system"
    else:
        node["owner_email"] = None
    for ch in node["children"]:
        _stamp_emails(ch, emails)


# ── Host vitals (best-effort, no psutil) ─────────────────────────────────────

def _mem_numbers() -> dict:
    if sys.platform == "darwin":
        total = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3
        ).stdout.strip())
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"page size of (\d+) bytes", vm)
        page = int(m.group(1)) if m else 4096
        stats: dict[str, int] = {}
        for line in vm.splitlines():
            if ":" in line:
                k, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                if val.isdigit():
                    stats[k.strip()] = int(val)
        used_pages = (
            stats.get("Pages active", 0)
            + stats.get("Pages wired down", 0)
            + stats.get("Pages occupied by compressor", 0)
        )
        used = used_pages * page
    else:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[1].isdigit():
                    info[p[0].rstrip(":")] = int(p[1]) * 1024  # kB → bytes
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = max(total - avail, 0)
    if not total:
        return {}
    return {
        "mem_total_mb": round(total / 1e6),
        "mem_used_mb": round(used / 1e6),
        "mem_percent": round(used / total * 100, 1),
    }


def _host_vitals() -> dict:
    from captain_claw.flight_deck import server as _srv

    v: dict = {
        "cpu_count": os.cpu_count(),
        "load_avg": None,
        "mem_total_mb": None,
        "mem_used_mb": None,
        "mem_percent": None,
        "disk_free_gb": None,
        "disk_total_gb": None,
    }
    try:
        v["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        pass
    try:
        du = shutil.disk_usage(str(getattr(_srv, "DATA_DIR", "/")))
        v["disk_total_gb"] = round(du.total / 1e9, 1)
        v["disk_free_gb"] = round(du.free / 1e9, 1)
    except Exception:
        pass
    try:
        v.update(_mem_numbers())
    except Exception:
        pass
    return v


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/processes")
async def system_processes(request: Request, user: dict = Depends(get_current_user)):
    """Process forest visible to the caller.

    Non-admins see only trees they own; admins see everything, including the
    Flight Deck server and legacy/unattributed agents.
    """
    is_admin = user.get("role") == "admin"
    user_id = user["id"]

    loop = asyncio.get_event_loop()
    by_pid, children = await loop.run_in_executor(None, _snapshot_processes)
    roots = _collect_roots()
    forest = _build_forest(by_pid, children, roots)

    visible = [
        n for n in forest
        if is_admin or (n["owner"] and n["owner"] == user_id)
    ]

    # Resolve owner ids → emails for the labels (only touch the DB if needed).
    emails: dict[str, str] = {}
    owner_ids = {n["owner"] for n in visible if n["owner"]}
    if owner_ids:
        db = get_db()
        for oid in owner_ids:
            try:
                u = await db.get_user_by_id(oid)
            except Exception:
                u = None
            emails[oid] = (u or {}).get("email") or (u or {}).get("display_name") or oid[:8]
    for n in visible:
        _stamp_emails(n, emails)

    # Registry agents whose process is dead → "stopped" (context only).
    from captain_claw.flight_deck import server as _srv
    try:
        reg = _srv._load_process_registry()
    except Exception:
        reg = {}
    stopped = 0
    for entry in reg.values():
        owner = entry.get("owner") or None
        if not (is_admin or (owner and owner == user_id)):
            continue
        pid = entry.get("pid")
        if not pid or int(pid) not in by_pid:
            stopped += 1

    summary = {
        "roots": len(visible),
        "agents": sum(1 for n in visible if n["kind"] == "agent"),
        "hosted": sum(1 for n in visible if n["kind"] == "hosted-app"),
        "children": sum(n["descendant_count"] for n in visible),
        "total_cpu": round(sum(n["agg_cpu"] for n in visible), 1),
        "total_mem_mb": round(sum(n["agg_mem_mb"] for n in visible), 1),
        "stopped": stopped,
    }

    # Per-owner rollup (CPU + memory + process count), memory-heaviest first.
    # Whole subtrees are attributed to their root's owner — a child's memory
    # counts toward the user who owns the agent that spawned it.
    by_user: dict[str, dict] = {}
    for n in visible:
        key = n.get("owner_email") or "unknown"
        b = by_user.setdefault(key, {
            "owner_email": key,
            "owner": n["owner"],
            "roots": 0,
            "procs": 0,
            "cpu": 0.0,
            "mem_mb": 0.0,
        })
        b["roots"] += 1
        b["procs"] += 1 + n["descendant_count"]
        b["cpu"] += n["agg_cpu"]
        b["mem_mb"] += n["agg_mem_mb"]
    by_user_list = sorted(by_user.values(), key=lambda x: -x["mem_mb"])
    for b in by_user_list:
        b["cpu"] = round(b["cpu"], 1)
        b["mem_mb"] = round(b["mem_mb"], 1)

    return {
        "is_admin": is_admin,
        "available": bool(by_pid),
        "host": _host_vitals(),
        "summary": summary,
        "by_user": by_user_list,
        "trees": visible,
    }


def _perform_stop(pid: int, slug: str | None, is_root: bool, desc_pids: list[int]) -> list[int]:
    """Blocking stop — SIGTERM (→SIGKILL) descendants first, then the target.

    A tracked agent root is stopped via ``_do_stop_process(slug)`` so the
    registry is updated (marked intentionally stopped) rather than left dangling.
    """
    from captain_claw.flight_deck import server as _srv

    killed: list[int] = []
    for dpid in desc_pids:
        _srv._kill_pid(dpid)
        killed.append(dpid)
    if slug and is_root:
        _srv._do_stop_process(slug)
    else:
        _srv._kill_pid(pid)
    killed.append(pid)
    return killed


@router.post("/processes/{pid}/stop")
async def system_stop_process(
    pid: int,
    request: Request,
    tree: bool = Query(False, description="Also stop the target's descendants"),
    user: dict = Depends(get_current_user),
):
    """Stop a process the caller is allowed to stop.

    A non-admin may only stop processes under a tree they own; an admin may stop
    anything except the Flight Deck server itself and pid ≤ 1.
    """
    is_admin = user.get("role") == "admin"
    user_id = user["id"]

    if pid <= 1:
        raise HTTPException(400, "Refusing to signal pid ≤ 1")
    if pid == FD_SERVER_PID:
        raise HTTPException(400, "Refusing to stop the Flight Deck server itself")

    loop = asyncio.get_event_loop()
    by_pid, children = await loop.run_in_executor(None, _snapshot_processes)
    roots = _collect_roots()
    forest = _build_forest(by_pid, children, roots)

    # Locate the target and the root of the tree it lives in.
    target: dict | None = None
    owning_root: dict | None = None

    def walk(node: dict, root: dict) -> None:
        nonlocal target, owning_root
        if node["pid"] == pid:
            target, owning_root = node, root
            return
        for ch in node["children"]:
            walk(ch, root)

    for r in forest:
        walk(r, r)
        if target is not None:
            break

    if target is None or owning_root is None:
        raise HTTPException(404, "Process not found among Flight Deck processes")

    owner = owning_root["owner"]
    if not is_admin and (not owner or owner != user_id):
        raise HTTPException(403, "You can only stop your own processes")
    if owning_root["kind"] == "flight-deck" and target["pid"] == owning_root["pid"]:
        raise HTTPException(400, "Refusing to stop the Flight Deck server itself")

    desc_pids: list[int] = []
    if tree:
        def gather(n: dict) -> None:
            for ch in n["children"]:
                gather(ch)
                desc_pids.append(ch["pid"])
        gather(target)

    killed = await loop.run_in_executor(
        None, _perform_stop, pid, target.get("slug"), target["is_root"], desc_pids
    )
    return {
        "ok": True,
        "pid": pid,
        "killed": killed,
        "message": f"Stopped {len(killed)} process{'es' if len(killed) != 1 else ''}",
    }
