"""Freshness probe for Meta Ray-Ban Display Web Apps.

Run:  python3 server.py [--port 8765] [--host 0.0.0.0]

Every GET / response is server-rendered with values that change each request:
  - Page-render timestamp (server-side, baked into the HTML)
  - Server uptime + system load + memory
  - A random token shown big

If two consecutive loads on the glasses show the same token, the HTML was
cached. If the token changes on each reload, the app is loaded fresh from
the server every time.

A /api/status JSON endpoint and a "Refresh" button let you separately test
whether live network calls work after the page has loaded.

No external deps. Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import secrets
import socket
import string
import time
from datetime import datetime, timezone

UTC = timezone.utc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROCESS_STARTED_AT = time.time()
REQUEST_COUNTER = {"n": 0}


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _system_uptime() -> float | None:
    """System (not process) uptime in seconds, where available."""
    try:
        with open("/proc/uptime", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except FileNotFoundError:
        pass
    try:  # macOS / BSD
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))

        class Timeval(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]

        tv = Timeval()
        size = ctypes.c_size_t(ctypes.sizeof(tv))
        mib = (ctypes.c_int * 2)(1, 21)  # CTL_KERN, KERN_BOOTTIME
        if libc.sysctl(mib, 2, ctypes.byref(tv), ctypes.byref(size), None, 0) != 0:
            return None
        return time.time() - tv.tv_sec
    except Exception:
        return None


def _load_avg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


def _memory() -> dict[str, int] | None:
    """Best-effort memory snapshot in bytes; returns None if unsupported."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {}
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.strip().split()
                if not parts:
                    continue
                info[key.strip()] = int(parts[0]) * 1024
            total = info.get("MemTotal")
            avail = info.get("MemAvailable") or info.get("MemFree")
            if total and avail is not None:
                return {"total": total, "available": avail, "used": total - avail}
    except FileNotFoundError:
        pass
    try:  # macOS
        import subprocess

        ps = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        total = int(ps.stdout.strip())
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
        page_size = 4096
        free_pages = active_pages = inactive_pages = wired_pages = 0
        for line in vm.stdout.splitlines():
            if "page size of" in line:
                page_size = int(line.split("page size of")[1].split()[0])
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages active:"):
                active_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages wired down:"):
                wired_pages = int(line.split(":")[1].strip().rstrip("."))
        avail = (free_pages + inactive_pages) * page_size
        used = (active_pages + wired_pages) * page_size
        return {"total": total, "available": avail, "used": used}
    except Exception:
        return None


def _hr_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{n} B"


def _random_token() -> str:
    return "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def _snapshot() -> dict:
    now_dt = datetime.now(UTC)
    proc_up = time.time() - PROCESS_STARTED_AT
    sys_up = _system_uptime()
    load = _load_avg()
    mem = _memory()

    REQUEST_COUNTER["n"] += 1

    snap = {
        "server_time_iso": now_dt.isoformat(),
        "server_time_unix": now_dt.timestamp(),
        "process_uptime_seconds": proc_up,
        "process_uptime_human": _format_seconds(proc_up),
        "system_uptime_seconds": sys_up,
        "system_uptime_human": _format_seconds(sys_up) if sys_up is not None else None,
        "load_avg": list(load) if load else None,
        "memory": mem,
        "memory_human": (
            {k: _hr_bytes(v) for k, v in mem.items()} if mem else None
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "random_token": _random_token(),
        "random_int": random.randint(100000, 999999),
        "request_number": REQUEST_COUNTER["n"],
    }
    return snap


# Glasses-friendly: dark bg, very large monospace, no images, no external assets.
PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>MRBD Freshness Probe</title>
<style>
  :root {{ color-scheme: dark; }}
  html, body {{
    margin: 0; padding: 0; background: #000; color: #eaffea;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 22px; line-height: 1.35;
  }}
  body {{ padding: 18px 22px; }}
  h1 {{
    font-size: 26px; margin: 0 0 12px 0; color: #9fff9f;
    letter-spacing: 0.5px;
  }}
  .token {{
    font-size: 64px; font-weight: 700; letter-spacing: 4px;
    color: #ffeb6b; background: #1a1a00; padding: 10px 16px;
    border-radius: 8px; display: inline-block; margin: 6px 0 14px 0;
  }}
  .row {{ margin: 4px 0; }}
  .k {{ color: #88c; }}
  .v {{ color: #fff; }}
  .small {{ font-size: 18px; color: #99a; margin-top: 12px; }}
  button {{
    font: inherit; font-size: 22px;
    background: #0a3a0a; color: #cfc; border: 1px solid #4a8;
    padding: 10px 18px; margin-top: 14px; border-radius: 6px;
  }}
  #live {{ margin-top: 10px; color: #ffd; }}
</style>
</head>
<body>
<h1>MRBD freshness probe</h1>

<div class="row"><span class="k">SSR token  </span><span class="v"><span class="token">{token}</span></span></div>
<div class="row"><span class="k">SSR random </span><span class="v">{rand_int}</span></div>
<div class="row"><span class="k">Req #      </span><span class="v">{req_n}</span></div>

<div class="row"><span class="k">server time</span> <span class="v">{server_time}</span></div>
<div class="row"><span class="k">proc uptime</span> <span class="v">{proc_up}</span></div>
<div class="row"><span class="k">sys  uptime</span> <span class="v">{sys_up}</span></div>
<div class="row"><span class="k">load avg   </span> <span class="v">{load}</span></div>
<div class="row"><span class="k">memory     </span> <span class="v">{mem}</span></div>

<div class="small">{host} · {plat} · py {pyv}</div>

<button id="b">Refresh live (fetch /api/status)</button>
<div id="live"></div>

<script>
  // Local JS clock — if this ticks but the SSR token above is stale on
  // reload, the HTML is cached but JS still runs.
  const ssrUnix = {ssr_unix};
  const localStart = Date.now();
  const live = document.getElementById('live');
  function tick() {{
    const elapsed = (Date.now() - localStart) / 1000;
    const projected = new Date((ssrUnix + elapsed) * 1000).toISOString();
    live.textContent = 'js clock (since load): ' + elapsed.toFixed(1) + 's  ·  projected server: ' + projected;
  }}
  setInterval(tick, 500); tick();

  document.getElementById('b').addEventListener('click', async () => {{
    const r = await fetch('/api/status', {{cache: 'no-store'}});
    const j = await r.json();
    live.textContent = 'live token: ' + j.random_token + '  ·  time: ' + j.server_time_iso;
  }});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MRBDProbe/1.0"

    def _no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._render_page()
        elif self.path == "/api/status":
            self._render_status()
        elif self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self._no_cache_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self._no_cache_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found")

    def _render_page(self) -> None:
        snap = _snapshot()
        load = snap["load_avg"]
        load_str = " ".join(f"{x:.2f}" for x in load) if load else "n/a"
        mem_h = snap["memory_human"]
        mem_str = (
            f"used {mem_h['used']} / total {mem_h['total']} (avail {mem_h['available']})"
            if mem_h else "n/a"
        )
        html = PAGE_TEMPLATE.format(
            token=snap["random_token"],
            rand_int=snap["random_int"],
            req_n=snap["request_number"],
            server_time=snap["server_time_iso"],
            proc_up=snap["process_uptime_human"],
            sys_up=snap["system_uptime_human"] or "n/a",
            load=load_str,
            mem=mem_str,
            host=snap["hostname"],
            plat=snap["platform"],
            pyv=snap["python"],
            ssr_unix=snap["server_time_unix"],
        ).encode("utf-8")
        self.send_response(200)
        self._no_cache_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _render_status(self) -> None:
        body = json.dumps(_snapshot(), default=str).encode("utf-8")
        self.send_response(200)
        self._no_cache_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        print(f"[{ts}] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MRBD freshness probe listening on http://{args.host}:{args.port}/", flush=True)
    print("Endpoints:  /  (HTML page)   /api/status (JSON)   /healthz", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srv.server_close()


if __name__ == "__main__":
    main()
