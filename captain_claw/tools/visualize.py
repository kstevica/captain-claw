"""Visualize tool: HTML in, the page and a PNG picture of it out.

Asked for a chart, a model used to write a plotting script, run it and often
end with nothing to show: a package that wasn't installed, a script that
saved no file, a picture nobody looked at. This tool takes the visual as HTML
(a fragment, or a whole page), saves it, renders it in headless Chromium and
screenshots it, so one call gives the interactive page for a browser and a
PNG for chats, documents and email. What went wrong while drawing — a script
error, a library that didn't load, a canvas left empty, a blank picture —
comes back in the result, for the model to fix and render again.

The page never touches the host. It is served from memory at an https
address that doesn't exist (so file:// is out of reach), and every other
request it makes is answered through the public-only HTTP client
(:func:`speaker.make_public_http_client`), so a page can't reach localhost,
the deck or the LAN. WebSockets are closed,
service workers blocked, and anything that slips past the router meets a dead
proxy.
"""

from __future__ import annotations

import asyncio
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool

log = get_logger(__name__)

# Pinned builds, loaded with `libraries` so the model never guesses a CDN URL.
# Chart.js comes with the date-fns adapter: time axes need one.
_LIBRARIES: dict[str, tuple[str, ...]] = {
    "chartjs": (
        "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js",
        "https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js",
    ),
    "echarts": ("https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js",),
    "plotly": ("https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",),
    "d3": ("https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",),
    "mermaid": ("https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js",),
    "leaflet": (
        "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css",
        "https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js",
    ),
}
_LIBRARY_ALIASES = {
    "chart.js": "chartjs", "chart": "chartjs", "apache-echarts": "echarts",
    "plotly.js": "plotly", "d3.js": "d3", "d3js": "d3",
}

# Where the page is served from: the router answers it, and .invalid never
# resolves, so nothing else can.
_PAGE_URL = "https://page.visualize.invalid/"
_PAGE_HOST = "page.visualize.invalid"
_DEFAULT_WIDTH = 1000
_DEFAULT_SCALE = 2.0
_START_HEIGHT = 800
# Telegram refuses a photo whose width + height pass 10,000 px.
_MAX_SIDES_PX = 9800
_MAX_REQUESTS = 200
_MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_MAX_PROBLEMS = 10
# Pinned library builds never change: keep them in memory between renders.
_CACHE_HOSTS = frozenset({"cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"})
_CACHE_MAX_BYTES = 40 * 1024 * 1024
_cache: dict[str, tuple[dict[str, str], bytes]] = {}
# Each render is its own Chromium; a few at once is plenty.
_renders = asyncio.Semaphore(2)

_DROP_RESPONSE_HEADERS = frozenset({
    "content-encoding", "content-length", "transfer-encoding", "connection",
    "keep-alive", "set-cookie", "strict-transport-security", "alt-svc",
})
_FORWARD_REQUEST_HEADERS = frozenset({"accept", "accept-language", "content-type"})

# Runs before the page's own scripts. A picture is one frame, so charts draw
# their final state at once: Chart.js and ECharts get animation off as their
# globals appear, ApexCharts reads window.Apex, CSS motion ends immediately.
_INIT_SCRIPT = r"""
(() => {
  const still = (lib, apply) => {
    let value;
    Object.defineProperty(window, lib, {
      configurable: true,
      get() { return value; },
      set(v) { value = v; try { apply(v); } catch (e) {} },
    });
  };
  still('Chart', (C) => { if (C && C.defaults) C.defaults.animation = false; });
  const stillInit = (init) => function (...args) {
    const chart = init.apply(this, args);
    const setOption = chart.setOption;
    chart.setOption = function (option, ...rest) {
      if (option && typeof option === 'object') option.animation = false;
      return setOption.call(this, option, ...rest);
    };
    return chart;
  };
  // ECharts' UMD build sets window.echarts to an empty object, then fills it.
  still('echarts', (E) => {
    if (!E || typeof E !== 'object') return;
    if (typeof E.init === 'function') { E.init = stillInit(E.init); return; }
    let init;
    Object.defineProperty(E, 'init', {
      configurable: true, enumerable: true,
      get() { return init; },
      set(v) { init = typeof v === 'function' ? stillInit(v) : v; },
    });
  });
  window.Apex = Object.assign({ chart: { animations: { enabled: false } } }, window.Apex || {});
  document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = '*,*::before,*::after{animation-duration:0s!important;'
      + 'animation-delay:0s!important;transition-duration:0s!important;'
      + 'transition-delay:0s!important}';
    (document.head || document.documentElement).appendChild(style);
  });
})();
"""

# Canvases a chart should have drawn on but didn't (a WebGL one can't be
# read this way, so it is left out).
_EMPTY_CANVASES_JS = r"""
() => {
  let total = 0, empty = 0;
  for (const c of document.querySelectorAll('canvas')) {
    const r = c.getBoundingClientRect();
    if (r.width < 4 || r.height < 4 || !c.width || !c.height) { total++; empty++; continue; }
    let ctx = null;
    try { ctx = c.getContext('2d'); } catch (e) {}
    if (!ctx) continue;
    total++;
    try {
      const t = document.createElement('canvas'); t.width = 48; t.height = 48;
      const x = t.getContext('2d'); x.drawImage(c, 0, 0, 48, 48);
      const d = x.getImageData(0, 0, 48, 48).data;
      let drawn = false;
      for (let i = 3; i < d.length; i += 4) if (d[i]) { drawn = true; break; }
      if (!drawn) empty++;
    } catch (e) {}
  }
  return { total, empty };
}
"""

# Where the content is: the box of the body's children (absolute ones too)
# plus the body's own padding, so a short chart isn't padded out to the
# window's height and a 600 px one doesn't sit in a 1000 px picture.
_CONTENT_BOX_JS = r"""
() => {
  const doc = document.documentElement, body = document.body;
  const width = doc.clientWidth;
  const scroll = Math.max(doc.scrollHeight, body ? body.scrollHeight : 0);
  if (!body) return { x: 0, width, height: scroll };
  const cs = getComputedStyle(body);
  const padL = Math.max(16, parseFloat(cs.paddingLeft || 0) + parseFloat(cs.marginLeft || 0));
  const padR = Math.max(16, parseFloat(cs.paddingRight || 0) + parseFloat(cs.marginRight || 0));
  let bottom = body.getBoundingClientRect().bottom + parseFloat(cs.marginBottom || 0);
  let left = Infinity, right = -Infinity;
  for (const el of body.children) {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    bottom = Math.max(bottom, r.bottom);
    left = Math.min(left, r.left);
    right = Math.max(right, r.right);
  }
  bottom = Math.ceil(bottom + window.scrollY);
  const height = bottom >= 40 ? Math.min(bottom, scroll) : scroll;
  const bareText = [...body.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
  if (!(right > left) || bareText) {
    return { x: 0, width, height };
  }
  const x = Math.max(0, Math.floor(left - padL));
  const w = Math.min(width, Math.ceil(right + padR)) - x;
  return w >= 100 ? { x, width: w, height } : { x: 0, width, height };
}
"""

_MERMAID_DONE_JS = (
    "() => !window.mermaid || "
    "!document.querySelector('.mermaid:not([data-processed])')"
)


@dataclass
class _Render:
    """What happened while the page drew."""

    problems: list[str] = field(default_factory=list)
    requests: int = 0

    def note(self, text: str) -> None:
        text = " ".join(str(text).split())[:300]
        if text and text not in self.problems and len(self.problems) < _MAX_PROBLEMS:
            self.problems.append(text)


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:48].strip("-") or "visualization"


def _libraries(raw: Any) -> tuple[list[str], list[str]]:
    """``(known, unknown)`` library names from the call."""
    if isinstance(raw, str):
        raw = [part for part in re.split(r"[\s,]+", raw) if part]
    known: list[str] = []
    unknown: list[str] = []
    for item in raw or []:
        name = str(item).strip().lower()
        name = _LIBRARY_ALIASES.get(name, name)
        if name in _LIBRARIES:
            if name not in known:
                known.append(name)
        elif name:
            unknown.append(str(item))
    return known, unknown


def _library_tags(names: list[str]) -> str:
    tags: list[str] = []
    for name in names:
        for url in _LIBRARIES[name]:
            if url.endswith(".css"):
                tags.append(f'<link rel="stylesheet" href="{url}">')
            else:
                tags.append(f'<script src="{url}"></script>')
        if name == "mermaid":
            tags.append("<script>mermaid.initialize({startOnLoad: true});</script>")
    return "\n".join(tags)


def _style_css() -> str:
    """CSS for the configured visualization style: palette as variables, fonts."""
    try:
        from captain_claw.visualization_style import load_visualization_style

        style = load_visualization_style()
    except Exception:
        return ""
    rules: list[str] = []
    colors = [c for c in style.color_palette if re.fullmatch(r"[#\w(),.%\s-]{1,40}", c)]
    if colors:
        rules.append(":root{" + "".join(f"--viz-{i}:{c};" for i, c in enumerate(colors, 1)) + "}")
    font = re.sub(r"[{}<>;]", "", style.font_primary).strip()
    if font:
        rules.append(f"body{{font-family:{font},system-ui,sans-serif}}")
    heading = re.sub(r"[{}<>;]", "", style.font_headings).strip()
    if heading:
        rules.append(f"h1,h2,h3,h4{{font-family:{heading},system-ui,sans-serif}}")
    return "".join(rules)


_BASE_CSS = (
    "*{box-sizing:border-box}"
    "html{background:#fff}"
    "body{margin:0;padding:24px;background:#fff;color:#1f2328;"
    'font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}'
    "h1,h2,h3,h4{margin:0 0 12px;line-height:1.25}"
)


def build_document(html_text: str, title: str, libraries: list[str]) -> str:
    """The page to save and render: a fragment wrapped in one, or the
    caller's own document with the libraries added to its head."""
    import html as _html

    head_bits = _library_tags(libraries)
    if re.search(r"<!doctype|<html[\s>]", html_text[:4000], re.I):
        if not head_bits:
            return html_text
        head = re.search(r"<head(?:\s[^>]*)?>", html_text, re.I)
        if head:
            return html_text[: head.end()] + "\n" + head_bits + html_text[head.end():]
        root = re.search(r"<html[^>]*>", html_text, re.I)
        if root:
            return (html_text[: root.end()] + f"\n<head>\n{head_bits}\n</head>"
                    + html_text[root.end():])
        return head_bits + "\n" + html_text
    page_title = _html.escape(title or "Visualization")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{page_title}</title>\n"
        f"<style>{_BASE_CSS}{_style_css()}</style>\n"
        + (head_bits + "\n" if head_bits else "")
        + "</head>\n<body>\n" + html_text.strip() + "\n</body>\n</html>\n"
    )


def _cached(url: str) -> tuple[dict[str, str], bytes] | None:
    return _cache.get(url)


def _remember(url: str, headers: dict[str, str], body: bytes) -> None:
    if (urlsplit(url).hostname or "").lower() not in _CACHE_HOSTS:
        return
    if len(body) > _CACHE_MAX_BYTES // 4:
        return
    _cache[url] = (headers, body)
    while sum(len(b) for _, b in _cache.values()) > _CACHE_MAX_BYTES and len(_cache) > 1:
        _cache.pop(next(iter(_cache)))


def _is_blank(png: bytes) -> bool:
    """One colour from edge to edge: nothing was drawn."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png)) as img:
            small = img.convert("RGB").resize((64, 64))
            return len(small.getcolors(64 * 64) or []) <= 1
    except Exception:
        return False


async def _launch(playwright: Any) -> Any:
    """Headless Chromium whose own network goes nowhere: every request the
    page makes is answered by the router, so a dead proxy (loopback
    included) only catches what the router can't see."""
    from captain_claw.tools.browser_session import _is_missing_browser_error, ensure_chromium_installed

    args = [
        "--proxy-server=http://127.0.0.1:9",
        "--proxy-bypass-list=<-loopback>",
        "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
        "--disable-background-networking",
        "--no-first-run",
    ]
    try:
        return await playwright.chromium.launch(headless=True, args=args)
    except Exception as exc:
        if not _is_missing_browser_error(exc):
            raise
        ok, detail = await ensure_chromium_installed()
        if not ok:
            raise RuntimeError(
                f"Chromium isn't installed and installing it failed ({detail}). "
                "Run `playwright install chromium` on the host."
            ) from exc
        return await playwright.chromium.launch(headless=True, args=args)


class VisualizeTool(Tool):
    """Render HTML visuals (charts, diagrams, dashboards) to an HTML page and a PNG."""

    name = "visualize"
    timeout_seconds = 120.0
    description = (
        "Make a chart, graph, diagram, dashboard, map, table or any other visual from HTML "
        "and get BOTH back: the HTML page (interactive, for a browser) and a PNG picture of "
        "it (for chats, documents, email). Use this instead of a Python/matplotlib script "
        "whenever an answer needs something visual — or part of one, e.g. a chart for a "
        "report. Pass `html` as a fragment (it is put in a clean page) or a whole document; "
        "put the data inline. `libraries` loads pinned builds — don't add <script src> for "
        "them yourself. Size elements explicitly (e.g. a chart in a div with a fixed "
        "height) and let them fill the page — set `width` for the picture's width. The "
        "result gives `Path:` (the PNG — it is shown/sent to the user) and "
        "`HTML:`; put the PNG in a .docx or .md with ![caption](path). Script errors, "
        "failed loads, empty canvases and a blank picture come back in the result: fix the "
        "HTML and call again. The page can load public URLs only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "html": {
                "type": "string",
                "description": (
                    "The visual as HTML: body content (with <style>/<script> as needed) or a "
                    "full document. Raw markup, not escaped."
                ),
            },
            "title": {
                "type": "string",
                "description": "Short title: the page title and the file name.",
            },
            "libraries": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_LIBRARIES)},
                "description": (
                    "Libraries to load before your code: chartjs (Chart.js 4 + date adapter), "
                    "echarts (5), plotly (2), d3 (7), mermaid (10 — write "
                    "<pre class=\"mermaid\">…</pre>, it renders itself), leaflet (1.9, with "
                    "its CSS)."
                ),
            },
            "width": {
                "type": "integer",
                "description": f"Page width in CSS pixels, 320-2400 (default {_DEFAULT_WIDTH}).",
            },
            "height": {
                "type": "integer",
                "description": (
                    "Fixed page height in CSS pixels: the picture is exactly width x height. "
                    "Leave out to capture the whole page."
                ),
            },
            "selector": {
                "type": "string",
                "description": "CSS selector of one element to picture alone (e.g. '#chart').",
            },
            "scale": {
                "type": "number",
                "description": "Pixel density of the PNG, 1-3 (default 2: sharp on phones).",
            },
            "wait_ms": {
                "type": "integer",
                "description": (
                    "Extra wait after the page settles, up to 10000 ms, for drawing that "
                    "takes time. Or set window.vizReady = true in your script when done."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional .html path (saved under saved/showcase/<session>/ by default); "
                    "the PNG goes next to it. The same path again replaces your earlier render."
                ),
            },
        },
        "required": ["html"],
    }

    async def execute(
        self,
        html: str = "",
        title: str = "",
        libraries: Any = None,
        width: Any = None,
        height: Any = None,
        selector: str = "",
        scale: Any = None,
        wait_ms: Any = None,
        output_path: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        import html as _html

        text = str(html or "")
        if "<" not in text and "&lt;" in text:
            text = _html.unescape(text)
        if not text.strip():
            return ToolResult(success=False, error="Missing required argument: html (the visual as HTML).")
        try:
            from playwright.async_api import async_playwright  # noqa: F401
        except ImportError:
            return ToolResult(success=False, error=(
                "Playwright isn't installed on this host, so nothing can be rendered. "
                "Install it with `pip install playwright && playwright install chromium`."))

        title = " ".join(str(title or "").split())[:120]
        names, unknown = _libraries(libraries)
        page_width = _clamp_int(width, _DEFAULT_WIDTH, 320, 2400)
        page_height = _clamp_int(height, 0, 0, 9000) or None
        if page_height is not None:
            page_height = max(page_height, 100)
        density = _clamp_float(scale, _DEFAULT_SCALE, 1.0, 3.0)
        extra_wait = _clamp_int(wait_ms, 0, 0, 10000)

        try:
            html_path = self._target(output_path, title, kwargs)
        except PermissionError as exc:
            return ToolResult(success=False, error=str(exc))
        png_path = html_path.with_suffix(".png")

        document = build_document(text, title, names)
        # Local pictures (<img src="saved/media/…">) go in as base64, as the
        # write tool does for an HTML report, so the page stands alone.
        document, images_note = await WriteTool._report_images(document, html_path, ".html", kwargs)
        self._save(html_path, document.encode("utf-8"))
        self._register(html_path, kwargs)

        render = _Render()
        for item in unknown:
            render.note(f"unknown library '{item}' (not loaded) — use one of: {', '.join(sorted(_LIBRARIES))}")
        try:
            async with _renders:
                png, size = await self._render(
                    document, render,
                    width=page_width, height=page_height, scale=density,
                    selector=str(selector or "").strip(), extra_wait=extra_wait,
                    wait_mermaid="mermaid" in names, wait_flag="vizReady" in document,
                )
        except Exception as exc:
            log.warning("visualize render failed", error=str(exc)[:300])
            reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]
            return ToolResult(success=False, error=(
                f"The page couldn't be rendered: {reason}\nThe page as sent: {html_path}"
                + _problem_lines(render)))

        if _is_blank(png):
            return ToolResult(success=False, error=(
                "The picture came out blank — nothing was drawn. Fix the HTML and call "
                f"visualize again.\nThe page as sent: {html_path}" + _problem_lines(render)))

        self._save(png_path, png)
        self._register(png_path, kwargs)
        css_w, css_h, px_w, px_h = size
        lines = [
            "Rendered the visualization.",
            f"Path: {png_path}",
            f"HTML: {html_path}",
            f"Picture: {css_w}x{css_h} CSS px, {px_w}x{px_h} PNG, {len(png) / 1024:.1f} KB",
        ]
        if images_note:
            lines.append(images_note)
        if render.problems:
            lines.append("Problems while drawing — check the picture shows everything; if not, "
                         "fix the HTML and call visualize again:")
            lines.extend(f"- {p}" for p in render.problems)
        else:
            # Without this a model may keep re-rendering, trying to "show" it.
            lines.append("Done — don't render it again. Reply now, naming the PNG and the HTML "
                         "file (the interactive version).")
        return ToolResult(success=True, content="\n".join(lines))

    # ── Files ──────────────────────────────────────────────────────

    @staticmethod
    def _target(output_path: str, title: str, kwargs: dict[str, Any]) -> Path:
        """Where the HTML goes. A path the model names is replaced when it may
        write there; otherwise a new name next to what is already there."""
        from captain_claw import speaker

        requested = str(output_path or "").strip()
        named = bool(requested)
        if not named:
            requested = f"showcase/{_slug(title)}.html"
        target = WriteTool.resolve_local_target(requested, kwargs)
        if target.suffix.lower() not in (".html", ".htm"):
            target = target.with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)

        def taken(path: Path) -> bool:
            if not (path.exists() or path.with_suffix(".png").exists()):
                return False
            if not named:
                return True
            return speaker.member_bound() and not (
                speaker.member_may_write(path) and speaker.member_may_write(path.with_suffix(".png")))

        if taken(target):
            stem = target.stem
            for n in range(2, 1000):
                candidate = target.with_name(f"{stem}-{n}{target.suffix}")
                if not (candidate.exists() or candidate.with_suffix(".png").exists()):
                    target = candidate
                    break
        if speaker.member_bound() and not speaker.member_may_write(target):
            raise PermissionError(
                speaker.PATH_REFUSED_PREFIX + "you can only save in your own folders — "
                "leave output_path out, or pass a new file name.")
        return target

    @staticmethod
    def _save(path: Path, data: bytes) -> None:
        from captain_claw import saved_attribution
        from captain_claw.vfs import record_author

        prior = saved_attribution.prior_creator(path)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        saved_attribution.note_write(path, prior)
        record_author(path)

    @staticmethod
    def _register(path: Path, kwargs: dict[str, Any]) -> None:
        registry = kwargs.get("_file_registry")
        if registry is None:
            return
        try:
            registry.register(logical_path=str(path), physical_path=str(path),
                              task_id=str(kwargs.get("_task_id", "")))
        except Exception:
            pass

    # ── Rendering ──────────────────────────────────────────────────

    async def _render(
        self, document: str, render: _Render, *, width: int, height: int | None,
        scale: float, selector: str, extra_wait: int, wait_mermaid: bool, wait_flag: bool,
    ) -> tuple[bytes, tuple[int, int, int, int]]:
        from playwright.async_api import async_playwright

        from captain_claw import speaker

        client = speaker.make_public_http_client()
        try:
            async with async_playwright() as pw:
                browser = await _launch(pw)
                try:
                    context = await browser.new_context(
                        viewport={"width": width, "height": height or _START_HEIGHT},
                        device_scale_factor=scale,
                        service_workers="block",
                        accept_downloads=False,
                        reduced_motion="reduce",
                    )

                    async def answer(route: Any) -> None:
                        await self._answer(route, client, render, document)

                    await context.route("**/*", answer)
                    if hasattr(context, "route_web_socket"):  # Playwright 1.48+
                        await context.route_web_socket(re.compile(".*"), _close_socket)
                    await context.add_init_script(_INIT_SCRIPT)
                    page = await context.new_page()
                    page.on("console", lambda msg: msg.type == "error" and render.note(f"console: {msg.text}"))
                    page.on("pageerror", lambda exc: render.note(f"script error: {exc}"))
                    await page.goto(_PAGE_URL, wait_until="load", timeout=30_000)
                    await self._settle(page, render, wait_mermaid, wait_flag, extra_wait)
                    return await self._shoot(page, render, width=width, height=height,
                                             scale=scale, selector=selector)
                finally:
                    await browser.close()
        finally:
            await client.aclose()

    async def _answer(self, route: Any, client: Any, render: _Render, document: str) -> None:
        """Answer one request of the page: the page itself from memory, the
        rest through the public-only client."""
        request = route.request
        url = request.url
        parts = urlsplit(url)
        if (parts.hostname or "").lower() == _PAGE_HOST:
            if parts.path in ("", "/") and request.resource_type == "document":
                await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=document)
            else:
                render.note(f"{parts.path[:120]} is a relative address — the page has no folder; "
                            "use a full https:// URL, or put local pictures in as "
                            "<img src=\"saved/…\"> (they are embedded)")
                await route.fulfill(status=404, body="")
            return
        if parts.scheme not in ("http", "https"):
            render.note(f"blocked {url[:120]} — the page can load public http(s) URLs only "
                        "(put local pictures in as <img src=\"saved/…\">; they are embedded)")
            await route.abort("blockedbyclient")
            return
        render.requests += 1
        if render.requests > _MAX_REQUESTS:
            render.note(f"more than {_MAX_REQUESTS} requests — the rest were dropped")
            await route.abort("blockedbyclient")
            return
        if request.method == "GET":
            hit = _cached(url)
            if hit is not None:
                await route.fulfill(status=200, headers=hit[0], body=hit[1])
                return
        headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_REQUEST_HEADERS}
        try:
            async with client.stream(request.method, url, headers=headers,
                                     content=request.post_data_buffer) as resp:
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > _MAX_RESPONSE_BYTES:
                        raise ValueError(f"larger than {_MAX_RESPONSE_BYTES // (1024 * 1024)} MB")
                    chunks.append(chunk)
                body = b"".join(chunks)
                status = resp.status_code
                out_headers = {k: v for k, v in resp.headers.items()
                               if k.lower() not in _DROP_RESPONSE_HEADERS}
        except Exception as exc:
            render.note(f"couldn't load {url[:160]}: {str(exc)[:160] or type(exc).__name__}")
            try:
                await route.abort("failed")
            except Exception:
                pass
            return
        if status >= 400:
            render.note(f"{url[:160]} answered HTTP {status}")
        elif request.method == "GET" and status == 200:
            _remember(url, out_headers, body)
        try:
            await route.fulfill(status=status, headers=out_headers, body=body)
        except Exception:
            pass

    @staticmethod
    async def _settle(page: Any, render: _Render, wait_mermaid: bool, wait_flag: bool,
                      extra_wait: int) -> None:
        """Let late requests, fonts and async drawing finish."""
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            render.note("the page was still loading after 10 s — pictured as it was")
        try:
            await page.evaluate("document.fonts ? document.fonts.ready.then(() => true) : true")
        except Exception:
            pass
        if wait_mermaid:
            try:
                await page.wait_for_function(_MERMAID_DONE_JS, timeout=10_000)
            except Exception:
                render.note("mermaid didn't finish drawing within 10 s (check the diagram syntax)")
        if wait_flag:
            try:
                await page.wait_for_function("() => window.vizReady === true", timeout=10_000)
            except Exception:
                render.note("window.vizReady wasn't set to true within 10 s")
        # Two frames, so what the last script changed is painted.
        await page.evaluate(
            "new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))")
        if extra_wait:
            await page.wait_for_timeout(extra_wait)
        try:
            canvases = await page.evaluate(_EMPTY_CANVASES_JS)
            if canvases and canvases.get("empty"):
                render.note(f"{canvases['empty']} of {canvases['total']} canvas element(s) drew "
                            "nothing (a chart that failed, or a container with no height)")
        except Exception:
            pass

    @staticmethod
    async def _shoot(page: Any, render: _Render, *, width: int, height: int | None,
                     scale: float, selector: str) -> tuple[bytes, tuple[int, int, int, int]]:
        """The PNG and its size: ``(css_w, css_h, px_w, px_h)``."""
        if selector:
            target = page.locator(selector).first
            if await page.locator(selector).count() == 0:
                render.note(f"selector {selector!r} matched nothing — pictured the whole page")
            else:
                box = await target.bounding_box() or {"width": width, "height": 0}
                css_w, css_h = int(box["width"]), int(box["height"])
                if css_w < 2 or css_h < 2:
                    render.note(f"{selector!r} has no size on the page — pictured the whole page")
                else:
                    use_css = (css_w + css_h) * scale > _MAX_SIDES_PX
                    png = await target.screenshot(type="png", timeout=15_000,
                                                  scale="css" if use_css else "device")
                    return png, _sized(css_w, css_h, 1.0 if use_css else scale)

        if height is not None:
            use_css = (width + height) * scale > _MAX_SIDES_PX
            png = await page.screenshot(type="png", timeout=15_000,
                                        scale="css" if use_css else "device")
            return png, _sized(width, height, 1.0 if use_css else scale)

        box = await page.evaluate(_CONTENT_BOX_JS) or {}
        x = int(box.get("x") or 0)
        css_w = int(box.get("width") or width)
        full = int(box.get("height") or _START_HEIGHT)
        use_css = (css_w + full) * scale > _MAX_SIDES_PX
        css_h = full
        if css_w + full > _MAX_SIDES_PX:
            css_h = _MAX_SIDES_PX - css_w
            render.note(f"the page is {full} px tall; only the top {css_h} px are pictured — "
                        "split it into several visuals, or picture one part with `selector`")
        png = await page.screenshot(
            type="png", timeout=15_000, full_page=True,
            clip={"x": x, "y": 0, "width": css_w, "height": css_h},
            scale="css" if use_css else "device",
        )
        return png, _sized(css_w, css_h, 1.0 if use_css else scale)


async def _close_socket(ws: Any) -> None:
    await ws.close()


def _sized(css_w: int, css_h: int, scale: float) -> tuple[int, int, int, int]:
    return css_w, css_h, round(css_w * scale), round(css_h * scale)


def _problem_lines(render: _Render) -> str:
    if not render.problems:
        return ""
    return "\nProblems while drawing:\n" + "\n".join(f"- {p}" for p in render.problems)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))
