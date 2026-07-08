"""Web fetch tools for retrieving web page content."""

import asyncio
import hashlib
import os
import re
import shutil
import time
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


# ── R10: source corpus ───────────────────────────────────────────────
# When a research run opts into the source corpus (env CLAW_SOURCE_CORPUS set at
# spawn), a primary fetch saves the FULL page text into the run's shared VFS
# folder (sources/) — which the Research Map then indexes — and returns only a
# head + a pointer. That gives depth (nothing is lost to truncation; the reporter
# and the claim-checker can page-read every source) WITHOUT blowing up any one
# worker's context. Off by default → identical to today's behaviour.

def _corpus_enabled() -> bool:
    return str(os.environ.get("CLAW_SOURCE_CORPUS", "")).strip().lower() in ("1", "true", "yes")


def _slugify_url(url: str) -> str:
    s = re.sub(r"^https?://", "", url or "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s[:80] or "page"


def _save_source_to_corpus(url: str, content: str) -> str | None:
    """Save full fetched text to ``vfs:<project>/sources/`` for the whole run to
    reuse. Returns the ``vfs:`` pointer saved, or ``None`` if disabled / on error."""
    if not _corpus_enabled() or not (content or "").strip():
        return None
    try:
        from captain_claw import vfs
        root = vfs.project_root(create=True)
        sources = root / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:8]
        name = f"{_slugify_url(url)}-{h}.md"
        header = f"# Source: {url}\n\n<!-- fetched {int(time.time())} · {len(content)} chars -->\n\n"
        (sources / name).write_text(header + content, encoding="utf-8")
        return f"vfs:{vfs.default_project()}/sources/{name}"
    except Exception as e:  # noqa: BLE001 — corpus is best-effort, never break a fetch
        log.warning("web_fetch corpus save failed", url=url, error=str(e))
        return None


_CORPUS_HEAD_CHARS = 8000  # how much of a corpus-saved page to inline as a preview


def _corpus_output(url: str, status_code: int, mode: str, raw_len: int,
                   content: str, saved: str) -> str:
    """Build the head+pointer response for a corpus-saved fetch."""
    head = content[:_CORPUS_HEAD_CHARS]
    out = (f"[URL: {url}]\n[Status: {status_code}]\n[Mode: {mode}]\n"
           f"[Size: {raw_len} chars · {len(content)} text chars]\n"
           f"[Full text saved to {saved} — search it with the `researchmap` tool "
           f"or read the file for anything past the preview below]\n\n{head}")
    if len(content) > _CORPUS_HEAD_CHARS:
        out += f"\n\n... [preview only — the full {len(content)} chars are in {saved}]"
    return out

# ── Optional Playwright dependency ───────────────────────────────────

try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:  # pragma: no cover
    _HAS_PLAYWRIGHT = False
    async_playwright = None  # type: ignore[assignment,misc]

# ── Google Drive URL blocking ─────────────────────────────────────────

_GDRIVE_HOSTS = (
    "docs.google.com", "drive.google.com",
    "sheets.google.com", "slides.google.com",
)


def _is_google_drive_url(url: str) -> bool:
    """Return True if *url* points to Google Drive/Docs/Sheets/Slides."""
    try:
        host = urlparse(url).hostname or ""
        return any(host == h or host.endswith("." + h) for h in _GDRIVE_HOSTS)
    except Exception:
        return False


_GDRIVE_FETCH_BLOCK_MSG = (
    "Cannot fetch Google Drive/Docs URLs via web_fetch (requires authentication). "
    "Use the gws tool instead:\n"
    "  - gws(action='docs_read', file_id='...') to read Google Docs content\n"
    "  - gws(action='drive_download', file_id='...') to download files\n"
    "  - gws(action='drive_info', file_id='...') for file metadata\n"
    "The docs_read action returns the full document text inline."
)


def _make_http_client() -> httpx.AsyncClient:
    """Create a shared-style HTTP client."""
    return httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Captain Claw/0.1.0 (Web Fetch Tool)",
        },
    )


def _extract_readable_text(html: str, base_url: str | None = None) -> str:
    """Extract human-readable text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas"]):
        tag.decompose()

    # Preserve links in readable text so later turns can reference sources.
    for anchor in soup.find_all("a"):
        href = (anchor.get("href") or "").strip()
        label = anchor.get_text(" ", strip=True)
        if not href:
            continue
        absolute = urljoin(base_url, href) if base_url else href
        if label:
            anchor.replace_with(f"{label} ({absolute})")
        else:
            anchor.replace_with(absolute)

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    raw_text = soup.get_text(separator="\n")
    lines: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    if title and not text.startswith(title):
        return f"{title}\n\n{text}" if text else title
    return text


_LOAD_MORE_PATTERNS = [
    "load more", "show more", "view more", "see more",
    "more results", "load all", "show all", "view all",
]

_LOAD_MORE_SELECTORS = [
    # Common CSS selectors for "load more" buttons
    "button",
    "a.load-more", "a.show-more", "a.view-more",
    "[class*='load-more']", "[class*='loadmore']",
    "[class*='show-more']", "[class*='showmore']",
    "[class*='view-more']", "[class*='viewmore']",
    "[data-action='load-more']", "[data-action='loadmore']",
]


# ── One-time, best-effort Chromium auto-install ──────────────────────
# The browser binary is often absent even when the Playwright python package
# is installed. Rather than depend on the agent self-healing (unreliable), we
# kick off `playwright install chromium` ONCE in the background on the first
# deep-fetch need. The triggering call falls back to fast content; subsequent
# calls get a working browser. Fire-and-forget so we never hang a tool call.

_browser_install_attempted = False
_browser_install_done = False
_browser_install_task = None  # hold a reference so the task isn't GC'd mid-run


async def _do_browser_install() -> None:
    global _browser_install_done
    import sys
    try:
        log.info("web_fetch: installing Playwright Chromium in the background (one-time)…")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "playwright", "install", "chromium",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        _browser_install_done = rc == 0
        log.info("web_fetch: Chromium install finished (rc=%s)", rc)
    except Exception as e:  # pragma: no cover
        log.warning("web_fetch: Chromium install failed: %s", e)


def _kick_browser_install() -> bool:
    """Start a one-time background Chromium install. Sync + atomic (no await),
    so concurrent callers can't double-trigger. Returns True if an install is
    running/queued, False if not possible (no Playwright pkg / no loop)."""
    global _browser_install_attempted, _browser_install_task
    if not _HAS_PLAYWRIGHT or _browser_install_done:
        return False
    if _browser_install_attempted:
        return True
    _browser_install_attempted = True
    try:
        _browser_install_task = asyncio.create_task(_do_browser_install())
        return True
    except RuntimeError:
        return False


async def _deep_fetch_page(page, url: str, max_scrolls: int = 20, scroll_pause: float = 1.0, max_seconds: float = 22.0) -> str:
    """Render *url* in an already-open Playwright *page*: navigate, auto-scroll,
    click 'load more', and return the final HTML. Reusable across a shared
    browser (one context/page per URL) so batch fetches don't relaunch Chromium.
    """
    # domcontentloaded fires when the HTML is parsed — reliable. networkidle
    # never settles on ad/tracker-heavy pages (a big chunk of the real web),
    # so it just burns the time budget and times out with nothing. Cap total
    # work with a deadline and return whatever has rendered on timeout.
    deadline = asyncio.get_running_loop().time() + max_seconds
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=min(20000, int(max_seconds * 1000)))
    except Exception as exc:
        log.debug("deep_fetch: goto did not fully settle for %s (%s); using partial content", url, exc)

    prev_height = 0
    stable_rounds = 0

    for _ in range(max_scrolls):
        if asyncio.get_running_loop().time() >= deadline:
            break
        # Try clicking a "load more" button first.
        clicked = False
        for selector in _LOAD_MORE_SELECTORS:
            try:
                elements = await page.query_selector_all(selector)
                for el in elements:
                    text = (await el.inner_text()).strip().lower()
                    if any(pat in text for pat in _LOAD_MORE_PATTERNS):
                        if await el.is_visible():
                            await el.click()
                            clicked = True
                            await asyncio.sleep(scroll_pause)
                            break
            except Exception:
                continue
            if clicked:
                break

        # Scroll to bottom.
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(scroll_pause)

        # Check if page height grew.
        cur_height = await page.evaluate("document.body.scrollHeight")
        if cur_height == prev_height and not clicked:
            stable_rounds += 1
            if stable_rounds >= 2:
                break  # No new content after 2 stable rounds.
        else:
            stable_rounds = 0
        prev_height = cur_height

    return await page.content()


async def _deep_fetch(url: str, max_scrolls: int = 20, scroll_pause: float = 1.0, max_seconds: float = 22.0) -> str:
    """Single-URL deep fetch: launch a browser, render one page, return HTML."""
    if not _HAS_PLAYWRIGHT:
        raise RuntimeError(
            "deep_fetch requires Playwright. "
            "Install with: pip install playwright && playwright install chromium"
        )

    pw = await async_playwright().start()
    try:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception:
            # Browser binary missing — kick a one-time background install so the
            # next attempt works, then surface the actionable error this time.
            _kick_browser_install()
            raise
        page = await browser.new_page(
            user_agent="Captain Claw/0.1.0 (Web Fetch Tool)",
            viewport={"width": 1280, "height": 800},
        )
        html = await _deep_fetch_page(page, url, max_scrolls, scroll_pause, max_seconds)
    finally:
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass

    return html


class WebFetchTool(Tool):
    """Fetch web page content as clean readable text (ALWAYS text mode)."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return clean readable text (no HTML). "
        "This is the default tool for reading web pages. Always extracts text. "
        "Uses a headless browser by default (deep_fetch=true) for JS-rendered pages; "
        "pass deep_fetch=false for a faster plain HTTP fetch on simple/static pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters to return (default from config, typically 100000)",
            },
            "deep_fetch": {
                "type": "boolean",
                "description": (
                    "Use a headless browser (Playwright) to render JavaScript, "
                    "auto-scroll, and click 'load more' buttons to capture all content. "
                    "Slower but necessary for JS-rendered or lazy-loaded pages. Default: true. "
                    "Set to false for a faster plain HTTP fetch on simple/static pages."
                ),
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = _make_http_client()

    async def execute(
        self,
        url: str,
        max_chars: int | None = None,
        deep_fetch: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page and extract readable text via BeautifulSoup.

        Google Drive/Docs URLs are blocked when the ``gws`` CLI is
        available because they require authentication that web_fetch
        cannot provide.

        Args:
            url: URL to fetch
            max_chars: Max characters to return
            deep_fetch: Use headless browser to render JS and auto-scroll

        Returns:
            ToolResult with extracted readable text
        """
        # Hard guard: strip any extract_mode — web_fetch ALWAYS returns text.
        kwargs.pop("extract_mode", None)

        # Block Google Drive URLs when gws is available.
        if _is_google_drive_url(url) and shutil.which("gws"):
            return ToolResult(success=False, error=_GDRIVE_FETCH_BLOCK_MSG)

        try:
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 100000))
            effective_max_chars = configured_max if max_chars is None else int(max_chars)
            effective_max_chars = max(1, effective_max_chars)

            if deep_fetch:
                log.info("Fetching URL (deep_fetch mode)", url=url)
                raw_html = await _deep_fetch(url)
                status_code = 200  # Playwright doesn't expose status easily
                mode = "deep"
            else:
                log.info("Fetching URL (text mode)", url=url)
                response = await self.client.get(url)
                response.raise_for_status()
                raw_html = response.text
                status_code = response.status_code
                mode = "text"

            content = _extract_readable_text(raw_html, base_url=url)

            # R10: in a corpus run, save the FULL text and return a head + pointer.
            saved = _save_source_to_corpus(url, content)
            if saved:
                return ToolResult(
                    success=True,
                    content=_corpus_output(url, status_code, mode, len(raw_html), content, saved),
                )

            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = f"[URL: {url}]\n"
            output += f"[Status: {status_code}]\n"
            output += f"[Mode: {mode}]\n"
            output += f"[Size: {len(raw_html)} chars]\n\n"
            output += content

            return ToolResult(
                success=True,
                content=output,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            log.error("Fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class WebGetTool(Tool):
    """Fetch raw HTML from a URL (for scraping, parsing, DOM inspection)."""

    name = "web_get"
    description = (
        "Fetch a URL and return the raw HTML source. "
        "Use ONLY when you need the actual HTML markup (scraping, DOM analysis, CSS selectors). "
        "For normal page reading, use web_fetch instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": "Maximum characters of raw HTML to return (default from config, typically 100000)",
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = _make_http_client()

    async def execute(
        self,
        url: str,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page and return raw HTML.

        Args:
            url: URL to fetch
            max_chars: Max characters to return

        Returns:
            ToolResult with raw HTML content
        """
        # Block Google Drive URLs when gws is available.
        if _is_google_drive_url(url) and shutil.which("gws"):
            return ToolResult(success=False, error=_GDRIVE_FETCH_BLOCK_MSG)

        try:
            log.info("Fetching URL (raw HTML mode)", url=url)
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 100000))
            effective_max_chars = configured_max if max_chars is None else int(max_chars)
            effective_max_chars = max(1, effective_max_chars)

            response = await self.client.get(url)
            response.raise_for_status()

            content = response.text
            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = f"[URL: {url}]\n"
            output += f"[Status: {response.status_code}]\n"
            output += f"[Mode: html]\n"
            output += f"[Size: {len(response.text)} chars]\n\n"
            output += content

            return ToolResult(
                success=True,
                content=output,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            log.error("Fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ── web_fetch_batch: parallel multi-URL fetch with fast→deep self-correction ──

_JS_SHELL_MARKERS = (
    "enable javascript", "please enable js", "javascript is required",
    "you need to enable javascript", "requires javascript",
)


def _looks_thin(content: str, status: int, min_chars: int, raw_html_len: int) -> bool:
    """Heuristic: did the fast HTTP fetch fail to capture real content
    (HTTP error, a JS shell, or content hidden behind client-side rendering)?
    If so, escalate to deep mode.

    A page with little text is only "thin" if its HTML is much larger than the
    extracted text (i.e. content is JS-rendered) — a genuinely short page
    (e.g. example.com) has little HTML *and* little text and is NOT escalated.
    """
    if status >= 400:
        return True
    text = (content or "").strip()
    low = text[:2000].lower()
    if any(m in low for m in _JS_SHELL_MARKERS):
        return True
    if len(text) < min_chars:
        # Short text — escalate only if the HTML is big relative to the text,
        # which signals JS-rendered content the plain fetch couldn't see.
        return raw_html_len > 4000 and len(text) < raw_html_len * 0.1
    return False


class _Outcome:
    __slots__ = ("url", "ok", "mode", "status", "content", "error", "needs_deep")

    def __init__(self, url: str):
        self.url = url
        self.ok = False
        self.mode = ""
        self.status = 0
        self.content = ""
        self.error = ""
        self.needs_deep = False


class WebFetchBatchTool(Tool):
    """Fetch many URLs in parallel, each self-correcting fast→deep."""

    name = "web_fetch_batch"
    description = (
        "Fetch MULTIPLE URLs in parallel and return clean readable text for each. "
        "Use this instead of repeated web_fetch calls when you have several URLs to "
        "read — e.g. the results of a web_search. Each URL self-corrects: a fast HTTP "
        "fetch first, escalating to a headless browser only when the page is thin or "
        "JS-rendered. Pass a 'urls' list; returns one labelled section per URL."
    )
    timeout_seconds = 120.0  # parallel batch; deep-mode escalation can be slow
    parameters = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs to fetch (processed in parallel)",
            },
            "max_chars": {
                "type": "number",
                "description": "Optional per-URL character cap (overrides config default)",
            },
            "deep_fetch": {
                "type": "boolean",
                "description": (
                    "Force headless-browser (deep) mode for every URL. Default: auto — "
                    "fast HTTP first, deep only when a page comes back thin."
                ),
            },
        },
        "required": ["urls"],
    }

    def __init__(self):
        self.client = _make_http_client()

    async def execute(
        self,
        urls: Any = None,
        max_chars: int | None = None,
        deep_fetch: bool | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        cfg = get_config().tools.web_fetch

        if isinstance(urls, str):
            urls = [urls]
        clean: list[str] = []
        seen: set[str] = set()
        for u in (urls or []):
            if isinstance(u, str) and u.strip() and u.strip() not in seen:
                seen.add(u.strip())
                clean.append(u.strip())
        if not clean:
            return ToolResult(success=False, error="web_fetch_batch requires a non-empty 'urls' list.")

        max_urls = max(1, int(getattr(cfg, "batch_max_urls", 10)))
        overflow = clean[max_urls:]
        targets = clean[:max_urls]

        per_url_cap = max(1, int(max_chars) if max_chars is not None else int(getattr(cfg, "batch_per_url_max_chars", 25000)))
        total_cap = max(1, int(getattr(cfg, "batch_total_max_chars", 150000)))
        min_useful = int(getattr(cfg, "batch_min_useful_chars", 500))
        fast_conc = max(1, int(getattr(cfg, "batch_fast_concurrency", 8)))
        deep_conc = max(1, int(getattr(cfg, "batch_deep_concurrency", 3)))
        fast_timeout = float(getattr(cfg, "batch_fast_timeout", 15.0))
        force_deep = bool(deep_fetch)

        outcomes = {u: _Outcome(u) for u in targets}

        # ── Phase 1: fast HTTP (skipped when deep is forced) ──
        if not force_deep:
            fast_sem = asyncio.Semaphore(fast_conc)

            async def _fast(u: str) -> None:
                async with fast_sem:
                    oc = outcomes[u]
                    if _is_google_drive_url(u) and shutil.which("gws"):
                        oc.error = _GDRIVE_FETCH_BLOCK_MSG
                        return
                    try:
                        resp = await self.client.get(u, timeout=fast_timeout)
                        oc.status = resp.status_code
                        text = _extract_readable_text(resp.text, base_url=u)
                        if _looks_thin(text, resp.status_code, min_useful, len(resp.text)):
                            oc.needs_deep = True
                            oc.content = text  # keep partial in case deep is unavailable
                        else:
                            oc.ok = True
                            oc.mode = "fast"
                            oc.content = text
                    except Exception as e:
                        oc.error = str(e)
                        oc.needs_deep = True

            await asyncio.gather(*[_fast(u) for u in targets])

        # ── Phase 2: deep (forced, or the thin ones) ──
        deep_urls = targets if force_deep else [u for u in targets if outcomes[u].needs_deep]
        # "deep unavailable" = Playwright python pkg missing, OR present but its
        # browser binary isn't installed (launch fails). Either way we surface
        # the actionable install signal so the agent can self-heal and retry.
        deep_unavailable = bool(deep_urls) and not _HAS_PLAYWRIGHT
        if deep_urls and _HAS_PLAYWRIGHT:
            launched = await self._deep_phase(deep_urls, outcomes, deep_conc)
            if not launched:
                deep_unavailable = True

        # Salvage: if deep couldn't complete a URL but the fast HTTP phase
        # captured content, surface that (same as web_fetch(deep_fetch=false))
        # instead of dropping the URL. The deep-unavailable note still flags
        # that the content may be a JS shell / partial.
        for u in targets:
            oc = outcomes[u]
            if not oc.ok and oc.content:
                oc.ok = True
                oc.mode = "fast"

        return self._aggregate(targets, outcomes, per_url_cap, total_cap, overflow, deep_unavailable)

    async def _deep_phase(self, deep_urls: list[str], outcomes: dict, deep_conc: int) -> bool:
        """Render the thin URLs in ONE shared browser, isolated per-URL context.
        Returns True if the browser launched; False if the browser binary is
        missing (so the caller can emit the install signal)."""
        sem = asyncio.Semaphore(deep_conc)
        pw = await async_playwright().start()
        browser = None
        try:
            try:
                browser = await pw.chromium.launch(headless=True)
            except Exception as e:
                log.warning("web_fetch_batch: browser launch failed (binary not installed?): %s", e)
                _kick_browser_install()  # background self-heal for next time
                return False

            async def _deep(u: str) -> None:
                async with sem:
                    oc = outcomes[u]
                    context = None
                    try:
                        context = await browser.new_context(
                            user_agent="Captain Claw/0.1.0 (Web Fetch Tool)",
                            viewport={"width": 1280, "height": 800},
                        )
                        page = await context.new_page()
                        html = await _deep_fetch_page(page, u)
                        oc.content = _extract_readable_text(html, base_url=u)
                        oc.ok = True
                        oc.mode = "deep"
                        oc.status = 200
                        oc.error = ""
                        oc.needs_deep = False
                    except Exception as e:
                        if not oc.content:
                            oc.error = (oc.error + " | " if oc.error else "") + f"deep fetch failed: {e}"
                    finally:
                        if context is not None:
                            try:
                                await context.close()
                            except Exception:
                                pass

            await asyncio.gather(*[_deep(u) for u in deep_urls])
            return True
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
            try:
                await pw.stop()
            except Exception:
                pass

    def _aggregate(self, targets, outcomes, per_url_cap, total_cap, overflow, deep_unavailable) -> ToolResult:
        ok_count = sum(1 for u in targets if outcomes[u].ok)
        deep_count = sum(1 for u in targets if outcomes[u].mode == "deep")
        fail_count = len(targets) - ok_count

        sections: list[str] = []
        total = 0
        budget_hit = False
        for u in targets:
            oc = outcomes[u]
            if oc.ok and oc.content:
                body = oc.content
                if len(body) > per_url_cap:
                    body = body[:per_url_cap] + "\n... [truncated]"
                remaining = total_cap - total
                if remaining <= 0:
                    budget_hit = True
                    sections.append(f"[URL: {u}]\n[Status: {oc.status}]\n[Mode: {oc.mode}]\n[Skipped: total output budget reached]")
                    continue
                if len(body) > remaining:
                    body = body[:remaining] + "\n... [truncated: total budget]"
                    budget_hit = True
                total += len(body)
                sections.append(f"[URL: {u}]\n[Status: {oc.status}]\n[Mode: {oc.mode}]\n[Size: {len(body)} chars]\n\n{body}")
            else:
                sections.append(f"[URL: {u}]\n[Status: {oc.status}]\n[FAILED: {oc.error or 'no content'}]")

        header = f"Fetched {ok_count}/{len(targets)} URLs ({deep_count} via deep browser, {fail_count} failed)."
        notes: list[str] = []
        if deep_unavailable:
            if _browser_install_attempted and not _browser_install_done:
                notes.append(
                    "Some URLs needed a headless browser (deep mode); a one-time Chromium "
                    "install is running in the background now. The fast-HTTP content is shown "
                    "below — retry web_fetch_batch shortly to deep-fetch them properly."
                )
            else:
                notes.append(
                    "Some URLs need a headless browser, but Playwright's browser isn't available. "
                    "Run `pip install playwright && playwright install chromium`, then retry "
                    "web_fetch_batch for the failed URLs."
                )
        if overflow:
            notes.append(
                f"{len(overflow)} URL(s) exceeded the per-call cap and were NOT fetched. "
                f"Call web_fetch_batch again with: {overflow}"
            )
        if budget_hit:
            notes.append("Output was truncated to fit the total size budget.")

        out = header
        if notes:
            out += "\n" + "\n".join(f"- {n}" for n in notes)
        out += "\n\n" + "\n\n---\n\n".join(sections)
        return ToolResult(success=True, content=out)

    async def close(self):
        await self.client.aclose()
