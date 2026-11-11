"""Web fetch tools for retrieving web page content."""

import asyncio
import hashlib
import os
import re
import time
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import httpx

from captain_claw import research_images
from captain_claw.config import get_config
from captain_claw.google_ids import google_drive_can_open, google_drive_redirect
from captain_claw.logging import get_logger
from captain_claw.tools.browser_session import (
    BrowserUnavailable,
    browser_unavailable_reason,
    launch_chromium,
)
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


_CORPUS_HEAD_CHARS = 50000  # how much of a corpus-saved page to inline as a preview


def _corpus_output(url: str, status_code: int, mode: str, raw_len: int,
                   content: str, saved: str, head_chars: int = _CORPUS_HEAD_CHARS) -> str:
    """Build the head+pointer response for a corpus-saved fetch."""
    head_chars = max(1, min(_CORPUS_HEAD_CHARS, int(head_chars)))
    head = content[:head_chars]
    out = (f"[URL: {url}]\n[Status: {status_code}]\n[Mode: {mode}]\n"
           f"[Size: {raw_len} chars · {len(content)} text chars]\n"
           f"[Full text saved to {saved} — search it with the `researchmap` tool "
           f"or read the file for anything past the preview below]\n\n{head}")
    if len(content) > head_chars:
        out += f"\n\n... [preview only — the full {len(content)} chars are in {saved}]"
    return out


# ── Page size ────────────────────────────────────────────────────────
# The configured max_chars is a FLOOR for what the model asked: models habitually
# pass max_chars=4000-6000 "to save context", which on a large-context model just
# throws the page away (and the duplicate-call guard then blocks a re-fetch). A
# smaller request is ignored, a larger one honoured. Whatever the number, one page
# stays within a share of THIS agent's context budget, so a small local window is
# never handed a page it would then drop from its history whole.
_PAGE_CONTEXT_SHARE = 0.35  # of context.max_tokens (≈4 chars per token)
_MIN_PAGE_CEILING = 20_000


def _context_char_ceiling(agent: Any = None) -> int:
    """Most characters one fetch result may carry for this agent (0 = unknown).

    Sized on the window the agent really has: ``context.max_tokens``, cut to its
    provider's ``num_ctx`` when that is smaller — a local Ollama model is clamped
    to what the box can hold, and the context builder budgets to that clamp
    (agent_context_mixin), so a page sized on the configured budget would be
    dropped from the history whole.
    """
    try:
        tokens = int(get_config().context.max_tokens or 0)
    except Exception:  # noqa: BLE001 — never let sizing break a fetch
        return 0
    provider_ctx = getattr(getattr(agent, "provider", None), "num_ctx", 0)
    if isinstance(provider_ctx, int) and not isinstance(provider_ctx, bool) \
            and 0 < provider_ctx < tokens:
        tokens = provider_ctx
    if tokens <= 0:
        return 0
    return max(_MIN_PAGE_CEILING, int(tokens * 4 * _PAGE_CONTEXT_SHARE))


def _page_char_limit(requested: Any, configured: int, agent: Any = None) -> int:
    """Characters to keep from one fetched page (see the note above)."""
    limit = max(1, int(configured))
    if requested is not None:
        try:
            limit = max(limit, int(requested))
        except (TypeError, ValueError):
            pass
    ceiling = _context_char_ceiling(agent)
    return min(limit, ceiling) if ceiling else limit


def _governed_page_output(url: str, status_code: int, mode: str, raw_len: int,
                          content: str, requested: Any, agent: Any) -> str:
    """A governed web_fetch result: the page saved in full (the run's shared
    sources/ when there is one, so teammates reuse it; else privately when it is
    too big to keep inline), returned as head + pointer."""
    from captain_claw import context_spill as _spill

    cap = _spill.inline_cap_chars(agent)
    try:
        if requested is not None and int(requested) < cap:
            cap = max(4_000, int(requested))
    except (TypeError, ValueError):
        pass
    page = (f"[URL: {url}]\n[Status: {status_code}]\n[Mode: {mode}]\n"
            f"[Size: {raw_len} chars]\n\n{content}")
    shared = _spill.run_sources_dir() is not None and _spill.shareable_page(content, status_code)
    pointer = None
    if content.strip() and (shared or len(page) > cap):
        pointer = _spill.save_source(agent, "web_fetch", url, content, shared=shared,
                                     status=status_code)
    if not pointer:
        return page if len(page) <= cap else page[:cap] + "\n... [truncated]"
    return _spill.spill_view("web_fetch", url, page, pointer, cap)


_MAX_CHARS_HELP = (
    "Optional — leave it out: the page comes back whole up to the configured limit "
    "(typically 200000 chars). A smaller value is ignored; pass one only to get MORE."
)

# ── Optional Playwright dependency ───────────────────────────────────

try:
    from playwright.async_api import async_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:  # pragma: no cover
    _HAS_PLAYWRIGHT = False
    async_playwright = None  # type: ignore[assignment,misc]

# ── Google Drive URL redirect ─────────────────────────────────────────
# A Drive/Docs URL naming a file goes through the google_drive tool (the
# owner's connection, exports, shared drives) instead of an anonymous fetch
# that lands on a sign-in page — when google_drive can open it: Google
# connected with a scope that reaches shared links (not drive.file alone).
# Otherwise the fetch proceeds as before, so public links still work; so do
# Forms and published (/d/e/) pages, which google_drive cannot read at all.


async def _gdrive_fetch_block(url: str, agent: Any = None) -> str | None:
    """The redirect message when *url* must go through google_drive, else None."""
    if not google_drive_can_open(url):
        return None
    from captain_claw.tools.google_drive import (
        agent_offers_google_drive,
        google_drive_reads_links,
    )

    if not agent_offers_google_drive(agent) or not await google_drive_reads_links():
        return None
    return google_drive_redirect(
        url,
        "Google Drive/Docs URLs are read with your Google connection, not web_fetch.",
        connected=True,
    )


# A shared-agent member's fetch reads at most this much of a body, and the
# whole fetch (every redirect hop included) must finish within this time —
# httpx's own timeout is per read, so a fast huge download never trips it.
MEMBER_FETCH_MAX_BYTES = 8 * 1024 * 1024
MEMBER_FETCH_DEADLINE_S = 45.0


async def _member_fetch(url: str) -> tuple[str, int, bool]:
    """GET *url* for a member: public addresses only, body capped at
    :data:`MEMBER_FETCH_MAX_BYTES`. Returns ``(text, status, truncated)``."""
    from captain_claw import speaker as _speaker

    async with _speaker.make_public_http_client() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            truncated = False
            async for chunk in response.aiter_bytes():
                room = MEMBER_FETCH_MAX_BYTES - size
                if len(chunk) > room:
                    # More body than the cap allows: keep the cap, stop reading.
                    chunks.append(chunk[:room])
                    size += room
                    truncated = True
                    break
                chunks.append(chunk)
                size += len(chunk)
            text = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
            return text, response.status_code, truncated


def _make_http_client() -> httpx.AsyncClient:
    """Create a shared-style HTTP client."""
    return httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Captain Claw/0.1.0 (Web Fetch Tool)",
        },
    )


def _extract_readable_text(html: str, base_url: str | None = None,
                           images: list | None = None) -> str:
    """Extract human-readable text from raw HTML.

    With *images*, the page's relevant-looking pictures (research_images) are
    appended to it — found on the same parse, before anything is stripped.
    """
    soup = BeautifulSoup(html, "html.parser")

    if images is not None:
        try:
            images.extend(research_images.find_page_images(soup, base_url))
        except Exception as e:  # noqa: BLE001 — pictures are extra, never break a fetch
            log.debug("page image scan failed", url=base_url, error=str(e))

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


# ── Pictures for reports (research_images) ────────────────────────────
# A fetched page's relevant-looking pictures are saved and listed at the end of
# the result, so a report can show them; a shared-agent member gets the list
# of original URLs only (nothing is written to the agent's folders for them).
_IMAGE_BUDGET_S = 12.0        # one page's downloads (web_fetch)
_BATCH_IMAGE_BUDGET_S = 15.0  # all pages' downloads together (web_fetch_batch)


def _images_on() -> bool:
    try:
        return bool(getattr(get_config().tools.web_fetch, "save_images", True))
    except Exception:  # noqa: BLE001
        return True


def _images_per_page() -> int:
    try:
        return max(0, int(getattr(get_config().tools.web_fetch, "images_per_page", 4)))
    except Exception:  # noqa: BLE001
        return 4


async def _page_images_block(page_url: str, images: list, kwargs: dict[str, Any], *,
                             limit: int, budget_s: float, member: bool,
                             client: Any = None) -> str:
    """The ``[Images …]`` block a fetch result ends with ('' when none)."""
    if not images or limit <= 0:
        return ""
    if member:
        return research_images.format_unsaved(images, limit)
    try:
        store = research_images.ImageStore.for_tool(kwargs)
        saved = await research_images.save_page_images(
            images, page_url, store, limit=limit, budget_s=budget_s, client=client)
    except Exception as e:  # noqa: BLE001 — pictures are extra, never break a fetch
        log.warning("page images not saved", url=page_url, error=str(e))
        return ""
    return research_images.format_saved(saved)


def _earlier_images_block(page_url: str, kwargs: dict[str, Any]) -> str:
    """Pictures already saved from *page_url* (a page the run fetched before)."""
    try:
        return research_images.format_saved(
            research_images.ImageStore.for_tool(kwargs).for_page(page_url), earlier=True)
    except Exception:  # noqa: BLE001
        return ""


def _with_block(content: str, block: str) -> str:
    return f"{content}\n\n{block}" if block else content


async def _save_image_bytes(data: bytes, url: str, kwargs: dict[str, Any]) -> Any:
    """Keep a picture web_fetch was pointed at directly (None if it isn't one)."""
    fitted = await asyncio.to_thread(research_images.normalize_image, data, check_size=False)
    if fitted is None:
        return None
    store = research_images.ImageStore.for_tool(kwargs)
    return await asyncio.to_thread(store.save, *fitted, url=url)


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

_BROWSER_DOWN_NOTE = (
    "[Browser: can't start on this server (system libraries missing), so this is the "
    "plain-HTTP text. Pages that build their content with scripts may be incomplete. "
    "Don't install anything yourself; tell the user the page needs the browser and the "
    "server's admin has to fix it.]\n"
)


def _deep_unavailable_note() -> str:
    """The batch note for URLs that needed the browser and didn't get it."""
    if browser_unavailable_reason():
        return ("Some URLs needed a headless browser, which can't start on this server "
                "(system libraries missing); their fast-HTTP text is shown. Don't install "
                "anything yourself: tell the user those pages need the browser and the "
                "server's admin has to fix it.")
    if not _HAS_PLAYWRIGHT:
        return ("Some URLs needed a headless browser, which isn't installed with this "
                "Captain; their fast-HTTP text is shown.")
    if _browser_install_attempted and not _browser_install_done:
        return ("Some URLs needed a headless browser (deep mode); a one-time Chromium "
                "install is running in the background now. The fast-HTTP content is shown "
                "below — retry web_fetch_batch shortly to deep-fetch them properly.")
    return ("Some URLs needed a headless browser, which isn't available here; their "
            "fast-HTTP text is shown. Don't install anything yourself.")


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
    browser = None
    try:
        try:
            browser = await launch_chromium(pw.chromium, headless=True, auto_install=False)
        except BrowserUnavailable:
            raise  # the host lacks Chromium's libraries: an install won't help
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


async def _wall_note(kwargs: dict[str, Any], url: str, status: int | None, text: str, html: str) -> str:
    """A fetch that came back with a bot wall (or nothing) instead of the page
    is noted for the session; the note says what to try next — at last,
    asking the person for their Chrome (web_blocked.py). ``""`` for a page."""
    from captain_claw.tools import web_blocked

    why = web_blocked.block_reason(status=status, text=text, html=html)
    if why:
        return await web_blocked.note_failure(kwargs, url, web_blocked.FETCH, why)
    if not str(text or "").strip():
        return await web_blocked.note_failure(kwargs, url, web_blocked.FETCH, "the page came back empty",
                                              strong=False)
    web_blocked.forget(kwargs, url, web_blocked.FETCH)
    return ""


async def _fetch_error_note(kwargs: dict[str, Any], url: str, exc: BaseException) -> str:
    from captain_claw.tools import web_blocked

    why, strong = web_blocked.fetch_error_reason(exc)
    if not why:
        return ""
    return await web_blocked.note_failure(kwargs, url, web_blocked.FETCH, why, strong=strong)


class WebFetchTool(Tool):
    """Fetch web page content as clean readable text (ALWAYS text mode)."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return clean readable text (no HTML). "
        "This is the default tool for reading web pages. Always extracts text. "
        "Uses a headless browser by default (deep_fetch=true) for JS-rendered pages; "
        "pass deep_fetch=false for a faster plain HTTP fetch on simple/static pages. "
        "Pictures on the page that could matter (described by alt text, a caption or "
        "the text around them) are saved for reports and listed at the end; fetching "
        "an image URL saves that picture."
    )
    timeout_seconds = 50.0  # a deep fetch (≤ ~25 s) plus its pictures (≤ 12 s)
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "number",
                "description": _MAX_CHARS_HELP,
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

        Google Drive/Docs file URLs are redirected to the ``google_drive`` tool
        while Google is connected — it authenticates as the owner, which
        web_fetch cannot.

        Args:
            url: URL to fetch
            max_chars: Max characters to return
            deep_fetch: Use headless browser to render JS and auto-scroll

        Returns:
            ToolResult with extracted readable text
        """
        # Hard guard: strip any extract_mode — web_fetch ALWAYS returns text.
        kwargs.pop("extract_mode", None)

        # A shared-agent member's fetch: plain HTTP only (no headless
        # browser), public addresses only (checked at connect time on every
        # hop), no Google redirect and no corpus write.
        from captain_claw import speaker as _speaker

        member = _speaker.principal_for(kwargs.get("_agent")) is not None
        if member:
            deep_fetch = False
            url_error = _speaker.check_public_url(url)
            if url_error:
                return ToolResult(success=False, error=url_error)
        else:
            # Redirect Google Drive URLs to google_drive while Google is connected.
            gdrive_block = await _gdrive_fetch_block(url, kwargs.get("_agent"))
            if gdrive_block:
                return ToolResult(success=False, error=gdrive_block)

        body_truncated = False
        want_images = _images_on()
        # An image URL is a picture to keep for a report, not a page to read.
        if want_images and not member and research_images.looks_like_image_url(url):
            return await self._fetch_image(url, kwargs)
        # In-turn context governor: a page the run already fetched comes back
        # as its saved copy's pointer + a short preview, and a new page is
        # fetched whole, saved to the run's sources/, and returned as head +
        # pointer (context_spill).
        from captain_claw import context_spill as _spill

        _agent = kwargs.get("_agent")
        governed = (not member) and _spill.governor_enabled()
        if governed:
            found = _spill.find_fetched(url)
            if found:
                view = _spill.dedupe_view(url, found, _agent)
                if want_images:
                    view = _with_block(view, _earlier_images_block(url, kwargs))
                return ToolResult(success=True, content=view)
        images: list | None = [] if want_images else None
        try:
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 200000))
            effective_max_chars = _page_char_limit(max_chars, configured_max, _agent)
            if governed:
                effective_max_chars = _spill.SPILL_SOURCE_MAX

            browser_note = ""
            if deep_fetch:
                log.info("Fetching URL (deep_fetch mode)", url=url)
                try:
                    raw_html = await _deep_fetch(url)
                    status_code = 200  # Playwright doesn't expose status easily
                    mode = "deep"
                except BrowserUnavailable:
                    # Fall back to the plain fetch below, and say so once.
                    log.warning("deep_fetch: browser unavailable on this host, using plain HTTP", url=url)
                    browser_note = _BROWSER_DOWN_NOTE
                    deep_fetch = False
            if deep_fetch:
                pass  # rendered above
            elif member:
                log.info("Fetching URL (member text mode)", url=url)
                try:
                    raw_html, status_code, body_truncated = await asyncio.wait_for(
                        _member_fetch(url), timeout=MEMBER_FETCH_DEADLINE_S,
                    )
                except TimeoutError:
                    return ToolResult(
                        success=False,
                        error=(
                            f"The page took longer than {int(MEMBER_FETCH_DEADLINE_S)} s "
                            "to load and was abandoned."
                        ),
                    )
                mode = "text"
            else:
                log.info("Fetching URL (text mode)", url=url)
                response = await self.client.get(url)
                response.raise_for_status()
                ctype = str(getattr(response, "headers", {}).get("content-type") or "").lower()
                if want_images and ctype.startswith("image/"):
                    entry = await _save_image_bytes(response.content, url, kwargs)
                    if entry is not None:
                        return ToolResult(success=True,
                                          content=research_images.format_direct(entry))
                    return ToolResult(success=False, error=(
                        f"{url} is a picture ({ctype}) that could not be read or saved."))
                raw_html = response.text
                status_code = response.status_code
                mode = "text"

            content = _extract_readable_text(raw_html, base_url=url, images=images)
            wall_note = await _wall_note(kwargs, url, None if mode == "deep" else status_code,
                                         content, raw_html)
            block = await _page_images_block(
                url, images or [], kwargs, limit=_images_per_page(),
                budget_s=_IMAGE_BUDGET_S, member=member)

            if governed:
                return ToolResult(success=True, content=browser_note + _with_block(_governed_page_output(
                    url, status_code, mode, len(raw_html), content[:_spill.SPILL_SOURCE_MAX],
                    max_chars, _agent), block) + wall_note)

            # R10: in a corpus run, save the FULL text and return a head + pointer.
            saved = None if member else _save_source_to_corpus(url, content)
            if saved:
                return ToolResult(
                    success=True,
                    content=browser_note + _with_block(_corpus_output(
                        url, status_code, mode, len(raw_html), content, saved,
                        head_chars=effective_max_chars), block) + wall_note,
                )

            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = browser_note + f"[URL: {url}]\n"
            output += f"[Status: {status_code}]\n"
            output += f"[Mode: {mode}]\n"
            output += f"[Size: {len(raw_html)} chars]\n"
            if body_truncated:
                output += (
                    f"[Body: only the first {MEMBER_FETCH_MAX_BYTES // (1024 * 1024)} MB "
                    "was read]\n"
                )
            output += "\n" + content

            return ToolResult(
                success=True,
                content=_with_block(output, block) + wall_note,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}" + await _fetch_error_note(kwargs, url, e),
            )
        except Exception as e:
            log.error("Fetch failed", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e) + await _fetch_error_note(kwargs, url, e),
            )

    async def _fetch_image(self, url: str, kwargs: dict[str, Any]) -> ToolResult:
        """Save the picture at *url* for a report (public addresses only)."""
        store = research_images.ImageStore.for_tool(kwargs)
        client = research_images.make_client()
        try:
            entry = await research_images.save_remote(store, client, url, check_size=False)
        finally:
            await client.aclose()
        if entry is None:
            return ToolResult(success=False, error=(
                f"Could not save the picture at {url}: it is unreachable (or on a private "
                "address), not an image, or larger than 12 MB."))
        return ToolResult(success=True, content=research_images.format_direct(entry))

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


_WEB_GET_TEXT_NOTE = (
    "[NOTE: raw HTML withheld on the first call. Weak models reach for web_get when "
    "they only want to READ a page — so the first web_get on a URL returns clean "
    "readable text, exactly like `web_fetch`. If the text above answers your question, "
    "you are done: use it and do NOT call web_get again. Only if you genuinely need the "
    "markup itself (CSS selectors, DOM structure, attributes, embedded JSON) call "
    "web_get on this same URL once more — that call returns the raw HTML.]"
)


class WebGetTool(Tool):
    """Fetch raw HTML from a URL (for scraping, parsing, DOM inspection).

    Guard: the FIRST call on a given URL returns stripped readable text (web_fetch
    behaviour); a SECOND call on the same URL returns the raw HTML. Weak models
    habitually pick web_get for plain page reading, which floods their context with
    markup they never use. Making them ask twice costs a strong model one extra call
    and saves a weak one an entire context window. The seen-set lives on the tool
    instance, which is registered per agent, so each session decides for itself.
    """

    name = "web_get"
    description = (
        "Fetch a URL and return the raw HTML source. "
        "Use ONLY when you need the actual HTML markup (scraping, DOM analysis, CSS selectors). "
        "For normal page reading, use web_fetch instead. "
        "NOTE: the first call on a URL returns readable text, not HTML — call it a "
        "second time on the same URL to get the raw HTML."
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
                "description": _MAX_CHARS_HELP,
            },
        },
        "required": ["url"],
    }

    def __init__(self):
        self.client = _make_http_client()
        # URLs this agent has already spent a web_get on — the second call is
        # the one that earns the markup. Same key shape as the duplicate-call
        # guard in agent_tool_loop_mixin (bare .strip()) so the two agree.
        self._seen_urls: set[str] = set()

    async def execute(
        self,
        url: str,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Fetch a web page: readable text on the first call for a URL, raw HTML on the second.

        Args:
            url: URL to fetch
            max_chars: Max characters to return

        Returns:
            ToolResult with readable text (first call) or raw HTML (repeat call)
        """
        # Redirect Google Drive URLs to google_drive while Google is connected.
        gdrive_block = await _gdrive_fetch_block(url, kwargs.get("_agent"))
        if gdrive_block:
            return ToolResult(success=False, error=gdrive_block)

        url_key = str(url or "").strip()
        raw_allowed = url_key in self._seen_urls

        try:
            log.info(
                "Fetching URL (web_get)",
                url=url,
                mode="html" if raw_allowed else "text (first call)",
            )
            cfg = get_config()
            configured_max = int(getattr(cfg.tools.web_fetch, "max_chars", 200000))
            effective_max_chars = _page_char_limit(max_chars, configured_max, kwargs.get("_agent"))

            response = await self.client.get(url)
            response.raise_for_status()

            raw_html = response.text
            # Only a fetch that actually returned the page counts as the first
            # look — a failed one must not burn the URL's free text pass.
            self._seen_urls.add(url_key)

            if raw_allowed:
                content = raw_html
                mode = "html"
            else:
                content = _extract_readable_text(raw_html, base_url=url)
                mode = "text (first web_get on this URL)"

            if len(content) > effective_max_chars:
                content = content[:effective_max_chars] + "\n... [truncated]"

            output = f"[URL: {url}]\n"
            output += f"[Status: {response.status_code}]\n"
            output += f"[Mode: {mode}]\n"
            output += f"[Size: {len(raw_html)} chars]\n\n"
            output += content
            if not raw_allowed:
                output += f"\n\n{_WEB_GET_TEXT_NOTE}"
            output += await _wall_note(kwargs, url, response.status_code,
                                       content if not raw_allowed else _extract_readable_text(raw_html),
                                       raw_html)

            return ToolResult(
                success=True,
                content=output,
            )

        except httpx.HTTPError as e:
            log.error("HTTP fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=f"HTTP error: {e}" + await _fetch_error_note(kwargs, url, e),
            )
        except Exception as e:
            log.error("Fetch failed (raw)", url=url, error=str(e))
            return ToolResult(
                success=False,
                error=str(e) + await _fetch_error_note(kwargs, url, e),
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
    __slots__ = ("url", "ok", "mode", "status", "content", "error", "needs_deep", "salvaged",
                 "images")

    def __init__(self, url: str):
        self.url = url
        self.ok = False
        self.mode = ""
        self.status = 0
        self.content = ""
        self.error = ""
        self.needs_deep = False
        self.salvaged = False
        self.images: list = []


class WebFetchBatchTool(Tool):
    """Fetch many URLs in parallel, each self-correcting fast→deep."""

    name = "web_fetch_batch"
    description = (
        "Fetch MULTIPLE URLs in parallel and return clean readable text for each. "
        "Use this instead of repeated web_fetch calls when you have several URLs to "
        "read — e.g. the results of a web_search. Each URL self-corrects: a fast HTTP "
        "fetch first, escalating to a headless browser only when the page is thin or "
        "JS-rendered. Pass a 'urls' list; returns one labelled section per URL, each "
        "ending with the pictures saved from that page for reports (if any)."
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
                "description": (
                    "Optional — leave it out: each page comes back whole up to the configured "
                    "per-URL limit. A smaller value is ignored; pass one only to get MORE."
                ),
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

        _agent = kwargs.get("_agent")
        per_url_cap = _page_char_limit(
            max_chars, int(getattr(cfg, "batch_per_url_max_chars", 100000)), _agent)
        total_cap = max(1, int(getattr(cfg, "batch_total_max_chars", 400000)))
        _ceiling = _context_char_ceiling(_agent)
        if _ceiling:
            total_cap = min(total_cap, _ceiling)  # the whole batch is ONE tool result
        min_useful = int(getattr(cfg, "batch_min_useful_chars", 500))
        fast_conc = max(1, int(getattr(cfg, "batch_fast_concurrency", 8)))
        deep_conc = max(1, int(getattr(cfg, "batch_deep_concurrency", 3)))
        fast_timeout = float(getattr(cfg, "batch_fast_timeout", 15.0))
        force_deep = bool(deep_fetch)

        outcomes = {u: _Outcome(u) for u in targets}

        # Drive/Docs URLs go to google_drive while Google is connected — before
        # either phase, so a forced deep fetch can't route around it.
        blocked: set[str] = set()
        for u in targets:
            gdrive_block = await _gdrive_fetch_block(u, kwargs.get("_agent"))
            if gdrive_block:
                outcomes[u].error = gdrive_block
                blocked.add(u)
        fetchable = [u for u in targets if u not in blocked]

        # Governed: URLs the run already fetched come back as their saved
        # copy (pointer + short preview); only the rest are fetched.
        from captain_claw import context_spill as _spill
        from captain_claw import speaker as _speaker

        governed = _spill.governor_enabled() and _speaker.principal_for(_agent) is None
        saved_hits: dict[str, dict] = {}
        if governed:
            for u in fetchable:
                found = _spill.find_fetched(u)
                if found:
                    saved_hits[u] = found
                    outcomes[u].ok = True
                    outcomes[u].mode = "saved"
            fetchable = [u for u in fetchable if u not in saved_hits]

        # ── Phase 1: fast HTTP (skipped when deep is forced) ──
        if not force_deep:
            fast_sem = asyncio.Semaphore(fast_conc)

            async def _fast(u: str) -> None:
                async with fast_sem:
                    oc = outcomes[u]
                    try:
                        resp = await self.client.get(u, timeout=fast_timeout)
                        oc.status = resp.status_code
                        oc.images = []
                        text = _extract_readable_text(resp.text, base_url=u, images=oc.images)
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

            await asyncio.gather(*[_fast(u) for u in fetchable])

        # ── Phase 2: deep (forced, or the thin ones) ──
        deep_urls = fetchable if force_deep else [u for u in fetchable if outcomes[u].needs_deep]
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
                oc.salvaged = True

        blocks = await self._image_blocks(targets, outcomes, saved_hits, kwargs,
                                          member=_speaker.principal_for(_agent) is not None)
        if governed:
            result = self._aggregate_governed(targets, outcomes, saved_hits, overflow,
                                              deep_unavailable, max_chars, _agent, blocks)
        else:
            result = self._aggregate(targets, outcomes, per_url_cap, total_cap, overflow,
                                     deep_unavailable, blocks)
        notes = await self._wall_notes([u for u in targets if u not in blocked and u not in saved_hits],
                                       outcomes, kwargs)
        if notes:
            if result.success:
                result.content = (result.content or "") + notes
            else:
                result.error = (result.error or "") + notes
        return result

    @staticmethod
    async def _wall_notes(urls: list[str], outcomes: dict, kwargs: dict[str, Any]) -> str:
        """The bot walls this batch met — noted per site, the notes once each."""
        from captain_claw.tools import web_blocked

        notes: list[str] = []
        for u in urls:
            oc = outcomes[u]
            if oc.ok:
                note = await _wall_note(kwargs, u, oc.status if oc.mode == "fast" else None, oc.content, "")
            else:
                why = (web_blocked.block_reason(status=oc.status, text=oc.content)
                       or web_blocked.net_reason(oc.error))
                note = await web_blocked.note_failure(kwargs, u, web_blocked.FETCH, why,
                                                      strong=why.startswith(("a bot check", "it ")))
            if note and note not in notes:
                notes.append(note)
        return "".join(notes)

    async def _image_blocks(self, targets, outcomes, saved_hits, kwargs, *,
                            member: bool) -> dict[str, str]:
        """Each page's ``[Images …]`` block: its pictures saved (all pages at
        once, within one budget), at most ``images_per_batch`` in the batch."""
        if not _images_on():
            return {}
        blocks: dict[str, str] = {}
        if not member:
            for u in saved_hits:
                blocks[u] = _earlier_images_block(u, kwargs)
        pages = [u for u in targets if outcomes[u].ok and outcomes[u].images
                 and u not in saved_hits]
        if not pages:
            return blocks
        try:
            total = max(0, int(getattr(get_config().tools.web_fetch, "images_per_batch", 16)))
        except Exception:  # noqa: BLE001
            total = 16
        per_page = min(_images_per_page(), max(1, total // len(pages))) if total else 0
        client = None if member else research_images.make_client()
        try:
            found = await asyncio.gather(*[
                _page_images_block(u, outcomes[u].images, kwargs, limit=per_page,
                                   budget_s=_BATCH_IMAGE_BUDGET_S, member=member, client=client)
                for u in pages
            ])
        finally:
            if client is not None:
                await client.aclose()
        blocks.update(zip(pages, found))
        return blocks

    def _aggregate_governed(self, targets, outcomes, saved_hits, overflow, deep_unavailable,
                            requested, agent, blocks=None) -> ToolResult:
        """Governed batch output: every page saved in full (the run's shared
        sources/ when there is one), each shown as its share of one inline
        result — head + pointer — so the batch never floods the context."""
        from captain_claw import context_spill as _spill

        ok = [u for u in targets if outcomes[u].ok and (outcomes[u].content or u in saved_hits)]
        cap = _spill.inline_cap_chars(agent)
        try:
            if requested is not None and int(requested) * max(1, len(ok)) < cap:
                cap = int(requested) * max(1, len(ok))
        except (TypeError, ValueError):
            pass
        per_url = max(2_000, cap // max(1, len(ok)))
        run_dir = _spill.run_sources_dir() is not None
        sections: list[str] = []
        for u in targets:
            oc = outcomes[u]
            if u in saved_hits:
                sections.append(_spill.dedupe_view(u, saved_hits[u], agent)[: per_url + 600])
                continue
            if not (oc.ok and oc.content):
                sections.append(f"[URL: {u}]\n[Status: {oc.status}]\n[FAILED: {oc.error or 'no content'}]")
                continue
            page = f"[URL: {u}]\n[Status: {oc.status}]\n[Mode: {oc.mode}]\n\n{oc.content}"
            # A thin page kept only because the browser couldn't render it is
            # never the run's copy.
            shared = run_dir and not oc.salvaged and _spill.shareable_page(oc.content, oc.status)
            pointer = None
            if shared or len(page) > per_url:
                pointer = _spill.save_source(agent, "web_fetch", u, oc.content, shared=shared,
                                             status=oc.status or None)
            if pointer:
                sections.append(_spill.spill_view("web_fetch", u, page, pointer, per_url))
            else:
                sections.append(page if len(page) <= per_url else page[:per_url] + "\n... [truncated]")
        ok_count = len(ok)
        header = (f"Fetched {ok_count}/{len(targets)} URLs "
                  f"({len(saved_hits)} already saved in this run, "
                  f"{len(targets) - ok_count} failed). Each page's full text is saved; "
                  "read a pointer with `read` (offset/limit) or `grep` for more than its preview.")
        notes: list[str] = []
        if deep_unavailable:
            notes.append(_deep_unavailable_note())
        if overflow:
            notes.append(f"{len(overflow)} URL(s) exceeded the per-call cap and were NOT fetched. "
                         f"Call web_fetch_batch again with: {overflow}")
        out = header + ("\n" + "\n".join(f"- {n}" for n in notes) if notes else "")
        blocks = blocks or {}
        out += "\n\n" + "\n\n---\n\n".join(
            _with_block(s, blocks.get(u, "")) for u, s in zip(targets, sections))
        return ToolResult(success=True, content=out)

    async def _deep_phase(self, deep_urls: list[str], outcomes: dict, deep_conc: int) -> bool:
        """Render the thin URLs in ONE shared browser, isolated per-URL context.
        Returns True if the browser launched; False if the browser binary is
        missing (so the caller can emit the install signal)."""
        sem = asyncio.Semaphore(deep_conc)
        pw = await async_playwright().start()
        browser = None
        try:
            try:
                browser = await launch_chromium(pw.chromium, headless=True, auto_install=False)
            except BrowserUnavailable:
                return False  # host lacks Chromium's libraries; the note says so
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
                        images: list = []
                        oc.content = _extract_readable_text(html, base_url=u, images=images)
                        oc.images = images
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

    def _aggregate(self, targets, outcomes, per_url_cap, total_cap, overflow, deep_unavailable,
                   blocks=None) -> ToolResult:
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
            notes.append(_deep_unavailable_note())
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
        blocks = blocks or {}
        out += "\n\n" + "\n\n---\n\n".join(
            _with_block(s, blocks.get(u, "")) for u, s in zip(targets, sections))
        return ToolResult(success=True, content=out)

    async def close(self):
        await self.client.aclose()
