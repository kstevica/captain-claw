"""Browser workflow recorder and replay engine.

Records user interactions (clicks, typing, navigation) in a browser session
and replays them later with variable substitution.  This enables "teach once,
run many" workflows where the user demonstrates a task and the agent replays
it with different data.

Architecture mirrors :mod:`browser_network` (recording) and
:mod:`browser_api_replay` (replay).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RecordedStep:
    """A single recorded user interaction."""

    seq: int
    action: str  # click, type, navigate, select, submit, press_key
    url: str
    value: str | None = None
    selectors: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "action": self.action,
            "url": self.url,
            "value": self.value,
            "selectors": self.selectors,
            "timestamp": self.timestamp,
        }


@dataclass
class StepResult:
    """Outcome of replaying a single step."""

    seq: int
    action: str
    success: bool
    error: str | None = None


@dataclass
class WorkflowReplayResult:
    """Outcome of replaying an entire workflow."""

    success: bool
    steps_completed: int
    step_results: list[StepResult] = field(default_factory=list)
    error: str | None = None

    def to_summary(self) -> str:
        total = len(self.step_results)
        lines = [
            f"Workflow replay: {'SUCCESS' if self.success else 'FAILED'}",
            f"Steps completed: {self.steps_completed}/{total}",
        ]
        for sr in self.step_results:
            status = "OK" if sr.success else f"FAILED: {sr.error}"
            lines.append(f"  Step {sr.seq}: {sr.action} — {status}")
        if self.error:
            lines.append(f"\nError: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recorder JavaScript
# ---------------------------------------------------------------------------

_RECORDER_JS = r"""
(function() {
  if (window.__clawRecorderActive) return;
  window.__clawRecorderActive = true;

  function getSelectors(el) {
    var s = {};

    // --- CSS selector ---
    if (el.id) {
      s.css = '#' + CSS.escape(el.id);
    } else {
      var parts = [];
      var cur = el;
      while (cur && cur !== document.body && cur !== document.documentElement) {
        var seg = cur.tagName.toLowerCase();
        if (cur.id) {
          parts.unshift('#' + CSS.escape(cur.id));
          break;
        }
        if (cur.className && typeof cur.className === 'string') {
          var cls = cur.className.trim().split(/\s+/).slice(0, 2).join('.');
          if (cls) seg += '.' + cls;
        }
        var parent = cur.parentElement;
        if (parent) {
          var siblings = Array.from(parent.children).filter(function(c) {
            return c.tagName === cur.tagName;
          });
          if (siblings.length > 1) {
            seg += ':nth-child(' + (Array.from(parent.children).indexOf(cur) + 1) + ')';
          }
        }
        parts.unshift(seg);
        cur = cur.parentElement;
      }
      s.css = parts.join(' > ');
    }

    // --- ARIA role ---
    var roleMap = {
      'A': 'link', 'BUTTON': 'button', 'INPUT': 'textbox',
      'SELECT': 'combobox', 'TEXTAREA': 'textbox'
    };
    var inputRole = el.tagName === 'INPUT' ? ({
      'checkbox': 'checkbox', 'radio': 'radio', 'submit': 'button',
      'search': 'searchbox', 'number': 'spinbutton', 'range': 'slider'
    })[el.type] || 'textbox' : null;
    s.role = el.getAttribute('role') || inputRole || roleMap[el.tagName] || '';

    // --- Accessible name ---
    s.role_name = el.getAttribute('aria-label')
      || el.getAttribute('title')
      || (el.labels && el.labels[0] ? el.labels[0].textContent.trim().substring(0, 100) : '')
      || '';

    // --- Visible text (for links/buttons) ---
    if (['A', 'BUTTON'].includes(el.tagName) || el.getAttribute('role') === 'button') {
      s.text = (el.textContent || '').trim().substring(0, 100);
    }

    // --- Placeholder ---
    if (el.placeholder) {
      s.placeholder = el.placeholder;
    }

    // --- Name attribute ---
    if (el.name) {
      s.name_attr = el.name;
    }

    return s;
  }

  function send(action, el, extra) {
    var data = {
      action: action,
      url: location.href,
      selectors: el ? getSelectors(el) : {}
    };
    if (extra) {
      for (var k in extra) { data[k] = extra[k]; }
    }
    try {
      window.__claw_record_step__(JSON.stringify(data));
    } catch (err) {
      // callback not available — silently ignore
    }
  }

  // --- Click capture ---
  document.addEventListener('click', function(e) {
    var el = e.target.closest(
      'a, button, [role="button"], [role="link"], [role="tab"], ' +
      '[role="menuitem"], [role="option"], [role="switch"], ' +
      'input[type="submit"], input[type="checkbox"], input[type="radio"], ' +
      'summary, label'
    );
    if (el) {
      send('click', el);
    }
  }, true);

  // --- Input value capture (on change — final value) ---
  document.addEventListener('change', function(e) {
    var el = e.target;
    if (el.tagName === 'INPUT' && !['checkbox', 'radio', 'submit'].includes(el.type)) {
      send('type', el, { value: el.value });
    } else if (el.tagName === 'TEXTAREA') {
      send('type', el, { value: el.value });
    } else if (el.tagName === 'SELECT') {
      send('select', el, { value: el.value });
    }
  }, true);

  // --- Form submission ---
  document.addEventListener('submit', function(e) {
    send('submit', e.target);
  }, true);

  // --- Key press (Enter, Tab, Escape) ---
  document.addEventListener('keydown', function(e) {
    if (['Enter', 'Tab', 'Escape'].includes(e.key)) {
      send('press_key', e.target, { value: e.key });
    }
  }, true);

  // --- SPA navigation detection ---
  var origPush = history.pushState;
  var origReplace = history.replaceState;
  history.pushState = function() {
    origPush.apply(this, arguments);
    send('navigate', null, { value: location.href });
  };
  history.replaceState = function() {
    origReplace.apply(this, arguments);
    send('navigate', null, { value: location.href });
  };
  window.addEventListener('popstate', function() {
    send('navigate', null, { value: location.href });
  });
})();
"""


# ---------------------------------------------------------------------------
# WorkflowRecorder
# ---------------------------------------------------------------------------

class WorkflowRecorder:
    """Captures user browser interactions by injecting JS event listeners.

    Lifecycle mirrors :class:`~browser_network.NetworkInterceptor`:
    ``attach`` → ``start_recording`` / ``stop_recording`` → ``detach``.
    """

    def __init__(self) -> None:
        self._steps: list[RecordedStep] = []
        self._is_recording: bool = False
        self._page: Any | None = None
        self._seq: int = 0
        self._function_exposed: bool = False

    # -- lifecycle ----------------------------------------------------------

    async def attach(self, page: Any) -> None:
        """Register callback and inject recorder JS into *page*."""
        self._page = page

        # expose_function can only be called once per name per page
        if not self._function_exposed:
            try:
                await page.expose_function(
                    "__claw_record_step__", self._on_step,
                )
                self._function_exposed = True
            except Exception:
                # already exposed (e.g. reattach after SPA navigation)
                self._function_exposed = True

        # re-inject JS after full page loads (MPA support)
        page.on("load", self._on_page_load)

        # inject immediately for the current page
        await self._inject_recorder_js(page)
        log.info("Workflow recorder attached")

    async def detach(self) -> None:
        """Remove event listeners.  Injected JS dies on next navigation."""
        if self._page is not None:
            try:
                self._page.remove_listener("load", self._on_page_load)
            except Exception:
                pass
            self._page = None
        self._is_recording = False
        log.info("Workflow recorder detached")

    # -- recording control --------------------------------------------------

    def start_recording(self) -> None:
        self._is_recording = True
        self._seq = len(self._steps)  # continue sequence
        log.info("Workflow recording started", existing_steps=len(self._steps))

    def stop_recording(self) -> None:
        self._is_recording = False
        log.info("Workflow recording stopped", total_steps=len(self._steps))

    def clear(self) -> None:
        self._steps.clear()
        self._seq = 0
        log.info("Workflow recording cleared")

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def steps(self) -> list[RecordedStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    # -- internal -----------------------------------------------------------

    async def _on_step(self, step_json: str) -> None:
        """Callback invoked from injected JS for each interaction."""
        if not self._is_recording:
            return
        try:
            data = json.loads(step_json)
        except json.JSONDecodeError:
            log.warning("Invalid step JSON from recorder", raw=step_json[:200])
            return

        step = RecordedStep(
            seq=self._seq,
            action=data.get("action", "unknown"),
            url=data.get("url", ""),
            value=data.get("value"),
            selectors=data.get("selectors", {}),
        )
        self._steps.append(step)
        self._seq += 1
        log.info(
            "Recorded step",
            seq=step.seq,
            action=step.action,
            value=step.value[:60] if step.value else None,
        )

    async def _on_page_load(self, _: Any = None) -> None:
        """Re-inject recorder JS after a full page navigation."""
        if self._is_recording and self._page is not None:
            try:
                await self._inject_recorder_js(self._page)
                log.debug("Re-injected recorder JS after page load")
            except Exception as exc:
                log.warning("Failed to re-inject recorder JS", error=str(exc))

    async def _inject_recorder_js(self, page: Any) -> None:
        await page.evaluate(_RECORDER_JS)

    # -- serialisation helpers ----------------------------------------------

    def steps_as_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._steps]

    def summary(self) -> str:
        if not self._steps:
            return "No steps recorded."
        action_counts: dict[str, int] = {}
        for s in self._steps:
            action_counts[s.action] = action_counts.get(s.action, 0) + 1
        parts = [f"{count}x {action}" for action, count in sorted(action_counts.items())]
        return (
            f"{len(self._steps)} steps recorded: {', '.join(parts)}.\n"
            f"First URL: {self._steps[0].url}\n"
            f"Last URL: {self._steps[-1].url}"
        )


# ---------------------------------------------------------------------------
# WorkflowReplayEngine
# ---------------------------------------------------------------------------

class WorkflowReplayEngine:
    """Replays a recorded workflow through :class:`BrowserSession` methods."""

    @classmethod
    async def replay(
        cls,
        session: Any,  # BrowserSession
        steps: list[dict[str, Any]],
        variables: dict[str, str] | None = None,
        step_delay: float = 1.0,
    ) -> WorkflowReplayResult:
        """Execute each recorded step in order with variable substitution.

        ``{{variable_name}}`` placeholders in step values and URLs are
        replaced with values from *variables*.
        """
        results: list[StepResult] = []
        var_map = variables or {}

        for i, step in enumerate(steps):
            action = step.get("action", "")
            value = step.get("value")
            selectors = step.get("selectors", {})

            # variable substitution in value
            if value:
                for vname, vval in var_map.items():
                    value = value.replace("{{" + vname + "}}", vval)

            try:
                if action == "navigate":
                    url = value or ""
                    for vname, vval in var_map.items():
                        url = url.replace("{{" + vname + "}}", vval)
                    if url:
                        await session.navigate(url)

                elif action == "click":
                    await cls._click_with_fallback(session, selectors)

                elif action == "type":
                    await cls._type_with_fallback(session, selectors, value or "")

                elif action == "select":
                    page = await session.ensure_page()
                    css = selectors.get("css", "")
                    if css:
                        await page.select_option(css, value=value)

                elif action == "press_key":
                    await session.press_key(value or "Enter")

                elif action == "submit":
                    await session.press_key("Enter")

                else:
                    log.warning("Unknown workflow step action", action=action, seq=i)

                results.append(StepResult(seq=i, action=action, success=True))

            except Exception as exc:
                results.append(StepResult(seq=i, action=action, success=False, error=str(exc)))
                return WorkflowReplayResult(
                    success=False,
                    steps_completed=i,
                    step_results=results,
                    error=f"Step {i} ({action}) failed: {exc}",
                )

            # brief pause for page stability
            if i < len(steps) - 1:
                await asyncio.sleep(step_delay)

        return WorkflowReplayResult(
            success=True,
            steps_completed=len(steps),
            step_results=results,
        )

    # -- selector fallback chains -------------------------------------------

    @classmethod
    async def _click_with_fallback(
        cls, session: Any, selectors: dict[str, str],
    ) -> None:
        errors: list[str] = []

        # Strategy 1: ARIA role + accessible name
        role = selectors.get("role", "")
        role_name = selectors.get("role_name", "")
        if role and role_name:
            try:
                await session.click_by_role(role, name=role_name, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"role({role}, {role_name}): {exc}")

        # Strategy 2: visible text
        text = selectors.get("text", "")
        if text:
            try:
                await session.click_by_text(text, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"text({text}): {exc}")

        # Strategy 3: CSS selector
        css = selectors.get("css", "")
        if css:
            try:
                await session.click(css, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"css({css}): {exc}")

        raise RuntimeError(
            f"All selector strategies failed for click: {'; '.join(errors)}"
        )

    @classmethod
    async def _type_with_fallback(
        cls, session: Any, selectors: dict[str, str], text: str,
    ) -> None:
        errors: list[str] = []

        # Strategy 1: ARIA role + accessible name
        role = selectors.get("role", "")
        role_name = selectors.get("role_name", "")
        if role and role_name:
            try:
                await session.type_by_role(role, name=role_name, text=text, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"role({role}, {role_name}): {exc}")

        # Strategy 2: placeholder attribute
        placeholder = selectors.get("placeholder", "")
        if placeholder:
            try:
                page = await session.ensure_page()
                escaped = placeholder.replace('"', '\\"')
                await page.fill(f'[placeholder="{escaped}"]', text, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"placeholder({placeholder}): {exc}")

        # Strategy 3: name attribute
        name_attr = selectors.get("name_attr", "")
        if name_attr:
            try:
                page = await session.ensure_page()
                escaped = name_attr.replace('"', '\\"')
                await page.fill(f'[name="{escaped}"]', text, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"name({name_attr}): {exc}")

        # Strategy 4: CSS selector
        css = selectors.get("css", "")
        if css:
            try:
                await session.type_text(css, text, timeout=5000)
                return
            except Exception as exc:
                errors.append(f"css({css}): {exc}")

        raise RuntimeError(
            f"All selector strategies failed for type: {'; '.join(errors)}"
        )
