"""Accessibility tree extraction for the browser tool.

Uses Playwright's ``aria_snapshot()`` API to extract a semantic tree of the page,
providing a clean view of page structure even for React/SPA apps with messy
DOM.  Also parses the snapshot to find interactive elements with their
selectors for the LLM to use in click/type actions.

Playwright 1.49+ uses ``page.locator('body').aria_snapshot()`` which returns a
YAML-like indented text representation of the accessibility tree — much cleaner
than the old ``page.accessibility.snapshot()`` dict-based API.
"""

from __future__ import annotations

import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Roles considered "interactive" — elements users can click, type into, toggle, etc.
_INTERACTIVE_ROLES: frozenset[str] = frozenset({
    "button", "link", "textbox", "checkbox", "radio",
    "combobox", "searchbox", "slider", "spinbutton",
    "switch", "tab", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "treeitem",
})

# Regex to parse a line from aria_snapshot output.
# Examples:
#   - heading "Example Domain" [level=1]
#   - textbox "Customer name:"
#   - link "Learn more":
#   - button "Submit order"
#   - checkbox "Bacon"
#   - radio "Small"
_ARIA_LINE_RE = re.compile(
    r"^(?P<indent>\s*)-\s+"
    r"(?P<role>[a-zA-Z]+)"
    r'(?:\s+"(?P<name>[^"]*)")?'
    r"(?:\s*\[(?P<attrs>[^\]]*)\])?"
    r"\s*:?\s*$"
)


def _suggest_selector(role: str, name: str) -> str:
    """Suggest a Playwright locator string for an interactive element.

    Prefers get_by_role() style since it's the most resilient for SPAs.
    """
    if not name:
        return f'get_by_role("{role}")'

    # Escape quotes in name for display
    safe_name = name.replace('"', '\\"')[:80]
    return f'get_by_role("{role}", name="{safe_name}")'


def _parse_interactive_elements(
    snapshot_text: str,
    max_items: int = 40,
) -> list[dict[str, str]]:
    """Parse aria_snapshot text to extract interactive elements.

    Returns a list of dicts with ``role``, ``name``, ``selector`` keys.
    """
    results: list[dict[str, str]] = []

    for line in snapshot_text.splitlines():
        if len(results) >= max_items:
            break

        m = _ARIA_LINE_RE.match(line)
        if not m:
            continue

        role = m.group("role").lower()
        name = m.group("name") or ""

        if role in _INTERACTIVE_ROLES:
            selector = _suggest_selector(role, name)
            results.append({
                "role": role,
                "name": name[:100] if name else "(no name)",
                "selector": selector,
            })

    return results


# ---------- public API -------------------------------------------------------


class AccessibilityExtractor:
    """Extract a condensed accessibility tree from a Playwright page.

    Uses ``page.locator('body').aria_snapshot()`` (Playwright 1.49+) which
    returns a clean YAML-like text representation.  This is already well-suited
    for LLM consumption — no custom formatting needed.
    """

    @staticmethod
    async def extract_tree(
        page: Any,
        max_depth: int = 6,
        max_lines: int = 150,
    ) -> str:
        """Extract accessibility tree as indented text.

        Args:
            page: Playwright Page object.
            max_depth: Maximum tree traversal depth (approximated by indent level).
            max_lines: Cap output to this many lines.

        Returns:
            Formatted string of the accessibility tree.
        """
        log.info("Extracting accessibility tree", max_depth=max_depth, max_lines=max_lines)
        try:
            snapshot = await page.locator("body").aria_snapshot()
            log.info("Accessibility snapshot obtained", snapshot_len=len(snapshot or ""))
        except Exception as e:
            log.warning("Accessibility snapshot failed", error=str(e))
            return f"(accessibility tree unavailable: {e})"

        if not snapshot or not snapshot.strip():
            log.info("Empty accessibility tree")
            return "(empty accessibility tree — page may still be loading)"

        lines = snapshot.splitlines()

        if not lines:
            return "(no meaningful accessibility nodes found)"

        # Optionally trim by max_depth (indent level ≈ depth)
        if max_depth > 0:
            filtered: list[str] = []
            for line in lines:
                # Count leading spaces; each indent level = 2 spaces
                stripped = line.lstrip()
                indent_chars = len(line) - len(stripped)
                depth = indent_chars // 2
                if depth <= max_depth:
                    filtered.append(line)
            lines = filtered

        if len(lines) > max_lines:
            total = len(lines)
            lines = lines[:max_lines]
            lines.append(f"  ... (showing {max_lines} of {total} lines)")

        return "\n".join(lines)

    @staticmethod
    async def find_interactive_elements(
        page: Any,
        max_items: int = 40,
    ) -> list[dict[str, str]]:
        """Return a list of interactive elements with selector suggestions.

        Each item has keys: ``role``, ``name``, ``selector``.

        Args:
            page: Playwright Page object.
            max_items: Maximum elements to return.

        Returns:
            List of dicts with role, name, and suggested Playwright selector.
        """
        log.info("Finding interactive elements", max_items=max_items)
        try:
            snapshot = await page.locator("body").aria_snapshot()
        except Exception as e:
            log.warning("Accessibility snapshot failed for interactive elements", error=str(e))
            return []

        if not snapshot:
            log.info("No snapshot for interactive elements")
            return []

        elements = _parse_interactive_elements(snapshot, max_items)
        log.info("Found interactive elements", count=len(elements))
        return elements

    @staticmethod
    def format_interactive_list(elements: list[dict[str, str]]) -> str:
        """Format interactive elements list for display."""
        if not elements:
            return "(no interactive elements found)"

        lines: list[str] = []
        for elem in elements:
            lines.append(f"  [{elem['role']}] {elem['name']}  →  {elem['selector']}")
        return "\n".join(lines)
