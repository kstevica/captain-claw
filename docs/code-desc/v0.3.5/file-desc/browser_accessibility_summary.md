# Summary: browser_accessibility.py

# browser_accessibility.py Summary

## Summary
This module extracts semantic accessibility trees from web pages using Playwright's `aria_snapshot()` API, converting the YAML-like output into a clean, LLM-friendly format. It identifies interactive elements (buttons, links, textboxes, etc.) and generates resilient Playwright selectors for programmatic interaction with web applications, particularly SPAs and React-based sites with complex DOM structures.

## Purpose
Solves the problem of navigating and interacting with modern web applications by:
- Providing a semantic view of page structure independent of messy underlying DOM
- Extracting interactive elements with their accessibility roles and names
- Generating robust `get_by_role()` style selectors that survive DOM mutations
- Filtering and formatting accessibility data for LLM consumption with configurable depth/line limits

## Most Important Functions/Classes

1. **`AccessibilityExtractor.extract_tree(page, max_depth, max_lines)`**
   - Retrieves the accessibility tree from a page via `aria_snapshot()` and filters by depth/line count
   - Returns formatted indented text representation suitable for LLM analysis
   - Handles errors gracefully with fallback messages

2. **`AccessibilityExtractor.find_interactive_elements(page, max_items)`**
   - Extracts interactive elements (buttons, links, textboxes, checkboxes, etc.) from the accessibility snapshot
   - Returns list of dicts containing role, name, and suggested Playwright selector for each element
   - Limits results to prevent overwhelming LLM context

3. **`_parse_interactive_elements(snapshot_text, max_items)`**
   - Parses YAML-like aria_snapshot text using regex pattern matching
   - Filters for roles in `_INTERACTIVE_ROLES` frozenset
   - Returns structured list with role, name (truncated to 100 chars), and selector suggestions

4. **`_suggest_selector(role, name)`**
   - Generates Playwright locator strings using `get_by_role()` pattern (most resilient for SPAs)
   - Escapes quotes in element names and truncates to 80 characters for readability
   - Fallback to role-only selector if name is unavailable

5. **`AccessibilityExtractor.format_interactive_list(elements)`**
   - Formats parsed interactive elements into human-readable display format
   - Shows role, name, and selector suggestion per line for easy reference

## Architecture & Dependencies
- **Core Dependency**: Playwright 1.49+ (uses modern `aria_snapshot()` API)
- **Logging**: Integrates with `captain_claw.logging` for structured logging
- **Regex-based Parsing**: `_ARIA_LINE_RE` pattern matches accessibility tree lines with groups for indent, role, name, and attributes
- **Role Classification**: Hardcoded frozenset of 17 interactive ARIA roles (button, link, textbox, checkbox, radio, combobox, searchbox, slider, spinbutton, switch, tab, menuitem variants, option, treeitem)
- **Static Methods**: All public methods are static, enabling use without instantiation; designed for integration with async Playwright workflows