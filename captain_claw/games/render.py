"""ASCII rendering of a per-character view.

This is a *human-channel* helper. Agents receive the structured view
dict directly; the renderer just composes a friendlier display for the
spectator UI.
"""

from __future__ import annotations

from typing import Any


def render_view(view: dict[str, Any]) -> str:
    """Render a view dict to a single ASCII string."""
    if "error" in view:
        return f"[error] {view['error']}"

    char = view["character"]
    cr = view.get("current_room")
    lines: list[str] = []

    lines.append(f"=== Tick {view['tick']} — {char['name']} ({char['glyph']}) ===")
    lines.append(f"Objective: {char['objective']}")
    lines.append("")

    if cr is None:
        lines.append("(you are nowhere)")
    else:
        lines.append(f"-- {cr['name']} --")
        if cr.get("ascii_tile"):
            lines.extend(cr["ascii_tile"])
            lines.append("")
        lines.append(cr["description"])
        lines.append("")
        if cr.get("entities"):
            lines.append("You see:")
            for e in cr["entities"]:
                tags = []
                if e.get("takeable"):
                    tags.append("takeable")
                if e.get("examinable"):
                    tags.append("examine" if not e.get("examined") else "examined")
                tag = f" ({', '.join(tags)})" if tags else ""
                lines.append(f"  {e['glyph']} {e['name']}{tag} [{e['id']}]")
        if cr.get("others_here"):
            lines.append("Also here:")
            for o in cr["others_here"]:
                lines.append(f"  {o['glyph']} {o['name']}")
        if cr.get("exits"):
            locked = cr.get("locked_exits", {})
            exit_parts = []
            for d, r in sorted(cr["exits"].items()):
                lock_tag = " 🔒" if locked.get(d) else ""
                exit_parts.append(f"{d}→{r}{lock_tag}")
            exits_str = ", ".join(exit_parts)
            lines.append(f"Exits: {exits_str}")

    inv = view.get("inventory", [])
    lines.append("")
    if inv:
        lines.append("Inventory: " + ", ".join(e["name"] for e in inv))
    else:
        lines.append("Inventory: (empty)")

    heard = view.get("heard", [])
    if heard:
        lines.append("")
        lines.append("You heard:")
        for h in heard:
            lines.append(f'  {h["actor_name"]}: "{h["text"]}"')

    if view.get("terminal"):
        lines.append("")
        lines.append("*** GAME OVER " + ("(WIN)" if view.get("win") else "") + " ***")

    return "\n".join(lines)
