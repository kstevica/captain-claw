"""Response types a browser renders as a script-capable document.

Shared by the agent's file routes (``/api/files/view``, ``/api/files/download``,
``/api/media``) and Flight Deck's agent-file-view proxy: an uploaded HTML or SVG
served inline on the agent's origin is stored XSS unless it is sandboxed.
"""

from __future__ import annotations

ACTIVE_VIEW_TYPES = frozenset({"text/html", "application/xhtml+xml", "image/svg+xml",
                               "text/xml", "application/xml", "text/xsl"})


def active_view_type(content_type: str) -> bool:
    """A response type a browser renders as a document that can run script:
    HTML, SVG and every XML type (browsers render any ``*+xml`` type — e.g.
    ``.rss``/``.atom``/``.xslt`` as the agent labels them — as XML, where
    XHTML-namespaced script runs)."""
    base = str(content_type or "").split(";", 1)[0].strip().lower()
    return base in ACTIVE_VIEW_TYPES or base.endswith("+xml")


# Sandboxed without allow-same-origin: the document runs on an opaque origin,
# so its script can't read the agent's cookies, storage or API.
ACTIVE_VIEW_CSP = "sandbox allow-scripts allow-popups allow-forms allow-modals allow-downloads"
