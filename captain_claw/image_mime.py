"""An image's real type, read from its first bytes.

A file's name can lie: a JPEG saved as ``photo.png`` is common (downloads,
renamed exports, test sets). Captain used to label a picture for the model by
its extension, and a provider that checks (Anthropic) refused the whole
request — "specified using the image/png media type, but the image appears to
be a image/jpeg image" — which ended the agent's turn. The bytes decide; the
name is only the fallback.
"""

from __future__ import annotations

_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"}


def sniff_image_mime(data: bytes) -> str:
    """The image type its bytes show ("image/jpeg" …), or "" when unknown."""
    head = bytes(data[:16])
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in _HEIF_BRANDS:
            return "image/heic"
    for signature, mime in _SIGNATURES:
        if head.startswith(signature):
            return mime
    return ""
