"""An image's type comes from its bytes, not its file name (image_mime): a
JPEG saved as .png sent as image/png was refused by the provider and ended
the agent's turn."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from captain_claw.image_mime import sniff_image_mime
from captain_claw.llm import vision


def _image_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 40, 40)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt, mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"),
    ("WEBP", "image/webp"), ("BMP", "image/bmp"), ("TIFF", "image/tiff"),
])
def test_the_type_is_read_from_the_bytes(fmt, mime):
    assert sniff_image_mime(_image_bytes(fmt)) == mime


def test_unknown_bytes_give_no_type():
    assert sniff_image_mime(b"not an image at all") == ""
    assert sniff_image_mime(b"") == ""


def test_a_jpeg_named_png_goes_to_the_model_as_a_jpeg(tmp_path):
    picture = tmp_path / "cat_sleeping_couch.png"
    picture.write_bytes(_image_bytes("JPEG"))
    encoded = vision.encode_image(str(picture))
    assert encoded is not None and encoded.mime == "image/jpeg"
    assert encoded.data_url.startswith("data:image/jpeg;base64,")


def test_a_correctly_named_png_stays_a_png(tmp_path):
    picture = tmp_path / "chart.png"
    picture.write_bytes(_image_bytes("PNG"))
    assert vision.encode_image(str(picture)).mime == "image/png"
