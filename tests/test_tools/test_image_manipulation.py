"""image_manipulation: pictures edited, joined and converted in-process.

The engine tests (captain_claw.image_ops) pin what each operation does to the
pixels. The tool tests run real calls against a temporary workspace: where
results land, that inputs never change, the ``Path:`` lines every chat surface
reads, and the refusals. The wiring tests pin that every agent has the tool,
that eco mode finds it, and that a sign-in deck rules its paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from captain_claw import image_ops as I
from captain_claw.tools.image_manipulation import ImageManipulationTool


def _solid(size=(200, 100), color="steelblue") -> Image.Image:
    return Image.new("RGB", size, color)


def _logo() -> Image.Image:
    """A red square with a white hole, on white."""
    img = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 30, 89, 89), fill="red")
    draw.rectangle((50, 50, 69, 69), fill="white")
    return img


def _apply(img: Image.Image, *ops: dict) -> Image.Image:
    out, _ = I.apply_operations(img, list(ops), I.Context())
    return out


# ── Reading arguments ────────────────────────────────────────────────


def test_lengths_are_pixels_percentages_or_fractions():
    assert I.length(120, 1000, "x") == 120
    assert I.length("120px", 1000, "x") == 120
    assert I.length("25%", 1000, "x") == 250
    assert I.length(0.5, 1000, "x") == 500
    assert I.length(0.5, 1000, "line", fractions=False) == 0
    assert I.length(None, 1000, "x", 7) == 7
    with pytest.raises(I.ImageOpError, match="pixels"):
        I.length("wide", 1000, "x")


def test_factors_accept_relative_percentages():
    assert I.factor(1.2, "brightness") == 1.2
    assert I.factor("120%", "brightness") == pytest.approx(1.2)
    assert I.factor("+20%", "brightness") == pytest.approx(1.2)
    assert I.factor("-20%", "brightness") == pytest.approx(0.8)
    with pytest.raises(I.ImageOpError, match="1 = unchanged"):
        I.factor(120, "brightness")


def test_colours_and_anchors():
    assert I.colour("red") == (255, 0, 0, 255)
    assert I.colour("#00000080") == (0, 0, 0, 128)
    assert I.colour("ff8800") == (255, 136, 0, 255)
    assert I.colour("transparent") == (0, 0, 0, 0)
    assert I.colour([10, 20, 30]) == (10, 20, 30, 255)
    with pytest.raises(I.ImageOpError, match="isn't a colour"):
        I.colour("blurple")
    assert I.anchor("bottom right") == I.anchor("bottom-right") == I.anchor("lower_right") == (1.0, 1.0)
    assert I.anchor("top center") == I.anchor("top") == (0.5, 0.0)
    assert I.anchor("centre") == (0.5, 0.5)
    with pytest.raises(I.ImageOpError, match="one of"):
        I.anchor("somewhere")
    assert I.ratio("16:9") == pytest.approx(16 / 9)
    assert I.ratio("square") == 1.0


# ── Geometry ─────────────────────────────────────────────────────────


def test_resize_keeps_proportions_and_fits():
    img = _solid((400, 200))
    assert _apply(img, {"op": "resize", "width": 100}).size == (100, 50)
    assert _apply(img, {"op": "resize", "height": "50%"}).size == (200, 100)
    assert _apply(img, {"op": "resize", "scale": 0.25}).size == (100, 50)
    assert _apply(img, {"op": "resize", "width": 100, "height": 100}).size == (100, 50)  # contain
    assert _apply(img, {"op": "resize", "width": 100, "height": 100, "fit": "cover"}).size == (100, 100)
    assert _apply(img, {"op": "resize", "width": 100, "height": 100, "fit": "fill"}).size == (100, 100)
    padded = _apply(img, {"op": "resize", "width": 100, "height": 100, "fit": "pad", "background": "black"})
    assert padded.size == (100, 100) and padded.getpixel((50, 5)) == (0, 0, 0)
    assert _apply(img, {"op": "resize", "width": 800, "shrink_only": True}).size == (400, 200)
    with pytest.raises(I.ImageOpError, match="width, height or scale"):
        _apply(img, {"op": "resize"})


def test_crop_by_box_aspect_and_insets():
    img = _solid((400, 200))
    assert _apply(img, {"op": "crop", "x": 10, "y": 20, "width": 100, "height": 50}).size == (100, 50)
    assert _apply(img, {"op": "crop", "box": [0, 0, 50, 60]}).size == (50, 60)
    assert _apply(img, {"op": "crop", "aspect": "1:1"}).size == (200, 200)
    assert _apply(img, {"op": "crop", "top": 10, "bottom": "10%", "left": 50}).size == (350, 170)
    with pytest.raises(I.ImageOpError, match="empty or outside"):
        _apply(img, {"op": "crop", "x": 500, "y": 0, "width": 10, "height": 10})


def test_rotate_is_clockwise_and_flip_mirrors():
    img = _solid((40, 20), "white")
    img.putpixel((0, 0), (255, 0, 0))
    turned = _apply(img, {"op": "rotate", "angle": 90})
    assert turned.size == (20, 40)
    assert turned.getpixel((19, 0)) == (255, 0, 0)  # top-left went to top-right
    assert _apply(img, {"op": "flip"}).getpixel((39, 0)) == (255, 0, 0)
    assert _apply(img, {"op": "mirror"}).getpixel((39, 0)) == (255, 0, 0)
    tilted = _apply(img, {"op": "rotate", "angle": 30})
    assert tilted.mode == "RGBA" and tilted.size[0] > 40  # expanded, corners see-through


def test_pad_border_trim_round_corners_and_circle():
    img = _solid((100, 50))
    assert _apply(img, {"op": "pad", "aspect": "1:1"}).size == (100, 100)
    assert _apply(img, {"op": "pad", "size": 10}).size == (120, 70)
    assert _apply(img, {"op": "border", "size": 5, "color": "red"}).getpixel((0, 0)) == (255, 0, 0)
    framed = _apply(img, {"op": "pad", "size": 15, "background": "white"})
    assert _apply(framed, {"op": "trim"}).size == (100, 50)
    rounded = _apply(img, {"op": "round_corners", "radius": 20})
    assert rounded.getpixel((0, 0))[3] == 0 and rounded.getpixel((50, 25))[3] == 255
    circle = _apply(img, {"op": "circle"})
    assert circle.size == (50, 50) and circle.getpixel((0, 0))[3] == 0 and circle.getpixel((25, 25))[3] == 255


def test_perspective_straightens_to_the_corner_distances():
    img = _solid((400, 300))
    out = _apply(img, {"op": "perspective", "corners": [[10, 10], [310, 20], [300, 220], [0, 210]]})
    assert out.size == (300, 200)  # the longer of each pair of opposite sides
    with pytest.raises(I.ImageOpError, match="four corners"):
        _apply(img, {"op": "perspective", "corners": [[0, 0], [1, 1]]})


# ── Colour, filters, regions ─────────────────────────────────────────


def test_adjustments_and_their_short_forms():
    img = _solid((10, 10), (100, 100, 100))
    assert _apply(img, {"op": "brightness", "amount": 2}).getpixel((0, 0)) == (200, 200, 200)
    assert _apply(img, {"op": "adjust", "brightness": "-50%"}).getpixel((0, 0)) == (50, 50, 50)
    assert _apply(_solid((4, 4), "red"), {"op": "grayscale"}).mode == "L"
    assert _apply(_solid((4, 4), "red"), {"op": "greyscale"}).mode == "L"
    assert _apply(img, {"op": "invert"}).getpixel((0, 0)) == (155, 155, 155)
    with pytest.raises(I.ImageOpError, match="give the amount"):
        _apply(img, {"op": "contrast"})
    with pytest.raises(I.ImageOpError, match="unknown op 'explode'"):
        _apply(img, {"op": "explode"})


def test_a_region_changes_only_that_part():
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), ((x * 7) % 256, (y * 13) % 256, 0))
    out = _apply(img, {"op": "pixelate", "size": 10, "region": {"x": 0, "y": 0, "width": "50%", "height": 50}})
    assert out.getpixel((60, 60)) == img.getpixel((60, 60))
    assert out.getpixel((1, 1)) == out.getpixel((8, 8))  # one block
    blurred = _apply(img, {"op": "blur", "radius": 5, "box": [50, 50, 100, 100]})
    assert blurred.getpixel((10, 10)) == img.getpixel((10, 10))


def test_transparent_follows_the_background_from_the_edges():
    out = _apply(_logo(), {"op": "remove_background"})
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0  # the white around
    assert out.getpixel((40, 40)) == (255, 0, 0, 255)
    assert out.getpixel((60, 60))[3] == 255  # the white hole inside stays
    assert _apply(_logo(), {"op": "transparent", "everywhere": True}).getpixel((60, 60))[3] == 0


def test_without_scipy_the_flood_fill_still_keeps_the_inside(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    big = _logo().resize((1500, 1500), Image.Resampling.NEAREST)  # downscaled flood fill
    out = _apply(big, {"op": "transparent"})
    assert out.getpixel((5, 5))[3] == 0 and out.getpixel((750, 750))[3] == 255


def test_gradient_tint_quantize_and_threshold():
    img = _solid((10, 100), "white")
    darker = _apply(img, {"op": "gradient", "colors": ["#00000000", "#000000ff"]})
    assert darker.getpixel((5, 0)) == (255, 255, 255) and darker.getpixel((5, 99))[0] < 10
    assert _apply(img, {"op": "tint", "color": "red", "amount": 1}).getpixel((0, 0)) == (255, 0, 0)
    assert _apply(_solid(), {"op": "quantize", "colors": 4}).mode == "P"
    assert set(np.unique(np.asarray(_apply(_solid(), {"op": "threshold", "threshold": 100})))) <= {0, 255}


# ── Drawing ──────────────────────────────────────────────────────────


def test_text_is_drawn_in_a_contrasting_colour():
    dark = _solid((400, 200), "black")
    out = _apply(dark, {"op": "text", "text": "Hello", "position": "center", "size": 60})
    assert np.asarray(out)[..., 0].max() > 200  # light letters on black
    light = _solid((400, 200), "white")
    out = _apply(light, {"op": "text", "text": "Hello", "position": "center", "size": 60})
    assert np.asarray(out)[..., 0].min() < 60


def test_text_font_has_every_character_when_one_is_installed():
    if not I.font_index():
        pytest.skip("no fonts installed")
    font, note = I.Fonts().get(None, 40, text="Pozdrav čćžšđ ČĆŽŠĐ")
    if "no installed font has every character" in note:
        pytest.skip("no installed font covers Croatian letters")
    assert I._missing_glyphs(font, "čćžšđČĆŽŠĐ") == set()


def test_watermark_tiles_and_shapes_draw():
    img = _solid((300, 300), "white")
    tiled = _apply(img, {"op": "watermark", "text": "DRAFT", "tile": True, "angle": -30, "opacity": 1})
    assert int((np.asarray(tiled) != 255).any(axis=2).sum()) > 2000
    boxed = _apply(img, {"op": "rectangle", "x": 10, "y": 10, "width": 50, "height": 50, "fill": "blue"})
    assert boxed.getpixel((30, 30)) == (0, 0, 255)
    arrow = _apply(img, {"op": "arrow", "points": [[10, 150], [290, 150]], "line_width": 6})
    assert arrow.getpixel((150, 150)) != (255, 255, 255)


# ── Several pictures ─────────────────────────────────────────────────


def test_combine_side_by_side_matches_the_middle_height():
    images = [_solid((200, 100)), _solid((100, 200), "red"), _solid((50, 50), "green")]
    out, summary = I.combine(images, {"gap": 10, "background": "black"}, I.Context())
    # middle height 100: 200 + 50 + 100 wide, plus two gaps.
    assert out.size == (200 + 50 + 100 + 20, 100)
    assert "side by side" in summary
    tall, _ = I.combine(images, {"layout": "vertical", "width": 100}, I.Context())
    assert tall.width == 100
    labelled, _ = I.combine(images[:2], {"labels": ["Before", "After"]}, I.Context())
    assert labelled.height > 100


def test_combine_grid_and_stack():
    images = [_solid((100, 100))] * 5
    grid, summary = I.combine(images, {"layout": "grid", "columns": 3, "gap": 4}, I.Context())
    assert grid.size == (3 * 100 + 2 * 4, 2 * 100 + 4)
    assert "3×2 grid" in summary
    stack, _ = I.combine([_solid((100, 100)), _solid((20, 20), "red")], {"layout": "stack"}, I.Context())
    assert stack.size == (100, 100) and stack.getpixel((50, 50))[:3] == (255, 0, 0)
    with pytest.raises(I.ImageOpError, match="at least two"):
        I.combine([_solid()], {}, I.Context())


def test_overlay_places_scales_and_fades():
    base = _solid((400, 200), "black")
    sticker = _solid((100, 100), "white")
    out = I.overlay(base, [(sticker, {"position": "bottom-right", "scale": 0.1, "margin": 0})], I.Context())
    assert out.getpixel((395, 195))[:3] == (255, 255, 255) and out.getpixel((355, 195))[:3] == (0, 0, 0)
    faded = I.overlay(base, [(sticker, {"x": 0, "y": 0, "opacity": 0.5})], I.Context())
    assert 120 < faded.getpixel((10, 10))[0] < 135
    multiplied = I.overlay(_solid((10, 10), (200, 200, 200)), [(_solid((10, 10), (128, 128, 128)),
                                                                {"blend": "multiply"})], I.Context())
    assert multiplied.getpixel((5, 5))[0] == 200 * 128 // 255
    cut = I.overlay(base, [(_logo(), {"x": 0, "y": 0, "remove_background": True})], I.Context())
    assert cut.getpixel((5, 5))[:3] == (0, 0, 0) and cut.getpixel((40, 40))[:3] == (255, 0, 0)


def test_animate_split_and_pdf_pages():
    frames, durations = I.animate([_solid(), _solid(color="red"), _solid(color="green")],
                                  {"duration": 200, "bounce": True}, I.Context())
    assert len(frames) == 4 and durations == [200] * 4
    tiles = I.split(_solid((300, 200)), {"rows": 2, "columns": 3})
    assert [k for k, _ in tiles][:2] == ["r1c1", "r1c2"] and tiles[0][1].size == (100, 100)
    pages, dpi = I.pdf_pages([_solid((3000, 2000)), _solid((100, 200))], {"page_size": "a4"})
    assert dpi == 150 and pages[0].size == (1754, 1240) and pages[1].size == (1240, 1754)


# ── Opening and saving ───────────────────────────────────────────────


def test_a_sideways_phone_photo_opens_upright(tmp_path):
    img = _solid((60, 40))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated: shown turned 90° clockwise
    path = tmp_path / "phone.jpg"
    img.save(path, exif=exif.tobytes())
    assert I.load(path).image.size == (40, 60)
    assert any("Stored sideways" in line for line in I.describe(path))


def test_formats_follow_transparency():
    ctx = I.Context()
    assert I.choose_format(asked=None, source="JPEG", alpha=True, animated=False)[0] == "PNG"
    assert I.choose_format(asked=None, source="PNG", alpha=False, animated=True)[0] == "PNG"
    assert I.choose_format(asked=None, source="JPEG", alpha=False, animated=True)[0] == "GIF"
    data = I.encode([Image.new("RGBA", (10, 10), (0, 0, 0, 0))], "JPEG", ctx)
    assert data[:2] == b"\xff\xd8" and any("no transparency" in n for n in ctx.notes)
    with pytest.raises(I.ImageOpError, match="format"):
        I.format_named("heic")


def test_too_large_is_refused_before_it_is_made():
    with pytest.raises(I.ImageOpError, match="megapixel"):
        _apply(_solid(), {"op": "resize", "width": 100_000})


# ── The tool ─────────────────────────────────────────────────────────


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    (root / "saved").mkdir(parents=True)
    _solid((640, 480)).save(root / "a.jpg")
    _solid((300, 600), "seagreen").save(root / "b.png")
    _logo().save(root / "logo.png")
    return root


def _kw(ws: Path, **extra) -> dict:
    return {"_runtime_base_path": str(ws), "_saved_base_path": str(ws / "saved"), "_session_id": "s1", **extra}


def _paths(content: str, label: str = "Path") -> list[Path]:
    return [Path(line.split(":", 1)[1].strip()) for line in content.splitlines() if line.startswith(f"{label}: ")]


async def test_edit_saves_a_new_file_and_never_changes_the_input(ws):
    before = (ws / "a.jpg").read_bytes()
    tool = ImageManipulationTool()
    ops = [{"op": "resize", "width": 320}, {"op": "text", "text": "Zagreb", "position": "bottom"}]
    result = await tool.execute(action="edit", path="a.jpg", operations=ops, **_kw(ws))
    assert result.success, result.error
    [out] = _paths(result.content)
    assert out == ws / "saved" / "media" / "s1" / "a-edited.jpg"
    assert Image.open(out).size == (320, 240)
    assert (ws / "a.jpg").read_bytes() == before
    assert "Applied: resize → 320×240, text" in result.content
    again = await tool.execute(action="edit", path="a.jpg", operations=json.dumps(ops), **_kw(ws))
    assert _paths(again.content) == [out.with_name("a-edited-2.jpg")]
    replaced = await tool.execute(action="edit", path="a.jpg", operations=ops, output_path="a-edited.jpg",
                                  overwrite=True, **_kw(ws))
    assert _paths(replaced.content) == [out]


async def test_transparency_switches_jpeg_to_png_and_says_so(ws):
    result = await ImageManipulationTool().execute(action="edit", path="a.jpg", operations=[{"op": "circle"}],
                                                   **_kw(ws))
    [out] = _paths(result.content)
    assert out.suffix == ".png" and Image.open(out).mode == "RGBA"
    assert "saved as PNG to keep the transparency" in result.content


async def test_an_op_name_as_the_action_is_one_edit(ws):
    result = await ImageManipulationTool().execute(action="resize", path="a.jpg", width="50%", **_kw(ws))
    assert result.success, result.error
    assert Image.open(_paths(result.content)[0]).size == (320, 240)


async def test_convert_and_batch_names(ws):
    tool = ImageManipulationTool()
    result = await tool.execute(action="convert", paths=["a.jpg", "b.png"], format="webp", **_kw(ws))
    assert [p.name for p in _paths(result.content)] == ["a.webp", "b.webp"]
    folder = await tool.execute(action="edit", paths=["a.jpg", "b.png"], output_path="small",
                                operations=[{"op": "resize", "width": 50}], **_kw(ws))
    assert [p.parent.name for p in _paths(folder.content)] == ["small", "small"]
    pattern = await tool.execute(action="edit", paths=["a.jpg", "b.png"], output_path="thumb.png",
                                 operations=[{"op": "resize", "width": 50}], **_kw(ws))
    assert [p.name for p in _paths(pattern.content)] == ["thumb-1.png", "thumb-2.png"]
    refused = await tool.execute(action="convert", path="a.jpg", **_kw(ws))
    assert not refused.success and "format" in refused.error


async def test_combine_overlay_and_create(ws):
    tool = ImageManipulationTool()
    joined = await tool.execute(action="combine", paths=["a.jpg", "b.png"], gap=10, output_path="pair.png",
                                labels=["A", "B"], **_kw(ws))
    assert joined.success, joined.error
    [pair] = _paths(joined.content)
    assert pair.name == "pair.png" and "From: a.jpg, b.png" in joined.content
    collage = await tool.execute(action="collage", paths=["a.jpg", "b.png", "logo.png", "a.jpg"], columns=2,
                                 **_kw(ws))
    assert "2×2 grid" in collage.content
    stamped = await tool.execute(action="overlay", path="a.jpg", paths=["logo.png", "b.png"],
                                 layers=[{"image": 1, "position": "top-left", "scale": 0.1, "remove_background": True},
                                         {"image": 2, "x": "50%", "y": 0, "width": 100, "opacity": 0.5}], **_kw(ws))
    assert stamped.success, stamped.error and "Put 2 pictures on a.jpg" in stamped.content
    made = await tool.execute(action="create", width=600, height=300, background="navy",
                              operations=[{"op": "text", "text": "Hi"}], **_kw(ws))
    assert Image.open(_paths(made.content)[0]).size == (600, 300)
    lonely = await tool.execute(action="overlay", path="a.jpg", **_kw(ws))
    assert not lonely.success and "paths" in lonely.error


async def test_animate_frames_split_and_pdf(ws):
    tool = ImageManipulationTool()
    anim = await tool.execute(action="animate", paths=["a.jpg", "b.png", "logo.png"], duration=150, **_kw(ws))
    [gif] = _paths(anim.content)
    assert gif.suffix == ".gif" and Image.open(gif).n_frames == 3
    resized = await tool.execute(action="edit", path=str(gif), operations=[{"op": "resize", "width": 64}], **_kw(ws))
    assert Image.open(_paths(resized.content)[0]).n_frames == 3  # still animated
    frames = await tool.execute(action="frames", path=str(gif), **_kw(ws))
    assert _paths(frames.content) == [] and len(_paths(frames.content, "File")) == 3  # not posted to the chat
    tiles = await tool.execute(action="split", path="a.jpg", rows=2, columns=2, **_kw(ws))
    assert [p.name for p in _paths(tiles.content)] == ["a-r1c1.jpg", "a-r1c2.jpg", "a-r2c1.jpg", "a-r2c2.jpg"]
    pdf = await tool.execute(action="pdf", paths=["a.jpg", "b.png"], **_kw(ws))
    [doc] = _paths(pdf.content)
    assert doc.suffix == ".pdf" and doc.read_bytes().startswith(b"%PDF") and "2-page PDF" in pdf.content


async def test_info_and_refusals(ws):
    tool = ImageManipulationTool()
    info = await tool.execute(action="info", paths=["a.jpg", "logo.png"], **_kw(ws))
    assert "a.jpg: 640×480 px, JPEG" in info.content and "logo.png: 120×120 px, PNG" in info.content
    missing = await tool.execute(action="edit", path="nope.png", operations=[{"op": "invert"}], **_kw(ws))
    assert not missing.success and "not found" in missing.error
    web = await tool.execute(action="edit", path="https://example.com/x.png", operations=[{"op": "invert"}], **_kw(ws))
    assert not web.success and "web address" in web.error
    (ws / "notes.txt").write_text("hello")
    text = await tool.execute(action="edit", path="notes.txt", operations=[{"op": "invert"}], **_kw(ws))
    assert not text.success and "isn't a picture" in text.error
    unknown = await tool.execute(action="sing", path="a.jpg", **_kw(ws))
    assert not unknown.success and "Unknown action" in unknown.error


async def test_a_model_that_sees_images_gets_the_results(ws):
    result = await ImageManipulationTool().execute(action="edit", path="a.jpg", operations=[{"op": "flip"}],
                                                   **_kw(ws, _look_inline=True))
    assert result.images == [str(_paths(result.content)[0])]
    plain = await ImageManipulationTool().execute(action="edit", path="a.jpg", operations=[{"op": "flip"}], **_kw(ws))
    assert plain.images == []


async def test_every_chat_surface_reads_its_pictures(ws):
    from captain_claw import reply_images
    from captain_claw.platform_adapter import _PICTURE_MAKERS, extract_image_paths_from_tool_output
    from captain_claw.tools import whatsapp_send_file
    from captain_claw.web import chat_handler

    result = await ImageManipulationTool().execute(action="combine", paths=["a.jpg", "b.png"], **_kw(ws))
    assert extract_image_paths_from_tool_output(result.content) == _paths(result.content)
    assert "image_manipulation" in reply_images._PICTURE_TOOLS
    assert "image_manipulation" in _PICTURE_MAKERS
    assert "image_manipulation" in whatsapp_send_file._DELIVERED_IMAGE_TOOLS
    assert "image_manipulation" in chat_handler._CARDED_IMAGE_TOOLS


# ── Wiring ───────────────────────────────────────────────────────────


def test_every_agent_has_it():
    from captain_claw.config import ToolsConfig

    assert "image_manipulation" in ToolsConfig().enabled
    assert "image_manipulation" in ToolsConfig(enabled=["read"]).enabled


@pytest.mark.parametrize("message", [
    "resize this photo to 1080 wide",
    "put these two pictures side by side",
    "make a collage of the screenshots",
    "add our logo as a watermark",
    "crop report.png to 16:9",
    "spoji ove dvije slike",
    "napravi kolaž od fotografija",
    "stavi vodeni žig na fotku",
])
def test_eco_mode_finds_it(message):
    from captain_claw.agent_orchestration_mixin import _eco_select_tools_by_intent

    assert "image_manipulation" in _eco_select_tools_by_intent(message)


def test_the_schema_suits_every_provider():
    def walk(schema, where):
        for key, value in (schema.get("properties") or {}).items():
            assert "type" in value, f"{where}.{key} has no type"
            if value["type"] == "array":
                assert value.get("items"), f"{where}.{key} has no items"
                if isinstance(value["items"], dict):
                    walk(value["items"], f"{where}.{key}[]")
            if value["type"] == "object":
                walk(value, f"{where}.{key}")

    walk(ImageManipulationTool.parameters, "image_manipulation")
    json.dumps(ImageManipulationTool.parameters)
    ops = ImageManipulationTool.parameters["properties"]["operations"]["items"]["properties"]["op"]["enum"]
    assert set(I.OPS) <= set(ops)


def test_risk_and_rehearsal():
    from captain_claw import rehearsal_turn
    from captain_claw.flight_deck.tool_risk import risk_for

    assert risk_for("image_manipulation", {"action": "info"}) == "read"
    assert risk_for("image_manipulation", {"action": "combine"}) == "write"
    assert rehearsal_turn.classify("image_manipulation", {"action": "edit"})[0] == "allow"


def test_a_sign_in_deck_rules_every_path(tmp_path, monkeypatch):
    from captain_claw import deck_confine

    agent, work = tmp_path / "agent", tmp_path / "agent" / "data" / "workspace"
    work.mkdir(parents=True)
    other = tmp_path / "other-tenant"
    other.mkdir()
    monkeypatch.setenv("CLAW_CONFINE", "user")
    monkeypatch.setenv("CLAW_CONFINE_HOME", str(agent))
    monkeypatch.setenv("CLAW_CONFINE_WORKSPACE", str(work))
    for name in ("a.png", "b.png"):
        _solid().save(work / name)
    check = deck_confine.check_tool_paths
    assert check("image_manipulation", {"action": "combine", "paths": ["a.png", "b.png"],
                                        "output_path": "out.png"}, work) is None
    assert check("image_manipulation", {"action": "combine", "paths": ["a.png", str(other / "x.png")]}, work)
    assert check("image_manipulation", {"action": "edit", "path": str(other / "x.png")}, work)
    assert check("image_manipulation", {"action": "edit", "path": "a.png", "output_path": str(other / "y.png")}, work)
