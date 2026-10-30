"""Render the StopWatch or PaperColor screens to PNGs (and one contact sheet).

    pio run -e preview
    uv run --no-project --with pillow python preview/render.py [out dir]

    pio run -e preview_paper
    uv run --no-project --with pillow python preview/render.py paper [out dir]

    pio run -e preview_epaper154 -e preview_epaper154_touch
    uv run --no-project --with pillow python preview/render.py epaper154

Runs the preview program (the real ui_round.cpp / ui_paper.cpp built for the
Mac), then shows every frame the way it looks on the device: the StopWatch's
round screen in its bezel; the PaperColor's six inks in the muted colours an
E Ink Spectra 6 panel really has, in its frame. Frames: <out>/NN_name.png;
all of them: <out>/sheet.png.
"""

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
D = 466
BEZEL = 26


def sample_picture(path: Path) -> None:
    """A landscape picture as the deck sends it with round=1: filling the
    circle (its shorter side the screen's), 4:3 here."""
    w, h = round(D * 4 / 3), D
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        t = y / h
        for x in range(w):
            px[x, y] = (int(250 - 120 * t), int(150 - 60 * t + 20 * (x / w)), int(90 + 110 * t))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.62, h * 0.22, w * 0.62 + 70, h * 0.22 + 70), fill=(255, 230, 160))
    d.polygon([(0, h), (0, h * 0.62), (w * 0.25, h * 0.38), (w * 0.48, h * 0.66), (w * 0.7, h * 0.45),
               (w, h * 0.7), (w, h)], fill=(40, 40, 80))
    d.polygon([(0, h), (0, h * 0.8), (w * 0.35, h * 0.62), (w * 0.6, h * 0.82), (w, h * 0.74), (w, h)],
              fill=(20, 22, 45))
    im.save(path, "JPEG", quality=85)


def sample_background(path: Path) -> None:
    """A picture made the way Flight Deck makes a background
    (flight_deck/device_backgrounds.prepare): filling the circle (the watch
    darkens it behind its screens itself)."""
    w = h = D
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        t = y / h
        for x in range(w):
            px[x, y] = (int(40 + 200 * t), int(90 + 60 * t + 30 * (x / w)), int(200 - 90 * t))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.6, h * 0.16, w * 0.6 + 90, h * 0.16 + 90), fill=(255, 236, 170))
    d.polygon([(0, h), (0, h * 0.6), (w * 0.3, h * 0.4), (w * 0.55, h * 0.62), (w * 0.8, h * 0.44),
               (w, h * 0.58), (w, h)], fill=(60, 50, 110))
    d.polygon([(0, h), (0, h * 0.78), (w * 0.4, h * 0.64), (w * 0.7, h * 0.8), (w, h * 0.72), (w, h)],
              fill=(25, 30, 60))
    im.save(path, "JPEG", quality=88)


def framed(ppm: Path) -> Image.Image:
    """The frame as seen on the watch: round, inside a dark bezel."""
    screen = Image.open(ppm).convert("RGB")
    size = D + 2 * BEZEL
    out = Image.new("RGB", (size, size), (16, 16, 18))
    mask = Image.new("L", (D * 4, D * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, D * 4 - 1, D * 4 - 1), fill=255)
    mask = mask.resize((D, D), Image.LANCZOS)
    body = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(body).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    body = body.resize((size, size), Image.LANCZOS)
    case = Image.new("RGB", (size, size), (34, 35, 39))
    bg = Image.new("RGB", (size, size), (244, 244, 246))
    bg.paste(case, (0, 0), body)
    bg.paste(screen, (BEZEL, BEZEL), mask)
    return bg


# The panel's inks as the firmware draws them → as they look on the paper.
PAPER_INKS = [
    ((0, 0, 0), (40, 40, 46)),
    ((255, 255, 255), (228, 226, 216)),
    ((255, 243, 56), (224, 198, 46)),
    ((191, 0, 0), (170, 40, 38)),
    ((100, 64, 255), (52, 74, 152)),
    ((67, 138, 28), (60, 114, 68)),
]
PAPER_MARGIN = 46


def sample_paper_picture(path: Path) -> None:
    """A 4:3 picture as the deck sends it with paper=1 (its own shape,
    fitted to 400 wide): a sunset over hills and the sea."""
    w, h = 400, 300
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        t = y / h
        for x in range(w):
            px[x, y] = (int(255 - 90 * t), int(170 - 110 * t + 30 * (x / w)), int(80 + 140 * t))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.58, h * 0.30, w * 0.58 + 70, h * 0.30 + 70), fill=(255, 236, 150))
    d.polygon([(0, h * 0.72), (w * 0.22, h * 0.42), (w * 0.4, h * 0.6), (w * 0.62, h * 0.38), (w * 0.82, h * 0.58),
               (w, h * 0.5), (w, h * 0.78), (0, h * 0.78)], fill=(70, 60, 110))
    d.rectangle((0, h * 0.78, w, h), fill=(30, 70, 130))
    for i in range(9):
        y = h * 0.8 + i * 7
        d.line((w * 0.6 - 30 + i * 2, y, w * 0.6 + 60 - i * 3, y), fill=(250, 200, 120), width=2)
    im.save(path, "JPEG", quality=88)


def paper_framed(ppm: Path) -> Image.Image:
    """The frame as it looks on the PaperColor: inks mapped to the panel's
    muted colours, inside the device's light frame."""
    screen = Image.open(ppm).convert("RGB")
    pal = Image.new("P", (1, 1))
    pal.putpalette([c for ink, _ in PAPER_INKS for c in ink] + [0] * (768 - 3 * len(PAPER_INKS)))
    q = screen.quantize(palette=pal, dither=Image.Dither.NONE)
    q.putpalette([c for _, look in PAPER_INKS for c in look] + [0] * (768 - 3 * len(PAPER_INKS)))
    screen = q.convert("RGB")
    w, h = screen.size
    m = PAPER_MARGIN
    size = (w + 2 * m, h + 2 * m)
    out = Image.new("RGB", size, (244, 244, 246))
    body = Image.new("L", (size[0] * 4, size[1] * 4), 0)
    ImageDraw.Draw(body).rounded_rectangle((0, 0, size[0] * 4 - 1, size[1] * 4 - 1), radius=120, fill=255)
    body = body.resize(size, Image.LANCZOS)
    out.paste(Image.new("RGB", size, (206, 207, 211)), (0, 0), body)
    d = ImageDraw.Draw(out)
    d.rectangle((m - 6, m - 6, m + w + 5, m + h + 5), fill=(196, 196, 198))
    out.paste(screen, (m, m))
    return out


def mono_framed(ppm: Path, scale: int = 2) -> Image.Image:
    """The frame as it looks on the Waveshare 1.54": black on the panel's grey
    paper, every pixel doubled so its pixels can be judged."""
    screen = Image.open(ppm).convert("L").point(lambda v: 228 if v > 127 else 38)
    w, h = screen.size
    screen = screen.resize((w * scale, h * scale), Image.NEAREST).convert("RGB")
    m = 18
    out = Image.new("RGB", (screen.width + 2 * m, screen.height + 2 * m), (40, 42, 48))
    out.paste(screen, (m, m))
    return out


def render_epaper154() -> None:
    """Both builds' screens, each to its own folder, and a contact sheet."""
    for name in ("epaper154", "epaper154_touch"):
        out = ROOT / "preview" / "out" / name
        out.mkdir(parents=True, exist_ok=True)
        for old in list(out.glob("*.ppm")) + list(out.glob("*.png")):
            old.unlink()
        sample_paper_picture(out / "big.jpg")
        Image.open(out / "big.jpg").resize((200, 150), Image.LANCZOS).save(out / "sample.jpg", "JPEG", quality=88)
        (out / "big.jpg").unlink()
        program = ROOT / ".pio" / "build" / ("preview_" + name) / "program"
        subprocess.run([str(program), str(out)], check=True, stdout=subprocess.DEVNULL)
        frames = []
        for ppm in sorted(out.glob("*.ppm")):
            img = mono_framed(ppm)
            img.save(ppm.with_suffix(".png"))
            ppm.unlink()
            frames.append((ppm.stem, img))
        sheet(out, frames, cols=6, cell=(frames[0][1].width, frames[0][1].height))


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "epaper154":
        render_epaper154()
        return
    mode = args[0] if args and args[0] in ("paper", "paper_b") else ""
    paper = bool(mode)
    if paper:
        args = args[1:]
    default = ROOT / "preview" / "out" / (mode if paper else "")
    out = Path(args[0] if args else default).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.ppm"):
        old.unlink()
    (sample_paper_picture if paper else sample_picture)(out / "sample.jpg")
    if not paper:
        sample_background(out / "background.jpg")
    program = ROOT / ".pio" / "build" / ("preview_" + mode if paper else "preview") / "program"
    subprocess.run([str(program), str(out), str(ROOT)], check=True, stdout=subprocess.DEVNULL)
    frames = []
    for ppm in sorted(out.glob("*.ppm")):
        img = paper_framed(ppm) if paper else framed(ppm)
        img.save(ppm.with_suffix(".png"))
        ppm.unlink()
        frames.append((ppm.stem, img))
    if paper:
        cell = (max(f.width for _, f in frames), max(f.height for _, f in frames))  # landscape too
        sheet(out, frames, cols=5, cell=cell)
    else:
        sheet(out, frames, cols=5, cell=(D + 2 * BEZEL, D + 2 * BEZEL))


def sheet(out: Path, frames: list, cols: int, cell: tuple[int, int]) -> None:
    """All frames on one contact sheet, each named under it."""
    cw, ch = cell
    pad, label = 24, 34
    rows = math.ceil(len(frames) / cols)
    board = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + label + pad) + pad), (244, 244, 246))
    d = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (name, img) in enumerate(frames):
        x = pad + (i % cols) * (cw + pad)
        y = pad + (i // cols) * (ch + label + pad)
        board.paste(img, (x + (cw - img.width) // 2, y + (ch - img.height) // 2))
        d.text((x + cw // 2, y + ch + 6), name.split("_", 1)[1].replace("_", " "), fill=(60, 62, 70),
               font=font, anchor="ma")
    board.save(out / "sheet.png")
    print(f"{len(frames)} screens → {out}/sheet.png")


if __name__ == "__main__":
    main()
