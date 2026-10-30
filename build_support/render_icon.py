#!/usr/bin/env python3
"""Render the Captain icon from build_support/icon.svg and write every icon file.

The icon is the Bot character in a captain's cap (build_support/brand/bot), on
the navy slab. icon.svg is rasterised in Playwright's headless Chromium at 2048 px
and downsampled to 1024, in two framings:

  fit    the rounded slab fills the frame, corners transparent
  bleed  the slab fills the whole square, no transparency (PWA "maskable", Apple touch)

Files written (relative to the repo):

  build_support/source_icon.png                     fit 1024, the master generate_icons.py reads
  desktop/icons/*                                   via generate_icons.py (png, ico, icns)
  captain_claw/flight_deck/static/icon-96.png        fit
  captain_claw/flight_deck/static/icon-192.png       bleed
  captain_claw/flight_deck/static/icon-512.png       bleed
  captain_claw/flight_deck/static/apple-touch-icon.png  bleed 180
  captain_claw/flight_deck/static/favicon.svg        the vector itself (fit)
  flight-deck/public/favicon.svg                     same file, for the Vite build
  captain_claw/web/static/{ccicon.png,favicon.svg,logo.svg}
  docs/ccicon.png

Usage:
  python build_support/render_icon.py [--website DIR] [--no-desktop]

  --website DIR   also write ccicon.png, favicon.svg and favicon.ico into that folder
                  (the Astro site's public/)
  --no-desktop    skip generate_icons.py

Requires: Pillow and playwright (+ `playwright install chromium`).
"""

import argparse
import io
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
SOURCE = SCRIPT_DIR / "icon.svg"
RENDER_PX = 2048
OUT_PX = 1024


def variant_svg(framing: str) -> str:
    """icon.svg as text, with the slab squared off for the bleed framing."""
    svg = SOURCE.read_text()
    if framing == "bleed":
        svg = re.sub(r'(<rect id="slab[^"]*"[^>]*?)rx="\d+"', r'\1rx="0"', svg)
    return svg


def render() -> dict[str, Image.Image]:
    """Render the fit and bleed variants; returns {name: 1024px RGBA image}."""
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name in ("fit", "bleed"):
            page = browser.new_page(viewport={"width": RENDER_PX, "height": RENDER_PX}, device_scale_factor=1)
            svg = re.sub(r'width="1024" height="1024"', f'width="{RENDER_PX}" height="{RENDER_PX}"', variant_svg(name), count=1)
            page.set_content(f'<body style="margin:0;background:transparent">{svg}</body>')
            png = page.locator("svg").screenshot(omit_background=True)
            page.close()
            out[name] = Image.open(io.BytesIO(png)).convert("RGBA").resize((OUT_PX, OUT_PX), Image.LANCZOS)
            print(f"  rendered {name}")
        browser.close()
    return out


def rel(dest: Path) -> str:
    return str(dest.relative_to(REPO)) if dest.is_relative_to(REPO) else str(dest)


def write_png(img: Image.Image, size: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.resize((size, size), Image.LANCZOS).save(dest, optimize=True)
    print(f"  {rel(dest)}  {size}px")


def write_svg(dest: Path) -> None:
    """The vector itself (fit framing), so the favicon stays sharp at any size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(variant_svg("fit"))
    print(f"  {rel(dest)}  svg")


def write_ico(img: Image.Image, dest: Path, sizes=(16, 32, 48)) -> None:
    frames = [img.resize((s, s), Image.LANCZOS) for s in sizes]
    frames[-1].save(dest, format="ICO", append_images=frames[:-1])
    print(f"  {rel(dest)}  ico {sizes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--website", type=Path, help="the website's public/ folder: writes ccicon.png, favicon.svg and favicon.ico there")
    ap.add_argument("--no-desktop", action="store_true", help="skip generate_icons.py")
    args = ap.parse_args()

    print(f"Rendering {rel(SOURCE)}")
    r = render()
    fit, bleed = r["fit"], r["bleed"]

    print("Master")
    fit.save(SCRIPT_DIR / "source_icon.png", optimize=True)
    print(f"  {rel(SCRIPT_DIR / 'source_icon.png')}  1024px")

    print("Flight Deck")
    fd = REPO / "captain_claw" / "flight_deck" / "static"
    write_png(fit, 96, fd / "icon-96.png")
    write_png(bleed, 192, fd / "icon-192.png")
    write_png(bleed, 512, fd / "icon-512.png")
    write_png(bleed, 180, fd / "apple-touch-icon.png")
    write_svg(fd / "favicon.svg")
    shutil.copy(fd / "favicon.svg", REPO / "flight-deck" / "public" / "favicon.svg")
    print("  flight-deck/public/favicon.svg  (copy)")

    print("Web UI and docs")
    web = REPO / "captain_claw" / "web" / "static"
    write_png(fit, 1024, web / "ccicon.png")
    write_svg(web / "favicon.svg")
    write_svg(web / "logo.svg")
    write_png(fit, 1024, REPO / "docs" / "ccicon.png")

    if args.website:
        print(f"Website {args.website}")
        write_png(fit, 1024, args.website / "ccicon.png")
        write_svg(args.website / "favicon.svg")
        write_ico(fit, args.website / "favicon.ico")

    if not args.no_desktop:
        print("Desktop")
        sys.path.insert(0, str(SCRIPT_DIR))
        import generate_icons  # noqa: E402

        sys.argv = [sys.argv[0]]
        generate_icons.main()


if __name__ == "__main__":
    main()
