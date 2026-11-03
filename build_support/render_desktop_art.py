#!/usr/bin/env python3
"""The desktop app's own artwork: the product tour on its first-start page and the
background of its macOS install window.

- ``desktop/assets/tour/*.webp``: four dark product shots from the website
  (``~/Dev/captain-claw.com/website/public``), 1600 px wide, shown while the first
  start installs Captain (``desktop/setup.html``).
- ``desktop/icons/dmg-background.png`` and ``@2x``: the window the DMG opens in.
  electron-builder joins the two into one Retina TIFF; Finder draws the icons and
  their names on top, at the places ``package.json`` ``build.dmg.contents`` gives.

Run it with the repo's venv (Playwright renders the background; Pillow resizes):

    .venv/bin/python build_support/render_desktop_art.py [--website DIR]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
WEBSITE = Path.home() / "Dev" / "captain-claw.com" / "website" / "public"

# The shots and the order setup.html shows them in. Fictional data (Velora, Maya Horvat).
TOUR = {
    "chat": "screenshots/kiosk/kiosk-chat-skill-cards-dark.webp",
    "library": "screenshots/kiosk/kiosk-file-library-dark.webp",
    "taskboard": "media/0.8.4/taskboard.webp",
    "forge": "media/0.8.5/forge-ready.webp",
}

# The install window's content, in points. Keep in step with build.dmg in
# desktop/package.json: the app at (180, 196), Applications at (480, 196), icons
# 112 pt, and a window 28 pt taller than this for Finder's title bar.
WIDTH, HEIGHT = 660, 400

DMG_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
@font-face { font-family: Sora; src: url("FONT") format("woff2"); font-weight: 600 800; }
html, body { margin: 0; width: WIDTHpx; height: HEIGHTpx; overflow: hidden; }
body {
  background:
    radial-gradient(60% 70% at 50% 0%, rgba(99,102,241,.30), transparent 70%),
    radial-gradient(45% 55% at 85% 100%, rgba(34,211,238,.14), transparent 70%),
    radial-gradient(40% 50% at 10% 95%, rgba(216,180,254,.10), transparent 70%),
    linear-gradient(180deg, #0d0d26 0%, #07071a 100%);
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
  color: #e2e8f0; position: relative;
}
.grid { position: absolute; inset: 0;
  background-image: linear-gradient(rgba(148,163,184,.06) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148,163,184,.06) 1px, transparent 1px);
  background-size: 28px 28px;
  -webkit-mask-image: radial-gradient(70% 70% at 50% 45%, #000 30%, transparent 85%); }
.word { position: absolute; top: 34px; width: 100%; text-align: center; }
.word span { display: inline-block; font: 800 34px/1.1 Sora, sans-serif; letter-spacing: -0.03em;
  background: linear-gradient(110deg, #67e8f9 0%, #818cf8 52%, #d8b4fe 100%);
  -webkit-background-clip: text; color: transparent; }
.tag { position: absolute; top: 78px; width: 100%; text-align: center; font-size: 13px; color: #94a3b8; letter-spacing: .01em; }
/* Soft wells the icons sit in. */
.well { position: absolute; top: 106px; width: 180px; height: 180px; margin-left: -90px; border-radius: 50%;
  background: radial-gradient(closest-side, rgba(129,140,248,.26), rgba(129,140,248,.06) 62%, transparent); }
/* Finder writes each icon's name under it, black in Light Mode and white in Dark
   Mode: a mid-tone chip keeps both readable. */
.chip { position: absolute; top: 262px; height: 22px; margin-left: -62px; width: 124px; border-radius: 11px;
  background: rgba(124,135,196,.78); box-shadow: 0 0 0 1px rgba(255,255,255,.08) inset; }
.arrow { position: absolute; left: 262px; top: 180px; width: 136px; height: 32px; }
/* High enough that Finder's path and status bars, when someone has them on, don't cover it. */
.foot { position: absolute; top: 306px; width: 100%; text-align: center; font-size: 12px; color: #7c89a3; }
</style></head><body>
<div class="grid"></div>
<div class="word"><span>Captain</span></div>
<div class="tag">Your team of agents, on your own computer</div>
<div class="well" style="left:180px"></div><div class="well" style="left:480px"></div>
<div class="chip" style="left:180px"></div><div class="chip" style="left:480px"></div>
<svg class="arrow" viewBox="0 0 136 32" fill="none">
  <defs><linearGradient id="a" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="136" y2="0"><stop offset="0" stop-color="#67e8f9" stop-opacity=".25"/>
  <stop offset=".6" stop-color="#818cf8"/><stop offset="1" stop-color="#d8b4fe"/></linearGradient></defs>
  <path d="M4 16 H120" stroke="url(#a)" stroke-width="3" stroke-linecap="round" stroke-dasharray="2 9"/>
  <path d="M112 6 L126 16 L112 26" stroke="#d8b4fe" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<div class="foot">Drag Captain into Applications</div>
</body></html>"""


def copy_tour(website: Path) -> None:
    from PIL import Image

    out = DESKTOP / "assets" / "tour"
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in TOUR.items():
        src = website / rel
        if not src.is_file():
            raise SystemExit(f"Missing {src}: pass --website to the site's public/ folder")
        img = Image.open(src).convert("RGB")
        if img.width > 1600:
            img = img.resize((1600, round(img.height * 1600 / img.width)), Image.LANCZOS)
        dest = out / f"{name}.webp"
        img.save(dest, "WEBP", quality=82, method=6)
        print(f"  {dest.relative_to(ROOT)}  {img.width}x{img.height}  {dest.stat().st_size // 1024} KB")


async def render_dmg() -> None:
    from playwright.async_api import async_playwright

    font = (DESKTOP / "assets" / "fonts" / "sora-latin.woff2").resolve().as_uri()
    html = DMG_HTML.replace("FONT", font).replace("WIDTH", str(WIDTH)).replace("HEIGHT", str(HEIGHT))
    page_file = DESKTOP / "icons" / ".dmg-background.html"
    page_file.write_text(html)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            for scale, suffix in ((1, ""), (2, "@2x")):
                page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=scale)
                await page.goto(page_file.resolve().as_uri())
                await page.evaluate("document.fonts.ready")
                dest = DESKTOP / "icons" / f"dmg-background{suffix}.png"
                await page.screenshot(path=str(dest))
                print(f"  {dest.relative_to(ROOT)}  {WIDTH * scale}x{HEIGHT * scale}")
                await page.close()
            await browser.close()
    finally:
        page_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--website", type=Path, default=WEBSITE, help="The website's public/ folder")
    args = parser.parse_args()
    print("Tour shots:")
    copy_tour(args.website.expanduser())
    print("Install window:")
    asyncio.run(render_dmg())


if __name__ == "__main__":
    main()
