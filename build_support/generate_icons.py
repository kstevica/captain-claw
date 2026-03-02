#!/usr/bin/env python3
"""Generate Captain Claw app icons for all platforms.

Converts a source image into all icon formats needed by Electron desktop
packaging:

  desktop/icons/icon.png      – 512×512 PNG  (electron-builder / Linux)
  desktop/icons/icon.ico      – multi-size Windows ICO
  desktop/icons/icon.icns     – macOS ICNS   (requires macOS + iconutil)
  desktop/icons/icon_1024.png – 1024×1024 high-res source

Usage:
  python build_support/generate_icons.py [path/to/source.png]

If no source path is given the script looks for
  build_support/source_icon.png

Requires: Pillow  (pip install Pillow)
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ICONS_DIR = SCRIPT_DIR.parent / "desktop" / "icons"
DEFAULT_SOURCE = SCRIPT_DIR / "source_icon.png"


def load_source(path: Path) -> Image.Image:
    """Load and prepare the source image as a square RGBA 1024×1024."""
    img = Image.open(path).convert("RGBA")

    # If not square, centre-crop to the shorter edge
    w, h = img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

    # Resize to 1024×1024 if needed
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), Image.LANCZOS)

    return img


def create_ico(source: Image.Image, path: Path) -> None:
    """Create a Windows .ico file with multiple sizes."""
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []
    for s in sizes:
        img = source.copy().resize((s, s), Image.LANCZOS)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        images.append(img)
    # Pillow ICO: save the largest as base, append the rest
    images[-1].save(str(path), format="ICO", append_images=images[:-1])


def create_icns(source: Image.Image, path: Path) -> None:
    """Create a macOS .icns file using iconutil (macOS only)."""
    if sys.platform != "darwin":
        print("  Note: .icns requires macOS iconutil – skipping.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "icon.iconset"
        iconset.mkdir()

        icon_sizes = [16, 32, 64, 128, 256, 512]
        for s in icon_sizes:
            source.resize((s, s), Image.LANCZOS).save(
                str(iconset / f"icon_{s}x{s}.png")
            )
            if s * 2 <= 1024:
                source.resize((s * 2, s * 2), Image.LANCZOS).save(
                    str(iconset / f"icon_{s}x{s}@2x.png")
                )

        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  iconutil failed: {result.stderr}")
        else:
            print(f"  Created {path}")


def main() -> None:
    # Determine source image path
    if len(sys.argv) > 1:
        src_path = Path(sys.argv[1]).expanduser().resolve()
    elif DEFAULT_SOURCE.exists():
        src_path = DEFAULT_SOURCE
    else:
        print(
            f"Error: No source image.\n"
            f"  Usage: {sys.argv[0]} <source.png>\n"
            f"  Or place source image at: {DEFAULT_SOURCE}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not src_path.exists():
        print(f"Error: Source image not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading source: {src_path}")
    source = load_source(src_path)

    # Also copy source into build_support/ for future runs
    if src_path != DEFAULT_SOURCE:
        source.save(str(DEFAULT_SOURCE))
        print(f"  Saved copy to {DEFAULT_SOURCE}")

    # 1024×1024
    source.save(str(ICONS_DIR / "icon_1024.png"))
    print(f"  Created {ICONS_DIR / 'icon_1024.png'}")

    # 512×512 PNG (Linux / electron-builder)
    png_path = ICONS_DIR / "icon.png"
    source.resize((512, 512), Image.LANCZOS).save(str(png_path))
    print(f"  Created {png_path}")

    # Windows ICO
    ico_path = ICONS_DIR / "icon.ico"
    create_ico(source, ico_path)
    print(f"  Created {ico_path}")

    # macOS ICNS
    icns_path = ICONS_DIR / "icon.icns"
    create_icns(source, icns_path)

    print("Done!")


if __name__ == "__main__":
    main()
