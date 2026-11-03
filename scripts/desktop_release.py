#!/usr/bin/env python3
"""Make what the desktop app installs and updates from: a wheel, a lock, a manifest.

The desktop app ships without Python (see desktop/runtime.js). On first launch, and
for every update after, it reads ``desktop-manifest.json`` (by default the one on the
latest GitHub release) and installs from it:

- **uv**, pinned by version and checksum for each platform;
- **Python** (``python``), installed by that uv;
- **the dependencies**: ``desktop-lock-<version>.txt``, every package pinned with its
  hashes, for every platform (``uv pip compile --universal``), so each install gets
  exactly the set that was tested;
- **Captain Claw itself**: the wheel, which carries the backend and the Flight Deck
  frontend (``flight_deck/static`` is package data).

The manifest pins the lock and the wheel by sha256, so everything an install runs is
checked against the one file read over HTTPS. An update whose lock didn't change
downloads only the wheel; uv's cache supplies the rest.

Usage:
    python scripts/desktop_release.py                       # build the wheel, write dist/desktop-release/<version>/
    python scripts/desktop_release.py --wheel dist/captain_claw-0.8.6-py3-none-any.whl
    python scripts/desktop_release.py --base-url http://127.0.0.1:8765/   # a local test release

Then upload the three files to the release (the script prints the command). The
manifest has to be on the release that GitHub calls latest, or the app won't see it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The uv every install uses; 0.13.1 was released 2026-10-13. Bump it here, and the next
# release's manifest moves every app to it.
UV_VERSION = "0.13.1"
UV_ASSETS = {
    "darwin-arm64": "uv-aarch64-apple-darwin.tar.gz",
    "darwin-x64": "uv-x86_64-apple-darwin.tar.gz",
    "win32-x64": "uv-x86_64-pc-windows-msvc.zip",
    "win32-arm64": "uv-aarch64-pc-windows-msvc.zip",
    "linux-x64": "uv-x86_64-unknown-linux-gnu.tar.gz",
    "linux-arm64": "uv-aarch64-unknown-linux-gnu.tar.gz",
}
PYTHON = "3.11"
# What the app's shell must understand to install this release (desktop/runtime.js
# SHELL_PROTOCOL). Raise both together when the manifest changes in a way an older
# app can't follow; that app then asks its user to download the new one.
MIN_SHELL = 1
REPO = "kstevica/captain-claw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str, cwd: Path = ROOT, stdin: str | None = None) -> str:
    print("  $", " ".join(args))
    done = subprocess.run(args, cwd=cwd, input=stdin, capture_output=True, text=True)  # noqa: S603
    if done.returncode:
        sys.exit(f"Failed ({done.returncode}):\n{done.stderr or done.stdout}")
    return done.stdout


def find_uv(given: str | None) -> str:
    uv = given or shutil.which("uv")
    if not uv:
        sys.exit("uv isn't on PATH. Install it (https://docs.astral.sh/uv/) or pass --uv.")
    return uv


def build_wheel(uv: str, out: Path) -> Path:
    print("Building the wheel (with the Flight Deck bundle as committed)...")
    run(uv, "build", "--wheel", "--out-dir", str(out))
    wheels = sorted(out.glob("captain_claw-*.whl"))
    if len(wheels) != 1:
        sys.exit(f"Expected one wheel in {out}, found {[w.name for w in wheels]}")
    return wheels[0]


def wheel_version(wheel: Path) -> str:
    match = re.match(r"captain_claw-([^-]+)-", wheel.name)
    if not match:
        sys.exit(f"{wheel.name} isn't a captain_claw wheel")
    return match.group(1)


def compile_lock(uv: str, wheel: Path, python: str, dest: Path) -> None:
    """Every dependency of *wheel*, pinned with hashes, for every platform; the wheel
    itself is left out (the app installs it on its own, checked by its own hash)."""
    print("Locking the dependencies for every platform...")
    text = run(
        uv, "pip", "compile", "-", "--universal", "--generate-hashes", "--python-version", python,
        "--no-emit-package", "captain-claw", "--no-header", "--quiet", "--no-config",
        stdin=f"captain-claw @ {wheel.resolve().as_uri()}\n",
    )
    if re.search(r"^captain-claw\b", text, re.M):
        sys.exit("The lock still names captain-claw itself")
    dest.write_text(
        f"# Captain Claw {wheel_version(wheel)}: the desktop app's dependencies (scripts/desktop_release.py)\n" + text
    )


def uv_platforms(version: str) -> dict[str, dict[str, str]]:
    print(f"Reading uv {version}'s checksums...")
    base = f"https://github.com/astral-sh/uv/releases/download/{version}/"
    out: dict[str, dict[str, str]] = {}
    for platform, asset in UV_ASSETS.items():
        with urllib.request.urlopen(base + asset + ".sha256", timeout=60) as resp:  # noqa: S310
            digest = resp.read().decode().split()[0].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            sys.exit(f"Odd checksum for {asset}: {digest!r}")
        out[platform] = {"url": base + asset, "sha256": digest, "archive": "zip" if asset.endswith(".zip") else "tar.gz"}
    return out


def summary(version: str, notes_file: Path | None) -> str:
    """The first paragraph of the release notes, for the app's update dialog."""
    path = notes_file or ROOT / "release-notes" / f"RELEASE_NOTES_{version}.md"
    if not path.is_file():
        return ""
    for block in re.split(r"\n\s*\n", path.read_text()):
        text = block.strip()
        # Skip headings, pictures, tables and the "**Release title:**" lines.
        if text and not text.startswith(("#", "!", "<", "|", "---")) and not re.match(r"\*\*Release \w+:", text):
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
            text = re.sub(r"[*_`]", "", " ".join(text.split()))
            return text[:600]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, help="Use this wheel instead of building one")
    parser.add_argument("--base-url", help="Where the files will be downloaded from (default: the GitHub release v<version>)")
    parser.add_argument("--out", type=Path, help="Output folder (default: dist/desktop-release/<version>)")
    parser.add_argument("--notes-file", type=Path, help="Release notes to take the summary from")
    parser.add_argument("--uv", help="The uv to build and lock with (default: the one on PATH)")
    parser.add_argument("--uv-version", default=UV_VERSION, help=f"The uv the app installs (default {UV_VERSION})")
    parser.add_argument("--python", default=PYTHON, help=f"The Python the app installs (default {PYTHON})")
    args = parser.parse_args()

    uv = find_uv(args.uv)
    with tempfile.TemporaryDirectory() as tmp:
        wheel = args.wheel.resolve() if args.wheel else build_wheel(uv, Path(tmp))
        if not wheel.is_file():
            sys.exit(f"No wheel at {wheel}")
        version = wheel_version(wheel)
        if not args.wheel:
            declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
            if declared != version:
                sys.exit(f"pyproject says {declared}, the wheel {version}")
        out = (args.out or ROOT / "dist" / "desktop-release" / version).resolve()
        out.mkdir(parents=True, exist_ok=True)
        shipped_wheel = out / wheel.name
        if shipped_wheel != wheel:
            shutil.copy2(wheel, shipped_wheel)

    lock = out / f"desktop-lock-{version}.txt"
    compile_lock(uv, shipped_wheel, args.python, lock)

    base = args.base_url or f"https://github.com/{REPO}/releases/download/v{version}/"
    if not base.endswith("/"):
        base += "/"
    manifest = {
        "schema": 1,
        "version": version,
        "released": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_shell": MIN_SHELL,
        "python": args.python,
        "summary": summary(version, args.notes_file),
        "notes_url": f"https://github.com/{REPO}/releases/tag/v{version}",
        "wheel": {"url": base + shipped_wheel.name, "sha256": sha256(shipped_wheel), "size": shipped_wheel.stat().st_size},
        "lock": {"url": base + lock.name, "sha256": sha256(lock), "size": lock.stat().st_size},
        "uv": {"version": args.uv_version, "platforms": uv_platforms(args.uv_version)},
    }
    (out / "desktop-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    packages = sum(1 for line in lock.read_text().splitlines() if re.match(r"^[A-Za-z0-9]", line))
    print(f"\nCaptain Claw {version} for the desktop app, in {out}:")
    for name in (shipped_wheel.name, lock.name, "desktop-manifest.json"):
        print(f"  {name}  ({(out / name).stat().st_size / 1e6:.1f} MB)")
    print(f"  {packages} dependencies locked; uv {args.uv_version}; Python {args.python}")
    if not args.base_url:
        print("\nUpload them to the release (it must be the one GitHub calls latest):")
        print(f"  gh release upload v{version} '{shipped_wheel}' '{lock}' '{out / 'desktop-manifest.json'}' --clobber")


if __name__ == "__main__":
    main()
