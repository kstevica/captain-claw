"""PyInstaller runtime hook for Captain Claw.

Runs before any application code.  Patches environment variables so that
the ``Path(__file__).parent.parent / "instructions"`` pattern used in
``instructions.py`` resolves correctly inside a frozen binary.
"""

import os
import sys


def _get_bundle_dir() -> str:
    """Return the directory where bundled data files live.

    In --onedir mode this is the folder containing the executable.
    In --onefile mode this is the temporary _MEIxxxxxx extraction dir
    (pointed to by sys._MEIPASS).
    """
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))


if getattr(sys, "frozen", False):
    bundle_dir = _get_bundle_dir()

    # Point the InstructionLoader at the bundled instructions directory
    # so the ``Path(__file__).parent.parent / "instructions"`` fallback
    # is overridden.
    os.environ.setdefault(
        "CAPTAIN_CLAW_INSTRUCTIONS_DIR",
        os.path.join(bundle_dir, "captain_claw", "instructions"),
    )

    # When running inside an Electron shell the static files are in the
    # same bundle; make sure the web server can find them.
    os.environ.setdefault(
        "CAPTAIN_CLAW_STATIC_DIR",
        os.path.join(bundle_dir, "captain_claw", "web", "static"),
    )

    # Playwright bundled Chromium browser.
    pw_browsers = os.path.join(bundle_dir, "pw-browsers")
    if os.path.isdir(pw_browsers):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = pw_browsers
