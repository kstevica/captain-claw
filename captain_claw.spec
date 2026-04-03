# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Captain Claw.

Builds three executables:
  - captain-claw       (CLI + Web UI, default entry point)
  - captain-claw-web   (Web UI only)
  - captain-claw-orchestrate (headless orchestrator)

Usage:
  pyinstaller captain_claw.spec

The spec produces a single dist/ folder with all three binaries
sharing one set of bundled dependencies.
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")
PKG_DIR = os.path.join(PROJECT_ROOT, "captain_claw")
STATIC_DIR = os.path.join(PKG_DIR, "web", "static")
FD_STATIC_DIR = os.path.join(PKG_DIR, "flight_deck", "static")
INSTRUCTIONS_DIR = os.path.join(PKG_DIR, "instructions")

# ── Data files to bundle ────────────────────────────────────────
# (source, dest_in_bundle)
datas = [
    (STATIC_DIR, os.path.join("captain_claw", "web", "static")),
    (FD_STATIC_DIR, os.path.join("captain_claw", "flight_deck", "static")),
    (INSTRUCTIONS_DIR, os.path.join("captain_claw", "instructions")),
] + collect_data_files("litellm") + collect_data_files("playwright")

# NOTE: Playwright Chromium browser binaries are NOT bundled here.
# PyInstaller fails to codesign the Chromium .app bundle on macOS.
# Instead, build.sh copies pw-browsers/ into dist/ after PyInstaller.

# ── Hidden imports ──────────────────────────────────────────────
# These are all the lazy / dynamic imports that PyInstaller's
# static analysis cannot detect.
hidden_imports = [
    # ── tiktoken (namespace-package plugin discovery) ──
    "tiktoken_ext",
    "tiktoken_ext.openai_public",

    # ── litellm lazy imports (provider backends) ──
    "litellm.llms",
    "litellm.llms.openai",
    "litellm.llms.anthropic",
    "litellm.llms.ollama",
    "litellm.llms.gemini",
    "litellm.cost_calculator",
    "litellm.litellm_core_utils",
    "litellm.litellm_core_utils.streaming_handler",
    "litellm.litellm_core_utils.llm_request_utils",
    "litellm.main",
    "litellm.router",
    "litellm.utils",
    "litellm.types",
    "litellm.exceptions",

    # ── aiohttp & its C-extension sub-deps ──
    "aiohttp",
    "aiohttp.web",
    "multidict",
    "multidict._multidict",
    "yarl",
    "yarl._quoting",
    "frozenlist",
    "frozenlist._frozenlist",
    "aiosignal",

    # ── pydantic (compiled core) ──
    "pydantic",
    "pydantic_core",
    "pydantic_settings",

    # ── captain_claw lazy imports (web_server.py delegation) ──
    "captain_claw.web",
    "captain_claw.web.ws_handler",
    "captain_claw.web.rest_instructions",
    "captain_claw.web.rest_config",
    "captain_claw.web.rest_sessions",
    "captain_claw.web.rest_settings",
    "captain_claw.web.rest_orchestrator",
    "captain_claw.web.rest_cron",
    "captain_claw.web.rest_entities",
    "captain_claw.web.rest_workflows",
    "captain_claw.web.rest_reflections",
    "captain_claw.web.rest_loops",
    "captain_claw.web.openai_proxy",
    "captain_claw.web.google_oauth",
    "captain_claw.web.static_pages",
    "captain_claw.web.telegram",
    "captain_claw.web.chat_handler",
    "captain_claw.web.slash_commands",

    # ── captain_claw lazy imports (main.py / agent) ──
    "captain_claw.web_server",
    "captain_claw.orchestrator_cli",
    "captain_claw.agent_tool_loop_mixin",
    "captain_claw.runtime_context",
    "captain_claw.cron_dispatch",
    "captain_claw.local_command_dispatch",
    "captain_claw.platform_lifecycle",
    "captain_claw.prompt_execution",
    "captain_claw.onboarding",
    "captain_claw.session_export",
    "captain_claw.reflections",
    "captain_claw.google_oauth",
    "captain_claw.google_oauth_manager",

    # ── stdlib extras sometimes missed ──
    "asyncio",
    "sqlite3",
    "email.mime.text",
    "email.mime.multipart",

    # ── playwright (browser tool) ──
    "playwright",
    "playwright.async_api",
    "playwright._impl",
    "playwright._impl._driver",

    # ── flight_deck (FastAPI server + sub-modules) ──
    "captain_claw.flight_deck",
    "captain_claw.flight_deck.server",
    "captain_claw.flight_deck.auth",
    "captain_claw.flight_deck.auth_routes",
    "captain_claw.flight_deck.db",
    "captain_claw.flight_deck.rate_limiter",
    "captain_claw.flight_deck.settings_routes",
    "captain_claw.flight_deck.chat_routes",
    "captain_claw.flight_deck.admin_routes",
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.staticfiles",
    "fastapi.responses",
    "uvicorn",
    "uvicorn.main",
    "uvicorn.config",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.logging",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "starlette.staticfiles",
    "starlette.responses",
    "starlette.websockets",
    "docker",
    "docker.models",
    "docker.models.containers",

    # ── other deps ──
    "httpx",
    "httpx._transports",
    "httpx._transports.default",
    "beautifulsoup4",
    "bs4",
    "pypdf",
    "aiofiles",
    "structlog",
    "yaml",
    "rich",
    "blessed",
    "dotenv",
    "tenacity",
    "pocket_tts",
    "glob2",
    "aiosqlite",
]

# ── Excludes (reduce binary size) ───────────────────────────────
excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "IPython",
    "jupyter",
    "notebook",
    "test",
    "tests",
    "pytest",
    "mypy",
    "ruff",
]

# ── Common Analysis kwargs ──────────────────────────────────────
common_kwargs = dict(
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(PROJECT_ROOT, "build_support", "runtime_hook.py")],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)


# ── Analysis for each entry point ───────────────────────────────
a_main = Analysis(
    [os.path.join(PKG_DIR, "main.py")],
    **common_kwargs,
)

a_web = Analysis(
    [os.path.join(PKG_DIR, "web_server.py")],
    **common_kwargs,
)

a_orchestrate = Analysis(
    [os.path.join(PKG_DIR, "orchestrator_cli.py")],
    **common_kwargs,
)

a_fd = Analysis(
    [os.path.join(PKG_DIR, "flight_deck", "server.py")],
    **common_kwargs,
)

# ── Merge to share common modules ──────────────────────────────
MERGE(
    (a_main, "captain-claw", "captain-claw"),
    (a_web, "captain-claw-web", "captain-claw-web"),
    (a_orchestrate, "captain-claw-orchestrate", "captain-claw-orchestrate"),
    (a_fd, "captain-claw-fd", "captain-claw-fd"),
)

# ── PYZ (bytecode archive) ─────────────────────────────────────
pyz_main = PYZ(a_main.pure, cipher=block_cipher)
pyz_web = PYZ(a_web.pure, cipher=block_cipher)
pyz_orchestrate = PYZ(a_orchestrate.pure, cipher=block_cipher)
pyz_fd = PYZ(a_fd.pure, cipher=block_cipher)

# ── EXE definitions ────────────────────────────────────────────
exe_main = EXE(
    pyz_main,
    a_main.scripts,
    [],
    exclude_binaries=True,
    name="captain-claw",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

exe_web = EXE(
    pyz_web,
    a_web.scripts,
    [],
    exclude_binaries=True,
    name="captain-claw-web",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

exe_orchestrate = EXE(
    pyz_orchestrate,
    a_orchestrate.scripts,
    [],
    exclude_binaries=True,
    name="captain-claw-orchestrate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

exe_fd = EXE(
    pyz_fd,
    a_fd.scripts,
    [],
    exclude_binaries=True,
    name="captain-claw-fd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

# ── COLLECT into one dist folder ────────────────────────────────
coll = COLLECT(
    exe_main, a_main.binaries, a_main.datas,
    exe_web, a_web.binaries, a_web.datas,
    exe_orchestrate, a_orchestrate.binaries, a_orchestrate.datas,
    exe_fd, a_fd.binaries, a_fd.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="captain-claw",
)
