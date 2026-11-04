# Captain Claw v0.8.7 Release Notes

**Release title:** Shakedown
**Release date:** 2026-11-04

A patch on the day 0.8.6 came out. A fresh install's agents work again, the
command-line binaries are built and attached to the release for the first time
since 0.8.0, and the desktop app installs exactly what was released.

**If you installed Captain for desktop 0.8.6 and an agent answers "No module named
'mcp'", this is the fix:** Check for Updates… installs it.

3 pull requests, the same day as 0.8.6.

## Fixed

- **Agents on a fresh install.** LiteLLM 1.108, its newest version, loads the
  `mcp` package on every call that offers tools, which is every agent turn, but
  lists `mcp` only as an optional extra. A fresh install (the desktop app, or a
  new `pip install`) got LiteLLM 1.108 without it, and every turn failed with
  "No module named 'mcp'". Captain now depends on `litellm[mcp]`. A deck that
  hasn't reinstalled its packages kept its older LiteLLM and never saw this.
- **Command-line binaries, built again.** The PyInstaller `captain-claw` stopped
  on its first line ("No module named 'typer'"), so every build since 0.8.0
  failed its check and no release carried binaries. It now starts the way the
  `captain-claw` command pip installs does, and macOS, Windows, and Linux x64 and
  arm64 all pass. On a tag they're attached to the release you publish; the
  workflow no longer creates, renames or drafts a release, which it would have
  done to a published one.
- **The desktop app installs exactly the release.** Its wheel is now built from
  a clean export of the tagged commit. Built in a working checkout, it could
  carry files a later commit had deleted (the first 0.8.6 build had six old
  Flight Deck files in it, and was caught before release).

## Upgrading

1. **The desktop app**: **Check for Updates…** (it also offers 0.8.7 within a
   minute of starting). It installs beside 0.8.6, test-starts it and switches
   on a restart.
2. **Each Flight Deck**: pull and `pip install .` (this adds `mcp`), or use
   **Console → Updates**; then restart Flight Deck and every agent.
3. **From PyPI**: `pip install captain-claw` still gets 0.8.4, which has the same
   LiteLLM problem. Install from the repository, or add `mcp` (`pip install
   "mcp>=2.2,<3"`), until PyPI has a newer version.

Nothing else changed: no new settings, firmware stays **0.20.4** / PaperColor B
**1.0.31**, and BotPort keeps protocol 2. Everything new in 0.8.6 is in its
[release notes](https://github.com/kstevica/captain-claw/blob/main/release-notes/RELEASE_NOTES_0.8.6.md).
