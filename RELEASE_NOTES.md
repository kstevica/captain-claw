# Captain Claw v0.2.6.3 Release Notes

**Release date:** 2026-03-01

## Highlights

Packaging and installation overhaul — lighter default install, pre-configured multi-model onboarding, resilient port binding, and full pip compatibility.

## New Features

### Pre-Configured Multi-Model Setup
- Onboarding now seeds **12 default allowed models** across OpenAI, Anthropic, and Gemini — including image generation, OCR, and vision models
- Fresh installs get a ready-to-use multi-model configuration out of the box
- Both TUI and web onboarding wizards inform users about the pre-configured models
- Users can still add custom models during onboarding or later in Settings

### Port Auto-Retry
- Web server now automatically tries the next port (up to 10 attempts) if the configured port is already in use
- New `--port` CLI argument to override the web server port at launch

### Lighter Default Install
- `pocket-tts` (and its heavy PyTorch/NumPy/SciPy dependencies) moved to an optional extra
- Default `pip install captain-claw` is now significantly lighter (~2 GB+ smaller)
- Install TTS support explicitly with `pip install captain-claw[tts]`

### Example Configuration
- New `config.yaml.example` with full annotated reference of all configuration options
- Enriched master `config.yaml` with additional settings documentation

### Usage Guide
- New `USAGE.md` with detailed usage instructions and examples

## Fixes

### pip Install Compatibility
- **Instruction templates now load correctly** when installed via pip — previously the `instructions/` folder lived at the project root and was not included in the package distribution
- Moved `instructions/` into the `captain_claw/` package and updated path resolution, package-data, PyInstaller spec, and CI verification
- All 6 missing tools (`todo`, `contacts`, `scripts`, `apis`, `typesense`, `datastore`) are now included in the default enabled tools list for fresh installations

### Onboarding
- Updated Gemini default model to `gemini-3-flash-preview`
- Summary step now shows the count of pre-configured models

## Internal

- Updated `pyproject.toml` package-data to include `instructions/**/*`
- Updated PyInstaller spec (`captain_claw.spec`) for new instructions path
- Updated GitHub Actions build workflow verification path
- Updated `README.md` with optional extras install instructions and corrected architecture paths
