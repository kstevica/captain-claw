# OpenCV vision tool (deterministic local CV under the LLM vision layer)

Add a single `vision` tool backed by **OpenCV 5** — a cheap, deterministic, local
computer-vision layer that sits *under* the existing multimodal-LLM vision tools.
It doesn't replace them; it pre-processes and measures for them (cutting LLM spend)
and does pixel-exact things an LLM can't do reliably (diffs, QR/barcode decode,
geometric measurement, template matching).

Motivation: **every visual operation in Captain Claw today routes through a
multimodal LLM.** `image_vision`, `image_ocr`, `video_vision` (one LLM call *per
sampled frame*), `browser_vision`, and `screen_capture` analysis all base64 an image
and pay for a model round-trip. There is no deterministic pixel-level layer at all.
Two payoffs, both landing on things we already track: (1) it cuts LLM spend (feeds
the run-cost accounting) by pre-filtering/deduping frames and gating "watch this
page" loops so a model fires only when pixels actually changed; (2) it unlocks
deterministic operations LLMs are bad at. CPU-only and light — matches the
"light on resources" constraint that prompted this.

Status: **Phases 1–3 shipped & committed on branch `feat/opencv-vision-tool`**,
verified locally against cv2 4.13 (Phase 2 detectors downloaded + run for real;
Phase 3 helpers + wiring unit-tested). Decisions locked 2026-07-14. All phases done.

## Locked decisions (2026-07-14, with the user)

1. **One `vision` verb-tool**, matching the consolidated-verb convention
   (`vfs`, `facts`, `datastore`, `topics`) rather than a tool per operation.
2. **Classical CV + small bundled ONNX detectors only.** OpenCV 5 can run
   LLMs/VLMs inside its DNN module — we deliberately do **not** use that: it's
   CPU-only and would compete with the LiteLLM multi-model routing. We use the
   rewritten DNN engine only for small local detectors (YOLO / face / text region).
3. **Deterministic → zero run-cost.** Ops that touch no model spend nothing and are
   not costed. `detect` (ONNX) runs locally on CPU, also no LLM spend.
4. **Optional dependency, guarded import.** `try: import cv2` with a graceful
   `_HAS_CV2` fallback (mirrors `_HAS_PILLOW` in `image_ocr.py`). Nothing else in the
   system changes when OpenCV is absent; the tool just reports it's unavailable.
5. **Derived images go to the shared VFS**, so agents / Basna / Vatra / Code can pass
   cropped/annotated/cleaned outputs between each other by path.

## What OpenCV 5 brings (and what we take)

- **Classical CV** (the workhorse for us): SSIM/pixel diff, contours, perceptual
  hashing, deskew/threshold/enhance, `QRCodeDetector` + barcode decode, template
  matching, color/blur/quality metrics.
- **Rewritten DNN engine** — ONNX coverage jumped ~23% (4.x) → **80%+**, with a
  CPU-tuned Hardware Acceleration Layer (SSE/AVX/NEON). Small bundled ONNX detectors
  (YOLO, face, EAST/DB text) load and run reliably on CPU through one dependency.
- **Not taken:** VLM/LLM-in-DNN inference (see decision 2).

Packaging is settled: `opencv-python-headless==5.0.0.93` is on PyPI with manylinux
x86_64/aarch64 + macOS wheels, and OpenCV is **already a declared optional dep** in
the `faces` extra (`pyproject.toml`, `opencv-python-headless>=4.9.0`). The headless
wheel is ~40–90 MB, CPU-only, no GUI deps.

## The gap it fills

| Need | Today | With `vision` |
| --- | --- | --- |
| "Describe this image" | `image_vision` (LLM) ✓ | unchanged — LLM still wins |
| "Did the page change between these two shots?" | LLM call every poll | SSIM diff, **$0**, returns change boxes |
| "Sample the *meaningful* frames of a video" | fixed every 6s → up to 20 LLM calls | scene-cut + dedupe → 3–5 LLM calls |
| "Read this QR / barcode" | LLM (unreliable) | `QRCodeDetector`, deterministic |
| "Deskew/clean a scan before OCR" | none | better OCR, sometimes skips the LLM |
| "Where's this button on screen?" | LLM guess | template match → exact coords |
| "Is this image blurry / blank / a dupe?" | none | blur score / pHash, **$0** |

## The tool

`vision(op=..., ...)` — deterministic; writes derivatives to the VFS; returns
structured JSON (boxes, hashes, scores, paths).

- `diff` — two images → SSIM score + bounding boxes of changed regions.
- `keyframes` — video → scene-cut + dedupe frame list (feeds `video_vision`).
- `dedupe` — images → perceptual-hash near-duplicate clusters.
- `prep` — deskew / threshold / crop / enhance → cleaned image in VFS (pre-OCR).
- `qr` — decode QR + barcodes → text/payloads.
- `detect` — bundled ONNX (objects / faces / text regions) → boxes. **(Phase 2)**
- `measure` — dimensions, dominant colors, blur score, brightness → metrics.
- `locate` — template-match a needle image in a haystack → coords + confidence.
- `annotate` — draw boxes/labels on an image → new image in VFS (for reports).

Standard `Tool` shape: `name = "vision"`, JSON-schema `parameters` with an `op`
enum + per-op args, `async def execute(**kwargs) -> ToolResult`, registered in
`tools/__init__.py` (import + `__all__`). Heavy calls (`cv2` is sync) run via
`asyncio.to_thread` to keep the event loop free.

## Where it plugs in (highest-leverage first)

1. **`video_vision` keyframe pre-pass** — replace fixed-interval sampling with
   `keyframes` (scene-cut + dedupe). Fewer LLM calls per video; cleanest first win.
2. **Diff-gate for watch/autonomy loops** — `browser_vision` / `screen_capture` and
   the autonomous-work / Jarvis polling loops call the LLM **only when the image
   changed** (`diff` above a threshold). Recurring cost cut on anything that polls.
3. **OCR pre-processing** — `prep` (deskew/crop/threshold) before `image_ocr`, plus
   `qr` decode that OCR and LLMs both miss.
4. **Deterministic click targets** — `locate` gives the browser/desktop tools exact
   coordinates instead of an LLM guess.
5. **(Future) Iskra beings** — a per-tick perception primitive cheap enough to run
   continuously, if beings ever get "eyes." Flagged, not in scope here.

## Cost behaviour

Every Phase-1 op is deterministic and local → **zero tokens, not costed.** `detect`
(Phase 2) runs a small ONNX model on CPU → still no LLM spend. Net effect on the
run-cost ledger is *negative* (fewer LLM frames), never positive.

## Phases

- **Phase 1 — classical, no models. DONE (uncommitted).** `vision` tool with
  `diff` (SSIM + change boxes), `dedupe` (dHash clustering), `measure` (size/blur/
  brightness/blank/dominant-colors), `prep` (grayscale/deskew/threshold/denoise/
  enhance/autocrop/crop), `qr` (QR + barcode-if-present), `locate` (template match),
  `annotate` (boxes/labels), plus a standalone `keyframes` (histogram scene-cut).
  Guarded `cv2` import; VFS-aware input (`vfs:`/real) and output; sync work in
  `asyncio.to_thread`. Registered + `_ALWAYS_ENABLED` (parity with `video_vision`).
  `video_vision` gets a `dedupe_frame_indices` keyframe pre-pass (drops near-identical
  frames before the per-frame LLM loop; no-ops without cv2). New `cv` extra. 11 tests
  (`tests/test_tools/test_vision.py`, importorskip). Verified end-to-end on cv2 4.13:
  all ops produce correct output; `locate` hits conf 1.0; QR decodes; diff SSIM correct.
- **Phase 2 — local ONNX detectors. DONE (committed).** `detect what=faces|text|objects`
  via OpenCV's DNN engine — no LLM, no token spend. **faces** = YuNet (native
  `FaceDetectorYN`, ~232 KB); **text** = PP-OCRv3 DB (native `TextDetectionModel_DB`,
  ~2.4 MB, returns text-region boxes to feed `image_ocr`); **objects** = generic
  YOLOv8/v5 ONNX decode + `NMSBoxes` + COCO-80 labels, **bring-your-own model** (a COCO
  detector worth bundling is heavier than the "light on resources" bar — no default
  auto-download). Model cache infra: `_MODEL_REGISTRY` + `_ensure_model` (best-effort,
  LFS-aware download that rejects git-lfs pointer files) + `_models_dir()`
  (`CAPTAIN_CLAW_VISION_MODELS` env → `fd-data/models/vision` → `~/.captain-claw/...`)
  so prod can pre-place models offline. Optional `out` annotates. 7 new tests (YOLO
  decode on synthetic v8/v5 tensors + model-dir + no-model error + network-gated
  faces/text). Verified on cv2 4.13: text fires on rendered text, YuNet loads/runs,
  cache reused on 2nd call. NOTE: reused the OpenCV DNN face detector, **not** the
  heavier `faces` extra (insightface/onnxruntime) — lighter and no extra dep.
- **Phase 3 — pipeline integration. DONE (committed).** Reality-check up front:
  there is *no* existing screenshot-poll-to-LLM loop to retrofit (`screen_capture` is
  single-shot; the only vision poll, `video_vision`, was already diff-gated in Phase 1).
  So Phase 3 shipped as three additive, opt-in, fail-open wirings:
  - **OCR pre-processing** — `image_ocr` gains an opt-in `preprocess` param (+ config
    `image_ocr.preprocess`/`preprocess_enhance`): an in-memory `preprocess_ocr_bytes`
    (bytes→bytes) deskews (and optionally CLAHE-enhances) the image before the LLM
    sees it. Gated to `image_ocr` (NOT `image_vision` — binarizing/deskewing hurts
    natural-image description). Fails open (returns input bytes) without OpenCV.
  - **Diff-gate** — `screen_capture` gains an opt-in `baseline` param: capture, then
    `images_differ(baseline, new)` (SSIM); if unchanged, **skip the vision LLM** and
    report "no change". A watch/poll loop passes the previous shot as `baseline` and
    only pays for the LLM when the screen actually changed. Fail-open: absent OpenCV /
    unreadable baseline → treated as changed, so it analyses (no regression).
  - **`locate` → click** — `desktop_action screenshot_click` gains an opt-in `template`
    image path: finds the element by pixel-exact OpenCV template match (`vision locate`,
    no LLM) and clicks its center via the existing coordinate-click path. Browser was
    left out deliberately — it has no coordinate-click path (Playwright selectors only),
    so `locate` doesn't plug in without new machinery; desktop already clicks by (x,y).
  - Two new public helpers in `vision.py` (`preprocess_ocr_bytes`, `images_differ`),
    both guarded + fail-open. 8 new tests (helpers + `_locate_by_template` parsing) →
    24 total in `test_vision.py`. Verified on cv2 4.13.

## Safety / back-compat

Absent OpenCV, the tool reports unavailable and every existing tool behaves exactly
as today (guarded import, no hard dependency added to core). The `video_vision`
pre-pass falls back to fixed-interval sampling when `cv2` is missing, so no regression.

## Touch list (Phase 1)

- `captain_claw/tools/vision.py` — new `VisionTool` (verb dispatch, guarded `cv2`).
- `captain_claw/tools/__init__.py` — import + `__all__` entry.
- `captain_claw/tools/video_vision.py` — optional `keyframes` pre-pass before the
  frame-description loop (guarded; falls back to current sampling).
- `pyproject.toml` — add a `cv` optional-extra (or bump the `faces`-shared pin) with
  `opencv-python-headless>=5.0`; keep it out of core deps.

## Post-ship: tool/name disambiguation (2026-07-14)

A live run surfaced a **naming collision**: a user said "use the vision tool on this
image" with a non-vision model whose multimodal peer was down; the agent reached for
this `vision` tool and ran `measure` + `detect what=text` — which found *where* text
was (region boxes) but of course couldn't read it or describe the image. `vision` is
easy to confuse with `image_vision`, and (being in `_ALWAYS_ENABLED`) is sometimes the
only image tool present in a constrained session. Fix — sharpened the guidance so an
agent picks the right tool, in three places:

- **`vision` tool description** — leads with "NOT for looking at, reading, or
  understanding a picture — use image_vision to describe, image_ocr to read text";
  `detect what=text` now says it returns *regions, not the words*.
- **Flight Deck attachment hint** (`ChatPanel.tsx`) and **web/WhatsApp hint**
  (`chat_handler.py`) — both now name `image_ocr` for reading text and explicitly say
  "do NOT use the 'vision' tool for this (pixel ops only)". FD bundle rebuilt.

Root-cause option still open: **rename the tool** (e.g. `cv`/`imagecv`/`pixels`) to
remove the collision with `image_vision` entirely — cheap now (branch not merged), but
a product-naming call, so left to the user.

## Follow-ups / known limits

- `cv2` is not importable in the current dev env — Phase 1 must add the extra and the
  install step; verify the OpenCV 5 headless wheel resolves on the prod platform
  (manylinux/macOS/arm) before relying on `detect`.
- Barcode decode via OpenCV alone is weaker than a dedicated lib (e.g. `pyzbar`);
  if 1D barcodes matter, revisit in Phase 2. QR is solid in-box.
- The new DNN engine is **CPU-only** at the OpenCV 5.0 launch (GPU planned later);
  keep bundled detectors small so `detect` stays "light on resources."
