#!/usr/bin/env python3
"""Pre-download an mflux model and generate a test image.

Usage:
    python scripts/download_mflux_model.py                # downloads schnell (default)
    python scripts/download_mflux_model.py fibo-lite       # downloads fibo-lite
    python scripts/download_mflux_model.py flux2-klein-4b  # downloads flux2-klein-4b
"""

import sys
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "schnell"
VALID = ("schnell", "fibo-lite", "flux2-klein-4b")
STEPS = {"schnell": 4, "fibo-lite": 4, "flux2-klein-4b": 4}

if MODEL not in VALID:
    print(f"Unknown model '{MODEL}'. Choose from: {', '.join(VALID)}")
    sys.exit(1)

print(f"==> Loading mflux model '{MODEL}' (will download on first run)...")

from mflux.models.flux.cli.flux_generate import Flux1, ModelConfig  # noqa: E402

flux = Flux1(
    model_config=ModelConfig.from_name(MODEL),
    quantize=8,
)

print("==> Model loaded! Generating a test image...")

image = flux.generate_image(
    seed=42,
    prompt="A cozy tavern interior with wooden tables and warm candlelight, fantasy game art",
    num_inference_steps=STEPS[MODEL],
    width=512,
    height=512,
)

out = Path(f"test_{MODEL}.png")
image.save(path=str(out), export_json_metadata=False, overwrite=True)
print(f"==> Done! Test image saved to {out}")
