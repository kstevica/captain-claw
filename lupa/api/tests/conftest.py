import sys
from pathlib import Path

# Make `lupa_api` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
