"""Minimal import smoke test for the ESCT package scripts.

Run from the repository root:
    python scripts/import_check.py
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import paths  # noqa: E402
import esct_model  # noqa: E402
import run_esct_devgamma_clean  # noqa: E402

print("ESCT imports OK; repo root:", paths.REPO_ROOT)
