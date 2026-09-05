#!/usr/bin/env python3
"""Run isolated mutations of production and validation code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.math_validation.mutations import main

if __name__ == "__main__":
    raise SystemExit(main())
