"""Build the optional C++17 formatter: python -m jamma.io._compile_matrix_text."""

import sys
from pathlib import Path

from jamma._build_support.compile_and_link import (
    MATRIX_TEXT_SPEC,
    BuildReport,
    compile_extension,
)
from jamma._build_support.load_proof import load_proof

if __name__ == "__main__":
    built = compile_extension(
        MATRIX_TEXT_SPEC,
        Path(__file__).parents[1],
        BuildReport.to_stream(sys.stdout, verbose=True),
    )
    raise SystemExit(0 if built and load_proof(MATRIX_TEXT_SPEC) else 1)
