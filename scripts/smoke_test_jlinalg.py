"""Smoke test for the jlinalg C extension wheel — used by CIBW_TEST_COMMAND.

Verifies:
1. The compiled _jlinalg C extension imports successfully (ABI match)
2. jlinalg_isa and HAS_OPENMP constants are present and correct type
3. dgemm and dsyrk produce correct results when vendor BLAS is available

Exit 0 on success, exit 1 on any failure.

Usage (cibuildwheel):
    CIBW_TEST_COMMAND = "python scripts/smoke_test_jlinalg.py"
"""

import sys

import numpy as np

# Public jlinalg API for dgemm/dsyrk (vendor BLAS or numpy fallback).
# v5.0: Level 1/2 functions (daxpy, ddot, etc.) removed from C extension —
# they are now numpy-only in jlinalg/__init__.py.
from jamma.jlinalg import dgemm, dsyrk

try:
    from jamma.jlinalg._jlinalg import (
        HAS_OPENMP,
        blas_backend,
        blas_is_ilp64,
        jlinalg_isa,
    )
except ImportError as exc:
    print(
        f"FAIL: _jlinalg import failed (ABI mismatch or missing .so): {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"_jlinalg OK, ISA={jlinalg_isa!r}, OpenMP={bool(HAS_OPENMP)}")
print(f"BLAS dispatch: backend={blas_backend!r}, ilp64={blas_is_ilp64}")

# Step 2: Sanity-check constant types.
if not isinstance(jlinalg_isa, str):
    print(f"FAIL: jlinalg_isa is {type(jlinalg_isa)}, expected str", file=sys.stderr)
    sys.exit(1)

valid_isa = {"AVX2", "NEON", "generic"}
if jlinalg_isa not in valid_isa:
    print(f"FAIL: jlinalg_isa {jlinalg_isa!r} not in {valid_isa}", file=sys.stderr)
    sys.exit(1)

if not isinstance(HAS_OPENMP, bool):
    print(f"FAIL: HAS_OPENMP is {type(HAS_OPENMP)}, expected bool", file=sys.stderr)
    sys.exit(1)

# Step 3: Numerical sanity checks on dgemm and dsyrk (vendor BLAS only).
# v5.0: dgemm/dsyrk require vendor BLAS — in manylinux containers without
# vendor BLAS, they return zeros. Use the public API (which falls back to numpy).
rng = np.random.default_rng(42)


def check_close(name, got, expected, atol=1e-10):
    """Check max absolute error and exit on failure."""
    max_err = np.max(np.abs(got - expected))
    if max_err > atol:
        print(
            f"FAIL: {name} numerical mismatch: max_err={max_err:.2e}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"{name}: OK (max_err={max_err:.2e})")


A = rng.standard_normal((100, 50))
B = rng.standard_normal((50, 80))
check_close("dgemm", dgemm(A, B), A @ B)

X = rng.standard_normal((60, 40))
check_close("dsyrk", dsyrk(X), X @ X.T)

print("Smoke test passed")
sys.exit(0)
