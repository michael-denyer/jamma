"""Smoke test for the jblas C extension wheel — used by CIBW_TEST_COMMAND.

Verifies:
1. The compiled _jblas C extension imports successfully (ABI match)
2. jblas_isa and HAS_OPENMP constants are present and correct type
3. ddot produces a correct result on synthetic data (numerical sanity)

Exit 0 on success, exit 1 on any failure.

Usage (cibuildwheel):
    CIBW_TEST_COMMAND = "python scripts/smoke_test_jblas.py"
"""

import sys

import numpy as np

# Step 1: Import the compiled C extension directly (not the fallback).
try:
    from jamma.jblas._jblas import HAS_OPENMP, ddot, jblas_isa
except ImportError as exc:
    print(
        f"FAIL: _jblas import failed (ABI mismatch or missing .so): {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"_jblas OK, ISA={jblas_isa!r}, OpenMP={bool(HAS_OPENMP)}")

# Step 2: Sanity-check constant types.
if not isinstance(jblas_isa, str):
    print(f"FAIL: jblas_isa is {type(jblas_isa)}, expected str", file=sys.stderr)
    sys.exit(1)

valid_isa = {"AVX2", "NEON", "generic"}
if jblas_isa not in valid_isa:
    print(f"FAIL: jblas_isa {jblas_isa!r} not in {valid_isa}", file=sys.stderr)
    sys.exit(1)

if not isinstance(HAS_OPENMP, bool):
    print(f"FAIL: HAS_OPENMP is {type(HAS_OPENMP)}, expected bool", file=sys.stderr)
    sys.exit(1)

# Step 3: Numerical sanity check on ddot.
rng = np.random.default_rng(42)
n = 10_000
x = rng.standard_normal(n)
y = rng.standard_normal(n)

result = ddot(x, y)
expected = np.dot(x, y)

if not isinstance(result, float):
    print(f"FAIL: ddot returned {type(result)}, expected float", file=sys.stderr)
    sys.exit(1)

rel_err = abs(result - expected) / max(abs(expected), 1e-300)
if rel_err > 1e-12:
    print(
        f"FAIL: ddot numerical mismatch: got {result}, expected {expected}, "
        f"rel_err={rel_err:.2e}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"ddot numerical sanity: OK (rel_err={rel_err:.2e})")
print("Smoke test passed")
sys.exit(0)
