"""Smoke test for the jblas C extension wheel — used by CIBW_TEST_COMMAND.

Verifies:
1. The compiled _jblas C extension imports successfully (ABI match)
2. jblas_isa and HAS_OPENMP constants are present and correct type
3. All 6 operations produce correct results on synthetic data (numerical sanity)

Exit 0 on success, exit 1 on any failure.

Usage (cibuildwheel):
    CIBW_TEST_COMMAND = "python scripts/smoke_test_jblas.py"
"""

import sys

import numpy as np

# Step 1: Import the compiled C extension directly (not the fallback).
try:
    from jamma.jblas._jblas import (
        HAS_OPENMP,
        blas_backend,
        blas_is_ilp64,
        daxpy,
        ddot,
        dgemm,
        dgemv,
        dnrm2,
        dscal,
        jblas_isa,
    )
except ImportError as exc:
    print(
        f"FAIL: _jblas import failed (ABI mismatch or missing .so): {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"_jblas OK, ISA={jblas_isa!r}, OpenMP={bool(HAS_OPENMP)}")
print(f"BLAS dispatch: backend={blas_backend!r}, ilp64={blas_is_ilp64}")

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

# Step 3: Numerical sanity checks on all 6 operations.
rng = np.random.default_rng(42)
n = 10_000


def check_close(name, got, expected, rtol=1e-12):
    """Check relative error and exit on failure."""
    rel_err = abs(got - expected) / max(abs(expected), 1e-300)
    if rel_err > rtol:
        print(
            f"FAIL: {name} numerical mismatch: got {got}, expected {expected}, "
            f"rel_err={rel_err:.2e}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"{name}: OK (rel_err={rel_err:.2e})")


# ddot
x = rng.standard_normal(n)
y = rng.standard_normal(n)
check_close("ddot", ddot(x, y), np.dot(x, y))

# dnrm2
x = rng.standard_normal(n)
check_close("dnrm2", dnrm2(x), np.linalg.norm(x))

# daxpy (in-place, check result vector)
x = rng.standard_normal(n)
y = rng.standard_normal(n)
y_ref = y + 2.5 * x
daxpy(2.5, x, y)
max_err = np.max(np.abs(y - y_ref))
if max_err > 1e-12:
    print(f"FAIL: daxpy max_err={max_err:.2e}", file=sys.stderr)
    sys.exit(1)
print(f"daxpy: OK (max_err={max_err:.2e})")

# dscal (in-place)
x = rng.standard_normal(n)
x_ref = x * 3.14
dscal(3.14, x)
max_err = np.max(np.abs(x - x_ref))
if max_err > 1e-12:
    print(f"FAIL: dscal max_err={max_err:.2e}", file=sys.stderr)
    sys.exit(1)
print(f"dscal: OK (max_err={max_err:.2e})")

# dgemv
A = rng.standard_normal((100, 50))
x = rng.standard_normal(50)
result = dgemv(A, x)
expected = A @ x
max_err = np.max(np.abs(result - expected))
if max_err > 1e-10:
    print(f"FAIL: dgemv max_err={max_err:.2e}", file=sys.stderr)
    sys.exit(1)
print(f"dgemv: OK (max_err={max_err:.2e})")

# dgemm
A = rng.standard_normal((100, 50))
B = rng.standard_normal((50, 80))
result = dgemm(A, B)
expected = A @ B
max_err = np.max(np.abs(result - expected))
if max_err > 1e-10:
    print(f"FAIL: dgemm max_err={max_err:.2e}", file=sys.stderr)
    sys.exit(1)
print(f"dgemm: OK (max_err={max_err:.2e})")

print("Smoke test passed")
sys.exit(0)
