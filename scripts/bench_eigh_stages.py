"""Benchmark jblas eigh stages (DSYTRD, DSTEDC, DORMTR) vs numpy.linalg.eigh.

Times each stage of the jblas three-step eigendecomposition independently
and compares total wall-clock time against numpy's LAPACK dsyevd.

Usage:
    uv run python scripts/bench_eigh_stages.py
"""

from __future__ import annotations

import ctypes
import os
import time

import numpy as np


def _find_jblas_so() -> str:
    """Find the _jblas shared library path."""
    import jamma.jblas._jblas as mod

    return mod.__file__


class _EighStatus(ctypes.Structure):
    """ctypes mirror of jblas_eigh_status_t."""

    _fields_ = [
        ("dstedc_ws_fallback", ctypes.c_int),
        ("dsytrd_mirror_fallback", ctypes.c_int),
        ("secular_failures", ctypes.c_int),
        ("qr_fallback", ctypes.c_int),
    ]


def _load_c_functions(so_path: str) -> tuple:
    """Load jblas C functions via ctypes for per-stage benchmarking.

    Returns:
        Tuple of (dsytrd, dstedc, dormtr, eigh) ctypes function objects.
    """
    lib = ctypes.CDLL(so_path)

    # int jblas_dsytrd_c(npy_intp N, double *A, npy_intp lda,
    #                    double *d, double *e, double *tau,
    #                    jblas_workspace_t *ws, jblas_eigh_status_t *status)
    dsytrd = lib.jblas_dsytrd_c
    dsytrd.restype = ctypes.c_int
    dsytrd.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_void_p,  # double *A
        ctypes.c_longlong,  # npy_intp lda
        ctypes.c_void_p,  # double *d
        ctypes.c_void_p,  # double *e
        ctypes.c_void_p,  # double *tau
        ctypes.c_void_p,  # jblas_workspace_t *ws (NULL = global mutex)
        ctypes.c_void_p,  # jblas_eigh_status_t *status (NULL = no status)
    ]

    # int jblas_dstedc_c(npy_intp N, double *d, double *e,
    #                    double *Z, npy_intp ldz, jblas_workspace_t *ws)
    dstedc = lib.jblas_dstedc_c
    dstedc.restype = ctypes.c_int
    dstedc.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_void_p,  # double *d
        ctypes.c_void_p,  # double *e
        ctypes.c_void_p,  # double *Z
        ctypes.c_longlong,  # npy_intp ldz
        ctypes.c_void_p,  # jblas_workspace_t *ws (NULL = use global mutex)
    ]

    # int jblas_dormtr_c(npy_intp N, npy_intp M,
    #                    const double *A, npy_intp lda, const double *tau,
    #                    double *C, npy_intp ldc)
    dormtr = lib.jblas_dormtr_c
    dormtr.restype = ctypes.c_int
    dormtr.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_longlong,  # npy_intp M
        ctypes.c_void_p,  # const double *A
        ctypes.c_longlong,  # npy_intp lda
        ctypes.c_void_p,  # const double *tau
        ctypes.c_void_p,  # double *C
        ctypes.c_longlong,  # npy_intp ldc
    ]

    # jblas_eigh_c for status reporting
    eigh_c = lib.jblas_eigh_c
    eigh_c.restype = ctypes.c_int
    eigh_c.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_void_p,  # double *K
        ctypes.c_longlong,  # npy_intp ldk
        ctypes.c_void_p,  # double *eigenvalues
        ctypes.c_void_p,  # double *eigenvectors
        ctypes.c_longlong,  # npy_intp ldz
        ctypes.POINTER(_EighStatus),  # jblas_eigh_status_t *status
    ]

    return dsytrd, dstedc, dormtr, eigh_c


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Get ctypes void pointer to numpy array data."""
    return ctypes.c_void_p(arr.ctypes.data)


def _random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive definite matrix."""
    A = rng.standard_normal((n, n))
    K = A @ A.T / n
    return np.ascontiguousarray(K, dtype=np.float64)


def bench_numpy_eigh(K: np.ndarray, n_runs: int) -> float:
    """Time numpy.linalg.eigh over n_runs, return best time."""
    best = float("inf")
    for _ in range(n_runs):
        K_copy = K.copy()
        t0 = time.perf_counter()
        np.linalg.eigh(K_copy)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def bench_jblas_eigh(K: np.ndarray, n_runs: int) -> float:
    """Time jblas eigh (full) over n_runs, return best time."""
    from jamma.jblas import eigh

    best = float("inf")
    for _ in range(n_runs):
        K_copy = K.copy()
        t0 = time.perf_counter()
        eigh(K_copy)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best


def bench_jblas_stages(
    K: np.ndarray,
    dsytrd_fn,
    dstedc_fn,
    dormtr_fn,
    n_runs: int,
) -> dict[str, float]:
    """Time each jblas eigh stage individually. Return dict of best times."""
    N = K.shape[0]

    best_dsytrd = float("inf")
    best_dstedc = float("inf")
    best_dormtr = float("inf")

    for _ in range(n_runs):
        # --- DSYTRD ---
        K_copy = K.copy()
        d = np.empty(N, dtype=np.float64)
        e = np.empty(N, dtype=np.float64)
        tau = (
            np.empty(N - 1, dtype=np.float64)
            if N > 1
            else np.empty(0, dtype=np.float64)
        )

        t0 = time.perf_counter()
        ret = dsytrd_fn(N, _ptr(K_copy), N, _ptr(d), _ptr(e), _ptr(tau), None, None)
        t_dsytrd = time.perf_counter() - t0
        assert ret == 0, f"dsytrd failed with ret={ret}"
        best_dsytrd = min(best_dsytrd, t_dsytrd)

        # Save tau and Householder vectors for dormtr
        tau_save = tau.copy()
        K_householder = K_copy.copy()

        # --- DSTEDC ---
        Z = np.empty((N, N), dtype=np.float64)

        t0 = time.perf_counter()
        ret = dstedc_fn(N, _ptr(d), _ptr(e), _ptr(Z), N, None)
        t_dstedc = time.perf_counter() - t0
        assert ret == 0, f"dstedc failed with ret={ret}"
        best_dstedc = min(best_dstedc, t_dstedc)

        # --- DORMTR ---
        # Need K_householder (from dsytrd) and tau
        # Z now has eigenvectors of T; dormtr transforms them back
        t0 = time.perf_counter()
        ret = dormtr_fn(N, N, _ptr(K_householder), N, _ptr(tau_save), _ptr(Z), N)
        t_dormtr = time.perf_counter() - t0
        assert ret == 0, f"dormtr failed with ret={ret}"
        best_dormtr = min(best_dormtr, t_dormtr)

    return {
        "dsytrd": best_dsytrd,
        "dstedc": best_dstedc,
        "dormtr": best_dormtr,
        # Note: total_staged sums per-stage bests across independent runs,
        # so it may understate actual single-run wall time.
        "total_staged": best_dsytrd + best_dstedc + best_dormtr,
    }


def main() -> None:
    sizes = [200, 500, 1000, 4096, 10000, 20000]
    # VALID-05 also specifies 46k, but that requires ILP64 MKL numpy (LP64 int32
    # overflow at ~46k x 46k). Run on ILP64 systems with JBLAS_BENCH_MAX_GB=200.
    n_runs = 3
    rng = np.random.default_rng(42)

    MAX_MATRIX_GB = float(os.environ.get("JBLAS_BENCH_MAX_GB", "8.0"))

    # Load C functions
    so_path = _find_jblas_so()
    print(f"jblas .so: {so_path}")
    dsytrd_fn, dstedc_fn, dormtr_fn, eigh_c_fn = _load_c_functions(so_path)

    # Check jblas is using C extension
    from jamma.jblas import HAS_C_EXTENSION, jblas_isa

    print(f"HAS_C_EXTENSION: {HAS_C_EXTENSION}, ISA: {jblas_isa}")
    print("numpy BLAS config:")
    try:
        config = np.show_config(mode="dicts")
        if isinstance(config, dict):
            blas = config.get("Build Dependencies", {}).get("blas", {})
            print(f"  BLAS: {blas.get('name', 'unknown')}")
    except Exception:
        print("  (could not read config)")
    print()

    # Header
    print(
        f"{'N':>6}  {'numpy eigh':>11}  {'jblas eigh':>11}  "
        f"{'dsytrd':>9}  {'dstedc':>9}  {'dormtr':>9}  "
        f"{'staged tot':>11}  {'ratio':>7}  {'bottleneck':>12}"
    )
    print("-" * 110)

    all_stages: dict[int, dict[str, float]] = {}

    for N in sizes:
        matrix_gb = (N * N * 8 * 3) / (1024**3)  # K + eigvecs + workspace
        if matrix_gb > MAX_MATRIX_GB:
            limit = MAX_MATRIX_GB
            print(
                f"\n--- N={N} skipped "
                f"(estimated {matrix_gb:.1f} GB > {limit:.1f} GB limit) ---"
            )
            print(f"    Set JBLAS_BENCH_MAX_GB={matrix_gb + 1:.0f} to enable")
            continue

        K = _random_spd(N, rng)

        # Warmup
        _ = np.linalg.eigh(K.copy())
        from jamma.jblas import eigh

        _ = eigh(K.copy())

        # Benchmark
        t_numpy = bench_numpy_eigh(K, n_runs)
        t_jblas = bench_jblas_eigh(K, n_runs)
        stages = bench_jblas_stages(K, dsytrd_fn, dstedc_fn, dormtr_fn, n_runs)
        all_stages[N] = stages

        # Identify bottleneck
        stage_times = {
            "dsytrd": stages["dsytrd"],
            "dstedc": stages["dstedc"],
            "dormtr": stages["dormtr"],
        }
        bottleneck = max(stage_times, key=stage_times.get)
        bottleneck_pct = stage_times[bottleneck] / stages["total_staged"] * 100

        ratio = t_jblas / t_numpy

        # Run eigh_c with status to report secular failures and QR fallback
        K_status = K.copy()
        eigenvalues = np.empty(N, dtype=np.float64)
        eigenvectors = np.empty((N, N), dtype=np.float64)
        status = _EighStatus()
        eigh_c_fn(
            N,
            _ptr(K_status),
            N,
            _ptr(eigenvalues),
            _ptr(eigenvectors),
            N,
            ctypes.byref(status),
        )

        print(
            f"{N:>6}  "
            f"{t_numpy:>10.4f}s  "
            f"{t_jblas:>10.4f}s  "
            f"{stages['dsytrd']:>8.4f}s  "
            f"{stages['dstedc']:>8.4f}s  "
            f"{stages['dormtr']:>8.4f}s  "
            f"{stages['total_staged']:>10.4f}s  "
            f"{ratio:>6.1f}x  "
            f"{bottleneck:>8} ({bottleneck_pct:.0f}%)  "
            f"sec_fail={status.secular_failures}  "
            f"qr_fb={status.qr_fallback}"
        )

    # Detailed breakdown from already-collected data
    print()
    print("Stage breakdown (% of jblas staged total):")
    print(f"{'N':>6}  {'dsytrd %':>9}  {'dstedc %':>9}  {'dormtr %':>9}")
    print("-" * 42)

    for N, stages in all_stages.items():
        total = stages["total_staged"]
        print(
            f"{N:>6}  "
            f"{stages['dsytrd'] / total * 100:>8.1f}%  "
            f"{stages['dstedc'] / total * 100:>8.1f}%  "
            f"{stages['dormtr'] / total * 100:>8.1f}%"
        )


if __name__ == "__main__":
    main()
