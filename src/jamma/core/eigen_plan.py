"""Eigendecomposition driver planning and the shared sizing primitives.

Below :mod:`jamma.core.memory` in the layering: the cost model imports the
peak formulas and the margin from here, never the reverse. The margin and
the GB-per-array helpers live here because both layers apply them.
"""

from typing import Literal, NamedTuple

from jamma.core.constants import env_flag


def array_gb(*shape: int) -> float:
    """Memory (GB) for a float64 array of the given shape."""
    total = 8
    for dim in shape:
        total *= dim
    return total / 1e9


def forced_numpy_fallback() -> bool:
    """Return True if JLINALG_NO_VENDOR_LAPACK forces the numpy eigendecomp path.

    Presence-based, matching ``docs/CONFIGURATION.md`` and the sibling
    ``JAMMA_FORCE_NUMPY_FALLBACK``: *any* value other than unset/``""``/``"0"``
    forces numpy — including ``"false"``, ``"no"``, and ``"off"``. Set the var to
    ``0`` (or leave it unset) to keep vendor LAPACK; do not expect ``"false"`` to
    mean off. The resolved decision is logged at runtime in
    ``eigendecompose_kinship``.

    Shared by the runtime path (``eigendecompose_kinship``) and the pre-flight
    estimators so both agree on whether vendor LAPACK is bypassed — otherwise a
    forced-numpy run could pass pre-flight on a smaller vendor estimate and
    then OOM.
    """
    return env_flag("JLINALG_NO_VENDOR_LAPACK")


def _dsyevd_workspace_gb(n: int) -> float:
    """DSYEVD workspace in GB: (1+6N+2N^2) float64s + (3+5N) int64s (upper bound)."""
    lwork_bytes = (1 + 6 * n + 2 * n * n) * 8  # float64
    # int64 on ILP64, int32 on LP64; use 8 to avoid underestimating
    liwork_bytes = (3 + 5 * n) * 8
    return (lwork_bytes + liwork_bytes) / 1e9


def _dsyevr_workspace_gb(n: int) -> float:
    """DSYEVR workspace in GB: max(1, 26*N) float64s + max(1, 10*N) int64s.

    DSYEVR (MRRR algorithm) uses O(N) workspace vs DSYEVD's O(N^2).
    At 125k samples: ~0.036 GB vs ~250 GB (excludes isuppz, 2*N ints, negligible).
    """
    lwork_bytes = max(1, 26 * n) * 8  # float64
    liwork_bytes = max(1, 10 * n) * 8  # int64 (ILP64 upper bound)
    return (lwork_bytes + liwork_bytes) / 1e9


def square_matrix_gb(n: int) -> float:
    """Memory (GB) for an n×n float64 matrix."""
    return n * n * 8 / 1e9


def _memory_margin_gb(peak_gb: float) -> float:
    """Safety margin: 10% of peak, capped at 10GB absolute.

    The single spelling of the margin — the estimators' sufficiency verdict
    and check_memory_available both apply exactly this.
    """
    return min(peak_gb * 0.1, 10.0)


def _eigendecomp_workspace_gb(n: int) -> float:
    """Return eigendecomp workspace in GB (DSYEVD, the default driver)."""
    return _dsyevd_workspace_gb(n)


def _eigendecomp_eigvec_gb(kinship_gb: float) -> float:
    """Return eigenvector memory (GB) for eigendecomp (non-inplace path).

    The in-place path avoids this allocation — see _dsyevd_inplace_peak_gb.
    """
    return kinship_gb


def _dsyevd_inplace_peak_gb(n: int) -> float:
    """Peak memory (GB) for in-place DSYEVD eigendecomposition.

    When inplace=True, K is reused as the eigenvector output buffer.
    Peak is: K (input/output) + DSYEVD workspace. No separate U allocation.
    Saves one full N x N matrix compared to the default path.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    return square_matrix_gb(n) + _dsyevd_workspace_gb(n)


def _dsyevd_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVD eigendecomposition (non-inplace).

    Peak is: K (scratch) + U (eigenvectors) + DSYEVD workspace.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    kinship_gb = square_matrix_gb(n)
    return kinship_gb + _eigendecomp_eigvec_gb(kinship_gb) + _dsyevd_workspace_gb(n)


def dsyevr_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVR eigendecomposition.

    On the Python path, jlinalg_dsyevr_ext writes vendor output directly into
    the caller-owned eigenvector buffer and transposes in place, so peak is:
    K (overwritten as scratch) + U (caller output) + O(N).
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    kinship_gb = square_matrix_gb(n)
    return kinship_gb + _eigendecomp_eigvec_gb(kinship_gb) + _dsyevr_workspace_gb(n)


class EigenDriverPlan(NamedTuple):
    """Chosen eigendecomposition driver and its peak-memory estimate.

    Single source of truth for the DSYEVD-inplace -> DSYEVD -> DSYEVR -> numpy
    driver decision. The runtime path (``eigendecompose_kinship``) builds its
    plan here, so a pre-flight caller using the same function cannot drift from
    it. The chosen driver can still differ per caller when they pass different
    ``inplace_eligible`` inputs.

    Attributes:
        driver: Chosen driver name (one of the four ``Literal`` values).
        use_inplace: Pass ``inplace=True`` to ``jlinalg.eigh`` (K reused as the
            eigenvector output buffer).
        use_dsyevr: DSYEVR was selected — either as the memory-pressure fallback
            from DSYEVD, or because it is the only available vendor driver.
        no_vendor: No vendor LAPACK will run (``np.linalg.eigh`` fallback).
        required_gb: Peak memory (GB) for the chosen driver. For the ``numpy``
            fallback this is a conservative DSYEVD-sized proxy, not numpy's exact
            peak.
        pre_fallback_gb: ``required_gb`` before any DSYEVR fallback (used to log
            which driver we fell back from).
        dsyevr_peak_gb: DSYEVR peak (GB).
        inplace_peak_gb: In-place DSYEVD peak (GB).
    """

    driver: Literal["DSYEVD-inplace", "DSYEVD", "DSYEVR", "numpy"]
    use_inplace: bool
    use_dsyevr: bool
    no_vendor: bool
    required_gb: float
    pre_fallback_gb: float
    dsyevr_peak_gb: float
    inplace_peak_gb: float


def plan_eigen_driver(
    n_samples: int,
    available_gb: float,
    *,
    has_dsyevd: bool,
    has_dsyevr: bool,
    no_vendor: bool,
    inplace_eligible: bool,
) -> EigenDriverPlan:
    """Select the eigendecomposition driver from memory and capability flags.

    Prefers in-place DSYEVD (smallest footprint), falls back to non-inplace
    DSYEVD, then to DSYEVR (O(N) workspace) when the DSYEVD peak plus safety
    margin would not fit. When only vendor DSYEVR is available, plans DSYEVR
    directly. With no vendor DSYEVD/DSYEVR (or a caller-forced ``no_vendor``),
    reports the numpy fallback and its conservative DSYEVD-sized footprint.

    Pure function — takes flags, returns a plan, performs no I/O. The runtime
    caller passes the real ``inplace_eligible`` (K is float64, C-contiguous,
    writeable); the pre-flight estimator passes ``inplace_eligible=True`` because
    the kinship matrix is not built yet and will normally be in-place eligible.

    Args:
        n_samples: Kinship matrix dimension.
        available_gb: Available memory (GB).
        has_dsyevd: Vendor DSYEVD available.
        has_dsyevr: Vendor DSYEVR available.
        no_vendor: Force the numpy fallback (e.g. JLINALG_NO_VENDOR_LAPACK set).
        inplace_eligible: K can be overwritten in place (float64, C-contiguous,
            writeable).

    Returns:
        EigenDriverPlan with the chosen driver, flags, and peak estimates.
    """
    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    dsyevr_peak = dsyevr_peak_gb(n_samples)
    inplace_peak = _dsyevd_inplace_peak_gb(n_samples)

    # No vendor DSYEVD *and* no vendor DSYEVR -> numpy fallback.
    if not no_vendor and not has_dsyevd and not has_dsyevr:
        no_vendor = True

    if no_vendor:
        return EigenDriverPlan(
            driver="numpy",
            use_inplace=False,
            use_dsyevr=False,
            no_vendor=True,
            required_gb=dsyevd_peak,
            pre_fallback_gb=dsyevd_peak,
            dsyevr_peak_gb=dsyevr_peak,
            inplace_peak_gb=inplace_peak,
        )

    # Only vendor DSYEVR is available (has_dsyevd is False, but the no-vendor
    # check above means has_dsyevr is True): jlinalg.eigh dispatches to DSYEVR
    # directly — there is no in-place path and no DSYEVD peak to reserve.
    if not has_dsyevd:
        return EigenDriverPlan(
            driver="DSYEVR",
            use_inplace=False,
            use_dsyevr=True,
            no_vendor=False,
            required_gb=dsyevr_peak,
            pre_fallback_gb=dsyevr_peak,
            dsyevr_peak_gb=dsyevr_peak,
            inplace_peak_gb=inplace_peak,
        )

    use_inplace = inplace_eligible
    required_gb = inplace_peak if use_inplace else dsyevd_peak
    pre_fallback_gb = required_gb
    use_dsyevr = False

    if required_gb + _memory_margin_gb(required_gb) > available_gb and has_dsyevr:
        pre_fallback_gb = required_gb
        required_gb = dsyevr_peak
        use_inplace = False
        use_dsyevr = True

    driver = "DSYEVR" if use_dsyevr else ("DSYEVD-inplace" if use_inplace else "DSYEVD")
    return EigenDriverPlan(
        driver=driver,
        use_inplace=use_inplace,
        use_dsyevr=use_dsyevr,
        no_vendor=False,
        required_gb=required_gb,
        pre_fallback_gb=pre_fallback_gb,
        dsyevr_peak_gb=dsyevr_peak,
        inplace_peak_gb=inplace_peak,
    )
