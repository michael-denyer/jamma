"""Eigendecomposition driver planning: peak formulas and the driver choice.

Above :mod:`jamma.core.memory` in the layering: it prices with
``array_gb`` and gates with ``fits`` from there, never the reverse.
"""

from enum import StrEnum
from typing import NamedTuple

from jamma.core.constants import env_flag
from jamma.core.memory import array_gb, fits


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


def _dsyevd_inplace_peak_gb(n: int) -> float:
    """Peak memory (GB) for in-place DSYEVD eigendecomposition.

    When inplace=True, K is reused as the eigenvector output buffer.
    Peak is: K (input/output) + DSYEVD workspace. No separate U allocation.
    Saves one full N x N matrix compared to the default path.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    return array_gb(n, n) + _dsyevd_workspace_gb(n)


def _dsyevd_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVD eigendecomposition (non-inplace).

    Peak is: K (scratch) + U (eigenvectors) + DSYEVD workspace. U is the same
    (n, n) shape as K, so it costs the same ``array_gb(n, n)``.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    kinship_gb = array_gb(n, n)
    return 2 * kinship_gb + _dsyevd_workspace_gb(n)


def dsyevr_peak_gb(n: int) -> float:
    """Peak memory (GB) for DSYEVR eigendecomposition.

    On the Python path, jlinalg_dsyevr_ext writes vendor output directly into
    the caller-owned eigenvector buffer and transposes in place, so peak is:
    K (overwritten as scratch) + U (caller output) + O(N). U is the same
    (n, n) shape as K, so it costs the same ``array_gb(n, n)``.
    """
    if n < 0:
        raise ValueError(f"n_samples must be >= 0, got {n}")
    return 2 * array_gb(n, n) + _dsyevr_workspace_gb(n)


class EigenDriver(StrEnum):
    DSYEVD_INPLACE = "DSYEVD-inplace"
    DSYEVD = "DSYEVD"
    DSYEVR = "DSYEVR"
    NUMPY = "numpy"


class EigenDriverPlan(NamedTuple):
    """Chosen eigendecomposition driver, its peak-memory estimate, and why.

    Single source of truth for the DSYEVD-inplace -> DSYEVD -> DSYEVR -> numpy
    driver decision. The runtime path (``eigendecompose_kinship``) builds its
    plan here, so a pre-flight caller using the same function cannot drift from
    it.

    Attributes:
        required_gb: Peak memory (GB) for the chosen driver. For the ``numpy``
            fallback this is a conservative DSYEVD-sized proxy, not numpy's exact
            peak.
        reason: Written by the branch that chose ``driver``; ``describe`` prints
            it verbatim.
    """

    driver: EigenDriver
    required_gb: float
    reason: str

    @property
    def use_inplace(self) -> bool:
        return self.driver is EigenDriver.DSYEVD_INPLACE

    @property
    def use_dsyevr(self) -> bool:
        return self.driver is EigenDriver.DSYEVR

    @property
    def no_vendor(self) -> bool:
        return self.driver is EigenDriver.NUMPY

    def describe(self, available_gb: float) -> str:
        return (
            f"Eigendecomp memory ({self.driver}): estimated {self.required_gb:.1f}GB, "
            f"available {available_gb:.1f}GB ({self.reason})"
        )


def plan_eigen_driver(
    n_samples: int,
    available_gb: float,
    *,
    has_dsyevd: bool,
    has_dsyevr: bool,
    forced_numpy: bool,
    inplace_blocker: str | None,
    budget_gb: float | None = None,
) -> EigenDriverPlan:
    """Select the eigendecomposition driver from memory and capability flags.

    Prefers in-place DSYEVD (smallest footprint), falls back to non-inplace
    DSYEVD, then to DSYEVR (O(N) workspace) when the DSYEVD peak plus safety
    margin would not fit. When only vendor DSYEVR is available, plans DSYEVR
    directly. With no vendor DSYEVD/DSYEVR, or when ``forced_numpy`` is set,
    reports the numpy fallback and its conservative DSYEVD-sized footprint.

    Pure function — takes flags, returns a plan, performs no I/O.

    Args:
        n_samples: Kinship matrix dimension.
        available_gb: Available memory (GB).
        has_dsyevd: Vendor DSYEVD available.
        has_dsyevr: Vendor DSYEVR available.
        forced_numpy: ``JLINALG_NO_VENDOR_LAPACK`` forces the numpy fallback.
        inplace_blocker: Why K cannot be overwritten in place (not float64, not
            C-contiguous, not writeable), or ``None`` when it can.
        budget_gb: User-set ceiling in GB, or None for no ceiling. Falls back to
            DSYEVR when the DSYEVD peak exceeds it, the same way an available-RAM
            shortfall does.

    Returns:
        EigenDriverPlan with the chosen driver, its peak estimate, and the reason.
    """
    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    dsyevr_peak = dsyevr_peak_gb(n_samples)
    inplace_peak = _dsyevd_inplace_peak_gb(n_samples)

    if forced_numpy:
        return EigenDriverPlan(
            EigenDriver.NUMPY,
            dsyevd_peak,
            "JLINALG_NO_VENDOR_LAPACK set, using np.linalg.eigh; "
            "estimate is DSYEVD-sized",
        )
    if not has_dsyevd and not has_dsyevr:
        return EigenDriverPlan(
            EigenDriver.NUMPY,
            dsyevd_peak,
            "no vendor DSYEVD or DSYEVR, using np.linalg.eigh; "
            "estimate is DSYEVD-sized",
        )

    if not has_dsyevd:
        return EigenDriverPlan(
            EigenDriver.DSYEVR, dsyevr_peak, "vendor DSYEVD unavailable"
        )

    if inplace_blocker is None:
        driver, required_gb = EigenDriver.DSYEVD_INPLACE, inplace_peak
        reason = "kinship in memory, overwriting in place"
    else:
        driver, required_gb = EigenDriver.DSYEVD, dsyevd_peak
        reason = inplace_blocker

    if has_dsyevr and (
        not fits(required_gb, available_gb)
        or (budget_gb is not None and required_gb > budget_gb)
    ):
        return EigenDriverPlan(
            EigenDriver.DSYEVR,
            dsyevr_peak,
            f"{driver}={required_gb:.1f}GB would not fit",
        )

    return EigenDriverPlan(
        driver, required_gb, f"{reason}; DSYEVR fallback={dsyevr_peak:.1f}GB"
    )
