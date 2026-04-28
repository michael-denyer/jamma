"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Tier markers every test file must declare (per-test or via pytestmark).
# Mirrors the markers list in pyproject.toml [tool.pytest.ini_options].
# See docs/TESTING.md §1.6 for the policy.
_REQUIRED_TIER_MARKERS = frozenset({"tier0", "tier1", "tier2", "slow", "benchmark"})

# Files exempt from the tier-marker requirement. Keep this list empty if
# possible; the right fix is almost always to add a marker, not an exemption.
_TIER_MARKER_EXEMPT_FILES: frozenset[str] = frozenset()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Fail collection if any test file has zero tier/slow/benchmark markers.

    Enforces docs/TESTING.md §1.6: every test file must declare at least
    one tier marker, either per-test or via module-level ``pytestmark``.
    This catches files that silently default into the tier0+tier1 CI run
    without anyone classifying them.

    Skipped on xdist workers — they only see a partition of the items, so
    a file with markers can appear marker-less from a single worker's view.
    The controller process sees the full collection and runs the check.
    """
    if hasattr(config, "workerinput"):
        return  # xdist worker: skip, controller will run the check
    files_with_marker: dict[str, bool] = {}
    for item in items:
        path = str(item.path) if hasattr(item, "path") else str(item.fspath)
        has_required = any(
            m.name in _REQUIRED_TIER_MARKERS for m in item.iter_markers()
        )
        files_with_marker[path] = files_with_marker.get(path, False) or has_required

    missing = [
        path
        for path, ok in files_with_marker.items()
        if not ok and Path(path).name not in _TIER_MARKER_EXEMPT_FILES
    ]
    if missing:
        rel = sorted(Path(p).relative_to(Path(__file__).parent.parent) for p in missing)
        listing = "\n  ".join(str(p) for p in rel)
        raise pytest.UsageError(
            "The following test files have no tier marker "
            "(tier0/tier1/tier2/slow/benchmark):\n  "
            f"{listing}\n\n"
            "Add `pytestmark = pytest.mark.tier0` (or per-test markers). "
            "See docs/TESTING.md §1.6."
        )


def pytest_configure(config: pytest.Config) -> None:
    """Warn at session start if any C extension is stale vs its source.

    The editable install picks up Python source edits automatically, but C
    source edits require an explicit rebuild. Without this check, an edit
    to e.g. ``_lmm_accel.c`` would be silently ignored — tests run against
    the old compiled .so. We warn rather than fail so the session still
    starts; pre-push hook (scripts/check_c_extension_freshness.py) is the
    blocking gate.
    """
    del config
    # Import guarded: script lives outside the package and may be missing
    # in some install layouts (e.g. a sdist-only install). Missing script
    # is not a test failure — just skip the check.
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    if not (script_dir / "check_c_extension_freshness.py").exists():
        return
    sys.path.insert(0, str(script_dir))
    try:
        import check_c_extension_freshness as freshness
    except ImportError:
        return
    finally:
        # Don't pollute sys.path past this function.
        if sys.path and sys.path[0] == str(script_dir):
            sys.path.pop(0)

    stale = [r for r in freshness.check_all() if r.is_stale]
    if not stale:
        return
    for r in stale:
        assert r.newest_source is not None  # guaranteed by is_stale
        sys.stderr.write(
            f"\n\033[33m[jamma] WARNING: C extension '{r.spec.label}' is "
            f"stale relative to {r.newest_source.name} — tests will run "
            f"against the OLD compiled .so. Rebuild with:\n"
            f"    {r.spec.rebuild_command}\033[0m\n"
        )
    sys.stderr.write(
        "\033[33m[jamma] If this is unexpected, run "
        "scripts/check_c_extension_freshness.py for full drift report.\033[0m\n\n"
    )


def load_phenotypes_from_fam(fam_path: Path) -> np.ndarray:
    """Load phenotypes from FAM file (column 6, 0-indexed column 5).

    Handles both GEMMA's missing-phenotype marker (-9) and literal 'NA'
    strings. Returns float64 array with NaN for missing values.

    Args:
        fam_path: Path to .fam PLINK file.

    Returns:
        Array of phenotype values (float64), with -9 and NA replaced by NaN.
    """
    from jamma.core.constants import PHENOTYPE_MISSING

    data = np.loadtxt(fam_path, usecols=5, dtype=str)
    missing = np.isin(data, [str(int(PHENOTYPE_MISSING)), "NA"])
    pheno = np.where(missing, "0", data).astype(np.float64)
    pheno[missing] = np.nan
    return pheno


if TYPE_CHECKING:
    from jamma.validation import ToleranceConfig

# Tier marker policy lives in docs/TESTING.md §1.5 (source of truth) and
# pyproject.toml [tool.pytest.ini_options].markers. The enforcement gate at
# the top of this file fails collection if a test lacks tier0/tier1/tier2.


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from test fixtures.

    Returns:
        Path prefix for gemma_synthetic PLINK files (without .bed/.bim/.fam extension)
    """
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for test results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to output directory
    """
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tolerance_config() -> ToleranceConfig:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        ToleranceConfig with default tolerance values for different comparison types
    """
    from jamma.validation import ToleranceConfig

    return ToleranceConfig()


def _build_synthetic_covariate_data(
    n_cvt: int,
    n_samples: int = 200,
    n_snps: int = 50,
    seed: int = 42,
) -> dict:
    """Build synthetic rotated data for C extension testing.

    Generates eigenvalues, rotated covariates (UtW), phenotype (Uty),
    genotypes (UtG), and computes Uab_batch for the given n_cvt.

    Args:
        n_cvt: Number of covariates.
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: eigenvalues, UtW, Uty, UtG, Uab_batch,
        n_samples, n_snps, n_cvt.
    """
    from jamma.lmm.likelihood import compute_Uab

    rng = np.random.default_rng(seed)

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]  # descending
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Compute Uab for each SNP
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_batch[i] = compute_Uab(UtW, Uty, UtG[:, i])

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "UtG": UtG,
        "Uab_batch": Uab_batch,
        "n_samples": n_samples,
        "n_snps": n_snps,
        "n_cvt": n_cvt,
    }


@pytest.fixture
def synthetic_covariate_data_ncvt2() -> dict:
    """Synthetic data with 2 covariates for C extension testing.

    200 samples, 50 SNPs, 2 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=2, seed=42)


@pytest.fixture
def synthetic_covariate_data_ncvt4() -> dict:
    """Synthetic data with 4 covariates for C extension testing.

    200 samples, 50 SNPs, 4 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=4, seed=99)
