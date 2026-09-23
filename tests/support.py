from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.lmm import accel

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Shared plumbing every scripts/check_*.py lint imports. install_lint_script()
# below copies it alongside whichever lint a test is driving.
_LINT_COMMON = _REPO_ROOT / "scripts" / "_lint_common.py"

# The one C-extension seam every LMM test drives through. Replaces 26
# `skipif(compute_numpy._accel is None, ...)` decorators, 5 module-level flags
# that all re-spelled the same bit, 14 inline `pytest.skip("C extension ...")`
# calls, and ~24 hand-written `orig = ...; try: ... finally:` or
# `monkeypatch.setattr(cn, "_accel", None)` hold-outs across 12 files. See
# docs/TESTING.md §2.7.
#
# Read once at collection, same as jlinalg's HAS_C_EXTENSION: the extension
# does not appear or disappear mid-session, only the ``no_c_kernels`` fixture holds
# it out for the span of one test.
requires_c = pytest.mark.skipif(
    not accel.available(), reason="C extension _lmm_accel not available"
)


def require_fixture(*paths: Path) -> None:
    """Assert committed fixture paths exist, raising rather than skipping.

    Everything under ``tests/fixtures/`` is committed and sha256-verified by
    the ``GEMMA fixture sha256 manifest`` pre-commit hook. A path that does
    not exist therefore means the test names the wrong path, never that the
    data is unavailable, so the correct response is a failure and not a skip.
    Guarding with ``pytest.skip`` hid two GEMMA-parity tests for their entire
    lifetime: the directory and the filename were both wrong, one
    ``.exists()`` check collapsed both misses into one skip reason, and the
    run stayed green (#147).

    Pass every path the caller is about to read, in one call. A wrong
    directory then reports all of its files at once instead of stopping at
    the first, which is the half of #147 a single check could not show.

    Args:
        paths: Fixture files that must be present.

    Raises:
        FileNotFoundError: If any path is absent. Every missing path is
            named, relative to the repository root.
    """
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    listing = "\n  ".join(
        str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else str(p)
        for p in missing
    )
    raise FileNotFoundError(
        f"{len(missing)} of {len(paths)} required test fixture(s) are "
        f"missing:\n  {listing}\n\n"
        "Everything under tests/fixtures/ is committed, so these paths are "
        "wrong rather than absent. Fix the paths; do not skip the test. "
        "See docs/TESTING.md §1.11."
    )


def install_lint_script(script: Path, scripts_dir: Path) -> Path:
    """Copy a ``scripts/check_*.py`` lint into ``scripts_dir`` and return it.

    Every lint test drives its script against a synthetic tree, which means
    reproducing the layout the script expects: it derives the repository root
    from its own location, and it imports ``scripts/_lint_common.py`` from
    ``sys.path[0]``. Copying the lint alone gives a tree where every test
    fails on ImportError, so the shared module travels with it.

    Args:
        script: The real lint under ``scripts/``.
        scripts_dir: The synthetic ``scripts/`` directory, created if absent.

    Returns:
        Path to the copied lint, ready to run.
    """
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_LINT_COMMON, scripts_dir / _LINT_COMMON.name)
    destination = scripts_dir / script.name
    shutil.copy2(script, destination)
    return destination


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
    from jamma.lmm.pab import compute_Uab

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


def make_runner_synthetic_data(
    n_samples: int = 100, n_snps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Create synthetic data for runner-level tests."""
    rng = np.random.default_rng(seed)
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
    kinship = (kinship + kinship.T) / 2
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]
    return genotypes, phenotypes, kinship, snp_info


def preflight(config, execution):  # type: ignore[no-untyped-def]
    """Run memory_preflight the way PipelineRunner.run does: from the resolved plan."""
    from jamma.pipeline_memory import memory_preflight
    from jamma.pipeline_plan import resolve_analysis_plan

    analysis = resolve_analysis_plan(
        config, execution=execution, snps_indices=None, ksnps_indices=None
    )
    return memory_preflight(analysis, check_memory=config.check_memory)
