"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

import ast
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

_TESTS_DIR = Path(__file__).resolve().parent


def _module_level_marker_names(tree: ast.Module) -> set[str]:
    """Return the set of marker names assigned to ``pytestmark`` at module level.

    Recognises both single-mark (``pytestmark = pytest.mark.tier0``) and
    list-of-marks (``pytestmark = [pytest.mark.tier0, pytest.mark.slow]``)
    forms. Anything else (computed expressions, function calls) is
    conservatively treated as no markers — the file should declare its
    classification statically.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "pytestmark"
        ):
            continue
        candidates: list[ast.expr] = []
        if isinstance(node.value, ast.List | ast.Tuple):
            candidates.extend(node.value.elts)
        else:
            candidates.append(node.value)
        for c in candidates:
            # pytest.mark.<name>
            if (
                isinstance(c, ast.Attribute)
                and isinstance(c.value, ast.Attribute)
                and isinstance(c.value.value, ast.Name)
                and c.value.value.id == "pytest"
                and c.value.attr == "mark"
            ):
                names.add(c.attr)
            # pytest.mark.<name>(...)
            elif (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Attribute)
                and isinstance(c.func.value.value, ast.Name)
                and c.func.value.value.id == "pytest"
                and c.func.value.attr == "mark"
            ):
                names.add(c.func.attr)
    return names


def _per_test_marker_names(tree: ast.Module) -> set[str]:
    """Return the set of @pytest.mark.<name> decorators on any function or class."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "pytest"
                and target.value.attr == "mark"
            ):
                names.add(target.attr)
    return names


def _file_declares_tier_marker(path: Path) -> bool:
    """Return True if ``path`` has at least one tier/slow/benchmark marker.

    Source-parsed (not collection-based) so the check is invariant under
    xdist, ``-k``, ``-m`` filters, and any other collection-time filtering.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        # Treat unparsable test files as marker-less so the gate flags
        # them; surfacing via the same channel keeps diagnostics together.
        return False
    if _module_level_marker_names(tree) & _REQUIRED_TIER_MARKERS:
        return True
    return bool(_per_test_marker_names(tree) & _REQUIRED_TIER_MARKERS)


def _enforce_tier_markers() -> None:
    """Source-parse every test file under ``tests/`` and fail on missing markers.

    Called from ``pytest_configure`` (before xdist forks workers) so the
    enforcement runs exactly once per session, regardless of distribution
    mode or CLI filters. The previous implementation used
    ``pytest_collection_modifyitems`` and was empirically a no-op under
    ``-n`` (xdist's controller hook receives an empty items list — see
    tests/test_conftest_tier_gate.py for the regression tests).
    """
    missing: list[Path] = []
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        if path.name in _TIER_MARKER_EXEMPT_FILES:
            continue
        if not _file_declares_tier_marker(path):
            missing.append(path)
    if missing:
        repo_root = _TESTS_DIR.parent
        listing = "\n  ".join(str(p.relative_to(repo_root)) for p in missing)
        raise pytest.UsageError(
            "The following test files have no tier marker "
            "(tier0/tier1/tier2/slow/benchmark):\n  "
            f"{listing}\n\n"
            "Add `pytestmark = pytest.mark.tier0` (or per-test markers). "
            "See docs/TESTING.md §1.6."
        )


def pytest_configure(config: pytest.Config) -> None:
    """Run session-start checks: tier-marker gate and stale-C-extension warn.

    The tier-marker gate runs in ``pytest_configure`` (not
    ``pytest_collection_modifyitems``) because xdist forks workers AFTER
    ``pytest_configure``; running the check here means it fires exactly
    once on the controller, before any partitioning. The previous
    collection-based hook silently no-op'd under xdist (controller's
    items list is empty; workers were skipped via ``workerinput`` guard).

    The stale-C-extension check is advisory: editable install picks up
    Python edits automatically, but C source edits require explicit
    rebuild. We warn rather than fail so the session still starts;
    pre-push hook (scripts/check_c_extension_freshness.py) is the
    blocking gate.
    """
    # xdist worker processes inherit ``pytest_configure`` invocations too.
    # Skip on workers — the controller already ran the gate, and a worker
    # raising UsageError mid-session would crash xdist.
    if not hasattr(config, "workerinput"):
        _enforce_tier_markers()

    # Import guarded: script lives outside the package and may be missing
    # in some install layouts (e.g. a sdist-only install). Missing script
    # is not a test failure — just skip the check.
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    if not (script_dir / "check_c_extension_freshness.py").exists():
        return
    sys.path.insert(0, str(script_dir))
    try:
        import check_c_extension_freshness as freshness
    except ImportError as exc:
        # The script exists on disk (we checked above) but failed to import.
        # That's a real bug — syntax error, broken refactor, missing dep.
        # We don't want to fail the whole session, but a silent return would
        # mask the bug indefinitely. Surface it instead.
        sys.stderr.write(
            f"\n\033[33m[jamma] WARNING: c-extension freshness check "
            f"could not be loaded ({type(exc).__name__}: {exc}). "
            f"Stale .so files will not be detected this session.\033[0m\n"
        )
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
