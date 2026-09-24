from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from bed_reader import to_bed

from tests.fixture_paths import SYNTHETIC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_c_extension_freshness as freshness


@pytest.fixture
def math_evidence_dir(tmp_path, request):
    """Keep each test's judged bundle for CI, or use its temporary directory."""
    if request.node.get_closest_marker("tier1") is None:
        raise pytest.UsageError(
            f"GEMMA comparison must be tier1: {request.node.nodeid}"
        )
    root = os.environ.get("JAMMA_MATH_EVIDENCE_DIR")
    if root is None:
        return tmp_path / "evidence"
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.nodeid)
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="attempt-", dir=directory)) / "bundle"


def pytest_configure(config: pytest.Config) -> None:
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


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from test fixtures.

    Returns:
        Path prefix for gemma_synthetic PLINK files (without .bed/.bim/.fam extension)
    """
    return SYNTHETIC.bfile


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
def asymmetric_plink(tmp_path: Path) -> Path:
    """SNPs cross MAF/missingness/monomorphism thresholds when rows are dropped."""
    rng = np.random.default_rng(327)
    genotypes = rng.binomial(2, np.linspace(0.05, 0.5, 61), (80, 61)).astype(float)
    genotypes[rng.random(genotypes.shape) < 0.04] = np.nan
    genotypes[:40, 0] = 0  # Polymorphic only outside the retained population.
    genotypes[40:, 0] = 2
    genotypes[:40, 1] = np.nan  # Missingness differs between populations.
    genotypes[:, 2] = 1  # Globally monomorphic.
    genotypes[:, 3] = np.nan  # Globally missing.
    bfile = tmp_path / "asymmetric"
    to_bed(
        bfile.with_suffix(".bed"),
        genotypes,
        properties={"chromosome": ["1"] * 20 + ["2"] * 20 + ["3"] * 21},
    )
    return bfile


@pytest.fixture
def no_c_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the C extension out for this test, so the NumPy path runs for real.

    ``jamma.lmm.accel.available()`` reads ``accel._accel`` at call time, not
    at import time, so clearing it
    here drives the fallback path rather than merely describing it. Every
    module that decides on C-vs-NumPy reads through ``accel``, so this one
    monkeypatch is the whole seam.
    """
    from jamma.lmm import accel

    monkeypatch.setattr(accel, "_accel", None)


@pytest.fixture
def synthetic_data_with_covariates(synthetic_data):
    """Load gemma_synthetic data plus covariates from gemma_covariate fixture.

    The covariates.txt file already includes the intercept column (first column
    is all 1.0), matching GEMMA's internal representation when -c is used.
    """
    genotypes, kinship, phenotypes, snp_info = synthetic_data
    covariates = np.loadtxt(SYNTHETIC.covariates)
    return genotypes, kinship, phenotypes, snp_info, covariates


@pytest.fixture
def synthetic_data():
    """Load gemma_synthetic float32 genotypes, kinship, phenotypes, and SnpMeta."""
    from jamma.genotype.dataset import GenotypeDataset
    from jamma.io import read_fam_phenotypes
    from jamma.kinship.io import read_kinship_matrix
    from tests.builders import read_plink_genotypes

    genotypes = read_plink_genotypes(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    snp_info = GenotypeDataset.open_plink(SYNTHETIC.bfile).variants
    return genotypes, kinship, phenotypes, snp_info
