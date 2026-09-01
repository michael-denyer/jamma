"""Pipeline-level eigendecomposition reuse and publication tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io import read_fam_phenotypes
from jamma.kinship import read_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.pipeline import PipelineConfig, PipelineRunner
from tests.fixture_paths import MOUSE

MOUSE_BFILE = MOUSE.bfile
MOUSE_KINSHIP_FILE = MOUSE.kinship


@pytest.mark.slow
@pytest.mark.tier1
class TestLMMEquivalence:
    """Loaded eigen data produces the same LMM result as fresh decomposition."""

    def test_loaded_eigen_matches_fresh_eigen_lmm(self, tmp_path: Path) -> None:
        fresh_dir = tmp_path / "fresh"
        fresh_result = PipelineRunner(
            PipelineConfig(
                bfile=MOUSE_BFILE,
                kinship_file=MOUSE_KINSHIP_FILE,
                output_dir=fresh_dir,
                output_prefix="fresh",
                check_memory=False,
                show_progress=False,
            )
        ).run()

        from jamma.io.plink import get_plink_metadata

        metadata = get_plink_metadata(MOUSE_BFILE)
        kinship = read_kinship_matrix(MOUSE_KINSHIP_FILE, n_samples=metadata.n_samples)
        phenotype = read_fam_phenotypes(MOUSE.fam)
        valid_mask = ~np.isnan(phenotype) & (phenotype != -9.0)
        eigenvalues, eigenvectors = eigendecompose_kinship(
            kinship[np.ix_(valid_mask, valid_mask)]
        )
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path / "eigen", prefix="test"
        )

        loaded_dir = tmp_path / "loaded"
        loaded_result = PipelineRunner(
            PipelineConfig(
                bfile=MOUSE_BFILE,
                eigenvalue_file=d_path,
                eigenvector_file=u_path,
                output_dir=loaded_dir,
                output_prefix="loaded",
                check_memory=False,
                show_progress=False,
            )
        ).run()

        from jamma.validation import compare_assoc_results, load_gemma_assoc

        assert fresh_result.n_samples == loaded_result.n_samples
        assert fresh_result.n_snps_tested == loaded_result.n_snps_tested
        comparison = compare_assoc_results(
            load_gemma_assoc(loaded_dir / "loaded.assoc.txt"),
            load_gemma_assoc(fresh_dir / "fresh.assoc.txt"),
        )
        assert comparison.passed, comparison

    def test_write_eigen_flag_creates_files(self, tmp_path: Path) -> None:
        PipelineRunner(
            PipelineConfig(
                bfile=MOUSE_BFILE,
                kinship_file=MOUSE_KINSHIP_FILE,
                output_dir=tmp_path,
                output_prefix="test",
                check_memory=False,
                show_progress=False,
                write_eigen=True,
            )
        ).run()

        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"
        assert d_path.stat().st_size > 0
        assert u_path.stat().st_size > 0
        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
        assert eigenvalues.shape[0] == eigenvectors.shape[0]
        assert eigenvectors.shape[0] == eigenvectors.shape[1]
