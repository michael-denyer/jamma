"""Tests for LOCO per-chromosome eigendecomposition caching.

Validates:
- _find_loco_eigen_cache: complete cache, partial cache rejection, .txt fallback,
  non-directory guard
- run_lmm_loco write_eigen: per-chr eigen file creation with correct dimensions
- write_eigen=True without eigen_dir raises ValueError
- Dimension mismatch on cached read raises ValueError
- Cached LOCO run produces identical results to non-cached run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.loco import _find_loco_eigen_cache

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MOUSE_HS1940_DIR = _FIXTURE_ROOT / "mouse_hs1940"
MOUSE_HS1940_BFILE = MOUSE_HS1940_DIR / "mouse_hs1940"


def _mouse_hs1940_exists() -> bool:
    return MOUSE_HS1940_BFILE.with_suffix(".bed").exists()


# ---------------------------------------------------------------------------
# _find_loco_eigen_cache tests
# ---------------------------------------------------------------------------


class TestFindLocoEigenCache:
    """Tests for _find_loco_eigen_cache helper function."""

    def test_non_directory_returns_none(self, tmp_path: Path) -> None:
        """Passing a file path instead of directory returns None."""
        fake_file = tmp_path / "not_a_dir.txt"
        fake_file.write_text("hello")
        result = _find_loco_eigen_cache(fake_file, "result", ["1", "2"])
        assert result is None

    def test_complete_cache_returns_dict(self, tmp_path: Path) -> None:
        """When all per-chr .npy files exist, returns dict mapping chr -> (d, u)."""
        n = 10
        chr_names = ["1", "2", "3"]
        prefix = "result"

        # Write per-chr eigen files
        for ch in chr_names:
            eigenvalues = np.random.default_rng(42).random(n)
            eigenvectors = np.eye(n)
            write_eigen_files(
                eigenvalues,
                eigenvectors,
                tmp_path,
                prefix=f"{prefix}.loco.chr{ch}",
            )

        result = _find_loco_eigen_cache(tmp_path, prefix, chr_names)
        assert result is not None
        assert set(result.keys()) == set(chr_names)
        for ch in chr_names:
            d_path, u_path = result[ch]
            assert d_path.exists()
            assert u_path.exists()

    def test_partial_cache_returns_none(self, tmp_path: Path) -> None:
        """When some chromosomes are missing, returns None."""
        n = 10
        chr_names = ["1", "2", "3"]
        prefix = "result"

        # Only write files for chr 1 and 2
        for ch in ["1", "2"]:
            eigenvalues = np.random.default_rng(42).random(n)
            eigenvectors = np.eye(n)
            write_eigen_files(
                eigenvalues,
                eigenvectors,
                tmp_path,
                prefix=f"{prefix}.loco.chr{ch}",
            )

        result = _find_loco_eigen_cache(tmp_path, prefix, chr_names)
        assert result is None

    def test_legacy_text_fallback(self, tmp_path: Path) -> None:
        """When legacy_text=True, checks for .txt files instead of .npy."""
        n = 10
        chr_names = ["1", "2"]
        prefix = "result"

        # Write as legacy text
        for ch in chr_names:
            eigenvalues = np.random.default_rng(42).random(n)
            eigenvectors = np.eye(n)
            write_eigen_files(
                eigenvalues,
                eigenvectors,
                tmp_path,
                prefix=f"{prefix}.loco.chr{ch}",
                legacy_text=True,
            )

        result = _find_loco_eigen_cache(tmp_path, prefix, chr_names, legacy_text=True)
        assert result is not None
        assert set(result.keys()) == set(chr_names)

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        """Empty directory returns None (no cache found)."""
        result = _find_loco_eigen_cache(tmp_path, "result", ["1", "2"])
        assert result is None


# ---------------------------------------------------------------------------
# run_lmm_loco write_eigen tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestLocoWriteEigen:
    """Tests for run_lmm_loco with write_eigen=True."""

    def test_write_eigen_produces_correct_per_chr_files(self, tmp_path: Path) -> None:
        """write_eigen=True writes eigenD/eigenU per chromosome with correct dims."""
        from jamma.io.plink import get_plink_metadata, partitions_from_metadata
        from jamma.lmm.loco import run_lmm_loco
        from jamma.lmm.prepare_common import compute_valid_mask
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        partitions = partitions_from_metadata(meta)
        unique_chrs = sorted(partitions.keys())

        valid_mask = compute_valid_mask(phenotypes, None)
        n_valid = int(np.sum(valid_mask))

        result = run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=tmp_path / "result.assoc.txt",
            check_memory=False,
            show_progress=False,
            write_eigen=True,
            eigen_dir=tmp_path,
            eigen_prefix="result",
        )
        assert result.n_tested > 0

        # Verify per-chromosome eigen files exist with correct dimensions
        for ch in unique_chrs:
            d_path = tmp_path / f"result.loco.chr{ch}.eigenD.npy"
            u_path = tmp_path / f"result.loco.chr{ch}.eigenU.npy"
            assert d_path.exists(), f"Missing eigenD for chr {ch}"
            assert u_path.exists(), f"Missing eigenU for chr {ch}"
            eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
            assert eigenvalues.shape == (n_valid,)
            assert eigenvectors.shape == (n_valid, n_valid)


# ---------------------------------------------------------------------------
# Integration tests: write + read cache round-trip
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestLocoEigenCacheIntegration:
    """End-to-end tests for LOCO eigen cache write/read cycle."""

    def test_cached_run_produces_identical_results(self, tmp_path: Path) -> None:
        """Write eigen, then read cache: results must be numerically identical."""
        from jamma.lmm.loco import run_lmm_loco
        from jamma.validation.compare import load_gemma_assoc
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        eigen_dir = tmp_path / "eigen_cache"
        eigen_dir.mkdir()

        # Run 1: compute + write eigen
        out1 = tmp_path / "run1.assoc.txt"
        run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=out1,
            check_memory=False,
            show_progress=False,
            write_eigen=True,
            eigen_dir=eigen_dir,
            eigen_prefix="result",
        )

        # Run 2: read cache (no kinship/eigendecomp)
        out2 = tmp_path / "run2.assoc.txt"
        run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=out2,
            check_memory=False,
            show_progress=False,
            eigen_dir=eigen_dir,
            eigen_prefix="result",
        )

        # Compare results — both should produce non-empty output
        results1 = load_gemma_assoc(out1)
        results2 = load_gemma_assoc(out2)
        assert len(results1) > 0
        assert len(results2) > 0

        # Build lookup by rs for the cached run
        r2_by_rs = {r.rs: r for r in results2}

        # All common SNPs should have identical statistics
        common_count = 0
        for r1 in results1:
            r2 = r2_by_rs.get(r1.rs)
            if r2 is None:
                continue
            common_count += 1
            np.testing.assert_allclose(
                r1.beta,
                r2.beta,
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"beta mismatch for {r1.rs}",
            )
            np.testing.assert_allclose(
                r1.se,
                r2.se,
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"se mismatch for {r1.rs}",
            )
            np.testing.assert_allclose(
                r1.p_wald,
                r2.p_wald,
                rtol=1e-8,
                atol=1e-14,
                err_msg=f"p_wald mismatch for {r1.rs}",
            )

        min_expected = int(0.99 * min(len(results1), len(results2)))
        assert common_count >= min_expected, (
            f"Only {common_count} common SNPs out of "
            f"{len(results1)}/{len(results2)} — expected >= {min_expected}"
        )


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestLocoEigenCacheFallback:
    """Integration tests for LOCO eigen cache fallback behavior."""

    def test_empty_eigen_dir_falls_back_to_compute(self, tmp_path: Path) -> None:
        """LOCO with eigen_dir pointing to empty dir runs normally."""
        from jamma.lmm.loco import run_lmm_loco
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        empty_dir = tmp_path / "empty_eigen"
        empty_dir.mkdir()

        result = run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=tmp_path / "result.assoc.txt",
            check_memory=False,
            show_progress=False,
            eigen_dir=empty_dir,
            eigen_prefix="result",
        )
        assert result.n_tested > 0

    def test_partial_cache_falls_back_to_compute(self, tmp_path: Path) -> None:
        """Partial cache (some chrs missing) falls back to full compute."""
        from jamma.io.plink import get_plink_metadata, partitions_from_metadata
        from jamma.lmm.loco import run_lmm_loco
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        partitions = partitions_from_metadata(meta)
        unique_chrs = sorted(partitions.keys())

        # First run: write all eigen files
        eigen_dir = tmp_path / "partial_eigen"
        eigen_dir.mkdir()
        run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=tmp_path / "full.assoc.txt",
            check_memory=False,
            show_progress=False,
            write_eigen=True,
            eigen_dir=eigen_dir,
            eigen_prefix="result",
        )

        # Delete one chromosome's files to simulate partial cache
        first_chr = unique_chrs[0]
        (eigen_dir / f"result.loco.chr{first_chr}.eigenD.npy").unlink()
        (eigen_dir / f"result.loco.chr{first_chr}.eigenU.npy").unlink()

        # Run with partial cache: should fall back to full compute
        result = run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=tmp_path / "partial.assoc.txt",
            check_memory=False,
            show_progress=False,
            eigen_dir=eigen_dir,
            eigen_prefix="result",
        )
        assert result.n_tested > 0


class TestLocoEigenCacheValidation:
    """Validation and error tests for LOCO eigen cache."""

    def test_write_eigen_without_eigen_dir_raises(self) -> None:
        """write_eigen=True with eigen_dir=None raises ValueError."""
        from jamma.lmm.loco import run_lmm_loco
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        if not fam_path.exists():
            pytest.skip("mouse_hs1940 fixture not available")
        phenotypes = load_phenotypes_from_fam(fam_path)

        with pytest.raises(ValueError, match="write_eigen=True requires eigen_dir"):
            run_lmm_loco(
                bed_path=MOUSE_HS1940_BFILE,
                phenotypes=phenotypes,
                lmm_mode=1,
                write_eigen=True,
                eigen_dir=None,
            )

    def test_dimension_mismatch_on_cached_eigen_raises(self, tmp_path: Path) -> None:
        """Cached eigen with wrong n_samples raises ValueError with chr context."""
        # Write eigen files for n=10 samples
        n_written = 10
        chr_names = ["1", "2"]
        for ch in chr_names:
            eigenvalues = np.random.default_rng(42).random(n_written)
            eigenvectors = np.eye(n_written)
            write_eigen_files(
                eigenvalues,
                eigenvectors,
                tmp_path,
                prefix=f"result.loco.chr{ch}",
            )

        # read_eigen_files with wrong n_samples should raise
        d_path = tmp_path / "result.loco.chr1.eigenD.npy"
        u_path = tmp_path / "result.loco.chr1.eigenU.npy"
        with pytest.raises(ValueError, match="have 10 samples but pipeline expects 20"):
            read_eigen_files(d_path, u_path, n_samples=20)


class TestEigenDirCLI:
    """Tests for --eigen-dir CLI flag."""

    def test_eigen_dir_accepted_by_pipeline_config(self) -> None:
        """PipelineConfig accepts eigen_dir field."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(
            bfile=Path("dummy"),
            lmm_mode=1,
            loco=True,
            eigen_dir=Path("/tmp/eigen"),
        )
        assert config.eigen_dir == Path("/tmp/eigen")

    def test_eigen_dir_none_by_default(self) -> None:
        """PipelineConfig.eigen_dir is None by default."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(
            bfile=Path("dummy"),
            lmm_mode=1,
        )
        assert config.eigen_dir is None
