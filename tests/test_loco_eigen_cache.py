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


class TestLocoWriteEigenAutoDefault:
    """PipelineConfig defaults eigen_dir for LOCO write_eigen (Python API parity).

    The CLI defaults --eigen-dir to the output directory when -loco -eigen
    is set; the Python API (gwas / PipelineConfig) must do the same so that
    ``gwas(..., loco=True, write_eigen=True)`` does not raise. Regression
    test for the broken eigen-cache API.
    """

    def test_loco_write_eigen_defaults_eigen_dir_to_output_dir(self) -> None:
        """loco + write_eigen + no eigen_dir → eigen_dir becomes output_dir."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(
            bfile=Path("dummy"),
            lmm_mode=1,
            loco=True,
            write_eigen=True,
            output_dir=Path("out"),
        )
        assert config.eigen_dir == Path("out")

    def test_explicit_eigen_dir_not_overridden(self) -> None:
        """An explicitly supplied eigen_dir is preserved, not defaulted."""
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(
            bfile=Path("dummy"),
            lmm_mode=1,
            loco=True,
            write_eigen=True,
            output_dir=Path("out"),
            eigen_dir=Path("custom_eigen"),
        )
        assert config.eigen_dir == Path("custom_eigen")

    def test_non_loco_write_eigen_does_not_default_eigen_dir(self) -> None:
        """Standard (non-LOCO) write_eigen leaves eigen_dir as None.

        The non-LOCO path writes eigen files to output_dir directly and never
        consults eigen_dir, so it must stay None.
        """
        from jamma.pipeline import PipelineConfig

        config = PipelineConfig(
            bfile=Path("dummy"),
            lmm_mode=1,
            loco=False,
            write_eigen=True,
            output_dir=Path("out"),
        )
        assert config.eigen_dir is None


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestGwasLocoWriteEigen:
    """End-to-end: gwas(loco=True, write_eigen=True) without eigen_dir works.

    Regression for the broken Python eigen-cache API: this call previously
    raised ``ValueError: write_eigen=True requires eigen_dir to be set``
    because the API never defaulted eigen_dir the way the CLI does.
    """

    def test_gwas_loco_write_eigen_writes_eigen_to_output_dir(
        self, tmp_path: Path
    ) -> None:
        """No eigen_dir given: per-chr eigen files land in output_dir."""
        from jamma import gwas
        from jamma.io.plink import get_plink_metadata, partitions_from_metadata

        out_dir = tmp_path / "out"
        result = gwas(
            str(MOUSE_HS1940_BFILE),
            loco=True,
            write_eigen=True,
            output_dir=str(out_dir),
            check_memory=False,
            show_progress=False,
        )
        assert result.n_snps_tested > 0

        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        unique_chrs = sorted(partitions_from_metadata(meta).keys())
        for ch in unique_chrs:
            assert (out_dir / f"result.loco.chr{ch}.eigenD.npy").exists(), (
                f"Missing eigenD for chr {ch} in output_dir"
            )
            assert (out_dir / f"result.loco.chr{ch}.eigenU.npy").exists(), (
                f"Missing eigenU for chr {ch} in output_dir"
            )


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestLocoLegacyText:
    """LOCO honors legacy_text for kinship and eigen artifacts.

    Regression test for GEMMA_DIVERGENCES §13: --legacy-text was ignored on
    the LOCO path, producing .npy instead of GEMMA-compatible .txt files.
    """

    def test_run_lmm_loco_legacy_text_writes_txt_artifacts(
        self, tmp_path: Path
    ) -> None:
        """legacy_text=True writes .txt eigen and kinship files."""
        from jamma.io.plink import get_plink_metadata, partitions_from_metadata
        from jamma.lmm.loco import run_lmm_loco
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        unique_chrs = sorted(partitions_from_metadata(meta).keys())

        run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=tmp_path / "result.assoc.txt",
            check_memory=False,
            show_progress=False,
            save_kinship=True,
            kinship_output_dir=tmp_path,
            kinship_output_prefix="result",
            write_eigen=True,
            eigen_dir=tmp_path,
            eigen_prefix="result",
            legacy_text=True,
        )

        for ch in unique_chrs:
            assert (tmp_path / f"result.loco.chr{ch}.eigenD.txt").exists(), (
                f"Missing .txt eigenD for chr {ch}"
            )
            assert (tmp_path / f"result.loco.chr{ch}.eigenU.txt").exists(), (
                f"Missing .txt eigenU for chr {ch}"
            )
            assert (tmp_path / f"result.loco.cXX.chr{ch}.txt").exists(), (
                f"Missing .txt kinship for chr {ch}"
            )


# ---------------------------------------------------------------------------
# Content + parameter cache-key tests (manifest-based stale-cache detection)
# ---------------------------------------------------------------------------


def _write_dummy_plink(
    prefix: Path,
    *,
    bed_size: int = 64,
    bim_lines: list[str] | None = None,
    bed_fill: int = 0,
) -> None:
    """Write minimal .bed/.bim files at ``prefix`` for cache-key unit tests.

    The cache-key function only stats .bed (name + size + mtime) and hashes
    .bim content, so these need not be valid PLINK binaries.
    """
    if bim_lines is None:
        bim_lines = [
            "1\trs1\t0\t100\tA\tG",
            "1\trs2\t0\t200\tC\tT",
            "2\trs3\t0\t300\tA\tT",
        ]
    prefix.with_suffix(".bed").write_bytes(bytes([bed_fill]) * bed_size)
    prefix.with_suffix(".bim").write_text("\n".join(bim_lines) + "\n")


def _compute_key(
    prefix: Path,
    *,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    valid_mask: np.ndarray | None = None,
    ksnps_indices: np.ndarray | None = None,
) -> str:
    """Call compute_eigen_cache_key with per-test overrides over fixed defaults.

    Explicit keyword forwarding (not dict-unpacking) keeps the call type-clean.
    """
    from jamma.lmm.eigen_cache import compute_eigen_cache_key

    if valid_mask is None:
        valid_mask = np.ones(20, dtype=bool)
    key, _components = compute_eigen_cache_key(
        prefix,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        valid_mask=valid_mask,
        ksnps_indices=ksnps_indices,
    )
    return key


class TestEigenCacheKey:
    """compute_eigen_cache_key changes iff a real eigen-pair determinant changes."""

    def test_key_is_stable_for_identical_inputs(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix)
        k2 = _compute_key(prefix)
        assert isinstance(k1, str)
        assert len(k1) > 0
        assert k1 == k2

    def test_key_changes_when_maf_threshold_changes(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        assert _compute_key(prefix) != _compute_key(prefix, maf_threshold=0.05)

    def test_key_changes_when_miss_threshold_changes(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        assert _compute_key(prefix) != _compute_key(prefix, miss_threshold=0.10)

    def test_key_changes_when_valid_mask_positions_change(self, tmp_path: Path) -> None:
        """Same valid COUNT, different valid POSITIONS -> different key.

        This is the silent-stale hole the manifest closes: two phenotypes with
        the same number of non-missing samples but a different missingness
        pattern select a different sample subset, hence a different K.
        """
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        m1 = np.ones(20, dtype=bool)
        m1[0] = False
        m2 = np.ones(20, dtype=bool)
        m2[1] = False
        assert int(m1.sum()) == int(m2.sum())  # same count
        k1 = _compute_key(prefix, valid_mask=m1)
        k2 = _compute_key(prefix, valid_mask=m2)
        assert k1 != k2

    def test_key_changes_when_valid_mask_length_changes(self, tmp_path: Path) -> None:
        """Different total sample count (.fam size) -> different key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix, valid_mask=np.ones(20, dtype=bool))
        k2 = _compute_key(prefix, valid_mask=np.ones(19, dtype=bool))
        assert k1 != k2

    def test_key_changes_when_bim_content_changes(self, tmp_path: Path) -> None:
        """Re-annotating a SNP's chromosome changes the LOCO partition -> key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix)
        _write_dummy_plink(
            prefix,
            bim_lines=[
                "1\trs1\t0\t100\tA\tG",
                "1\trs2\t0\t200\tC\tT",
                "3\trs3\t0\t300\tA\tT",  # chr 2 -> 3
            ],
        )
        assert k1 != _compute_key(prefix)

    def test_key_changes_when_bed_content_changes(self, tmp_path: Path) -> None:
        """A different .bed (here: different size) -> different key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix, bed_size=64)
        k1 = _compute_key(prefix)
        _write_dummy_plink(prefix, bed_size=128)
        assert k1 != _compute_key(prefix)

    def test_key_changes_when_ksnps_changes(self, tmp_path: Path) -> None:
        """Different kinship-SNP restriction -> different key; None differs too."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k_none = _compute_key(prefix)
        k_a = _compute_key(prefix, ksnps_indices=np.array([0, 1]))
        k_b = _compute_key(prefix, ksnps_indices=np.array([0, 2]))
        assert k_none != k_a
        assert k_none != k_b
        assert k_a != k_b

    def test_returns_canonical_components(self, tmp_path: Path) -> None:
        """Second return value is the exact hashed payload (for the manifest)."""
        import hashlib
        import json

        from jamma.lmm.eigen_cache import compute_eigen_cache_key

        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        key, components = compute_eigen_cache_key(
            prefix,
            maf_threshold=0.01,
            miss_threshold=0.05,
            valid_mask=np.ones(20, dtype=bool),
        )
        assert isinstance(components, dict)
        hashed_keys = {"bed_fingerprint", "bim_sha256", "valid_mask_sha256"}
        assert hashed_keys <= components.keys()
        canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert key == expected


class TestEigenCacheManifest:
    """Manifest read/write/validate behavior for stale-cache detection."""

    def test_absent_manifest_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import eigen_cache_is_valid

        ok, reason = eigen_cache_is_valid(tmp_path, "result", "somekey")
        assert ok is False
        assert "manifest" in reason.lower()

    def test_matching_key_is_valid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(tmp_path, "result", "KEY123", components={})
        ok, _reason = eigen_cache_is_valid(tmp_path, "result", "KEY123")
        assert ok is True

    def test_mismatched_key_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(tmp_path, "result", "KEY123", components={})
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "DIFFERENT")
        assert ok is False
        assert reason

    def test_manifest_roundtrip(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            read_eigen_cache_manifest,
            write_eigen_cache_manifest,
        )

        path = write_eigen_cache_manifest(
            tmp_path, "result", "KEY123", components={"maf_threshold": 0.01}
        )
        assert path.exists()
        manifest = read_eigen_cache_manifest(tmp_path, "result")
        assert manifest is not None
        assert manifest["cache_key"] == "KEY123"
        assert manifest["components"] == {"maf_threshold": 0.01}

    def test_corrupt_manifest_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            eigen_cache_manifest_path,
        )

        eigen_cache_manifest_path(tmp_path, "result").write_text("{ not json")
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "KEY")
        assert ok is False
        assert reason

    def test_invalidate_removes_present_manifest_and_no_ops_when_absent(
        self, tmp_path: Path
    ) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_manifest_path,
            invalidate_eigen_cache_manifest,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(tmp_path, "result", "KEY", components={})
        manifest = eigen_cache_manifest_path(tmp_path, "result")
        assert manifest.exists()

        invalidate_eigen_cache_manifest(tmp_path, "result")
        assert manifest.exists() is False

        invalidate_eigen_cache_manifest(tmp_path, "result")
        assert manifest.exists() is False


@pytest.mark.slow
@pytest.mark.skipif(
    not _mouse_hs1940_exists(), reason="mouse_hs1940 fixture not available"
)
class TestLocoEigenCacheStaleDetection:
    """End-to-end: a cache whose inputs changed must NOT be silently reused."""

    def test_changed_filter_invalidates_cache(self, tmp_path: Path) -> None:
        """Cache written at maf=0.01, then read with maf=0.05, must recompute.

        Proven by equality to a from-scratch maf=0.05 run: if the stale maf=0.01
        eigen cache were silently reused, results would differ.
        """
        from jamma.lmm.loco import run_lmm_loco
        from jamma.validation.compare import load_gemma_assoc
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        eigen_dir = tmp_path / "eigen_cache"
        eigen_dir.mkdir()

        common = {
            "bed_path": MOUSE_HS1940_BFILE,
            "phenotypes": phenotypes,
            "lmm_mode": 1,
            "check_memory": False,
            "show_progress": False,
            "miss_threshold": 0.05,
        }

        out_fresh = tmp_path / "fresh.assoc.txt"
        run_lmm_loco(**common, maf_threshold=0.05, output_path=out_fresh)

        run_lmm_loco(
            **common,
            maf_threshold=0.01,
            output_path=tmp_path / "populate.assoc.txt",
            write_eigen=True,
            eigen_dir=eigen_dir,
        )

        out_cached = tmp_path / "cached.assoc.txt"
        run_lmm_loco(
            **common,
            maf_threshold=0.05,
            output_path=out_cached,
            eigen_dir=eigen_dir,
        )

        fresh = {r.rs: r for r in load_gemma_assoc(out_fresh)}
        cached = {r.rs: r for r in load_gemma_assoc(out_cached)}
        assert fresh
        assert cached
        common_rs = set(fresh) & set(cached)
        assert common_rs
        for rs in common_rs:
            b_fresh = fresh[rs].beta
            b_cached = cached[rs].beta
            assert b_fresh is not None
            assert b_cached is not None
            np.testing.assert_allclose(
                b_cached,
                b_fresh,
                rtol=1e-8,
                atol=1e-14,
                err_msg=f"beta {rs}: stale maf=0.01 cache silently reused",
            )

    def test_interrupted_rewrite_leaves_no_stale_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupted write_eigen rewrite must leave no manifest.

        Run 1 writes a complete maf=0.01 cache + manifest. Run 2 rewrites with
        maf=0.05 but is interrupted on the second chromosome. The maf=0.01
        manifest must already be gone (invalidated before the loop), so a later
        read with the maf=0.01 inputs cannot validate the half-rewritten cache.
        """
        import jamma.lmm.loco as loco_mod
        from jamma.lmm.eigen_cache import eigen_cache_manifest_path
        from jamma.lmm.loco import run_lmm_loco
        from tests.conftest import load_phenotypes_from_fam

        fam_path = MOUSE_HS1940_BFILE.with_suffix(".fam")
        phenotypes = load_phenotypes_from_fam(fam_path)
        eigen_dir = tmp_path / "eigen_cache"
        eigen_dir.mkdir()

        common = {
            "bed_path": MOUSE_HS1940_BFILE,
            "phenotypes": phenotypes,
            "lmm_mode": 1,
            "check_memory": False,
            "show_progress": False,
            "miss_threshold": 0.05,
        }

        run_lmm_loco(
            **common,
            maf_threshold=0.01,
            output_path=tmp_path / "populate.assoc.txt",
            write_eigen=True,
            eigen_dir=eigen_dir,
        )
        manifest = eigen_cache_manifest_path(eigen_dir, "result")
        assert manifest.exists()

        real_write_eigen_files = loco_mod.write_eigen_files
        calls = {"n": 0}

        def interrupting_write_eigen_files(
            eigenvalues: np.ndarray,
            eigenvectors: np.ndarray,
            output_dir: Path,
            prefix: str = "result",
            *,
            legacy_text: bool = False,
        ) -> tuple[Path, Path]:
            calls["n"] += 1
            if calls["n"] == 1:
                return real_write_eigen_files(
                    eigenvalues,
                    eigenvectors,
                    output_dir,
                    prefix=prefix,
                    legacy_text=legacy_text,
                )
            raise RuntimeError("simulated interruption")

        monkeypatch.setattr(
            loco_mod, "write_eigen_files", interrupting_write_eigen_files
        )

        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_lmm_loco(
                **common,
                maf_threshold=0.05,
                output_path=tmp_path / "interrupted.assoc.txt",
                write_eigen=True,
                eigen_dir=eigen_dir,
            )

        assert calls["n"] >= 2, "interruption did not run the real writer first"
        assert manifest.exists() is False
