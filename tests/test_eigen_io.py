"""Tests for eigendecomposition file I/O and reuse.

Validates:
- GEMMA-compatible file format (.10g precision, no headers)
- Round-trip precision for eigenvalues and eigenvectors
- Dimension validation on read
- Edge cases (empty files, single value, nested dirs)
- LMM equivalence between fresh and loaded eigendecomposition
- Flag interaction rules (-d/-u pairing, -loco incompatibility)
- CLI help output for new flags
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.eigen_io import (
    _load_npy_cache,
    _write_npy_cache,
    read_eigen_files,
    read_eigenvalues,
    read_eigenvectors,
    write_eigen_files,
    write_eigenvalues,
    write_eigenvectors,
)
from tests.conftest import load_phenotypes_from_fam

# =============================================================================
# File format tests
# =============================================================================


@pytest.mark.tier0
class TestEigenvalueFormat:
    """Verify eigenvalue file format matches GEMMA .eigenD.txt."""

    def test_write_eigenvalues_format(self, tmp_path: Path) -> None:
        """Eigenvalue file: one value per line, .10g format, no header (legacy_text)."""
        values = np.array([0.001, 1.0, 2.5, 100.0, 12345.6789012345])
        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(values, path, legacy_text=True)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

        # Each line should be the .10g formatted value
        for i, line in enumerate(lines):
            expected = f"{values[i]:.10g}"
            assert line == expected, f"Line {i}: got {line!r}, expected {expected!r}"

    def test_write_eigenvectors_format(self, tmp_path: Path) -> None:
        """Eigenvector file: tab-separated, .10g format, no header (legacy_text)."""
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        path = tmp_path / "test.eigenU.txt"
        write_eigenvectors(matrix, path, legacy_text=True)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

        for i, line in enumerate(lines):
            parts = line.split("\t")
            assert len(parts) == 3
            for j, part in enumerate(parts):
                expected = f"{matrix[i, j]:.10g}"
                assert part == expected

    def test_eigenvalues_ascending_order_preserved(self, tmp_path: Path) -> None:
        """Ascending eigenvalue order from eigh is preserved through write/read."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 20))
        sym = A + A.T
        eigenvalues, _ = np.linalg.eigh(sym)

        # eigh returns ascending order
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

        path = tmp_path / "test.eigenD.npy"
        write_eigenvalues(eigenvalues, path)
        loaded = read_eigenvalues(path)

        # Order is preserved
        assert np.all(loaded[:-1] <= loaded[1:])
        np.testing.assert_array_equal(loaded, eigenvalues)


# =============================================================================
# Round-trip precision tests
# =============================================================================


@pytest.mark.tier0
class TestRoundTripPrecision:
    """Verify .10g format preserves sufficient precision for LMM."""

    def test_eigenvalue_round_trip_precision(self, tmp_path: Path) -> None:
        """100 random eigenvalues survive binary write/read with exact equality."""
        rng = np.random.default_rng(123)
        # Generate eigenvalues spanning several orders of magnitude
        original = np.sort(rng.uniform(0.001, 1000.0, size=100))

        path = tmp_path / "eigenD.npy"
        write_eigenvalues(original, path)
        loaded = read_eigenvalues(path)

        np.testing.assert_array_equal(loaded, original)

    def test_eigenvector_round_trip_precision(self, tmp_path: Path) -> None:
        """50x50 orthogonal matrix survives binary write/read with exact equality."""
        rng = np.random.default_rng(456)
        A = rng.standard_normal((50, 50))
        sym = A + A.T
        _, eigenvectors = np.linalg.eigh(sym)

        # Eigenvectors from eigh are orthonormal
        path = tmp_path / "eigenU.npy"
        write_eigenvectors(eigenvectors, path)
        loaded = read_eigenvectors(path)

        np.testing.assert_array_equal(loaded, eigenvectors)

    def test_eigen_files_round_trip(self, tmp_path: Path) -> None:
        """write_eigen_files + read_eigen_files round-trip both arrays exactly."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((30, 30))
        psd = A @ A.T  # PSD → non-negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(psd)

        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="roundtrip"
        )

        loaded_d, loaded_u = read_eigen_files(d_path, u_path)

        np.testing.assert_array_equal(loaded_d, eigenvalues)
        np.testing.assert_array_equal(loaded_u, eigenvectors)


# =============================================================================
# Dimension validation tests
# =============================================================================


@pytest.mark.tier0
class TestDimensionValidation:
    """Verify read_eigen_files catches dimension mismatches."""

    def test_read_eigen_files_dimension_mismatch(self, tmp_path: Path) -> None:
        """Mismatched eigenvalue count vs eigenvector dimensions raises ValueError."""
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(8), u_path)

        with pytest.raises(ValueError, match="does not match"):
            read_eigen_files(d_path, u_path)

    def test_read_eigen_files_n_samples_mismatch(self, tmp_path: Path) -> None:
        """n_samples validation catches wrong expected count."""
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        with pytest.raises(ValueError, match="pipeline expects 12"):
            read_eigen_files(d_path, u_path, n_samples=12)

    def test_read_eigen_files_consistent_dimensions(self, tmp_path: Path) -> None:
        """Consistent eigen pair with matching n_samples succeeds."""
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path, n_samples=10)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)

    def test_read_eigen_files_no_n_samples_validation(self, tmp_path: Path) -> None:
        """Omitting n_samples skips that validation."""
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.tier0
class TestEdgeCases:
    """Edge case handling for eigen I/O."""

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Writing to nested path creates parent directories."""
        nested = tmp_path / "a" / "b" / "c" / "test.eigenD.npy"
        write_eigenvalues(np.array([1.0, 2.0]), nested)
        assert nested.exists()
        loaded = read_eigenvalues(nested)
        assert len(loaded) == 2

    def test_read_eigenvalues_empty_file(self, tmp_path: Path) -> None:
        """Empty eigenD file raises ValueError."""
        path = tmp_path / "empty.eigenD.txt"
        path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_eigenvalues(path)

    def test_write_read_single_eigenvalue(self, tmp_path: Path) -> None:
        """1x1 matrix edge case preserves correct shapes."""
        eigenvalues = np.array([3.14])
        eigenvectors = np.array([[1.0]])

        d_path = tmp_path / "single.eigenD.npy"
        u_path = tmp_path / "single.eigenU.npy"

        write_eigenvalues(eigenvalues, d_path)
        write_eigenvectors(eigenvectors, u_path)

        loaded_d = read_eigenvalues(d_path)
        loaded_u = read_eigenvectors(u_path)

        # Readers guarantee correct shapes
        assert loaded_d.shape == (1,)
        np.testing.assert_array_equal(loaded_d, [3.14])

        assert loaded_u.shape == (1, 1)
        np.testing.assert_array_equal(loaded_u, [[1.0]])


# =============================================================================
# Reader validation tests
# =============================================================================


@pytest.mark.tier0
class TestReaderValidation:
    """Verify individual readers catch parse errors and bad shapes."""

    def test_read_eigenvalues_unparsable_includes_path(self, tmp_path: Path) -> None:
        """Non-numeric eigenvalue file includes path in error."""
        path = tmp_path / "bad.eigenD.txt"
        path.write_text("1.0\nhello\n3.0\n")

        with pytest.raises(ValueError, match=str(path)):
            read_eigenvalues(path)

    def test_read_eigenvectors_unparsable_includes_path(self, tmp_path: Path) -> None:
        """Non-numeric eigenvector file includes path in error."""
        path = tmp_path / "bad.eigenU.txt"
        path.write_text("1.0\t2.0\nfoo\tbar\n")

        with pytest.raises(ValueError, match=str(path)):
            read_eigenvectors(path)

    def test_read_eigenvectors_non_square_raises(self, tmp_path: Path) -> None:
        """Non-square eigenvector matrix raises ValueError at reader level."""
        path = tmp_path / "nonsquare.eigenU.txt"
        # 2 rows x 3 columns
        np.savetxt(path, np.ones((2, 3)), fmt="%.10g", delimiter="\t")

        with pytest.raises(ValueError, match="square"):
            read_eigenvectors(path)

    def test_read_eigenvectors_empty_file_raises(self, tmp_path: Path) -> None:
        """Empty eigenvector file raises ValueError."""
        path = tmp_path / "empty.eigenU.txt"
        path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_eigenvectors(path)

    def test_negative_eigenvalues_rejected(self, tmp_path: Path) -> None:
        """read_eigen_files rejects negative eigenvalues from external files.

        Kinship eigenvalues are non-negative by construction, but externally
        supplied -d/-u files could contain negatives. The C extension uses
        log(v) (not log(abs(v))), so negative eigenvalues produce NaN/domain
        errors. Validate at the input boundary.
        """
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        # One negative eigenvalue
        write_eigenvalues(np.array([-0.5, 1.0, 2.0]), d_path)
        write_eigenvectors(np.eye(3), u_path)

        with pytest.raises(ValueError, match="negative"):
            read_eigen_files(d_path, u_path)

    def test_zero_eigenvalues_accepted(self, tmp_path: Path) -> None:
        """Zero eigenvalues are valid (rank-deficient kinship)."""
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        write_eigenvalues(np.array([0.0, 0.0, 1.0]), d_path)
        write_eigenvectors(np.eye(3), u_path)

        eigenvalues, _ = read_eigen_files(d_path, u_path)
        assert eigenvalues[0] == 0.0

    def test_read_eigenvalues_npy_wrong_shape(self, tmp_path: Path) -> None:
        """2D .npy file raises ValueError when loaded as eigenvalues."""
        path = tmp_path / "bad.eigenD.npy"
        np.save(path, np.eye(3))  # 2D, not 1D
        with pytest.raises(ValueError, match="wrong shape"):
            read_eigenvalues(path)

    def test_read_eigenvectors_npy_non_square(self, tmp_path: Path) -> None:
        """Non-square .npy file raises ValueError when loaded as eigenvectors."""
        path = tmp_path / "bad.eigenU.npy"
        np.save(path, np.ones((2, 3)))
        with pytest.raises(ValueError, match="wrong shape"):
            read_eigenvectors(path)


# =============================================================================
# .npy sidecar cache tests
# =============================================================================


@pytest.mark.tier0
class TestNpyCache:
    """Verify .npy sidecar cache behavior."""

    def test_cache_written_on_first_read(self, tmp_path: Path) -> None:
        """Reading legacy text eigenvalues/eigenvectors creates .npy sidecar."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        # Use legacy_text=True to write actual .txt files for sidecar test
        write_eigenvalues(np.ones(5), d_path, legacy_text=True)
        write_eigenvectors(np.eye(5), u_path, legacy_text=True)

        # Delete .npy files that write_* creates
        d_path.with_suffix(".npy").unlink(missing_ok=True)
        u_path.with_suffix(".npy").unlink(missing_ok=True)

        read_eigenvalues(d_path)
        read_eigenvectors(u_path)

        assert d_path.with_suffix(".npy").exists(), ".eigenD.npy not created"
        assert u_path.with_suffix(".npy").exists(), ".eigenU.npy not created"

    def test_cache_used_on_second_read(self, tmp_path: Path) -> None:
        """Second read uses .npy cache (verified by data correctness)."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        eigenvalues = np.array([1.0, 2.0, 3.0])
        eigenvectors = np.eye(3) * 2.0
        # Use legacy_text=True to write actual .txt files + sidecar
        write_eigenvalues(eigenvalues, d_path, legacy_text=True)
        write_eigenvectors(eigenvectors, u_path, legacy_text=True)

        # First read (uses cache written by write_*)
        d1 = read_eigenvalues(d_path)
        u1 = read_eigenvectors(u_path)

        # Second read (uses cache)
        d2 = read_eigenvalues(d_path)
        u2 = read_eigenvectors(u_path)

        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(u1, u2)

    def test_stale_cache_invalidated(self, tmp_path: Path) -> None:
        """Modifying .txt file invalidates cache; fresh parse occurs."""
        import time

        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(np.array([1.0, 2.0]), path, legacy_text=True)
        first = read_eigenvalues(path)

        # Overwrite text file with different data (wait for mtime granularity)
        time.sleep(0.05)
        np.savetxt(path, np.array([10.0, 20.0]), fmt="%.10g")
        # Touch to ensure mtime is newer
        path.touch()

        second = read_eigenvalues(path)
        np.testing.assert_array_equal(second, [10.0, 20.0])
        assert not np.array_equal(first, second)

    def test_corrupt_npy_falls_back_to_text(self, tmp_path: Path) -> None:
        """Corrupted .npy cache triggers text re-parse for legacy text files."""
        path = tmp_path / "test.eigenD.txt"
        # Write text file with sidecar
        write_eigenvalues(np.array([1.0, 2.0, 3.0]), path, legacy_text=True)

        # Corrupt the .npy cache
        npy_path = path.with_suffix(".npy")
        npy_path.write_bytes(b"garbage data not a valid npy file")

        # Should still work by falling back to text
        data = read_eigenvalues(path)
        np.testing.assert_allclose(data, [1.0, 2.0, 3.0])

    def test_write_creates_npy_sidecar(self, tmp_path: Path) -> None:
        """write_eigenvalues/write_eigenvectors with legacy_text create .npy sidecar."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(3), d_path, legacy_text=True)
        write_eigenvectors(np.eye(3), u_path, legacy_text=True)

        assert d_path.with_suffix(".npy").exists()
        assert u_path.with_suffix(".npy").exists()

        # Verify .npy content matches
        d_cached = np.load(d_path.with_suffix(".npy"))
        u_cached = np.load(u_path.with_suffix(".npy"))
        np.testing.assert_array_equal(d_cached, np.ones(3))
        np.testing.assert_array_equal(u_cached, np.eye(3))

    def test_cache_survives_round_trip(self, tmp_path: Path) -> None:
        """Full write → read → cache-read round trip preserves data."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 20))
        psd = A @ A.T  # PSD → non-negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(psd)

        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="cache_rt"
        )

        # Read from cache (write_* already created .npy)
        loaded_d, loaded_u = read_eigen_files(d_path, u_path)

        np.testing.assert_allclose(loaded_d, eigenvalues, rtol=1e-9)
        np.testing.assert_allclose(loaded_u, eigenvectors, rtol=1e-9)

    def test_cache_load_is_read_only(self, tmp_path: Path) -> None:
        """Cache-loaded eigenvalues are read-only (mmap_mode='r' from _load_npy_cache).

        write_eigenvalues with legacy_text=True writes both the .txt file and
        a .npy sidecar via _write_npy_cache. Subsequent read_eigenvalues uses
        _load_npy_cache which returns np.load(..., mmap_mode='r').
        np.atleast_1d on a read-only memmap returns the same object unchanged.
        """
        d_path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(np.array([1.0, 2.0, 3.0]), d_path, legacy_text=True)

        # read_eigenvalues will use the .npy sidecar via _load_npy_cache
        result = read_eigenvalues(d_path)

        assert not result.flags.writeable, (
            "Cache-loaded eigenvalues should be read-only (mmap_mode='r'); "
            f"flags.writeable={result.flags.writeable}"
        )

    def test_cache_load_returns_memmap(self, tmp_path: Path) -> None:
        """_load_npy_cache returns np.memmap instance (demand-paged, not eager)."""
        arr = np.array([1.0, 2.0, 3.0])
        npy_path = tmp_path / "test.eigenD.npy"
        _write_npy_cache(arr, npy_path)

        result = _load_npy_cache(npy_path)
        assert result is not None
        assert isinstance(result, np.memmap), (
            f"Expected np.memmap from _load_npy_cache, got {type(result).__name__}"
        )
        np.testing.assert_array_equal(result, arr)


# =============================================================================
# Atomic .npy cache write tests
# =============================================================================


@pytest.mark.tier0
class TestAtomicCacheWrite:
    """Verify _write_npy_cache uses atomic rename and leaves no temp files."""

    def test_no_partial_npy_on_normal_write(self, tmp_path: Path) -> None:
        """Normal write_eigenvalues leaves .npy file and no .tmp.npy artifact."""
        d_path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(np.ones(5), d_path, legacy_text=True)

        npy_path = d_path.with_suffix(".npy")
        assert npy_path.exists(), ".eigenD.npy sidecar should exist after write"

        # No .tmp.npy temp file should remain
        tmp_npy = tmp_path / (npy_path.stem + ".tmp.npy")
        assert not tmp_npy.exists(), (
            f"Temp file {tmp_npy.name} should not exist after atomic rename completed"
        )

        # Verify content is correct
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, np.ones(5))

    def test_atomic_write_no_tmp_leftover(self, tmp_path: Path) -> None:
        """write_eigenvectors with legacy_text leaves no .tmp.npy in directory."""
        u_path = tmp_path / "test.eigenU.txt"
        write_eigenvectors(np.eye(5), u_path, legacy_text=True)

        npy_path = u_path.with_suffix(".npy")
        assert npy_path.exists(), ".eigenU.npy sidecar should exist after write"

        # Verify no temp artifact in directory
        tmp_files = [
            f.name for f in tmp_path.iterdir() if f.is_file() and ".tmp.npy" in f.name
        ]
        assert tmp_files == [], (
            f"Found unexpected .tmp.npy files after write: {tmp_files}"
        )

    def test_write_npy_cache_directly(self, tmp_path: Path) -> None:
        """_write_npy_cache writes .npy file atomically and cleans up temp."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        npy_path = tmp_path / "direct.eigenD.npy"

        _write_npy_cache(arr, npy_path)

        assert npy_path.exists(), "_write_npy_cache should create .npy file"

        # No .tmp.npy artifact should remain
        tmp_npy = tmp_path / (npy_path.stem + ".tmp.npy")
        assert not tmp_npy.exists(), (
            f"Temp file {tmp_npy.name} should not exist after _write_npy_cache"
        )

        # Verify content round-trips correctly
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, arr)

    def test_write_npy_cache_error_cleans_tmp(self, tmp_path: Path) -> None:
        """_write_npy_cache cleans up .tmp.npy when the atomic rename fails."""
        from unittest.mock import patch

        arr = np.array([1.0, 2.0, 3.0])
        npy_path = tmp_path / "fail.eigenD.npy"
        tmp_npy = tmp_path / "fail.eigenD.tmp.npy"

        # Patch Path.replace so the atomic rename step fails after the
        # temp file has been written. _write_npy_cache must then remove
        # the temp file in its finally block.
        with patch.object(Path, "replace", side_effect=OSError("mock")):
            _write_npy_cache(arr, npy_path)

        # The target .npy should not exist (rename failed)
        assert not npy_path.exists(), "Target .npy should not exist after failed rename"
        # The temp .tmp.npy should be cleaned up
        assert not tmp_npy.exists(), (
            f"Temp file {tmp_npy.name} should be cleaned up after rename failure"
        )


# =============================================================================
# LMM equivalence tests
# =============================================================================

# Fixture paths for mouse_hs1940 dataset
FIXTURES = Path(__file__).parent / "fixtures" / "mouse_hs1940"
BFILE = FIXTURES / "mouse_hs1940"
KINSHIP_FILE = FIXTURES / "mouse_hs1940_kinship.cXX.txt"


@pytest.mark.slow
@pytest.mark.tier1
class TestLMMEquivalence:
    """Verify loaded-eigen LMM results match fresh-eigen results."""

    @pytest.mark.tier1
    def test_loaded_eigen_matches_fresh_eigen_lmm(self, tmp_path: Path) -> None:
        """LMM with loaded eigen files matches LMM with fresh eigendecomp.

        This is the key validation: proves the multi-phenotype eigen reuse
        workflow produces correct results.
        """
        from jamma.kinship import read_kinship_matrix
        from jamma.lmm.eigen import eigendecompose_kinship
        from jamma.pipeline import PipelineConfig, PipelineRunner

        # 1. Run fresh-eigen pipeline (standard path with kinship)
        fresh_dir = tmp_path / "fresh"
        fresh_config = PipelineConfig(
            bfile=BFILE,
            kinship_file=KINSHIP_FILE,
            output_dir=fresh_dir,
            output_prefix="fresh",
            check_memory=False,
            show_progress=False,
        )
        fresh_result = PipelineRunner(fresh_config).run()

        # 2. Compute eigen from kinship (subsetted to valid-phenotype samples)
        from jamma.io.plink import get_plink_metadata

        meta = get_plink_metadata(BFILE)
        K = read_kinship_matrix(KINSHIP_FILE, n_samples=meta["n_samples"])

        # Subset to valid-phenotype samples (same as runner does internally)
        pheno = load_phenotypes_from_fam(Path(f"{BFILE}.fam"))
        valid_mask = ~np.isnan(pheno) & (pheno != -9.0)
        K_valid = K[np.ix_(valid_mask, valid_mask)]

        eigenvalues, eigenvectors = eigendecompose_kinship(K_valid)

        eigen_dir = tmp_path / "eigen"
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, eigen_dir, prefix="test"
        )

        # 3. Run loaded-eigen pipeline (no kinship, just eigen files)
        loaded_dir = tmp_path / "loaded"
        loaded_config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            output_dir=loaded_dir,
            output_prefix="loaded",
            check_memory=False,
            show_progress=False,
        )
        loaded_result = PipelineRunner(loaded_config).run()

        # 4. Compare results
        assert fresh_result.n_samples == loaded_result.n_samples
        assert fresh_result.n_snps_tested == loaded_result.n_snps_tested

        # Read output files and compare columns
        fresh_lines = (fresh_dir / "fresh.assoc.txt").read_text().strip().splitlines()
        loaded_lines = (
            (loaded_dir / "loaded.assoc.txt").read_text().strip().splitlines()
        )

        assert len(fresh_lines) == len(loaded_lines)
        assert len(fresh_lines) > 1  # header + data

        # Parse header
        header = fresh_lines[0].split("\t")
        beta_idx = header.index("beta")
        se_idx = header.index("se")
        p_wald_idx = header.index("p_wald")

        # Compare every SNP
        for i in range(1, len(fresh_lines)):
            fresh_cols = fresh_lines[i].split("\t")
            loaded_cols = loaded_lines[i].split("\t")

            # SNP identity must match
            assert fresh_cols[1] == loaded_cols[1], f"SNP mismatch at line {i}"

            fresh_beta = float(fresh_cols[beta_idx])
            loaded_beta = float(loaded_cols[beta_idx])
            fresh_se = float(fresh_cols[se_idx])
            loaded_se = float(loaded_cols[se_idx])
            fresh_p = float(fresh_cols[p_wald_idx])
            loaded_p = float(loaded_cols[p_wald_idx])

            # Handle NaN SNPs (degenerate)
            if np.isnan(fresh_beta):
                assert np.isnan(loaded_beta)
                continue

            # Standard tolerances from GEMMA_EQUIVALENCE.md tolerance table
            np.testing.assert_allclose(
                loaded_beta,
                fresh_beta,
                rtol=1e-2,
                err_msg=f"beta mismatch at SNP {fresh_cols[1]}",
            )
            np.testing.assert_allclose(
                loaded_se,
                fresh_se,
                rtol=1e-5,
                err_msg=f"se mismatch at SNP {fresh_cols[1]}",
            )
            np.testing.assert_allclose(
                loaded_p,
                fresh_p,
                rtol=1e-4,
                err_msg=f"p_wald mismatch at SNP {fresh_cols[1]}",
            )

    @pytest.mark.tier1
    def test_write_eigen_flag_creates_files(self, tmp_path: Path) -> None:
        """PipelineRunner with write_eigen=True creates eigenD/eigenU files."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        config = PipelineConfig(
            bfile=BFILE,
            kinship_file=KINSHIP_FILE,
            output_dir=tmp_path,
            output_prefix="test",
            check_memory=False,
            show_progress=False,
            write_eigen=True,
        )
        PipelineRunner(config).run()

        # Default binary format writes .npy files
        d_path = tmp_path / "test.eigenD.npy"
        u_path = tmp_path / "test.eigenU.npy"

        assert d_path.exists(), "eigenD file not created"
        assert u_path.exists(), "eigenU file not created"
        assert d_path.stat().st_size > 0
        assert u_path.stat().st_size > 0

        # Verify files are loadable
        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
        assert eigenvalues.shape[0] > 0
        assert eigenvectors.shape[0] == eigenvectors.shape[1]
        assert eigenvalues.shape[0] == eigenvectors.shape[0]


# =============================================================================
# Flag interaction tests (unit-level)
# =============================================================================


@pytest.mark.tier0
class TestFlagInteractions:
    """Verify flag validation rules for eigen reuse."""

    def test_validate_d_without_u_raises(self, tmp_path: Path) -> None:
        """Eigenvalue file without eigenvector file raises ValueError."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        # Create a dummy eigenvalue file
        d_path = tmp_path / "test.eigenD.txt"
        d_path.write_text("1.0\n2.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=None,
            check_memory=False,
        )
        with pytest.raises(ValueError, match=r"Both -d.*and -u.*must be provided"):
            PipelineRunner(config).validate_inputs()

    def test_validate_u_without_d_raises(self, tmp_path: Path) -> None:
        """Eigenvector file without eigenvalue file raises ValueError."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        u_path = tmp_path / "test.eigenU.txt"
        u_path.write_text("1.0\t0.0\n0.0\t1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=None,
            eigenvector_file=u_path,
            check_memory=False,
        )
        with pytest.raises(ValueError, match=r"Both -d.*and -u.*must be provided"):
            PipelineRunner(config).validate_inputs()

    def test_validate_eigen_with_loco_raises(self, tmp_path: Path) -> None:
        """Eigen files with -loco raises ValueError (use --eigen-dir instead)."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        d_path.write_text("1.0\n")
        u_path.write_text("1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            loco=True,
            check_memory=False,
        )
        with pytest.raises(ValueError, match="not supported with -loco"):
            PipelineRunner(config).validate_inputs()

    def test_validate_eigen_files_not_found_raises(self, tmp_path: Path) -> None:
        """Nonexistent eigenvalue file raises FileNotFoundError."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        d_path = tmp_path / "nonexistent.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        u_path.write_text("1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            check_memory=False,
        )
        with pytest.raises(FileNotFoundError, match="Eigenvalue file not found"):
            PipelineRunner(config).validate_inputs()

    def test_kinship_not_required_with_eigen_files(self, tmp_path: Path) -> None:
        """Kinship is optional when eigen files are provided."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        d_path.write_text("1.0\n2.0\n")
        u_path.write_text("1.0\t0.0\n0.0\t1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            kinship_file=None,
            check_memory=False,
        )
        # Should NOT raise -- kinship is optional with eigen files
        PipelineRunner(config).validate_inputs()


# =============================================================================
# Binary .npy eigen I/O tests (new default format)
# =============================================================================


@pytest.mark.tier0
class TestBinaryEigenIO:
    """Tests for binary .npy eigen I/O (new default format)."""

    def _make_eigen(self, n: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        psd = A @ A.T
        return np.linalg.eigh(psd)

    def test_binary_eigen_roundtrip(self, tmp_path: Path) -> None:
        """write_eigen_files writes .npy by default; read back is exact."""
        eigenvalues, eigenvectors = self._make_eigen(n=8)
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="bin"
        )

        assert d_path.suffix == ".npy", f"Expected .npy suffix, got {d_path.suffix}"
        assert u_path.suffix == ".npy", f"Expected .npy suffix, got {u_path.suffix}"

        D_loaded = np.load(d_path)
        U_loaded = np.load(u_path)
        np.testing.assert_array_equal(D_loaded, eigenvalues)
        np.testing.assert_array_equal(U_loaded, eigenvectors)

    def test_binary_eigen_read_back_via_read_eigen_files(self, tmp_path: Path) -> None:
        """read_eigen_files handles .npy paths from default write."""
        eigenvalues, eigenvectors = self._make_eigen(n=10)
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="rt"
        )

        d_loaded, u_loaded = read_eigen_files(d_path, u_path)
        np.testing.assert_array_equal(d_loaded, eigenvalues)
        np.testing.assert_array_equal(u_loaded, eigenvectors)

    def test_eigen_read_npy_path_directly(self, tmp_path: Path) -> None:
        """read_eigenvalues/read_eigenvectors handle .npy paths directly."""
        eigenvalues, eigenvectors = self._make_eigen(n=6)
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="direct"
        )

        D_loaded = read_eigenvalues(d_path)
        U_loaded = read_eigenvectors(u_path)
        np.testing.assert_array_equal(D_loaded, eigenvalues)
        np.testing.assert_array_equal(U_loaded, eigenvectors)

    def test_write_eigen_legacy_text(self, tmp_path: Path) -> None:
        """legacy_text=True writes .txt files (GEMMA-compatible)."""
        eigenvalues, eigenvectors = self._make_eigen(n=5)
        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="leg", legacy_text=True
        )

        assert d_path.suffix == ".txt", f"Expected .txt suffix, got {d_path.suffix}"
        assert u_path.suffix == ".txt", f"Expected .txt suffix, got {u_path.suffix}"
        assert d_path.exists()
        assert u_path.exists()

        # Content should be numeric text
        content = d_path.read_text()
        lines = [line for line in content.strip().splitlines() if line]
        assert len(lines) == len(eigenvalues)

    def test_existing_text_eigen_still_loadable(self, tmp_path: Path) -> None:
        """Text-only GEMMA eigen files (no .npy) are still readable."""
        eigenvalues, eigenvectors = self._make_eigen(n=7)

        # Write only text files directly (no sidecar creation)
        d_path = tmp_path / "old.eigenD.txt"
        u_path = tmp_path / "old.eigenU.txt"
        np.savetxt(d_path, eigenvalues, fmt="%.10g")
        np.savetxt(u_path, eigenvectors, fmt="%.10g", delimiter="\t")

        # Ensure no .npy exists
        d_path.with_suffix(".npy").unlink(missing_ok=True)
        u_path.with_suffix(".npy").unlink(missing_ok=True)

        D_loaded = read_eigenvalues(d_path)
        U_loaded = read_eigenvectors(u_path)

        np.testing.assert_allclose(D_loaded, eigenvalues, rtol=1e-9)
        np.testing.assert_allclose(U_loaded, eigenvectors, rtol=1e-9)

    def test_write_eigen_default_no_txt(self, tmp_path: Path) -> None:
        """Default binary write does NOT create .txt files."""
        eigenvalues, eigenvectors = self._make_eigen(n=5)
        write_eigen_files(eigenvalues, eigenvectors, tmp_path, prefix="nobintest")

        txt_d = tmp_path / "nobintest.eigenD.txt"
        txt_u = tmp_path / "nobintest.eigenU.txt"
        assert not txt_d.exists(), ".eigenD.txt should NOT be written in binary mode"
        assert not txt_u.exists(), ".eigenU.txt should NOT be written in binary mode"


# =============================================================================
# CLI flag tests
# =============================================================================


@pytest.mark.tier0
class TestCLIFlags:
    """Verify CLI help shows eigen flags."""

    def test_lmm_help_shows_eigen_flags(self) -> None:
        """--help output contains -d, -u, and -eigen flags."""
        from click.testing import CliRunner

        from jamma.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "-d" in result.output
        assert "-u" in result.output
        assert "-eigen" in result.output
