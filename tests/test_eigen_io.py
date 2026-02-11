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
    read_eigen_files,
    read_eigenvalues,
    read_eigenvectors,
    write_eigen_files,
    write_eigenvalues,
    write_eigenvectors,
)

# =============================================================================
# File format tests
# =============================================================================


class TestEigenvalueFormat:
    """Verify eigenvalue file format matches GEMMA .eigenD.txt."""

    def test_write_eigenvalues_format(self, tmp_path: Path) -> None:
        """Eigenvalue file has one value per line, .10g format, no header."""
        values = np.array([0.001, 1.0, 2.5, 100.0, 12345.6789012345])
        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(values, path)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

        # Each line should be the .10g formatted value
        for i, line in enumerate(lines):
            expected = f"{values[i]:.10g}"
            assert line == expected, f"Line {i}: got {line!r}, expected {expected!r}"

    def test_write_eigenvectors_format(self, tmp_path: Path) -> None:
        """Eigenvector file has tab-separated rows, .10g format, no header."""
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        path = tmp_path / "test.eigenU.txt"
        write_eigenvectors(matrix, path)

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

        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(eigenvalues, path)
        loaded = read_eigenvalues(path)

        # Order is preserved
        assert np.all(loaded[:-1] <= loaded[1:])
        np.testing.assert_allclose(loaded, eigenvalues, rtol=1e-9)


# =============================================================================
# Round-trip precision tests
# =============================================================================


class TestRoundTripPrecision:
    """Verify .10g format preserves sufficient precision for LMM."""

    def test_eigenvalue_round_trip_precision(self, tmp_path: Path) -> None:
        """100 random eigenvalues survive write/read within rtol=1e-9."""
        rng = np.random.default_rng(123)
        # Generate eigenvalues spanning several orders of magnitude
        original = np.sort(rng.uniform(0.001, 1000.0, size=100))

        path = tmp_path / "eigenD.txt"
        write_eigenvalues(original, path)
        loaded = read_eigenvalues(path)

        np.testing.assert_allclose(loaded, original, rtol=1e-9)

    def test_eigenvector_round_trip_precision(self, tmp_path: Path) -> None:
        """50x50 orthogonal matrix survives write/read within rtol=1e-9."""
        rng = np.random.default_rng(456)
        A = rng.standard_normal((50, 50))
        sym = A + A.T
        _, eigenvectors = np.linalg.eigh(sym)

        # Eigenvectors from eigh are orthonormal
        path = tmp_path / "eigenU.txt"
        write_eigenvectors(eigenvectors, path)
        loaded = read_eigenvectors(path)

        np.testing.assert_allclose(loaded, eigenvectors, rtol=1e-9)

    def test_eigen_files_round_trip(self, tmp_path: Path) -> None:
        """write_eigen_files + read_eigen_files round-trip both arrays."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((30, 30))
        sym = A + A.T
        eigenvalues, eigenvectors = np.linalg.eigh(sym)

        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="roundtrip"
        )

        loaded_d, loaded_u = read_eigen_files(d_path, u_path)

        np.testing.assert_allclose(loaded_d, eigenvalues, rtol=1e-9)
        np.testing.assert_allclose(loaded_u, eigenvectors, rtol=1e-9)


# =============================================================================
# Dimension validation tests
# =============================================================================


class TestDimensionValidation:
    """Verify read_eigen_files catches dimension mismatches."""

    def test_read_eigen_files_dimension_mismatch(self, tmp_path: Path) -> None:
        """Mismatched eigenvalue count vs eigenvector dimensions raises ValueError."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(8), u_path)

        with pytest.raises(ValueError, match="does not match"):
            read_eigen_files(d_path, u_path)

    def test_read_eigen_files_n_samples_mismatch(self, tmp_path: Path) -> None:
        """n_samples validation catches wrong expected count."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        with pytest.raises(ValueError, match="does not match expected n_samples=12"):
            read_eigen_files(d_path, u_path, n_samples=12)

    def test_read_eigen_files_consistent_dimensions(self, tmp_path: Path) -> None:
        """Consistent eigen pair with matching n_samples succeeds."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path, n_samples=10)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)

    def test_read_eigen_files_no_n_samples_validation(self, tmp_path: Path) -> None:
        """Omitting n_samples skips that validation."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling for eigen I/O."""

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Writing to nested path creates parent directories."""
        nested = tmp_path / "a" / "b" / "c" / "test.eigenD.txt"
        write_eigenvalues(np.array([1.0, 2.0]), nested)
        assert nested.exists()
        loaded = read_eigenvalues(nested)
        assert len(loaded) == 2

    def test_read_eigenvalues_empty_file(self, tmp_path: Path) -> None:
        """Empty eigenD file returns empty array (numpy warns but does not raise)."""
        path = tmp_path / "empty.eigenD.txt"
        path.write_text("")

        # numpy >= 2.0 returns empty float64 array with a UserWarning
        with pytest.warns(UserWarning, match="no data"):
            result = read_eigenvalues(path)

        assert result.size == 0
        assert result.dtype == np.float64

    def test_write_read_single_eigenvalue(self, tmp_path: Path) -> None:
        """1x1 matrix edge case preserves correct shapes."""
        eigenvalues = np.array([3.14])
        eigenvectors = np.array([[1.0]])

        d_path = tmp_path / "single.eigenD.txt"
        u_path = tmp_path / "single.eigenU.txt"

        write_eigenvalues(eigenvalues, d_path)
        write_eigenvectors(eigenvectors, u_path)

        loaded_d = read_eigenvalues(d_path)
        loaded_u = read_eigenvectors(u_path)

        # np.loadtxt on a single-line file returns a 0-d array for 1D
        # and a 1D array for a single-row matrix. Verify shape handling.
        assert loaded_d.ndim <= 1
        assert np.isclose(float(loaded_d), 3.14, rtol=1e-9)

        # Single row eigenvector file: np.loadtxt returns 1D
        assert np.isclose(float(loaded_u.flat[0]), 1.0)


# =============================================================================
# LMM equivalence tests
# =============================================================================

# Fixture paths for mouse_hs1940 dataset
FIXTURES = Path(__file__).parent / "fixtures" / "mouse_hs1940"
BFILE = FIXTURES / "mouse_hs1940"
KINSHIP_FILE = FIXTURES / "mouse_hs1940_kinship.cXX.txt"


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
        fam_data = np.loadtxt(f"{BFILE}.fam", dtype=str, usecols=(5,))
        missing = np.isin(fam_data, ["-9", "NA"])
        pheno = fam_data.copy()
        pheno[missing] = "0"
        pheno = pheno.astype(np.float64)
        pheno[missing] = np.nan
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

            # Standard tolerances from CLAUDE.md tolerance table
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

        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

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
        with pytest.raises(ValueError, match="Both -d.*and -u.*must be provided"):
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
        with pytest.raises(ValueError, match="Both -d.*and -u.*must be provided"):
            PipelineRunner(config).validate_inputs()

    def test_validate_eigen_with_loco_raises(self, tmp_path: Path) -> None:
        """Eigen files with -loco raises ValueError."""
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
# CLI flag tests
# =============================================================================


class TestCLIFlags:
    """Verify CLI help shows eigen flags."""

    def test_lmm_help_shows_eigen_flags(self) -> None:
        """lmm --help output contains -d, -u, and -eigen flags."""
        from typer.testing import CliRunner

        from jamma.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["lmm", "--help"])

        assert result.exit_code == 0
        assert "-d" in result.output
        assert "-u" in result.output
        assert "-eigen" in result.output

    def test_gk_help_shows_eigen_flag(self) -> None:
        """gk --help output contains -eigen flag."""
        from typer.testing import CliRunner

        from jamma.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["gk", "--help"])

        assert result.exit_code == 0
        assert "-eigen" in result.output
