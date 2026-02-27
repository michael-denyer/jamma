"""Error-path tests for PLINK I/O validation and LMM association writer.

Covers ERRP-03 (PLINK I/O error paths) and ERRP-04 (LMM I/O error paths).
Complements test_plink_validation.py (dimension/value tests) and
test_incremental_writer.py (disk-full retry logic) without duplicating them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import (
    get_chromosome_partitions,
    validate_genotype_values,
    validate_plink_dimensions,
)
from jamma.lmm.io import IncrementalAssocWriter, format_assoc_line
from jamma.lmm.schema import get_spec
from jamma.lmm.stats import AssocResult

FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"
LOCO_FIXTURES = Path(__file__).parent / "fixtures" / "gemma_loco"
LOCO_BFILE = LOCO_FIXTURES / "test"


@pytest.mark.tier0
class TestPlinkIOErrorPaths:
    """Error-path tests for PLINK I/O validation functions.

    Tests validate_plink_dimensions (truncated .bed, per-extension missing files),
    validate_genotype_values (out-of-range value counting), and
    get_chromosome_partitions (multi-chromosome handling).
    """

    def test_truncated_bed_raises(self, tmp_path: Path) -> None:
        """Truncated .bed raises ValueError with dimension mismatch message."""
        for ext in (".bed", ".bim", ".fam"):
            shutil.copy(FIXTURES / f"test{ext}", tmp_path / f"test{ext}")

        bed_path = tmp_path / "test.bed"
        original_size = bed_path.stat().st_size
        with open(bed_path, "r+b") as f:
            f.truncate(original_size - 10)

        with pytest.raises(
            ValueError, match="dimension mismatch|doesn't match expected"
        ):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_bed_raises(self, tmp_path: Path) -> None:
        """Missing .bed (with .bim and .fam present) raises FileNotFoundError."""
        shutil.copy(FIXTURES / "test.bim", tmp_path / "test.bim")
        shutil.copy(FIXTURES / "test.fam", tmp_path / "test.fam")

        with pytest.raises(FileNotFoundError):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_bim_raises(self, tmp_path: Path) -> None:
        """Missing .bim (with .bed and .fam present) raises FileNotFoundError."""
        shutil.copy(FIXTURES / "test.bed", tmp_path / "test.bed")
        shutil.copy(FIXTURES / "test.fam", tmp_path / "test.fam")

        with pytest.raises(FileNotFoundError):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_fam_raises(self, tmp_path: Path) -> None:
        """Missing .fam (with .bed and .bim present) raises FileNotFoundError."""
        shutil.copy(FIXTURES / "test.bed", tmp_path / "test.bed")
        shutil.copy(FIXTURES / "test.bim", tmp_path / "test.bim")

        with pytest.raises(FileNotFoundError):
            validate_plink_dimensions(tmp_path / "test")

    def test_validate_genotype_out_of_range(self) -> None:
        """Out-of-range values (3.0, -1.0) are counted; 0.0 and NaN are valid."""
        chunk = np.array([[0.0, 3.0, -1.0, np.nan]], dtype=np.float32)
        result = validate_genotype_values(chunk)
        assert result == 2

    def test_validate_genotype_all_valid(self) -> None:
        """All valid values (0, 1, 2, NaN) return count of 0."""
        chunk = np.array([[0.0, 1.0, 2.0, np.nan]], dtype=np.float32)
        assert validate_genotype_values(chunk) == 0

    def test_non_monotonic_chromosomes_handled(self) -> None:
        """get_chromosome_partitions handles multi-chromosome data without crash.

        Uses gemma_loco fixture which has chromosomes 1, 2, and 3.
        Verifies the returned dict has at least 1 key and total SNP count
        matches the BIM line count.
        """
        bfile = LOCO_BFILE if (LOCO_FIXTURES / "test.bed").exists() else BFILE
        partitions = get_chromosome_partitions(bfile)

        assert len(partitions) >= 1

        # Verify SNP counts are consistent with the BIM file
        bim_path = Path(f"{bfile}.bim")
        n_bim_snps = sum(1 for _ in open(bim_path))
        total_snps = sum(len(indices) for indices in partitions.values())
        assert total_snps == n_bim_snps


@pytest.mark.tier0
class TestLmmIOErrorPaths:
    """Error-path tests for LMM I/O association writer.

    Tests format_assoc_line (unknown test_type), IncrementalAssocWriter
    (invalid test_type), and write_arrays_batch (mode mismatch, missing
    stat keys, missing snp_info keys, length mismatch).
    """

    def _make_assoc_result(self) -> AssocResult:
        """Create a minimal AssocResult with dummy values for testing."""
        return AssocResult(
            chr="1",
            rs="rs1",
            ps=100,
            n_miss=0,
            allele1="A",
            allele0="G",
            af=0.3,
            beta=0.1,
            se=0.01,
            logl_H1=-100.0,
            l_remle=1.0,
            p_wald=0.05,
        )

    def test_format_assoc_line_unknown_type_raises(self) -> None:
        """format_assoc_line rejects unknown test_type with ValueError."""
        result = self._make_assoc_result()
        with pytest.raises(ValueError, match="Unknown test_type"):
            format_assoc_line(result, test_type="unknown")

    def test_incremental_writer_invalid_type_raises(self, tmp_path: Path) -> None:
        """IncrementalAssocWriter rejects invalid test_type with ValueError."""
        with pytest.raises(ValueError, match="Unknown test_type"):
            IncrementalAssocWriter(tmp_path / "out.txt", test_type="bad")

    def test_write_arrays_batch_mode_mismatch_raises(self, tmp_path: Path) -> None:
        """write_arrays_batch rejects lmm_mode whose test_type != writer's test_type."""
        snp_indices = np.array([0])
        snp_info = [{"chr": "1", "rs": "rs1", "pos": 100, "a1": "A", "a0": "G"}]
        afs = np.array([0.3])
        miss_counts = np.array([0])

        with pytest.raises(ValueError, match="does not match"):
            with IncrementalAssocWriter(
                tmp_path / "out.txt", test_type="wald"
            ) as writer:
                # lmm_mode=2 is LRT but writer expects wald
                writer.write_arrays_batch(
                    lmm_mode=2,
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays={},
                )

    def test_write_arrays_batch_missing_stat_keys_raises(self, tmp_path: Path) -> None:
        """write_arrays_batch rejects empty arrays (missing all wald stat keys)."""
        snp_indices = np.array([0])
        snp_info = [{"chr": "1", "rs": "rs1", "pos": 100, "a1": "A", "a0": "G"}]
        afs = np.array([0.3])
        miss_counts = np.array([0])

        with pytest.raises(ValueError, match="missing arrays"):
            with IncrementalAssocWriter(
                tmp_path / "out.txt", test_type="wald"
            ) as writer:
                # lmm_mode=1 is wald, but arrays={} is missing all required keys
                writer.write_arrays_batch(
                    lmm_mode=1,
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays={},
                )

    def test_write_arrays_batch_snp_info_missing_keys_raises(
        self, tmp_path: Path
    ) -> None:
        """write_arrays_batch raises KeyError when snp_info missing required keys."""
        snp_indices = np.array([0])
        # Only 'chr' key provided; missing rs, pos, a1, a0
        snp_info = [{"chr": "1"}]
        afs = np.array([0.3])
        miss_counts = np.array([0])

        # Build complete arrays for wald mode (lmm_mode=1) using get_spec
        spec = get_spec(1)
        arrays = {c.array_key: np.array([0.1]) for c in spec.stat_columns}

        with pytest.raises(KeyError, match="missing required keys"):
            with IncrementalAssocWriter(
                tmp_path / "out.txt", test_type="wald"
            ) as writer:
                writer.write_arrays_batch(
                    lmm_mode=1,
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays=arrays,
                )

    def test_write_arrays_batch_length_mismatch_raises(self, tmp_path: Path) -> None:
        """write_arrays_batch raises ValueError when afs length != snp_indices."""
        # 2 SNPs but afs has only 1 value
        snp_indices = np.array([0, 1])
        snp_info = [
            {"chr": "1", "rs": "rs1", "pos": 100, "a1": "A", "a0": "G"},
            {"chr": "1", "rs": "rs2", "pos": 200, "a1": "A", "a0": "G"},
        ]
        afs = np.array([0.3])  # length 1, mismatches snp_indices length 2
        miss_counts = np.array([0, 0])

        spec = get_spec(1)
        arrays = {c.array_key: np.array([0.1, 0.2]) for c in spec.stat_columns}

        with pytest.raises(ValueError, match="length"):
            with IncrementalAssocWriter(
                tmp_path / "out.txt", test_type="wald"
            ) as writer:
                writer.write_arrays_batch(
                    lmm_mode=1,
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays=arrays,
                )
