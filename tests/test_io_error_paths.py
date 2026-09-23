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

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.variants import SnpMeta
from jamma.io.plink import validate_plink_dimensions
from jamma.lmm.assoc_output import IncrementalAssocWriter
from jamma.lmm.schema import MODE_SPECS
from tests.fixture_paths import LOCO, SYNTHETIC
from tests.support import require_fixture

pytestmark = pytest.mark.tier0

BFILE = SYNTHETIC.bfile
LOCO_BFILE = LOCO.bfile


class TestPlinkIOErrorPaths:
    """Error-path tests for PLINK I/O validation functions.

    Tests validate_plink_dimensions (per-extension missing files) and
    GenotypeDataset.partitions (multi-chromosome handling).

    Truncated .bed and genotype value tests live in test_plink_validation.py
    to avoid duplication.
    """

    def test_missing_bed_raises(self, tmp_path: Path) -> None:
        """Missing .bed (with .bim and .fam present) raises FileNotFoundError."""
        shutil.copy(SYNTHETIC.bim, tmp_path / "test.bim")
        shutil.copy(SYNTHETIC.fam, tmp_path / "test.fam")

        with pytest.raises(FileNotFoundError, match=r"\.bed"):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_bim_raises(self, tmp_path: Path) -> None:
        """Missing .bim (with .bed and .fam present) raises FileNotFoundError."""
        shutil.copy(SYNTHETIC.bed, tmp_path / "test.bed")
        shutil.copy(SYNTHETIC.fam, tmp_path / "test.fam")

        with pytest.raises(FileNotFoundError, match=r"\.bim"):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_fam_raises(self, tmp_path: Path) -> None:
        """Missing .fam (with .bed and .bim present) raises FileNotFoundError."""
        shutil.copy(SYNTHETIC.bed, tmp_path / "test.bed")
        shutil.copy(SYNTHETIC.bim, tmp_path / "test.bim")

        with pytest.raises(FileNotFoundError, match=r"\.fam"):
            validate_plink_dimensions(tmp_path / "test")

    # Genotype value validation tests (out-of-range counting, all-valid)
    # live in test_plink_validation.py::TestValidateGenotypeValues.

    def test_multi_chromosome_partitions(self) -> None:
        """GenotypeDataset.partitions returns correct multi-chromosome partitions.

        Uses gemma_loco fixture which has chromosomes 1, 2, and 3.
        Verifies the returned dict has >= 3 keys and total SNP count
        matches the BIM line count.
        """
        require_fixture(LOCO_BFILE.with_suffix(".bed"), LOCO_BFILE.with_suffix(".bim"))

        partitions = GenotypeDataset.open_plink(LOCO_BFILE).partitions

        assert len(partitions) >= 3

        # Verify SNP counts are consistent with the BIM file
        bim_path = LOCO_BFILE.with_suffix(".bim")
        n_bim_snps = len(bim_path.read_text().strip().splitlines())
        total_snps = sum(len(indices) for indices in partitions.values())
        assert total_snps == n_bim_snps


class TestLmmIOErrorPaths:
    """Error-path tests for LMM I/O association writer.

    Tests write_arrays_batch (missing stat keys, missing snp_info keys,
    length mismatch).
    """

    def test_write_arrays_batch_missing_stat_keys_raises(self, tmp_path: Path) -> None:
        """write_arrays_batch rejects empty arrays (missing all wald stat keys)."""
        snp_indices = np.array([0])
        snp_info = SnpMeta.from_dicts(
            [{"chr": "1", "rs": "rs1", "pos": 100, "a1": "A", "a0": "G"}]
        )
        afs = np.array([0.3])
        miss_counts = np.array([0])

        with IncrementalAssocWriter(tmp_path / "out.txt", MODE_SPECS[1]) as writer:
            with pytest.raises(ValueError, match="missing arrays"):
                # arrays={} is missing all required wald stat keys
                writer.write_arrays_batch(
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays={},
                )

    def test_snp_meta_from_dicts_missing_keys_raises(self) -> None:
        """SnpMeta.from_dicts rejects a dict missing canonical keys."""
        # Only 'chr' key provided; missing rs, pos, a1, a0
        with pytest.raises(KeyError):
            SnpMeta.from_dicts([{"chr": "1"}])

    def test_write_arrays_batch_length_mismatch_raises(self, tmp_path: Path) -> None:
        """write_arrays_batch raises ValueError when afs length != snp_indices."""
        # 2 SNPs but afs has only 1 value
        snp_indices = np.array([0, 1])
        snp_info = SnpMeta.from_dicts(
            [
                {"chr": "1", "rs": "rs1", "pos": 100, "a1": "A", "a0": "G"},
                {"chr": "1", "rs": "rs2", "pos": 200, "a1": "A", "a0": "G"},
            ]
        )
        afs = np.array([0.3])  # length 1, mismatches snp_indices length 2
        miss_counts = np.array([0, 0])

        spec = MODE_SPECS[1]
        arrays = {c.array_key: np.array([0.1, 0.2]) for c in spec.stat_columns}

        with IncrementalAssocWriter(tmp_path / "out.txt", spec) as writer:
            with pytest.raises(ValueError, match="length"):
                writer.write_arrays_batch(
                    snp_indices=snp_indices,
                    snp_info=snp_info,
                    afs=afs,
                    miss_counts=miss_counts,
                    arrays=arrays,
                )
