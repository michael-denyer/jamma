"""PLINK binary format I/O using bed-reader.

This module provides loading of PLINK binary files (.bed/.bim/.fam) which is
the primary input format for GEMMA analysis.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bed_reader import open_bed


@dataclass
class PlinkData:
    """Container for PLINK binary data.

    Attributes:
        genotypes: Genotype matrix with shape (n_samples, n_snps).
            Values are 0.0 (hom ref), 1.0 (het), 2.0 (hom alt), or NaN (missing).
        iid: Sample IDs as 2D array with columns [FID, IID].
        sid: SNP IDs (variant identifiers).
        chromosome: Chromosome for each SNP.
        bp_position: Base pair position for each SNP.
        allele_1: Reference allele for each SNP.
        allele_2: Alternate allele for each SNP.
    """

    genotypes: np.ndarray
    iid: np.ndarray
    sid: np.ndarray
    chromosome: np.ndarray
    bp_position: np.ndarray
    allele_1: np.ndarray
    allele_2: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.genotypes.shape[0]

    @property
    def n_snps(self) -> int:
        """Number of SNPs in the dataset."""
        return self.genotypes.shape[1]


def load_plink_binary(bfile: Path) -> PlinkData:
    """Load PLINK binary files (.bed/.bim/.fam).

    Args:
        bfile: Path prefix for PLINK files (without .bed/.bim/.fam extension).
            For example, if files are data.bed, data.bim, data.fam, pass Path("data").

    Returns:
        PlinkData container with genotypes and metadata.

    Raises:
        FileNotFoundError: If the .bed file does not exist.

    Example:
        >>> data = load_plink_binary(Path("legacy/example/mouse_hs1940"))
        >>> print(f"{data.n_samples} samples, {data.n_snps} SNPs")
        1940 samples, 12226 SNPs
    """
    bed_path = Path(f"{bfile}.bed")

    if not bed_path.exists():
        raise FileNotFoundError(f"PLINK .bed file not found: {bed_path}")

    with open_bed(bed_path) as bed:
        # read() returns (n_samples, n_snps) float array
        # Values: 0.0 = hom ref, 1.0 = het, 2.0 = hom alt, NaN = missing
        genotypes = bed.read(dtype=np.float32)

        return PlinkData(
            genotypes=genotypes,
            iid=bed.iid,
            sid=bed.sid,
            chromosome=bed.chromosome,
            bp_position=bed.bp_position,
            allele_1=bed.allele_1,
            allele_2=bed.allele_2,
        )
