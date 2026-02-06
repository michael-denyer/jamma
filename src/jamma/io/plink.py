"""PLINK binary format I/O using bed-reader.

This module provides loading of PLINK binary files (.bed/.bim/.fam) which is
the primary input format for GEMMA analysis.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.core.progress import progress_iterator


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


def get_plink_metadata(bfile: Path) -> dict[str, Any]:
    """Get PLINK file metadata without loading genotypes.

    Opens the PLINK files to read dimensions and metadata arrays without
    loading the genotype matrix. Useful for streaming workflows that need
    to know dimensions before iteration.

    Args:
        bfile: Path prefix for PLINK files (without .bed/.bim/.fam extension).

    Returns:
        Dictionary with keys:
        - n_samples: Number of samples (individuals)
        - n_snps: Number of SNPs (variants)
        - iid: Sample IDs as 2D array with columns [FID, IID]
        - sid: SNP IDs (variant identifiers)
        - chromosome: Chromosome for each SNP
        - bp_position: Base pair position for each SNP
        - allele_1: Reference allele for each SNP
        - allele_2: Alternate allele for each SNP

    Raises:
        FileNotFoundError: If the .bed file does not exist.

    Example:
        >>> meta = get_plink_metadata(Path("legacy/example/mouse_hs1940"))
        >>> print(f"{meta['n_samples']} samples, {meta['n_snps']} SNPs")
        1940 samples, 12226 SNPs
    """
    bed_path = Path(f"{bfile}.bed")

    if not bed_path.exists():
        raise FileNotFoundError(f"PLINK .bed file not found: {bed_path}")

    with open_bed(bed_path) as bed:
        return {
            "n_samples": bed.iid_count,
            "n_snps": bed.sid_count,
            "iid": bed.iid,
            "sid": bed.sid,
            "chromosome": bed.chromosome,
            "bp_position": bed.bp_position,
            "allele_1": bed.allele_1,
            "allele_2": bed.allele_2,
        }


def stream_genotype_chunks(
    bed_path: Path,
    chunk_size: int = 10_000,
    dtype: type = np.float32,
    show_progress: bool = True,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Stream genotype chunks from disk without full matrix load.

    Opens the PLINK .bed file once and yields genotype chunks via windowed
    reads. The file handle stays open across all yields, avoiding the overhead
    of repeated metadata parsing.

    Memory: O(n_samples * chunk_size) per chunk, never O(n_samples * n_snps).

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        chunk_size: Number of SNPs per chunk (default 10,000).
        dtype: Output dtype for genotypes (default float32 for memory efficiency).
        show_progress: Whether to show progress bar (default True).

    Yields:
        Tuple of (genotypes_chunk, start_idx, end_idx):
        - genotypes_chunk: Array of shape (n_samples, chunk_snps)
        - start_idx: First SNP index (inclusive)
        - end_idx: Last SNP index (exclusive)

    Raises:
        FileNotFoundError: If the .bed file does not exist.

    Example:
        >>> chunks = stream_genotype_chunks(Path("data"), chunk_size=5000)
        >>> for chunk, start, end in chunks:
        ...     print(f"SNPs {start}-{end}: shape {chunk.shape}")
        SNPs 0-5000: shape (1940, 5000)
        SNPs 5000-10000: shape (1940, 5000)
    """
    bed_file = Path(f"{bed_path}.bed")

    if not bed_file.exists():
        raise FileNotFoundError(f"PLINK .bed file not found: {bed_file}")

    with open_bed(bed_file) as bed:
        n_samples = bed.iid_count
        n_snps = bed.sid_count
        n_chunks = (n_snps + chunk_size - 1) // chunk_size

        logger.info(
            f"Reading {n_snps} SNPs in {n_chunks} chunks of {chunk_size} "
            f"({n_samples} samples)"
        )

        iterator = range(0, n_snps, chunk_size)
        if show_progress:
            iterator = progress_iterator(
                iterator, total=n_chunks, desc="Reading genotypes"
            )

        for start in iterator:
            end = min(start + chunk_size, n_snps)
            # Windowed read: only reads bytes for SNPs [start:end]
            chunk = bed.read(index=np.s_[:, start:end], dtype=dtype)
            yield chunk, start, end
