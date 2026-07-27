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


def get_chromosome_partitions(bed_path: Path) -> dict[str, np.ndarray]:
    """Get SNP column indices grouped by chromosome from BIM file.

    Opens the PLINK .bed file to read the chromosome array from BIM metadata,
    then groups SNP indices by chromosome name. Chromosome names are preserved
    exactly as they appear in the BIM file (e.g., '1', 'chr1', 'X').

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).

    Returns:
        Dict mapping chromosome name (string) to sorted np.ndarray of SNP
        column indices. Keys are ordered by first appearance in the BIM file.

    Raises:
        FileNotFoundError: If the .bed file does not exist.

    Example:
        >>> partitions = get_chromosome_partitions(Path("data/mouse_hs1940"))
        >>> list(partitions.keys())[:3]
        ['1', '2', '3']
        >>> partitions['1'].shape
        (...)
    """
    bed_file = Path(f"{bed_path}.bed")

    if not bed_file.exists():
        raise FileNotFoundError(f"PLINK .bed file not found: {bed_file}")

    with open_bed(bed_file) as bed:
        chromosomes = bed.chromosome
        # Preserve BIM order: unique chromosomes by first appearance
        _, first_idx = np.unique(chromosomes, return_index=True)
        unique_chrs = [chromosomes[i] for i in np.sort(first_idx)]
        return {
            chr_name: np.where(chromosomes == chr_name)[0] for chr_name in unique_chrs
        }


def partitions_from_metadata(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """Derive chromosome partitions from already-loaded PLINK metadata.

    Equivalent to get_chromosome_partitions but avoids opening the BED
    file a second time. Use when get_plink_metadata has already been called.

    Args:
        meta: Metadata dict from get_plink_metadata (must contain 'chromosome').

    Returns:
        Dict mapping chromosome name to sorted array of SNP column indices.
        Keys ordered by first appearance in BIM file.
    """
    chromosomes = meta.get("chromosome")
    if chromosomes is None:
        raise ValueError(
            "partitions_from_metadata requires 'chromosome' key in metadata dict. "
            "Ensure this is the return value of get_plink_metadata()."
        )
    _, first_idx = np.unique(chromosomes, return_index=True)
    unique_chrs = [chromosomes[i] for i in np.sort(first_idx)]
    return {chr_name: np.where(chromosomes == chr_name)[0] for chr_name in unique_chrs}


def _count_lines_fast(path: Path, chunk_size: int = 1024 * 1024) -> int:
    """Count logical lines in a file using binary byte counting.

    Reads the file in binary mode and counts newline bytes in chunks.
    2-3x faster than text-mode iteration for large files because it
    avoids line decoding overhead. Handles files without a trailing
    newline by checking the last byte.

    Args:
        path: Path to the file to count lines in.
        chunk_size: Read buffer size in bytes (default 1 MB).

    Returns:
        Number of logical lines in the file.
    """
    count = 0
    last_byte = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        count += 1
    return count


def validate_plink_dimensions(bfile: Path) -> None:
    """Validate that PLINK .bed file size matches .fam and .bim counts.

    Computes the expected .bed file size from .fam sample count and .bim
    SNP count, then compares with the actual file size. This provides a
    more informative error message than bed-reader's built-in check when
    files are corrupted or truncated.

    Expected size: 3 (magic bytes) + ceil(n_fam / 4) * n_bim bytes.

    Args:
        bfile: Path prefix for PLINK files (without .bed/.bim/.fam extension).

    Raises:
        FileNotFoundError: If any of the .bed, .bim, or .fam files are missing.
        ValueError: If .bed file size does not match expected size from
            .fam and .bim dimensions.
    """
    bed_path = Path(f"{bfile}.bed")
    bim_path = Path(f"{bfile}.bim")
    fam_path = Path(f"{bfile}.fam")

    for path, ext in ((bed_path, ".bed"), (bim_path, ".bim"), (fam_path, ".fam")):
        if not path.exists():
            raise FileNotFoundError(f"PLINK {ext} file not found: {path}")

    # Count lines in .fam (= n_samples) and .bim (= n_snps)
    logger.info(f"Validating PLINK dimensions: {fam_path}")
    n_fam = _count_lines_fast(fam_path)
    n_bim = _count_lines_fast(bim_path)

    # Expected .bed size: 3 magic bytes + ceil(n_fam/4) bytes per SNP
    bytes_per_snp = (n_fam + 3) // 4
    expected_size = 3 + bytes_per_snp * n_bim
    actual_size = bed_path.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            f"PLINK dimension mismatch: .fam has {n_fam} samples, "
            f".bim has {n_bim} SNPs, but .bed file size ({actual_size} bytes) "
            f"doesn't match expected ({expected_size} bytes)"
        )


def validate_genotype_values(chunk: np.ndarray) -> int:
    """Check that all non-NaN genotype values are in {0.0, 1.0, 2.0}.

    Called per-chunk during pass-1 streaming. The caller accumulates
    the total count and logs a single summary warning at the end.

    Args:
        chunk: Genotype matrix chunk (n_samples, n_snps_chunk).

    Returns:
        Count of unexpected values (not in {0, 1, 2, NaN}).
    """
    # Count values outside {0, 1, 2, NaN}.
    # Valid genotypes are integers in [0, 2]; NaN is missing data (also valid).
    # Boolean equality checks avoid large temporary allocations from membership
    # tests when processing 100k x 10k float32 chunks.
    not_nan = ~np.isnan(chunk)
    valid_geno = (chunk == 0.0) | (chunk == 1.0) | (chunk == 2.0)
    return int(np.count_nonzero(not_nan & ~valid_geno))


def stream_genotype_chunks(
    bed_path: Path,
    chunk_size: int = 10_000,
    dtype: type = np.float32,
    show_progress: bool = True,
    snp_indices: np.ndarray | None = None,
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
        snp_indices: Sorted array of column indices to read. When provided,
            only these columns are read from the BED file. chunk_size applies
            to the filtered index space, not the total SNP count. Yields
            (chunk, global_start_idx, global_end_idx) where indices refer to
            positions in snp_indices, not BED file columns.

    Yields:
        Tuple of (genotypes_chunk, start_idx, end_idx):
        - genotypes_chunk: Array of shape (n_samples, chunk_snps)
        - start_idx: First SNP index (inclusive)
        - end_idx: Last SNP index (exclusive)
        When snp_indices is None, indices are absolute BED column positions.
        When snp_indices is provided, indices are positions within snp_indices.

    Raises:
        FileNotFoundError: If the .bed file does not exist.

    Example:
        >>> chunks = stream_genotype_chunks(Path("data"), chunk_size=5000)
        >>> for chunk, start, end in chunks:
        ...     print(f"SNPs {start}-{end}: shape {chunk.shape}")
        SNPs 0-5000: shape (1940, 5000)
        SNPs 5000-10000: shape (1940, 5000)
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    bed_file = Path(f"{bed_path}.bed")

    if not bed_file.exists():
        raise FileNotFoundError(f"PLINK .bed file not found: {bed_file}")

    with open_bed(bed_file) as bed:
        n_samples = bed.iid_count

        if snp_indices is not None:
            # Filtered mode: validate and read only requested columns
            if len(snp_indices) > 1 and np.any(np.diff(snp_indices) <= 0):
                raise ValueError(
                    "snp_indices must be sorted in strictly ascending order"
                )
            if len(snp_indices) > 0 and (
                snp_indices[0] < 0 or snp_indices[-1] >= bed.sid_count
            ):
                raise ValueError(
                    f"snp_indices out of bounds: range [{snp_indices[0]}, "
                    f"{snp_indices[-1]}], BED file has {bed.sid_count} SNPs"
                )
            n_total = len(snp_indices)
            label = "filtered SNPs"

            def read_chunk(start: int, end: int) -> np.ndarray:
                return bed.read(index=(np.s_[:], snp_indices[start:end]), dtype=dtype)
        else:
            # Unfiltered mode: read all columns sequentially
            n_total = int(bed.sid_count)
            label = "SNPs"

            def read_chunk(start: int, end: int) -> np.ndarray:
                return bed.read(index=np.s_[:, start:end], dtype=dtype)

        n_chunks = (n_total + chunk_size - 1) // chunk_size
        logger.info(
            f"Reading {n_total} {label} in {n_chunks} chunks "
            f"of {chunk_size} ({n_samples} samples)"
        )

        iterator = range(0, n_total, chunk_size)
        if show_progress:
            iterator = progress_iterator(
                iterator, total=n_chunks, desc="Reading genotypes"
            )

        for start in iterator:
            end = min(start + chunk_size, n_total)
            yield read_chunk(start, end), start, end


def prefetch_iterator(
    source: Iterator[tuple[np.ndarray, int, int]],
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Wrap a chunk iterator with one-item-ahead background prefetch.

    Uses a single background thread to read the next chunk while the
    current chunk is being processed. The ThreadPoolExecutor ensures
    proper lifecycle management and exception propagation.

    Memory impact: two chunks are live simultaneously during the overlap
    window. Callers should account for this in their memory budget
    (e.g., pipeline_buffers=2 in chunk sizing).

    Args:
        source: An iterator yielding (chunk, start, end) tuples.

    Yields:
        Same (chunk, start, end) tuples as the source, with I/O for
        the next item started in the background.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            nxt = next(source)
        except StopIteration:
            return

        while True:
            current = nxt
            fut = pool.submit(next, source)
            yield current
            try:
                nxt = fut.result()
            except StopIteration:
                return
