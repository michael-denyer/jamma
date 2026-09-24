"""PLINK binary format I/O using bed-reader.

``PlinkReader`` is the ``.bed`` strategy behind
``GenotypeDataset.open_plink``; the rest checks PLINK files and reads ``.fam``
phenotypes.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from bed_reader import open_bed
from loguru import logger

from jamma.io.phenotype import phenotype_values


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


class PlinkReader:
    """bed-reader behind ``GenotypeDataset``'s reader strategy."""

    def __init__(self, bfile: Path) -> None:
        self._bed = Path(f"{bfile}.bed")
        self._bim = Path(f"{bfile}.bim")

    def read(
        self,
        columns: np.ndarray,
        block_size: int,
        *,
        stats_only: bool,
        info_rows: np.ndarray | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield blocks of ``columns`` from one ``open_bed``, float32 for stats.

        ``info_rows`` is ignored: hard calls carry no INFO.

        A block of consecutive columns is read as a slice, any other block by
        index; both give the same values. Logs one ``Reading N SNPs`` line
        per pass, the user's record of how often the file was read.
        """
        dtype = np.float32 if stats_only else np.float64
        with open_bed(self._bed) as bed:
            label = "SNPs" if len(columns) == bed.sid_count else "filtered SNPs"
            n_blocks = (len(columns) + block_size - 1) // block_size
            logger.info(
                f"Reading {len(columns)} {label} in {n_blocks} chunks "
                f"of {block_size} ({bed.iid_count} samples)"
            )
            for start in range(0, len(columns), block_size):
                block = columns[start : start + block_size]
                first, last = int(block[0]), int(block[-1])
                if last - first + 1 == len(block):
                    yield bed.read(index=np.s_[:, first : last + 1], dtype=dtype)
                else:
                    yield bed.read(index=(np.s_[:], block), dtype=dtype)

    def fingerprint(self) -> dict[str, str]:
        """Return the LOCO eigen cache's ``bed_fingerprint`` and ``bim_sha256``."""
        st = self._bed.stat()
        with open(self._bim, "rb") as fh:
            bim_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
        return {
            "bed_fingerprint": f"{self._bed.name}:{st.st_size}:{st.st_mtime_ns}",
            "bim_sha256": bim_sha256,
        }


def parse_fam_phenotype_column(fam_data: np.ndarray, column: int) -> np.ndarray:
    """Return phenotype ``column`` (1-based) of a ``.fam`` read as strings.

    ``-9`` and ``NA`` become NaN. Columns 1 to 5 of the file are FID, IID,
    father, mother and sex, so phenotype column 1 is file column 6.

    Raises:
        ValueError: If the file has fewer phenotype columns than ``column``.
    """
    col_index = 4 + column
    n_cols = fam_data.shape[1]
    if col_index >= n_cols:
        n_pheno_cols = n_cols - 5
        raise ValueError(
            f"phenotype column {column} exceeds available columns "
            f"in .fam file ({n_pheno_cols} phenotype column"
            f"{'s' if n_pheno_cols != 1 else ''} available)"
        )
    return phenotype_values(fam_data[:, col_index])


def read_fam_phenotypes(fam_path: Path, column: int = 1) -> np.ndarray:
    """Read one phenotype column of a PLINK ``.fam`` file, NaN for missing."""
    fam_data = np.loadtxt(fam_path, dtype=str, ndmin=2)
    return parse_fam_phenotype_column(fam_data, column)
