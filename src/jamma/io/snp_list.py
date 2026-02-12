"""SNP list file I/O for JAMMA.

Provides functions for reading GEMMA-format SNP list files (one RS ID per line)
and resolving SNP IDs to column indices via BIM intersection.
"""

from pathlib import Path

import numpy as np
from loguru import logger


def read_snp_list_file(path: Path) -> set[str]:
    """Read a GEMMA-format SNP list file into a set of RS IDs.

    Parses a text file with one RS ID per line. Strips whitespace, skips
    empty lines, and uses the first whitespace-delimited token per line
    (matching GEMMA's strtok behavior for lines with extra columns).

    Args:
        path: Path to the SNP list file.

    Returns:
        Set of SNP RS IDs for O(1) membership testing.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains no valid IDs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SNP list file not found: {path}")

    snp_ids: set[str] = set()
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Use first whitespace-delimited token (matches GEMMA strtok)
            token = stripped.split()[0]
            snp_ids.add(token)

    if not snp_ids:
        raise ValueError(f"SNP list file is empty or contains no valid IDs: {path}")

    return snp_ids


def resolve_snp_list_to_indices(snp_ids: set[str], bim_sids: np.ndarray) -> np.ndarray:
    """Resolve SNP RS IDs to column indices via BIM intersection.

    Builds a lookup dict from bim_sids, then finds the index of each
    requested SNP ID. Case-sensitive comparison (matching GEMMA).

    Args:
        snp_ids: Set of SNP RS IDs to look up.
        bim_sids: Array of SNP IDs from the BIM file.

    Returns:
        Sorted array of column indices (dtype=np.intp) for matched SNPs.

    Raises:
        ValueError: If zero SNPs match.
    """
    sid_to_index = {sid: i for i, sid in enumerate(bim_sids)}

    indices = []
    for snp_id in snp_ids:
        idx = sid_to_index.get(snp_id)
        if idx is not None:
            indices.append(idx)

    n_found = len(indices)
    n_requested = len(snp_ids)

    if n_found == 0:
        raise ValueError(
            f"No SNPs from the list matched the BIM file (requested {n_requested} SNPs)"
        )

    if n_found < n_requested:
        n_missing = n_requested - n_found
        logger.warning(
            f"SNP list: {n_found}/{n_requested} SNPs found in BIM ({n_missing} missing)"
        )

    return np.array(sorted(indices), dtype=np.intp)
