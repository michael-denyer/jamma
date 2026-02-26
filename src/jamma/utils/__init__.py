"""Utility modules for JAMMA.

This package contains supporting utilities:
- logging: Loguru configuration and GEMMA-compatible log output
- chr_sort_key: Biological chromosome ordering (numeric, then X, Y, XY, MT)
"""

from loguru import logger

from jamma.utils.logging import setup_logging, write_gemma_log

# Chromosome ordering: numeric first (by integer value), then special (X, Y, XY, MT)
_CHR_SPECIAL_ORDER = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}


def chr_sort_key(chrom: str) -> tuple[int, str]:
    """Sort key for biological chromosome order (numeric, then X, Y, XY, MT).

    Numeric chromosomes sort by integer value, special chromosomes (X, Y, XY, MT)
    sort after all numeric ones, and unknown names sort last alphabetically.

    Args:
        chrom: Chromosome name (e.g., "1", "22", "X", "MT").

    Returns:
        Tuple suitable for use as a sort key.
    """
    upper = chrom.upper()
    if upper in _CHR_SPECIAL_ORDER:
        return (_CHR_SPECIAL_ORDER[upper], "")
    try:
        return (int(chrom), "")
    except ValueError:
        logger.info(
            f"Non-numeric chromosome '{chrom}' — sorting after known chromosomes"
        )
        return (1000, chrom)


__all__ = ["setup_logging", "write_gemma_log", "chr_sort_key"]
