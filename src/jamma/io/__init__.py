"""I/O modules for JAMMA.

This package contains modules for reading and writing various file formats:
- plink: PLINK binary format (.bed/.bim/.fam) I/O
- output: GEMMA-compatible output file writers
"""

from jamma.io.plink import (
    PlinkData,
    get_plink_metadata,
    load_plink_binary,
    stream_genotype_chunks,
)

__all__ = [
    "PlinkData",
    "get_plink_metadata",
    "load_plink_binary",
    "stream_genotype_chunks",
]
