"""I/O modules for JAMMA.

This package contains modules for reading and writing various file formats:
- plink: PLINK binary format (.bed/.bim/.fam) I/O
- covariate: GEMMA-format covariate file reading
- output: GEMMA-compatible output file writers
"""

from jamma.io.covariate import read_covariate_file
from jamma.io.plink import (
    PlinkData,
    get_chromosome_partitions,
    get_plink_metadata,
    load_plink_binary,
    stream_genotype_chunks,
)

__all__ = [
    "PlinkData",
    "get_chromosome_partitions",
    "get_plink_metadata",
    "load_plink_binary",
    "read_covariate_file",
    "stream_genotype_chunks",
]
