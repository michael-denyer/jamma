"""I/O modules for JAMMA.

This package contains modules for reading and writing various file formats:
- plink: PLINK binary format (.bed/.bim/.fam) I/O
- covariate: GEMMA-format covariate file reading
- output: GEMMA-compatible output file writers
"""

from jamma.io.covariate import read_covariate_file
from jamma.io.plink import (
    PlinkData,
    PlinkMetadata,
    get_plink_metadata,
    load_plink_binary,
    parse_fam_phenotype_column,
    read_fam_phenotypes,
    stream_genotype_chunks,
)
from jamma.io.snp_list import (
    read_snp_list_file,
    resolve_snp_list_file,
    resolve_snp_list_to_indices,
)
from jamma.io.weight import read_weight_file

__all__ = [
    "PlinkData",
    "PlinkMetadata",
    "get_plink_metadata",
    "load_plink_binary",
    "parse_fam_phenotype_column",
    "read_covariate_file",
    "read_fam_phenotypes",
    "read_snp_list_file",
    "read_weight_file",
    "resolve_snp_list_file",
    "resolve_snp_list_to_indices",
    "stream_genotype_chunks",
]
