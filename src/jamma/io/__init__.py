"""I/O modules for JAMMA.

This package contains modules for reading and writing various file formats:
- plink: PLINK file checks, the .bed reader strategy and .fam phenotypes
- covariate: GEMMA-format covariate file reading
"""

from jamma.io.covariate import read_covariate_file
from jamma.io.plink import parse_fam_phenotype_column, read_fam_phenotypes
from jamma.io.snp_list import (
    read_snp_list_file,
    resolve_snp_list_file,
    resolve_snp_list_to_indices,
)
from jamma.io.weight import read_weight_file

__all__ = [
    "parse_fam_phenotype_column",
    "read_covariate_file",
    "read_fam_phenotypes",
    "read_snp_list_file",
    "read_weight_file",
    "resolve_snp_list_file",
    "resolve_snp_list_to_indices",
]
