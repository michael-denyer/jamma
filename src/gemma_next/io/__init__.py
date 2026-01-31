"""I/O modules for GEMMA-Next.

This package contains modules for reading and writing various file formats:
- plink: PLINK binary format (.bed/.bim/.fam) I/O
- output: GEMMA-compatible output file writers
"""

from gemma_next.io.plink import PlinkData, load_plink_binary

__all__ = ["PlinkData", "load_plink_binary"]
