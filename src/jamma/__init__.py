"""JAMMA: Fast Mixed Model Association for GWAS.

JAMMA provides a Python-based implementation of genome-wide efficient mixed model
association analysis, targeting exact numerical compatibility with the original
GEMMA C++ implementation while scaling to modern biobank datasets.

Key features:
- Exact statistical output match with original GEMMA
- Cross-platform: NumPy backend with optional C extension acceleration
- Scalable to 200k+ samples with ILP64 vendor BLAS
- Modern Python packaging and CLI interface

Example:
    >>> from jamma import gwas
    >>> result = gwas("data/my_study", kinship_file="data/kinship.cXX.txt")
    >>> print(f"{result.n_snps_tested} SNPs in {result.timing['total_s']:.1f}s")
"""

import os

# Tolerate coexistence of Intel OpenMP (libiomp5, used by MKL numpy and
# jlinalg) with GNU OpenMP (libgomp, pulled in by scipy or system libs).
# Must be set before any C extension import triggers OpenMP initialization.
# On Databricks, both MKL and scipy are pre-loaded by the kernel, so this
# must happen at the earliest possible import point.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from importlib.metadata import version

__version__ = version("jamma")

try:
    from jamma._build_meta import BUILD_DATE as __release_date__
except ImportError:
    __release_date__ = "dev"

from jamma.gwas import GWASResult, gwas  # noqa: E402

__all__ = ["gwas", "GWASResult", "__version__", "__release_date__"]
