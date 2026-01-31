"""GEMMA-Next: Modern reimplementation of GEMMA for genome-wide association studies.

GEMMA-Next provides a Python-based implementation of the GEMMA (Genome-wide Efficient
Mixed Model Association) algorithm, targeting exact numerical compatibility with the
original C++ implementation while scaling to modern biobank datasets.

Key features:
- Exact statistical output match with original GEMMA
- Scalable to 200k+ samples using JAX
- Modern Python packaging and CLI interface

Example:
    >>> import gemma_next
    >>> gemma_next.__version__
    '0.1.0'
"""

__version__ = "0.1.0"
