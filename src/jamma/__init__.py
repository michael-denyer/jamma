"""JAMMA: JAX-Accelerated Mixed Model Association.

JAMMA provides a Python-based implementation of genome-wide efficient mixed model
association analysis, targeting exact numerical compatibility with the original
GEMMA C++ implementation while scaling to modern biobank datasets.

Key features:
- Exact statistical output match with original GEMMA
- Scalable to 200k+ samples using JAX
- Modern Python packaging and CLI interface

Example:
    >>> import jamma
    >>> jamma.__version__
    '0.1.0'
"""

__version__ = "0.1.0"
