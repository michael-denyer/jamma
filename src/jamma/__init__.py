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

import sys

from loguru import logger

__version__ = "0.1.0"

# Configure loguru with sensible defaults on import
# Uses stdout so output is visible in Databricks notebook cells (stderr may be buffered)
# Users can override by calling logger.remove()/add()
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss} | <level>{level: <8}</level> | {message}",
    colorize=True,
)
