"""GEMMA-literal scalar oracles.

Production runs the batch NumPy and C kernels. These per-SNP ports of GEMMA's
lmm.cpp keep the reference arithmetic one line per formula so tests can hold
the vectorised paths to it.
"""
