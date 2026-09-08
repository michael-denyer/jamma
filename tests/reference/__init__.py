"""GEMMA-literal scalar oracles.

Production runs the batch NumPy and C kernels. These per-SNP ports of GEMMA's
lmm.cpp keep the reference arithmetic one line per formula so tests can hold
the vectorised paths to it.
"""

# The oracle's own absolute floor on a projected residual, GEMMA master's
# `P_yy >= 0 && P_yy < 1e-8` form. Production replaces an exact zero only.
P_YY_FLOOR = 1e-8
