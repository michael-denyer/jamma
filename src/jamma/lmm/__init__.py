"""Linear Mixed Model (LMM) association testing.

This module implements GEMMA-compatible LMM association tests using JAX
for accelerated numerical computation. The core algorithm follows
Zhou & Stephens (2012) Nature Genetics.

Key components:
- eigendecompose_kinship: Eigendecomposition with GEMMA-compatible thresholding
- reml_log_likelihood: REML log-likelihood for variance component estimation
- optimize_lambda: Brent's method optimization for variance ratio
- calc_wald_test: Wald test statistics (beta, SE, p-value)
"""
