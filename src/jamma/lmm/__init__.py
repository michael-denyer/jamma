"""Linear Mixed Model (LMM) association testing.

This module implements GEMMA-compatible LMM association tests using JAX
for accelerated numerical computation. The core algorithm follows
Zhou & Stephens (2012) Nature Genetics.

Key components:
- eigendecompose_kinship: Eigendecomposition with GEMMA-compatible thresholding
- reml_log_likelihood: REML log-likelihood for variance component estimation
- mle_log_likelihood: MLE log-likelihood for LRT
- optimize_lambda: Brent's method optimization for variance ratio
- calc_wald_test: Wald test statistics (beta, SE, p-value)
- calc_lrt_test: Likelihood ratio test p-value
- run_lmm_association_jax: JAX-optimized batch processing
- run_lmm_association_streaming: JAX streaming with incremental disk writes

Test modes:
- lmm_mode=1: Wald test (default) - per-SNP REML optimization
- lmm_mode=2: LRT - null MLE once, per-SNP alternative MLE
- lmm_mode=3: Score test - null MLE once, no per-SNP optimization
- lmm_mode=4: All tests - Wald + LRT + Score in one pass with computation reuse
"""

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import write_assoc_results
from jamma.lmm.runner_jax import (
    auto_tune_chunk_size,
    run_lmm_association_jax,
    run_lmm_association_streaming,
)
from jamma.lmm.stats import AssocResult

__all__ = [
    "auto_tune_chunk_size",
    "run_lmm_association_jax",
    "run_lmm_association_streaming",
    "AssocResult",
    "eigendecompose_kinship",
    "write_assoc_results",
]
