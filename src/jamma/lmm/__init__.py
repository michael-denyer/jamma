"""Linear Mixed Model (LMM) association testing.

GEMMA-compatible LMM association tests using JAX for accelerated computation.
Core algorithm follows Zhou & Stephens (2012) Nature Genetics.

Modules:
- runner_jax: Batch processing (genotypes in memory)
- runner_streaming: Disk streaming (genotypes per chunk)
- chunk: Chunk size computation
- prepare: Shared setup (device, covariates, eigendecomp, null model)
- results: Result building functions
- likelihood_jax: JAX-optimized REML/MLE and optimization
- eigen: Eigendecomposition with GEMMA-compatible thresholding
- stats: AssocResult dataclass
- io: Result file I/O
"""

from jamma.lmm.chunk import auto_tune_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import write_assoc_results
from jamma.lmm.loco import run_lmm_loco
from jamma.lmm.runner_jax import run_lmm_association_jax
from jamma.lmm.runner_streaming import run_lmm_association_streaming
from jamma.lmm.stats import AssocResult

__all__ = [
    "auto_tune_chunk_size",
    "run_lmm_association_jax",
    "run_lmm_association_streaming",
    "run_lmm_loco",
    "AssocResult",
    "eigendecompose_kinship",
    "write_assoc_results",
]
