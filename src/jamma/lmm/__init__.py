"""Linear Mixed Model (LMM) association testing.

GEMMA-compatible LMM association tests using JAX for accelerated computation.
Core algorithm follows Zhou & Stephens (2012) Nature Genetics.

Modules:
- runner_jax: Batch processing (genotypes in memory)
- runner_streaming: Disk streaming (genotypes per chunk)
- runner_numpy: Pure-NumPy streaming runner (no JAX required)
- chunk: Chunk size computation
- prepare: Shared setup (device, covariates, eigendecomp, null model)
- results: Result building functions
- likelihood_jax: JAX-optimized REML/MLE and optimization
- eigen: Eigendecomposition with GEMMA-compatible thresholding
- stats: AssocResult dataclass
- io: Result file I/O
"""

# Always available (no JAX dependency):
from jamma.lmm.chunk import auto_tune_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.io import write_assoc_results
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.stats import AssocResult

# JAX-dependent runners — conditional import so `import jamma` works without JAX:
try:
    from jamma.lmm.loco import run_lmm_loco
    from jamma.lmm.runner_jax import run_lmm_association_jax
    from jamma.lmm.runner_streaming import run_lmm_association_streaming
except ImportError:
    pass  # JAX not installed; NumPy backend only

__all__ = [
    "auto_tune_chunk_size",
    "run_lmm_association_jax",
    "run_lmm_association_numpy",
    "run_lmm_association_streaming",
    "run_lmm_loco",
    "AssocResult",
    "eigendecompose_kinship",
    "read_eigen_files",
    "write_assoc_results",
    "write_eigen_files",
]
