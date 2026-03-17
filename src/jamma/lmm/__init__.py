"""Linear Mixed Model (LMM) association testing.

GEMMA-compatible LMM association tests with JAX and NumPy backends.
Core algorithm follows Zhou & Stephens (2012) Nature Genetics.

Modules:
- runner_jax: Batch processing (genotypes in memory, requires JAX)
- runner_jax_streaming: Disk streaming (genotypes per chunk, requires JAX)
- runner_numpy: Pure-NumPy batch runner (no JAX required)
- runner_numpy_streaming: Disk streaming with C extension (no JAX required)
- chunk: Chunk size computation
- prepare: JAX-specific setup (device sharding, placement)
- prepare_common: Shared setup (covariates, eigendecomp, null model)
- compute_numpy: NumPy mode dispatch for chunk computation
- likelihood_numpy: Pure-NumPy batch REML/MLE and optimization
- likelihood_jax: JAX-optimized REML/MLE and optimization
- special: Pure-stdlib betainc and chi2_sf (no numpy/scipy)
- results: Result building functions
- eigen: Eigendecomposition with GEMMA-compatible thresholding
- stats: AssocResult dataclass
- io: Result file I/O
"""

# Always available (no JAX dependency):
from jamma.core.backend import has_jax
from jamma.lmm.chunk import auto_tune_chunk_size
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.io import write_assoc_results
from jamma.lmm.loco import run_lmm_loco
from jamma.lmm.runner import ExecutionPlan, run_lmm, select_execution_mode
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.runner_numpy_streaming import run_lmm_association_numpy_streaming
from jamma.lmm.schema import LmmConfig, LmmRunResult
from jamma.lmm.stats import AssocResult

# JAX-dependent runners — probe for JAX first, then import unconditionally
# so that non-JAX import errors (typos, missing deps) propagate immediately.
_HAS_JAX = has_jax()

if _HAS_JAX:
    from jamma.lmm.runner_jax import run_lmm_association_jax
    from jamma.lmm.runner_jax_streaming import run_lmm_association_streaming

__all__ = [
    "auto_tune_chunk_size",
    "ExecutionPlan",
    "run_lmm",
    "run_lmm_association_numpy",
    "run_lmm_association_numpy_streaming",
    "run_lmm_loco",
    "select_execution_mode",
    "AssocResult",
    "eigendecompose_kinship",
    "LmmConfig",
    "LmmRunResult",
    "read_eigen_files",
    "write_assoc_results",
    "write_eigen_files",
]
if _HAS_JAX:
    __all__ += [
        "run_lmm_association_jax",
        "run_lmm_association_streaming",
    ]
