"""Linear Mixed Model (LMM) association testing.

GEMMA-compatible LMM association tests with NumPy backend.
Core algorithm follows Zhou & Stephens (2012) Nature Genetics.

Modules:
- runner_numpy: Pure-NumPy batch runner
- runner_numpy_streaming: Disk streaming with C extension
- chunk_runner_numpy: Shared NumPy chunk loop (orchestrator) for batch/streaming/LOCO
- chunk_sizing: RAM-budgeted chunk-size computation
- chunk_kernel: The one dispatch match, and the per-run state it needs
- chunk_pipeline: Rotation/compute thread split and overlapped pipeline driver
- chunk: Chunk size computation
- prepare_common: Shared setup (covariates, eigendecomp, null model)
- compute_numpy: NumPy mode dispatch for chunk computation
- likelihood_numpy: Pure-NumPy batch REML/MLE and optimization
- special: Pure-stdlib betainc and chi2_sf (no numpy/scipy)
- results: Result building functions
- eigen: Eigendecomposition with GEMMA-compatible thresholding
- stats: AssocResult dataclass
- io: Result file I/O
"""

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.eigen_io import read_eigen_files, write_eigen_files
from jamma.lmm.loco import DEFAULT_LOCO_CONFIG, LocoConfig, run_lmm_loco
from jamma.lmm.runner import ExecutionPlan, select_execution_mode
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from jamma.lmm.runner_numpy_streaming import run_lmm_association_numpy_streaming
from jamma.lmm.schema import LmmConfig, LmmRunResult
from jamma.lmm.stats import AssocResult

__all__ = [
    "DEFAULT_LOCO_CONFIG",
    "AssocResult",
    "ExecutionPlan",
    "LmmConfig",
    "LmmRunResult",
    "LocoConfig",
    "eigendecompose_kinship",
    "read_eigen_files",
    "run_lmm_association_numpy",
    "run_lmm_association_numpy_streaming",
    "run_lmm_loco",
    "select_execution_mode",
    "write_eigen_files",
]
