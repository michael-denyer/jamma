"""Loader for the `_lmm_accel` C extension.

The one place that imports, ABI-validates, and (on failure) auto-recompiles
`_lmm_accel`, through the shared seam in `jamma.core.recompile`. Every module
that needs to know whether the C extension is usable — `compute_numpy`,
`chunk_kernel`, `chunk_runner_numpy`, `runner`, `pipeline_memory`,
`pipeline_banner` — reads it from here instead of reaching into
`compute_numpy`'s private state, which used to be the only tenant of the
loader despite being a pure-NumPy fallback module in its own right.

``available()`` and ``HAS_OPENMP`` are read at call time, not cached at
import time in the caller, so a test that clears ``accel._accel`` (directly,
or through the ``no_c_kernels`` fixture in ``tests/conftest.py``) drives the
fallback for real rather than describing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jamma._build_support.compile_and_link import LMM_ACCEL_SPEC
from jamma.core.constants import env_flag
from jamma.core.recompile import _load_c_module

if TYPE_CHECKING:
    from types import ModuleType

_EXPECTED_ABI_VERSION = 20  # Must match ABI_VERSION in _lmm_accel.c

# Load and validate the C accelerator through the one shared seam in
# jamma.core.recompile. It honours JAMMA_FORCE_NUMPY_FALLBACK (returns None
# without importing, so ASAN never dlopens the .so), checks ABI_VERSION against
# _EXPECTED_ABI_VERSION, confirms the fused-kernel core symbols listed in
# LMM_ACCEL_SPEC.required_attrs are present, and rebuilds a stale .so once
# before giving up.
_accel: ModuleType | None = _load_c_module(LMM_ACCEL_SPEC, _EXPECTED_ABI_VERSION)

if _accel is None and not env_flag("JAMMA_FORCE_NUMPY_FALLBACK"):
    from loguru import logger as _logger

    _logger.warning(
        "C extension _lmm_accel not available — using pure-Python path "
        "(LMM may be slower without C extension; magnitude depends on "
        "dataset size and core count). To compile, run: "
        "python -m jamma.lmm._compile_accel"
    )
    del _logger

# HAS_OPENMP is a genuinely independent bit, unlike kernel availability: C
# sets it from #ifdef _OPENMP at build time, and plan_thread_budget reads it.
HAS_OPENMP: bool = bool(_accel is not None and _accel.HAS_OPENMP)


def available() -> bool:
    """Report whether the loaded C extension is usable right now."""
    return _accel is not None


def require() -> ModuleType:
    """Return the loaded C extension, or raise naming the fix.

    Every kernel entry point needs the same guard. Hand-writing it per symbol
    drifted before: some sites raised while others asserted, and an assert
    vanishes under ``python -O``, turning a clear diagnostic into a
    ``NoneType is not callable`` from inside the C call.
    """
    if _accel is None:
        raise RuntimeError(
            "This kernel requires the _lmm_accel C extension with ABI version "
            f"{_EXPECTED_ABI_VERSION}. Recompile: python -m jamma.lmm._compile_accel"
        )
    return _accel
