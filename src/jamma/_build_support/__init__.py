"""Internal build + runtime-recompile helpers for JAMMA.

Ships inside the installed package (``jamma._build_support``). Consumed by:

  * ``hatch_build.py`` — PEP 517 wheel build backend. Runs before jamma
    is installed in an isolated env, so it cannot do a normal
    ``from jamma._build_support import ...`` (which would trigger
    ``src/jamma/__init__.py`` and pull in loguru/numpy/jamma.lmm, none of
    which are in the build env). Instead it loads each helper module by
    file path with ``importlib.util.spec_from_file_location`` and
    registers it on ``sys.modules`` under a distinct
    ``jamma_build_support.*`` namespace so there's no collision with the
    real ``jamma.*`` package at runtime.
  * ``src/jamma/jlinalg/_compile_jlinalg.py`` — dev-mode and runtime
    recompile entry point for the jlinalg C extension. Imports from
    ``jamma._build_support`` normally (jamma is installed at runtime).
  * ``src/jamma/lmm/_compile_accel.py`` — dev-mode and runtime recompile
    entry point for the ``_lmm_accel`` C extension. Same import story.
  * ``src/jamma/core/recompile.py`` — runtime ABI-mismatch shim; calls
    the two compile_extension() entry points above.

The underscore prefix marks this as internal/unstable API — third-party
code must not import from ``jamma._build_support``.
"""
