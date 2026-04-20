"""Internal build + runtime-recompile helpers for JAMMA.

Ships inside the installed package (``jamma._build_support``). Consumed by:

  * ``hatch_build.py`` — PEP 517 wheel build backend. Runs before jamma
    is installed, so it imports via ``sys.path.insert(src_dir); from
    jamma._build_support...`` rather than a normal package import.
  * ``src/jamma/jlinalg/_compile_jlinalg.py`` — dev-mode and runtime
    recompile entry point for the jlinalg C extension.
  * ``src/jamma/lmm/_compile_accel.py`` — dev-mode and runtime recompile
    entry point for the ``_lmm_accel`` C extension.
  * ``src/jamma/core/recompile.py`` — runtime ABI-mismatch shim; calls
    the two compile_extension() entry points above.

The underscore prefix marks this as internal/unstable API — third-party
code must not import from ``jamma._build_support``.
"""
