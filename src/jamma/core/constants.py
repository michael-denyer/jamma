"""Domain constants shared across JAMMA modules."""

from __future__ import annotations

import os
from dataclasses import dataclass

# GEMMA encodes missing phenotypes as -9 in .fam files.
PHENOTYPE_MISSING: float = -9.0


def env_flag(name: str) -> bool:
    """Return whether an environment variable is set to a truthy value.

    Presence-based, matching ``docs/CONFIGURATION.md``: any value other than
    unset, ``""``, or ``"0"`` counts as on. That deliberately includes
    ``"false"``, ``"no"``, and ``"off"`` — set the variable to ``0`` or leave it
    unset to turn a flag off.

    Every JAMMA toggle shares this rule, so it lives in one place rather than
    being re-spelled per call site where the accepted values could drift apart.
    ``jamma._build_support`` cannot call this: it runs under PEP 517 build
    isolation with no runtime dependencies importable.
    """
    return os.environ.get(name, "").strip() not in ("", "0")


@dataclass(frozen=True)
class Env:
    """Every ``JAMMA_*`` runtime environment variable, parsed once per read.

    Construct with :meth:`Env.current`, never by caching the instance at
    import time: tests exercise every field with ``monkeypatch.setenv`` /
    ``delenv``, which only takes effect for reads that happen after the
    patch. A module-level singleton would freeze whatever the first import
    saw and silently ignore every later monkeypatch. ``Env.current()`` reads
    ``os.environ`` fresh each call, so it composes with that pattern: call it
    once per logical operation (a CLI invocation, a compute dispatch) rather
    than once per process.

    Boolean fields use :func:`env_flag`'s presence-based truthiness. The two
    integer fields (``blas_threads``, ``loco_workers``) keep the raw string
    behind ``*_raw`` so callers preserve their existing malformed-value
    warnings (``get_blas_thread_count``, ``get_loco_worker_count``) — folding
    the ``int()`` parse in here would either duplicate that logging or drop
    it.

    ``DO_NOT_TRACK`` (a non-``JAMMA_`` var with different truthiness: only
    ``"1"`` opts out, unlike ``env_flag``'s presence-based rule) is
    intentionally excluded — see ``telemetry.py``'s call site.
    """

    blas_threads_raw: str | None
    loco_workers_raw: str | None
    backend_raw: str | None
    force_numpy_fallback: bool
    no_telemetry: bool
    no_openmp: bool
    sanitize: str
    sentinel_ub: bool

    @classmethod
    def current(cls) -> Env:
        """Parse every JAMMA_* runtime env var from the current os.environ."""
        return cls(
            blas_threads_raw=os.environ.get("JAMMA_BLAS_THREADS"),
            loco_workers_raw=os.environ.get("JAMMA_LOCO_WORKERS"),
            backend_raw=os.environ.get("JAMMA_BACKEND"),
            force_numpy_fallback=env_flag("JAMMA_FORCE_NUMPY_FALLBACK"),
            no_telemetry=env_flag("JAMMA_NO_TELEMETRY"),
            no_openmp=env_flag("JAMMA_NO_OPENMP"),
            sanitize=os.environ.get("JAMMA_SANITIZE", "").strip(),
            sentinel_ub=env_flag("JAMMA_SENTINEL_UB"),
        )
