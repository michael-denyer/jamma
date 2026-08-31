"""Post-link import proof, shared by both dev-mode compile shims.

A successful compile+link does not guarantee a usable module. A bad RPATH, a
missing runtime library, an ABI mismatch with the host numpy, or a missing C
symbol all let the link pass while the import fails. Proving the import is
therefore part of "the build worked", not an extra.

The proof runs in a fresh subprocess rather than this interpreter so it never
re-executes ``jamma.lmm``/``jamma.jlinalg``'s own import machinery, which is
what caused the #181 self-deadlock when an in-process probe evicted the parent
package from ``sys.modules`` and re-imported it.

The statement to run is derived from the ``BuildSpec`` rather than hand-written
per shim: ``sys_module_key`` names the extension and ``required_attrs`` lists
symbols an ABI-matched build always exports, so importing those names is a
stronger proof than importing the module alone and cannot drift from the spec.
"""

from __future__ import annotations

import os
import subprocess
import sys

from .build_models import BuildSpec


def import_statement(spec: BuildSpec) -> str:
    """The import the load proof runs for ``spec``, derived from the spec.

    Imports ``spec.required_attrs`` from ``spec.sys_module_key`` so a build
    that links but omits an expected symbol fails the proof. Falls back to a
    bare module import when the spec lists no required attributes.
    """
    if spec.required_attrs:
        names = ", ".join(spec.required_attrs)
        return f"from {spec.sys_module_key} import {names}"
    return f"import {spec.sys_module_key}"


def load_proof(spec: BuildSpec, import_code: str | None = None) -> bool:
    """Prove the freshly compiled extension for ``spec`` imports.

    Skipped when JAMMA_SANITIZE is set: importing an ASan-instrumented ``.so``
    requires ``LD_PRELOAD=libasan.so``, which the sanitizer workflow exports
    only for the pytest step, not this compile step. The subprocess would abort
    with "ASan runtime does not come first in initial library list" (exit 134).
    The pytest step exercises the ``.so`` under the correct LD_PRELOAD.

    Args:
        spec: The ``BuildSpec`` whose extension was just built.
        import_code: Statement to run in the subprocess. Defaults to
            ``import_statement(spec)``. Overridable so a test can point the
            proof at a module that cannot import, without touching a real
            ``.so``.

    Returns:
        True if the extension imported (or the proof was skipped), False if
        the subprocess failed. On failure the subprocess stderr is printed, so
        a broken build never reports success silently.
    """

    label = spec.module_name or spec.output_stem

    if os.environ.get("JAMMA_SANITIZE", "").strip() not in ("", "0"):
        print(
            f"{label}: skipping post-link import proof — JAMMA_SANITIZE is set",
            file=sys.stderr,
        )
        return True

    statement = import_statement(spec) if import_code is None else import_code
    proc = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(
            f"ERROR: {label} compiled but failed to import in a fresh "
            f"interpreter (exit {proc.returncode}):",
            file=sys.stderr,
        )
        print(proc.stderr, file=sys.stderr)
        return False
    if proc.stdout.strip():
        print(proc.stdout.strip(), file=sys.stderr)
    return True
