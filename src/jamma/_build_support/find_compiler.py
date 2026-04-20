"""C compiler discovery for build-time AND runtime recompile.

Canonical location for ``find_c_compiler``. Consumers:

  * hatch_build.py (PEP 517 wheel build backend)
  * src/jamma/jlinalg/_compile_jlinalg.py (dev-mode + runtime recompile)
  * src/jamma/lmm/_compile_accel.py (dev-mode + runtime recompile)

Ships inside the installed package as ``jamma._build_support.find_compiler``
so runtime ABI-mismatch recompile via ``jamma.core.recompile`` reaches the
same discovery logic the wheel was built with — no separate minimal
fallback exists or should be added.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig


def _probe_compiler(cmd: str) -> bool:
    """Return True if *cmd* exists on PATH and ``--version`` exits 0."""
    path = shutil.which(cmd)
    if not path:
        return False
    try:
        probe = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
        if probe.returncode == 0:
            return True
        stderr = probe.stderr.decode(errors="replace").strip()[:200]
        print(
            f"Compiler '{cmd}' found at {path} but --version "
            f"returned exit code {probe.returncode}: {stderr}",
            file=sys.stderr,
        )
    except OSError as e:
        print(
            f"Compiler '{cmd}' found at {path} but failed to execute: {e}",
            file=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        print(
            f"Compiler '{cmd}' found at {path} but --version timed out (5s)",
            file=sys.stderr,
        )
    return False


def find_c_compiler() -> tuple[str, list[str]] | None:
    """Find a usable C compiler, trying multiple candidates.

    Checks in order:
    1. ``$CC`` environment variable — if set, it is the **only** candidate.
       An explicit ``$CC`` that fails verification returns None immediately
       (no fallback to other compilers).
    2. ``sysconfig`` configured compiler (what Python was built with)
    3. ``cc``, ``clang``, ``gcc`` as fallbacks

    Candidates are deduplicated by base command name; earlier entries take
    priority (e.g. if sysconfig returns ``gcc`` and ``gcc`` is also a
    fallback, only one probe is made).

    Each candidate must exist on PATH and respond to ``--version``
    to be considered usable.

    Returns:
        Tuple of (compiler_command, extra_flags) where extra_flags are
        additional arguments split from the compiler string (e.g.
        ``CC="gcc -pthread"`` yields ``("gcc", ["-pthread"])``).
        None if no usable compiler found.
    """
    # $CC is explicit — honour it or fail, don't silently substitute.
    cc_env = os.environ.get("CC")
    if cc_env:
        parts = cc_env.split()
        if not parts:
            return None
        cmd, extra = parts[0], parts[1:]
        if _probe_compiler(cmd):
            return cmd, extra
        print(
            f"$CC is set to '{cc_env}' but verification failed — "
            "not falling through to other compilers.",
            file=sys.stderr,
        )
        return None

    # Auto-detect: sysconfig, then common fallbacks (deduplicated).
    seen_cmds: set[str] = set()
    candidates: list[str] = []

    def _add(candidate: str) -> None:
        parts = candidate.split()
        if not parts:
            return
        cmd = parts[0]
        if cmd not in seen_cmds:
            seen_cmds.add(cmd)
            candidates.append(candidate)

    cc_sysconfig = sysconfig.get_config_var("CC")
    if cc_sysconfig:
        _add(cc_sysconfig)

    for fallback in ("cc", "clang", "gcc"):
        _add(fallback)

    for candidate in candidates:
        cmd, *extra = candidate.split()
        if _probe_compiler(cmd):
            return cmd, extra

    return None
