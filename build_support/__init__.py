"""PEP 517 build helpers for JAMMA.

This package lives at repo root, sibling to src/, and is intentionally
NOT part of the installed wheel. hatch_build.py (PEP 517 build backend)
cannot import from jamma.* at wheel-build time because the package is
not yet installed, so shared compile helpers live here instead.

Entry points (src/jamma/jlinalg/_compile_jlinalg.py,
src/jamma/lmm/_compile_accel.py, hatch_build.py) each do
`sys.path.insert(0, <repo_root>); from build_support import ...` to
pick up this package. On wheel install, build_support/ is absent; the
entry points tolerate ImportError and fall through to runtime shims in
jamma.core.recompile when appropriate.

Convention reference: numpy's build_support/, scipy's tools/.
"""
