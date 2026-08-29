"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jamma.lmm import accel

# Tier markers every test file must declare (per-test or via pytestmark).
# Mirrors the markers list in pyproject.toml [tool.pytest.ini_options].
# See docs/TESTING.md §1.6 for the policy.
_REQUIRED_TIER_MARKERS = frozenset({"tier0", "tier1", "tier2", "slow", "benchmark"})

# Files exempt from the tier-marker requirement. Keep this list empty if
# possible; the right fix is almost always to add a marker, not an exemption.
_TIER_MARKER_EXEMPT_FILES: frozenset[str] = frozenset()

_TESTS_DIR = Path(__file__).resolve().parent

# Shared plumbing every scripts/check_*.py lint imports. install_lint_script()
# below copies it alongside whichever lint a test is driving.
_LINT_COMMON = _TESTS_DIR.parent / "scripts" / "_lint_common.py"

# A skip reason naming a fixture claims something under tests/fixtures/ could
# not be found. Every one of those is committed, so the only way to reach such
# a skip is a wrong path in the test. That is a bug which presents as a green
# run: two GEMMA-parity tests sat behind one for their entire lifetime because
# the directory *and* the filename were both wrong (#147). require_fixture()
# below is the mechanism tests use, and it raises; this word is what the
# collection-time gate looks for in a skip reason so the guard cannot come
# back as a skip. See docs/TESTING.md §1.11.
_FIXTURE_WORD = "fixture"

# Predicates that answer "is there a file at this path?", covering both the
# pathlib methods and the os.path functions.
#
# A skip reached because one of these was False is the same bug as above wearing
# different words, and the word-based check cannot see it. TestDstedcNoAbort read
# a dstedc.c that 663a22b had deleted and skipped with the reason "source not
# available"; "fixture" never appeared, so the gate passed it and the test
# reported green on every run from that commit until #156 deleted it.
#
# Keying on the shape rather than the wording is what closes that off, because
# the shape is not something the author of the next guard gets to choose. The
# correct response to a path that should be there and is not is to fail --
# pytest.fail, an assert, or require_fixture below -- never to skip.
_PATH_PREDICATES = frozenset({"exists", "is_file", "is_dir", "isfile", "isdir"})

# Builtins that ask whether a name exists on an object. Harmless in an assert,
# which fails when the name is gone; in a skip condition they mean the guard
# silently turns itself off the day the name is renamed or deleted.
_ATTRIBUTE_PROBES = frozenset({"hasattr", "getattr"})


def _marker_name_from_decorator(dec: ast.expr) -> str | None:
    """Return ``name`` for a ``pytest.mark.<name>`` or ``...<name>(...)`` node."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Attribute)
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
        and target.value.attr == "mark"
    ):
        return target.attr
    return None


def _decorator_marker_names(decorator_list: list[ast.expr]) -> set[str]:
    """Return the ``pytest.mark.<name>`` marker names on a decorator list."""
    names: set[str] = set()
    for dec in decorator_list:
        name = _marker_name_from_decorator(dec)
        if name is not None:
            names.add(name)
    return names


def _module_level_marker_names(tree: ast.Module) -> set[str]:
    """Return the set of marker names assigned to ``pytestmark`` at module level.

    Recognises both single-mark (``pytestmark = pytest.mark.tier0``) and
    list-of-marks (``pytestmark = [pytest.mark.tier0, pytest.mark.slow]``)
    forms. Anything else (computed expressions, function calls) is
    conservatively treated as no markers — the file should declare its
    classification statically.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "pytestmark"
        ):
            continue
        candidates: list[ast.expr] = []
        if isinstance(node.value, ast.List | ast.Tuple):
            candidates.extend(node.value.elts)
        else:
            candidates.append(node.value)
        names |= _decorator_marker_names(candidates)
    return names


def _untiered_test_functions(tree: ast.Module) -> list[str]:
    """Return qualified names of ``test_*`` functions with no tier marker.

    For each function, class method, or async function whose name starts
    with ``test_``, unions the module-level ``pytestmark``, the enclosing
    class's decorators (if any), and the function's own decorators, then
    reports the function (as ``Class.test_x`` or ``test_x``) when that
    union has no member in ``_REQUIRED_TIER_MARKERS``.

    This is per-item, not per-file: a module ``pytestmark`` used to satisfy
    the whole file even when one function in it carried no marker of its
    own and the module marker didn't apply to it (there was no such case
    before, since a module marker always applies to every item in the
    file — the risk this closes is a *file* with per-function markers on
    some tests and none on a sibling test, which the old file-granular
    check could not see).
    """
    module_names = _module_level_marker_names(tree)
    untiered: list[str] = []

    def _check(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str) -> None:
        if not node.name.startswith("test_"):
            return
        own = _decorator_marker_names(node.decorator_list)
        if not (module_names | own) & _REQUIRED_TIER_MARKERS:
            untiered.append(f"{prefix}{node.name}")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            _check(node, "")
        elif isinstance(node, ast.ClassDef):
            class_names = module_names | _decorator_marker_names(node.decorator_list)
            for child in node.body:
                if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if not child.name.startswith("test_"):
                    continue
                own = _decorator_marker_names(child.decorator_list)
                if not (class_names | own) & _REQUIRED_TIER_MARKERS:
                    untiered.append(f"{node.name}.{child.name}")
    return untiered


def _file_untiered_functions(path: Path) -> list[str]:
    """Return the untiered ``test_*`` function names in ``path``.

    Source-parsed (not collection-based) so the check is invariant under
    xdist, ``-k``, ``-m`` filters, and any other collection-time filtering.
    An unparsable file reports a single ``"<file>"`` sentinel entry so it
    surfaces through the same channel rather than being silently skipped.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return ["<unparsable file>"]
    return _untiered_test_functions(tree)


def _enforce_tier_markers() -> None:
    """Source-parse every test file under ``tests/`` and fail on any untiered test.

    Called from ``pytest_configure`` (before xdist forks workers) so the
    enforcement runs exactly once per session, regardless of distribution
    mode or CLI filters. ``pytest_collection_modifyitems`` is not used
    because xdist forks workers AFTER ``pytest_configure``, and the
    collection-based hook is empirically a no-op under ``-n`` (xdist's
    controller hook receives an empty items list — see
    tests/test_conftest_tier_gate.py for the regression tests).

    Per-item, not per-file: unions module, class, and function markers for
    each ``test_*`` item and reports the function by name when that union
    carries none of ``_REQUIRED_TIER_MARKERS``.
    """
    missing: dict[Path, list[str]] = {}
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        if path.name in _TIER_MARKER_EXEMPT_FILES:
            continue
        untiered = _file_untiered_functions(path)
        if untiered:
            missing[path] = untiered
    if missing:
        repo_root = _TESTS_DIR.parent
        listing = "\n  ".join(
            f"{path.relative_to(repo_root)}: {', '.join(names)}"
            for path, names in missing.items()
        )
        raise pytest.UsageError(
            "The following tests have no tier marker "
            "(tier0/tier1/tier2/slow/benchmark):\n  "
            f"{listing}\n\n"
            "Add `pytestmark = pytest.mark.tier0` (or a per-test marker). "
            "See docs/TESTING.md §1.6."
        )


def _is_pytest_skip_call(node: ast.AST) -> bool:
    """True for a ``pytest.skip(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
    )


def _is_pytest_skipif_call(node: ast.AST) -> bool:
    """True for a ``pytest.mark.skipif(...)`` decorator call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skipif"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "mark"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "pytest"
    )


def _tests_a_path(expr: ast.expr) -> bool:
    """True if ``expr`` asks whether a filesystem path exists.

    Matches on the method or function name alone (``.exists()``,
    ``os.path.isfile(...)``), not on the receiver, because the receiver is
    usually a local whose type cannot be recovered from the source. Walking the
    whole expression means a negation or a boolean combination is caught too.
    """
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr in _PATH_PREDICATES
        for sub in ast.walk(expr)
    )


def _path_guarded_skip_lines(tree: ast.Module) -> list[int]:
    """Line numbers of skips that are control-dependent on a path check.

    Two shapes, which between them are how the guard gets written:

    - ``if not src.exists(): pytest.skip(...)``, and the ``else``-branch
      variant, reported at the line of the ``skip`` rather than the ``if`` so
      the message points at the statement to delete.
    - ``@pytest.mark.skipif(not SRC.exists(), reason=...)``.

    Deliberately says nothing about the reason string. That is the whole point:
    see the ``_PATH_PREDICATES`` comment above.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _tests_a_path(node.test):
            lines += [
                sub.lineno
                for branch in (node.body, node.orelse)
                for stmt in branch
                for sub in ast.walk(stmt)
                if isinstance(sub, ast.Call) and _is_pytest_skip_call(sub)
            ]
        elif _is_pytest_skipif_call(node):
            assert isinstance(node, ast.Call)  # narrowed by the predicate
            conditions = [
                *node.args,
                *(kw.value for kw in node.keywords if kw.arg == "condition"),
            ]
            if any(_tests_a_path(c) for c in conditions):
                lines.append(node.lineno)
    return sorted(set(lines))


def _module_level_bindings(tree: ast.Module) -> dict[str, ast.expr]:
    """Map each module-level ``NAME = expr`` to its right-hand side."""
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _probes_an_attribute(expr: ast.expr, bindings: dict[str, ast.expr]) -> bool:
    """True if ``expr`` asks whether a name exists on an object.

    Follows module-level bindings, because the availability flag is almost
    always computed once at import and referenced by the decorator
    (``AVAILABLE = ... hasattr(mod, "X") ...`` then
    ``@pytest.mark.skipif(not AVAILABLE, ...)``). Looking only inside the
    decorator's own expression would miss every real instance. Each name is
    followed at most once, so a self-referential binding terminates.
    """
    seen: set[str] = set()
    stack = [expr]
    while stack:
        current = stack.pop()
        for sub in ast.walk(current):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in _ATTRIBUTE_PROBES
            ):
                return True
            if isinstance(sub, ast.Name) and sub.id in bindings and sub.id not in seen:
                seen.add(sub.id)
                stack.append(bindings[sub.id])
    return False


def _attribute_probed_skip_lines(tree: ast.Module) -> list[int]:
    """Line numbers of skips gated on whether a name exists.

    The shape this catches: a capability flag built from ``hasattr`` against a
    module attribute, feeding a ``skipif``. It reads as a capability check and
    behaves like one right up until the attribute is renamed or deleted, at
    which point every test behind it skips and the run stays green. Nine tests
    covering the fused Wald kernel sat dormant that way once the flag they
    probed was removed (#182).

    Same two shapes as ``_path_guarded_skip_lines``, for the same reason: an
    ``if``/``pytest.skip`` pair and a ``skipif`` decorator are both how the
    guard gets written.
    """
    bindings = _module_level_bindings(tree)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _probes_an_attribute(node.test, bindings):
            lines += [
                sub.lineno
                for branch in (node.body, node.orelse)
                for stmt in branch
                for sub in ast.walk(stmt)
                if isinstance(sub, ast.Call) and _is_pytest_skip_call(sub)
            ]
        elif _is_pytest_skipif_call(node):
            assert isinstance(node, ast.Call)  # narrowed by the predicate
            conditions = [
                *node.args,
                *(kw.value for kw in node.keywords if kw.arg == "condition"),
            ]
            if any(_probes_an_attribute(c, bindings) for c in conditions):
                lines.append(node.lineno)
    return sorted(set(lines))


def _fixture_skip_lines(tree: ast.Module) -> list[int]:
    """Line numbers of skips whose reason names a fixture.

    Matches both ways of writing the guard: a ``pytest.skip(...)`` call and a
    ``@pytest.mark.skipif(..., reason=...)`` decorator. Only string literals are
    inspected, because a computed reason cannot be judged from the source and
    guessing would produce false failures.

    Complements ``_path_guarded_skip_lines``: this one catches a guard that
    names a fixture without checking a path, that one catches a guard that
    checks a path without naming anything. Neither subsumes the other.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not (_is_pytest_skip_call(node) or _is_pytest_skipif_call(node)):
            continue
        assert isinstance(node, ast.Call)  # narrowed by the predicates
        reasons = [*node.args, *(kw.value for kw in node.keywords)]
        for reason in reasons:
            if (
                isinstance(reason, ast.Constant)
                and isinstance(reason.value, str)
                and _FIXTURE_WORD in reason.value.lower()
            ):
                lines.append(node.lineno)
                break
    return lines


def _enforce_no_dormant_skips() -> None:
    """Source-parse every test file and reject skips that hide a wrong path.

    Three detectors, reported together. ``_fixture_skip_lines`` reads the reason
    string; ``_path_guarded_skip_lines`` reads the control flow. A guard has to
    evade both to stay hidden, and the two evasions pull in opposite directions:
    avoid the word and the shape still shows, keep the check implicit and the
    wording has nothing left to describe it with.

    ``_attribute_probed_skip_lines`` covers a third way for a guard to go quiet,
    which neither of the other two sees: the precondition is real and correctly
    worded, but it is expressed as ``hasattr`` against a name that later gets
    deleted. Nothing about the path or the wording changes; the condition just
    starts answering False forever (#182).

    Source-parsed and run once from ``pytest_configure``, for the same reason
    the tier gate above is: it then holds under xdist, ``-k``, ``-m``, and any
    other collection-time filtering, and it flags the guard even in a file whose
    tests never ran this session.

    This replaced a runtime backstop that matched the exact phrase "fixture not
    available" in skip reports as they arrived. That could only fire when the
    guarded test actually ran, and only for that one wording; none of the ~30
    legitimate skips in this suite phrase anything that way, and neither would
    most new guards. Reading the source instead catches the guard whatever it
    says, before a single test executes.

    Every category is collected before raising, for the reason
    ``require_fixture`` names every missing path at once: fixing one offender and
    re-running to discover the next is the slow way to clear a sweep.

    Raises:
        pytest.UsageError: Naming every file and line in every category, with
            the fix for each.
    """
    by_word: list[str] = []
    by_shape: list[str] = []
    by_probe: list[str] = []
    repo_root = _TESTS_DIR.parent
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            # Unparsable files are the tier gate's problem to report; flagging
            # them here too would double every message.
            continue
        rel = path.relative_to(repo_root)
        by_word += [f"{rel}:{line}" for line in _fixture_skip_lines(tree)]
        by_shape += [f"{rel}:{line}" for line in _path_guarded_skip_lines(tree)]
        by_probe += [f"{rel}:{line}" for line in _attribute_probed_skip_lines(tree)]
    if not (by_word or by_shape or by_probe):
        return
    parts: list[str] = []
    if by_word:
        listing = "\n  ".join(by_word)
        parts.append(
            "The following skips name a fixture in their reason:\n  "
            f"{listing}\n\n"
            "Everything under tests/fixtures/ is committed, so a fixture that "
            "cannot be found means the test names the wrong path. Call "
            "require_fixture(*paths) from tests/conftest.py, which raises "
            "instead of skipping, or assert the precondition."
        )
    if by_shape:
        listing = "\n  ".join(by_shape)
        parts.append(
            "The following skips are guarded by a filesystem check:\n  "
            f"{listing}\n\n"
            "A path that should be present and is not is a bug in the test, not "
            "a reason to skip: the run stays green and the test never executes "
            "again. Use pytest.fail, an assert, or require_fixture(*paths). If "
            "the file genuinely may be absent because it is a build output, "
            "gate on the build flag that predicts it (HAS_C_EXTENSION and the "
            "like) rather than on the path."
        )
    if by_probe:
        listing = "\n  ".join(by_probe)
        parts.append(
            "The following skips are gated on whether a name exists:\n  "
            f"{listing}\n\n"
            "hasattr and getattr answer False for a name that was deleted just "
            "as readily as for one that was never built, so the guard turns "
            "itself off during an unrelated rename and the run stays green. "
            "Gate on the capability instead (accel.available() for the C "
            "extension), and assert the attribute if the test needs it to be "
            "there."
        )
    parts.append("See docs/TESTING.md §1.11.")
    raise pytest.UsageError("\n\n".join(parts))


def pytest_configure(config: pytest.Config) -> None:
    """Run session-start checks: tier-marker gate and stale-C-extension warn.

    The tier-marker gate runs in ``pytest_configure`` (not
    ``pytest_collection_modifyitems``) because xdist forks workers AFTER
    ``pytest_configure``; running the check here means it fires exactly
    once on the controller, before any partitioning. The previous
    collection-based hook silently no-op'd under xdist (controller's
    items list is empty; workers were skipped via ``workerinput`` guard).

    The stale-C-extension check is advisory: editable install picks up
    Python edits automatically, but C source edits require explicit
    rebuild. We warn rather than fail so the session still starts;
    pre-push hook (scripts/check_c_extension_freshness.py) is the
    blocking gate.
    """
    # xdist worker processes inherit ``pytest_configure`` invocations too.
    # Skip on workers — the controller already ran the gate, and a worker
    # raising UsageError mid-session would crash xdist.
    if not hasattr(config, "workerinput"):
        _enforce_tier_markers()
        _enforce_no_dormant_skips()

    # Import guarded: script lives outside the package and may be missing
    # in some install layouts (e.g. a sdist-only install). Missing script
    # is not a test failure — just skip the check.
    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    if not (script_dir / "check_c_extension_freshness.py").exists():
        return
    sys.path.insert(0, str(script_dir))
    try:
        import check_c_extension_freshness as freshness
    except ImportError as exc:
        # The script exists on disk (we checked above) but failed to import.
        # That's a real bug — syntax error, broken refactor, missing dep.
        # We don't want to fail the whole session, but a silent return would
        # mask the bug indefinitely. Surface it instead.
        sys.stderr.write(
            f"\n\033[33m[jamma] WARNING: c-extension freshness check "
            f"could not be loaded ({type(exc).__name__}: {exc}). "
            f"Stale .so files will not be detected this session.\033[0m\n"
        )
        return
    finally:
        # Don't pollute sys.path past this function.
        if sys.path and sys.path[0] == str(script_dir):
            sys.path.pop(0)

    stale = [r for r in freshness.check_all() if r.is_stale]
    if not stale:
        return
    for r in stale:
        assert r.newest_source is not None  # guaranteed by is_stale
        sys.stderr.write(
            f"\n\033[33m[jamma] WARNING: C extension '{r.spec.label}' is "
            f"stale relative to {r.newest_source.name} — tests will run "
            f"against the OLD compiled .so. Rebuild with:\n"
            f"    {r.spec.rebuild_command}\033[0m\n"
        )
    sys.stderr.write(
        "\033[33m[jamma] If this is unexpected, run "
        "scripts/check_c_extension_freshness.py for full drift report.\033[0m\n\n"
    )


def require_fixture(*paths: Path) -> None:
    """Assert committed fixture paths exist, raising rather than skipping.

    Everything under ``tests/fixtures/`` is committed and sha256-verified by
    the ``GEMMA fixture sha256 manifest`` pre-commit hook. A path that does
    not exist therefore means the test names the wrong path, never that the
    data is unavailable, so the correct response is a failure and not a skip.
    Guarding with ``pytest.skip`` hid two GEMMA-parity tests for their entire
    lifetime: the directory and the filename were both wrong, one
    ``.exists()`` check collapsed both misses into one skip reason, and the
    run stayed green (#147).

    Pass every path the caller is about to read, in one call. A wrong
    directory then reports all of its files at once instead of stopping at
    the first, which is the half of #147 a single check could not show.

    Args:
        paths: Fixture files that must be present.

    Raises:
        FileNotFoundError: If any path is absent. Every missing path is
            named, relative to the repository root.
    """
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return
    repo_root = _TESTS_DIR.parent
    listing = "\n  ".join(
        str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
        for p in missing
    )
    raise FileNotFoundError(
        f"{len(missing)} of {len(paths)} required test fixture(s) are "
        f"missing:\n  {listing}\n\n"
        "Everything under tests/fixtures/ is committed, so these paths are "
        "wrong rather than absent. Fix the paths; do not skip the test. "
        "See docs/TESTING.md §1.11."
    )


def install_lint_script(script: Path, scripts_dir: Path) -> Path:
    """Copy a ``scripts/check_*.py`` lint into ``scripts_dir`` and return it.

    Every lint test drives its script against a synthetic tree, which means
    reproducing the layout the script expects: it derives the repository root
    from its own location, and it imports ``scripts/_lint_common.py`` from
    ``sys.path[0]``. Copying the lint alone gives a tree where every test
    fails on ImportError, so the shared module travels with it.

    Args:
        script: The real lint under ``scripts/``.
        scripts_dir: The synthetic ``scripts/`` directory, created if absent.

    Returns:
        Path to the copied lint, ready to run.
    """
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_LINT_COMMON, scripts_dir / _LINT_COMMON.name)
    destination = scripts_dir / script.name
    shutil.copy2(script, destination)
    return destination


if TYPE_CHECKING:
    from jamma.validation import ToleranceConfig

# Tier marker policy lives in docs/TESTING.md §1.5 (source of truth) and
# pyproject.toml [tool.pytest.ini_options].markers. The enforcement gate at
# the top of this file fails collection if a test lacks tier0/tier1/tier2.


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from test fixtures.

    Returns:
        Path prefix for gemma_synthetic PLINK files (without .bed/.bim/.fam extension)
    """
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for test results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to output directory
    """
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tolerance_config() -> ToleranceConfig:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        ToleranceConfig with default tolerance values for different comparison types
    """
    from jamma.validation import ToleranceConfig

    return ToleranceConfig()


# The one C-extension seam every LMM test drives through. Replaces 26
# `skipif(compute_numpy._accel is None, ...)` decorators, 5 module-level flags
# that all re-spelled the same bit, 14 inline `pytest.skip("C extension ...")`
# calls, and ~24 hand-written `orig = ...; try: ... finally:` or
# `monkeypatch.setattr(cn, "_accel", None)` hold-outs across 12 files. See
# docs/TESTING.md §2.7.
#
# Read once at collection, same as jlinalg's HAS_C_EXTENSION: the extension
# does not appear or disappear mid-session, only ``no_c_kernels`` below holds
# it out for the span of one test.
requires_c = pytest.mark.skipif(
    not accel.available(), reason="C extension _lmm_accel not available"
)


@pytest.fixture
def no_c_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the C extension out for this test, so the NumPy path runs for real.

    ``jamma.lmm.accel.available()`` and ``jamma.lmm.dispatch.select_current``
    read ``accel._accel`` at call time, not at import time, so clearing it
    here drives the fallback path rather than merely describing it. Every
    module that decides on C-vs-NumPy reads through ``accel``, so this one
    monkeypatch is the whole seam.
    """
    from jamma.lmm import accel

    monkeypatch.setattr(accel, "_accel", None)


def _build_synthetic_covariate_data(
    n_cvt: int,
    n_samples: int = 200,
    n_snps: int = 50,
    seed: int = 42,
) -> dict:
    """Build synthetic rotated data for C extension testing.

    Generates eigenvalues, rotated covariates (UtW), phenotype (Uty),
    genotypes (UtG), and computes Uab_batch for the given n_cvt.

    Args:
        n_cvt: Number of covariates.
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: eigenvalues, UtW, Uty, UtG, Uab_batch,
        n_samples, n_snps, n_cvt.
    """
    from jamma.lmm.likelihood import compute_Uab

    rng = np.random.default_rng(seed)

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]  # descending
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Compute Uab for each SNP
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_batch[i] = compute_Uab(UtW, Uty, UtG[:, i])

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "UtG": UtG,
        "Uab_batch": Uab_batch,
        "n_samples": n_samples,
        "n_snps": n_snps,
        "n_cvt": n_cvt,
    }


@pytest.fixture
def synthetic_covariate_data_ncvt2() -> dict:
    """Synthetic data with 2 covariates for C extension testing.

    200 samples, 50 SNPs, 2 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=2, seed=42)


@pytest.fixture
def synthetic_covariate_data_ncvt4() -> dict:
    """Synthetic data with 4 covariates for C extension testing.

    200 samples, 50 SNPs, 4 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=4, seed=99)


def make_runner_synthetic_data(
    n_samples: int = 100, n_snps: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Create synthetic data for runner-level tests."""
    rng = np.random.default_rng(seed)
    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
    kinship = (kinship + kinship.T) / 2
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]
    return genotypes, phenotypes, kinship, snp_info


@pytest.fixture
def synthetic_data_with_covariates(synthetic_data):
    """Load gemma_synthetic data plus covariates from gemma_covariate fixture.

    The covariates.txt file already includes the intercept column (first column
    is all 1.0), matching GEMMA's internal representation when -c is used.
    """
    from tests.fixture_paths import SYNTHETIC

    plink, kinship, phenotypes, snp_info = synthetic_data
    covariates = np.loadtxt(SYNTHETIC.covariates)
    return plink, kinship, phenotypes, snp_info, covariates


@pytest.fixture
def synthetic_data():
    """Load gemma_synthetic PLINK data, kinship, phenotypes, and snp_info."""
    from jamma.io import load_plink_binary, read_fam_phenotypes
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.schema import SnpMeta
    from tests.fixture_paths import SYNTHETIC

    plink = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship)
    phenotypes = read_fam_phenotypes(SYNTHETIC.fam)
    snp_info = SnpMeta.from_plink_meta(plink.meta)
    return plink, kinship, phenotypes, snp_info
