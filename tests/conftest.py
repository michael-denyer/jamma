"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

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
        for c in candidates:
            # pytest.mark.<name>
            if (
                isinstance(c, ast.Attribute)
                and isinstance(c.value, ast.Attribute)
                and isinstance(c.value.value, ast.Name)
                and c.value.value.id == "pytest"
                and c.value.attr == "mark"
            ):
                names.add(c.attr)
            # pytest.mark.<name>(...)
            elif (
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Attribute)
                and isinstance(c.func.value.value, ast.Name)
                and c.func.value.value.id == "pytest"
                and c.func.value.attr == "mark"
            ):
                names.add(c.func.attr)
    return names


def _per_test_marker_names(tree: ast.Module) -> set[str]:
    """Return the set of @pytest.mark.<name> decorators on any function or class."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "pytest"
                and target.value.attr == "mark"
            ):
                names.add(target.attr)
    return names


def _file_declares_tier_marker(path: Path) -> bool:
    """Return True if ``path`` has at least one tier/slow/benchmark marker.

    Source-parsed (not collection-based) so the check is invariant under
    xdist, ``-k``, ``-m`` filters, and any other collection-time filtering.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        # Treat unparsable test files as marker-less so the gate flags
        # them; surfacing via the same channel keeps diagnostics together.
        return False
    if _module_level_marker_names(tree) & _REQUIRED_TIER_MARKERS:
        return True
    return bool(_per_test_marker_names(tree) & _REQUIRED_TIER_MARKERS)


def _enforce_tier_markers() -> None:
    """Source-parse every test file under ``tests/`` and fail on missing markers.

    Called from ``pytest_configure`` (before xdist forks workers) so the
    enforcement runs exactly once per session, regardless of distribution
    mode or CLI filters. The previous implementation used
    ``pytest_collection_modifyitems`` and was empirically a no-op under
    ``-n`` (xdist's controller hook receives an empty items list — see
    tests/test_conftest_tier_gate.py for the regression tests).
    """
    missing: list[Path] = []
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        if path.name in _TIER_MARKER_EXEMPT_FILES:
            continue
        if not _file_declares_tier_marker(path):
            missing.append(path)
    if missing:
        repo_root = _TESTS_DIR.parent
        listing = "\n  ".join(str(p.relative_to(repo_root)) for p in missing)
        raise pytest.UsageError(
            "The following test files have no tier marker "
            "(tier0/tier1/tier2/slow/benchmark):\n  "
            f"{listing}\n\n"
            "Add `pytestmark = pytest.mark.tier0` (or per-test markers). "
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
            "Gate on the capability instead (compute_numpy._accel is not None "
            "for the C extension), and assert the attribute if the test needs "
            "it to be there."
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
