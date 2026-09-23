#!/usr/bin/env python3
"""Reject untiered tests and skips that hide a wrong path, across ``tests/``.

The tier gate names every ``test_*`` item whose module, class and own markers
carry no tier (docs/TESTING.md §1.6). The dormant-skip gate names every skip
that a wrong fixture path, a filesystem check, or an attribute probe could keep
silent (docs/TESTING.md §1.11).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from _lint_common import repo_root

# Tier markers every test file must declare (per-test or via pytestmark).
# Mirrors the markers list in pyproject.toml [tool.pytest.ini_options].
# See docs/TESTING.md §1.6 for the policy.
_REQUIRED_TIER_MARKERS = frozenset({"tier0", "tier1", "tier2", "slow", "benchmark"})

# Files exempt from the tier-marker requirement. Keep this list empty if
# possible; the right fix is almost always to add a marker, not an exemption.
_TIER_MARKER_EXEMPT_FILES: frozenset[str] = frozenset()

# A skip reason naming a fixture claims something under tests/fixtures/ could
# not be found. Every one of those is committed, so the only way to reach such
# a skip is a wrong path in the test. That is a bug which presents as a green
# run: two GEMMA-parity tests sat behind one for their entire lifetime because
# the directory *and* the filename were both wrong (#147). require_fixture()
# in tests/support.py is the mechanism tests use, and it raises; this word is
# what the gate looks for in a skip reason so the guard cannot come
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
# pytest.fail, an assert, or require_fixture -- never to skip.
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

    An unparsable file reports a single ``"<file>"`` sentinel entry so it
    surfaces through the same channel rather than being silently skipped.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError):
        return ["<unparsable file>"]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["<unparsable file>"]
    return _untiered_test_functions(tree)


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


def _tier_marker_message(tests_dir: Path) -> str | None:
    """Name every untiered test under ``tests_dir``, or return None.

    Per-item, not per-file: unions module, class, and function markers for
    each ``test_*`` item and reports the function by name when that union
    carries none of ``_REQUIRED_TIER_MARKERS``.
    """
    missing: dict[Path, list[str]] = {}
    for path in sorted(tests_dir.rglob("test_*.py")):
        if path.name in _TIER_MARKER_EXEMPT_FILES:
            continue
        untiered = _file_untiered_functions(path)
        if untiered:
            missing[path] = untiered
    if not missing:
        return None
    listing = "\n  ".join(
        f"{path.relative_to(tests_dir.parent)}: {', '.join(names)}"
        for path, names in missing.items()
    )
    return (
        "The following tests have no tier marker "
        "(tier0/tier1/tier2/slow/benchmark):\n  "
        f"{listing}\n\n"
        "Add `pytestmark = pytest.mark.tier0` (or a per-test marker). "
        "See docs/TESTING.md §1.6."
    )


def _dormant_skip_message(tests_dir: Path) -> str | None:
    """Name every skip under ``tests_dir`` that hides a wrong path, or return None.

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

    Every category is collected before returning, for the reason
    ``require_fixture`` names every missing path at once: fixing one offender and
    re-running to discover the next is the slow way to clear a sweep.
    """
    by_word: list[str] = []
    by_shape: list[str] = []
    by_probe: list[str] = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            # Unparsable files are the tier gate's problem to report; flagging
            # them here too would double every message.
            continue
        rel = path.relative_to(tests_dir.parent)
        by_word += [f"{rel}:{line}" for line in _fixture_skip_lines(tree)]
        by_shape += [f"{rel}:{line}" for line in _path_guarded_skip_lines(tree)]
        by_probe += [f"{rel}:{line}" for line in _attribute_probed_skip_lines(tree)]
    if not (by_word or by_shape or by_probe):
        return None
    parts: list[str] = []
    if by_word:
        listing = "\n  ".join(by_word)
        parts.append(
            "The following skips name a fixture in their reason:\n  "
            f"{listing}\n\n"
            "Everything under tests/fixtures/ is committed, so a fixture that "
            "cannot be found means the test names the wrong path. Call "
            "require_fixture(*paths) from tests/support.py, which raises "
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
    return "\n\n".join(parts)


def check(tests_dir: Path) -> list[str]:
    messages = (_tier_marker_message(tests_dir), _dormant_skip_message(tests_dir))
    return [message for message in messages if message is not None]


def main() -> int:
    messages = check(repo_root() / "tests")
    for message in messages:
        print(message, file=sys.stderr, end="\n\n")
    return 1 if messages else 0


if __name__ == "__main__":
    sys.exit(main())
