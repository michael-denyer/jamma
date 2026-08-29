"""Rewrite the inline rotated-covariate draw recipe onto ``rotated_lmm_inputs``.

Targets the mechanical shape ``rotated_lmm_inputs`` already covers: a single
seeded generator drawing ``UtW = rng.standard_normal((n, cvt))`` immediately
followed by ``Uty = rng.standard_normal(n)`` and, optionally, ``UtG =
rng.standard_normal((n, m))`` right after, with no other read of ``rng``
between the three lines. ``cvt`` must not be the literal ``1``: for
``n_cvt == 1`` the builder draws an intercept column (``np.ones``) instead of
a random one, so a literal-``1`` site is a different recipe wearing the same
variable names, and this codemod leaves it alone.

Every candidate is verified numerically before being rewritten: the matched
statements are executed in an isolated namespace and their output arrays are
compared byte-for-byte against a real call to ``rotated_lmm_inputs`` with the
inferred ``n_samples``/``n_snps``/``n_cvt``/``seed``. A syntactic match with a
numeric mismatch is reported, not rewritten -- rerun with ``--check`` to see
what is left after a rewrite pass.

Known gap: the codemod does not remove the ``rng = np.random.default_rng(...)``
binding line itself, even when the rewrite leaves it with no reader. Check the
diff for a now-dead ``rng =`` line after each run and delete it by hand; a
second, unrelated read of ``rng`` later in the same function is legal and the
codemod cannot always tell the two cases apart from source alone.

Usage::

    uv run python scripts/codemod_test_builders.py tests/            # rewrite
    uv run python scripts/codemod_test_builders.py --check tests/    # report only
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.builders import rotated_lmm_inputs

_UTW_RE = re.compile(
    r"^(?P<indent>[ \t]*)UtW = rng\.standard_normal\("
    r"\((?P<n>[^,]+), (?P<cvt>[^)]+)\)\)\s*$"
)
_UTY_RE = re.compile(r"^[ \t]*Uty = rng\.standard_normal\((?P<n>[^)]+)\)\s*$")
_UTG_RE = re.compile(
    r"^[ \t]*UtG = rng\.standard_normal\(\((?P<n>[^,]+), (?P<m>[^)]+)\)\)\s*$"
)
_RNG_RE = re.compile(r"^[ \t]*rng = np\.random\.default_rng\((?P<seed>[^)]+)\)\s*$")
_EIGENVALUES_RE = re.compile(
    r"^[ \t]*eigenvalues = np\.sort\(rng\.uniform\("
    r"(?P<lo>[^,]+), (?P<hi>[^,]+), (?P<n>[^)]+)\)\)\s*$"
)


@dataclass
class Candidate:
    """A syntactically matched draw group, before numeric verification."""

    file: Path
    utw_line: int  # 1-based, the line holding "UtW = ..."
    n_lines: int  # 2 (UtW+Uty) or 3 (UtW+Uty+UtG)
    indent: str
    n_expr: str
    cvt_expr: str
    m_expr: str | None
    seed: int
    rng_line: int
    eig_lo: str | None = None
    eig_hi: str | None = None
    eig_line: int | None = (
        None  # 1-based line of the "eigenvalues = ..." draw, if present
    )


def _find_rng_seed(lines: list[str], before_line: int) -> tuple[int, int] | None:
    """Find the nearest preceding literal ``rng = np.random.default_rng(N)``.

    Returns (seed, line_number) or None if the binding is not a literal int
    (a variable seed cannot be re-derived without executing more context, so
    those sites are left for hand migration).
    """
    for i in range(before_line - 2, -1, -1):
        m = _RNG_RE.match(lines[i])
        if m:
            seed_expr = m.group("seed").strip()
            if re.fullmatch(r"-?\d+", seed_expr):
                return int(seed_expr), i + 1
            return None
    return None


def find_candidates(path: Path) -> list[Candidate]:
    text = path.read_text()
    lines = text.splitlines()
    candidates = []
    for i, line in enumerate(lines):
        m_utw = _UTW_RE.match(line)
        if not m_utw:
            continue
        cvt_expr = m_utw.group("cvt").strip()
        if cvt_expr == "1":
            continue  # builder draws an intercept for n_cvt==1, different recipe
        if i + 1 >= len(lines):
            continue
        m_uty = _UTY_RE.match(lines[i + 1])
        if not m_uty or m_uty.group("n").strip() != m_utw.group("n").strip():
            continue
        seed_line = _find_rng_seed(lines, i + 1)
        if seed_line is None:
            continue
        seed, rng_line = seed_line

        n_lines = 2
        m_expr = None
        if i + 2 < len(lines):
            m_utg = _UTG_RE.match(lines[i + 2])
            if m_utg and m_utg.group("n").strip() == m_utw.group("n").strip():
                n_lines = 3
                m_expr = m_utg.group("m").strip()

        eig_lo = eig_hi = None
        eig_line = None
        if i > 0:
            m_eig = _EIGENVALUES_RE.match(lines[i - 1])
            if m_eig and m_eig.group("n").strip() == m_utw.group("n").strip():
                eig_lo, eig_hi = m_eig.group("lo").strip(), m_eig.group("hi").strip()
                eig_line = i  # 1-based line number of lines[i - 1]

        candidates.append(
            Candidate(
                file=path,
                utw_line=i + 1,
                n_lines=n_lines,
                indent=m_utw.group("indent"),
                n_expr=m_utw.group("n").strip(),
                cvt_expr=cvt_expr,
                eig_lo=eig_lo,
                eig_hi=eig_hi,
                eig_line=eig_line,
                m_expr=m_expr,
                seed=seed,
                rng_line=rng_line,
            )
        )
    return candidates


def _resolve_int_values(expr: str, bindings: dict[str, list[int]]) -> list[int] | None:
    """Resolve every possible value a variable name or literal int can take."""
    if re.fullmatch(r"-?\d+", expr):
        return [int(expr)]
    return bindings.get(expr)


def _enclosing_function(path: Path, line: int) -> ast.FunctionDef | None:
    """Return the innermost ``def`` whose body spans ``line``."""
    tree = ast.parse(path.read_text())
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        end = node.end_lineno or node.lineno
        if not (node.lineno <= line <= end):
            continue
        if (
            best is None
            or (end - node.lineno) < (best.end_lineno or best.lineno) - best.lineno
        ):
            best = node
    return best


def _parametrize_values(func: ast.FunctionDef, name: str) -> list[int] | None:
    """Read ``@pytest.mark.parametrize("name", [1, 2, 3])`` values for one arg.

    Only handles a single-argument parametrize with an int-literal list, the
    shape every n_cvt parametrize in this tree uses. Anything else (multi-arg
    tuples, non-literal values) returns None so the caller resolves it as
    unverifiable rather than guessing.
    """
    for dec in func.decorator_list:
        if not (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "parametrize"
        ):
            continue
        if len(dec.args) < 2:
            continue
        arg_names_node = dec.args[0]
        if not (
            isinstance(arg_names_node, ast.Constant) and arg_names_node.value == name
        ):
            continue
        values_node = dec.args[1]
        if not isinstance(values_node, ast.List):
            return None
        values = []
        for elt in values_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                values.append(elt.value)
            else:
                return None
        return values
    return None


def _local_int_bindings(path: Path, before_line: int) -> dict[str, list[int]]:
    """Collect ``NAME = <int literal>`` bindings visible before a line.

    Each name maps to the list of values it could hold: one value for a plain
    assignment, every value in an ``@pytest.mark.parametrize`` list for a
    same-named function parameter. Does not track control flow; a name
    reassigned between its binding and the draw is out of scope for this
    codemod and the candidate is skipped.
    """
    func = _enclosing_function(path, before_line)
    scope: ast.AST = func if func is not None else ast.parse(path.read_text())
    bindings: dict[str, list[int]] = {}
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or node.lineno >= before_line:
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, int
            ):
                bindings[node.targets[0].id] = [node.value.value]
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            if not isinstance(node.value, ast.Tuple):
                continue
            for target, value in zip(
                node.targets[0].elts, node.value.elts, strict=False
            ):
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, int)
                ):
                    bindings[target.id] = [value.value]

    if func is not None:
        for arg in func.args.args:
            if arg.arg in bindings:
                continue
            values = _parametrize_values(func, arg.arg)
            if values:
                bindings[arg.arg] = values
    return bindings


def _verify_one(
    cand: Candidate,
    n_samples: int,
    n_cvt: int,
    n_snps: int | None,
    eig_range: tuple[float, float] | None,
) -> bool:
    rng = np.random.default_rng(cand.seed)
    eigenvalues_actual = None
    if eig_range is not None:
        eigenvalues_actual = np.sort(rng.uniform(*eig_range, n_samples))
    utw_actual = rng.standard_normal((n_samples, n_cvt))
    uty_actual = rng.standard_normal(n_samples)
    utg_actual = None
    if n_snps is not None:
        utg_actual = rng.standard_normal((n_samples, n_snps))

    inputs = (
        rotated_lmm_inputs(
            n_samples=n_samples,
            n_snps=n_snps if n_snps is not None else 1,
            n_cvt=n_cvt,
            seed=cand.seed,
            eig_range=eig_range,
        )
        if eig_range is not None
        else rotated_lmm_inputs(
            n_samples=n_samples,
            n_snps=n_snps if n_snps is not None else 1,
            n_cvt=n_cvt,
            seed=cand.seed,
        )
    )

    if eigenvalues_actual is not None and not np.array_equal(
        inputs.eigenvalues, eigenvalues_actual
    ):
        return False
    if not np.array_equal(inputs.UtW, utw_actual):
        return False
    if not np.array_equal(inputs.Uty, uty_actual):
        return False
    return not (utg_actual is not None and not np.array_equal(inputs.UtG, utg_actual))


def verify_candidate(cand: Candidate) -> bool:
    """Execute the matched lines and compare against ``rotated_lmm_inputs``.

    Returns True only when every drawn array is byte-identical to the
    builder's output for every value the inferred parameters can take (a
    parametrized ``n_cvt`` is checked at each of its parametrize values, not
    just one).
    """
    bindings = _local_int_bindings(cand.file, cand.utw_line)
    n_samples_values = _resolve_int_values(cand.n_expr, bindings)
    n_cvt_values = _resolve_int_values(cand.cvt_expr, bindings)
    if not n_samples_values or not n_cvt_values:
        return False

    n_snps_values: list[int | None] = [None]
    if cand.n_lines == 3 and cand.m_expr is not None:
        resolved = _resolve_int_values(cand.m_expr, bindings)
        if not resolved:
            return False
        n_snps_values = list(resolved)

    eig_range: tuple[float, float] | None = None
    if cand.eig_line is not None:
        if cand.eig_lo is None or cand.eig_hi is None:
            return False
        if not (
            re.fullmatch(r"-?\d+(\.\d+)?", cand.eig_lo)
            and re.fullmatch(r"-?\d+(\.\d+)?", cand.eig_hi)
        ):
            return False  # variable bound, not a literal -- leave for hand migration
        eig_range = (float(cand.eig_lo), float(cand.eig_hi))

    return all(
        _verify_one(cand, n_samples, n_cvt, n_snps, eig_range)
        for n_samples in n_samples_values
        for n_cvt in n_cvt_values
        for n_snps in n_snps_values
    )


def rewrite_file(path: Path, candidates: list[Candidate]) -> int:
    """Apply verified rewrites bottom-to-top so earlier line numbers stay valid."""
    verified = [c for c in candidates if verify_candidate(c)]
    if not verified:
        return 0
    lines = path.read_text().splitlines()
    for cand in sorted(verified, key=lambda c: c.utw_line, reverse=True):
        start = (
            (cand.eig_line - 1) if cand.eig_line is not None else (cand.utw_line - 1)
        )
        end = (cand.utw_line - 1) + cand.n_lines
        n_snps_arg = f", n_snps={cand.m_expr}" if cand.m_expr else ", n_snps=1"
        eig_range_arg = (
            f", eig_range=({cand.eig_lo}, {cand.eig_hi})"
            if cand.eig_line is not None
            else ""
        )
        replacement = [
            f"{cand.indent}_inputs = rotated_lmm_inputs("
            f"n_samples={cand.n_expr}, n_cvt={cand.cvt_expr}, "
            f"seed={cand.seed}{n_snps_arg}{eig_range_arg})",
        ]
        names = ["eigenvalues"] if cand.eig_line is not None else []
        names += ["UtW", "Uty"] + (["UtG"] if cand.n_lines == 3 else [])
        targets = ", ".join(names)
        sources = ", ".join(f"_inputs.{n}" for n in names)
        replacement.append(f"{cand.indent}{targets} = {sources}")
        lines[start:end] = replacement
    path.write_text("\n".join(lines) + "\n")
    return len(verified)


def _ensure_import(path: Path) -> None:
    text = path.read_text()
    if re.search(r"from tests\.builders import[^\n]*\brotated_lmm_inputs\b", text):
        return
    if "rotated_lmm_inputs" not in text:
        return
    tree = ast.parse(text)
    lines = text.splitlines()
    insert_at = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_at = node.end_lineno or node.lineno
    lines.insert(insert_at, "from tests.builders import rotated_lmm_inputs")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="File or directory to scan")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report candidates without rewriting; exit 1 if any remain",
    )
    args = parser.parse_args()

    files = sorted(args.path.rglob("test_*.py")) if args.path.is_dir() else [args.path]

    total_rewritten = 0
    total_remaining = 0
    for f in files:
        candidates = find_candidates(f)
        if not candidates:
            continue
        if args.check:
            for c in candidates:
                verified = verify_candidate(c)
                status = "would rewrite" if verified else "remainder (verify failed)"
                print(f"{f}:{c.utw_line}: {status}")
                if verified:
                    total_rewritten += 1
                else:
                    total_remaining += 1
        else:
            n = rewrite_file(f, candidates)
            if n:
                _ensure_import(f)
                print(f"{f}: rewrote {n} site(s)")
            total_rewritten += n
            total_remaining += len(candidates) - n

    if args.check:
        print(f"{total_rewritten} site(s) would rewrite, {total_remaining} remain")
        return 1 if total_rewritten else 0
    print(f"{total_rewritten} site(s) rewritten, {total_remaining} left for hand fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
