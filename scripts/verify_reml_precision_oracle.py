"""Recompute the flat REML fixture's stationary points independently.

This developer-only check uses NumPy to read the fixture and mpmath for all
arithmetic. It imports no JAMMA likelihood, Pab, or optimizer code. Run in an
isolated environment with ``uv run --no-project --with numpy==2.4.6 --with
mpmath==1.4.1 python scripts/verify_reml_precision_oracle.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np


def profiled_reml(eigenvalues, covariate, phenotype, genotype):
    """Return a dense, intercept-plus-SNP REML objective in log(lambda)."""
    # mp.mpf(float) preserves the exact binary64 input value, rather than
    # interpreting its shorter, rounded decimal display as the input.
    d, w, y, g = (
        [mp.mpf(float(value)) for value in array]
        for array in (eigenvalues, covariate, phenotype, genotype)
    )
    df = mp.mpf(len(d) - 2)
    ww = mp.fsum(value * value for value in w)
    wg = mp.fsum(a * b for a, b in zip(w, g, strict=True))
    gg = mp.fsum(value * value for value in g)
    identity_det = ww * gg - wg * wg
    constant = df * (mp.log(df) - mp.log(2 * mp.pi) - 1) / 2

    def objective(log_lambda):
        lam = mp.exp(log_lambda)
        h = [1 / (1 + lam * value) for value in d]

        def product(left, right):
            return mp.fsum(
                weight * a * b for weight, a, b in zip(h, left, right, strict=True)
            )

        sww, swg, sgg = product(w, w), product(w, g), product(g, g)
        swy, sgy, syy = product(w, y), product(g, y), product(y, y)
        determinant = sww * sgg - swg * swg
        residual = (
            syy
            - (sgg * swy * swy - 2 * swg * swy * sgy + sww * sgy * sgy) / determinant
        )
        return (
            constant
            - mp.fsum(mp.log(1 + lam * value) for value in d) / 2
            - mp.log(determinant / identity_det) / 2
            - df * mp.log(residual) / 2
        )

    return objective


def verify(path: Path, digits: int) -> list[dict[str, str]]:
    """Check roots selected independently from a coarse likelihood grid."""
    mp.mp.dps = digits
    low, high = mp.log(mp.mpf("1e-5")), mp.log(mp.mpf("1e5"))
    grid = [low + (high - low) * index / 49 for index in range(50)]
    results = []
    with np.load(path, allow_pickle=False) as fixture:
        for index, snp in enumerate(fixture["snp_ids"]):
            objective = profiled_reml(
                fixture["eigenvalues"],
                fixture["UtW"][:, 0],
                fixture["Uty"],
                fixture["UtG"][:, index],
            )
            values = [objective(point) for point in grid]
            best = max(range(len(grid)), key=values.__getitem__)
            assert 0 < best < len(grid) - 1, f"{snp}: expected an interior peak"

            def score(point, objective=objective):
                return mp.diff(objective, point)

            root = mp.findroot(score, (grid[best - 1], grid[best + 1]))
            assert low < root < high, f"{snp}: root outside lambda bounds"
            assert mp.diff(objective, root, 2) < 0, f"{snp}: root is not a maximum"
            assert objective(root) >= max(values), f"{snp}: grid found a better peak"
            assert abs(score(root)) < mp.mpf("1e-50"), f"{snp}: score did not converge"
            optimum = mp.exp(root)
            stored = mp.mpf(float(fixture["oracle_lambdas"][index]))
            relative = abs(stored / optimum - 1)
            results.append(
                {
                    "snp": str(snp),
                    "lambda": mp.nstr(optimum, 30),
                    "stored_relative_error": mp.nstr(relative, 10),
                }
            )
            assert relative < mp.mpf("1e-10"), f"{snp}: stored root differs: {relative}"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/reml_flat_optima.npz",
    )
    parser.add_argument("--digits", type=int, default=80)
    args = parser.parse_args()
    if args.digits < 60:
        parser.error("--digits must be at least 60")
    print(json.dumps(verify(args.fixture, args.digits), indent=2))


if __name__ == "__main__":
    main()
