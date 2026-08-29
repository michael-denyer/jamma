"""Write GEMMA-format .assoc.txt files for load_gemma_assoc parser tests.

The seven format variants (Wald full/short, Score, LRT, LRT-full, all-tests,
all-tests-full) differ only in their column list; one writer plus one column
tuple per format replaces seven near-identical functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def write_assoc(
    path: Path, cols: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    """Write a tab-separated .assoc.txt with the given header and rows."""
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


WALD_FULL_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "logl_H1",
    "l_remle",
    "p_wald",
)
WALD_SHORT_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "l_remle",
    "p_wald",
)
SCORE_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "p_score",
)
LRT_COLS = ("chr", "rs", "ps", "n_miss", "allele1", "allele0", "af", "l_mle", "p_lrt")
LRT_FULL_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "logl_H1",
    "l_mle",
    "p_lrt",
)
ALL_TESTS_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "l_remle",
    "l_mle",
    "p_wald",
    "p_lrt",
    "p_score",
)
ALL_TESTS_FULL_COLS = (
    "chr",
    "rs",
    "ps",
    "n_miss",
    "allele1",
    "allele0",
    "af",
    "beta",
    "se",
    "logl_H1",
    "l_remle",
    "l_mle",
    "p_wald",
    "p_lrt",
    "p_score",
)
