"""I/O module for LMM association results.

Writes association results in GEMMA .assoc.txt format for
byte-identical output compatibility.
"""

from pathlib import Path

from jamma.lmm.stats import AssocResult


def format_assoc_line(result: AssocResult) -> str:
    """Format a single association result as tab-separated line.

    Matches GEMMA's WriteFiles formatting:
    - af: .3f (3 decimal places, fixed)
    - beta, se, logl_H1, l_remle, p_wald: .6e (scientific notation)
    - chr, rs: string as-is
    - ps, n_miss: integer as-is

    Args:
        result: AssocResult dataclass instance

    Returns:
        Tab-separated string (no newline)
    """
    return "\t".join(
        [
            result.chr,
            result.rs,
            str(result.ps),
            str(result.n_miss),
            result.allele1,
            result.allele0,
            f"{result.af:.3f}",
            f"{result.beta:.6e}",
            f"{result.se:.6e}",
            f"{result.logl_H1:.6e}",
            f"{result.l_remle:.6e}",
            f"{result.p_wald:.6e}",
        ]
    )


def write_assoc_results(results: list[AssocResult], path: Path) -> None:
    """Write association results in GEMMA .assoc.txt format.

    Output format matches GEMMA exactly:
    - Tab-separated columns
    - Header: chr, rs, ps, n_miss, allele1, allele0, af, beta, se, ...
    - Scientific notation for statistics (6 significant digits)

    Args:
        results: List of AssocResult dataclass instances
        path: Output file path (parent directories created if needed)
    """
    # Ensure parent directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # GEMMA header (tab-separated)
    header = (
        "chr\trs\tps\tn_miss\tallele1\tallele0\taf\tbeta\tse\tlogl_H1\tl_remle\tp_wald"
    )

    with open(path, "w") as f:
        f.write(header + "\n")
        for result in results:
            f.write(format_assoc_line(result) + "\n")
