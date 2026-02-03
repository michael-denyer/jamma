"""I/O module for LMM association results.

Writes association results in GEMMA .assoc.txt format for
byte-identical output compatibility.
"""

from pathlib import Path

from jamma.lmm.stats import AssocResult


def format_assoc_line(result: AssocResult) -> str:
    """Format a single Wald test result as tab-separated line.

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


def format_assoc_line_score(result: AssocResult) -> str:
    """Format Score test result as tab-separated line.

    GEMMA -lmm 3 format:
    chr  rs  ps  n_miss  allele1  allele0  af  beta  se  p_score

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
            f"{result.p_score:.6e}",
        ]
    )


def format_assoc_line_lrt(result: AssocResult) -> str:
    """Format LRT result as tab-separated line.

    GEMMA -lmm 2 format:
    chr  rs  ps  n_miss  allele1  allele0  af  l_mle  p_lrt

    Note: beta and se are NOT included in pure LRT output.

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
            f"{result.l_mle:.6e}",
            f"{result.p_lrt:.6e}",
        ]
    )


# GEMMA headers
HEADER_WALD = (
    "chr\trs\tps\tn_miss\tallele1\tallele0\taf\tbeta\tse\tlogl_H1\tl_remle\tp_wald"
)
HEADER_SCORE = "chr\trs\tps\tn_miss\tallele1\tallele0\taf\tbeta\tse\tp_score"
HEADER_LRT = "chr\trs\tps\tn_miss\tallele1\tallele0\taf\tl_mle\tp_lrt"


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

    with open(path, "w") as f:
        f.write(HEADER_WALD + "\n")
        for result in results:
            f.write(format_assoc_line(result) + "\n")


class IncrementalAssocWriter:
    """Write association results incrementally to disk.

    Context manager that writes results immediately as they are produced,
    avoiding memory accumulation for large GWAS. Output format matches
    write_assoc_results exactly for byte-identical output.

    Example:
        with IncrementalAssocWriter(Path("output.assoc.txt")) as writer:
            for result in compute_results():
                writer.write(result)
        print(f"Wrote {writer.count} results")

        # For Score test:
        with IncrementalAssocWriter(
            Path("output.assoc.txt"), test_type="score"
        ) as writer:
            ...

        # For LRT:
        with IncrementalAssocWriter(
            Path("output.assoc.txt"), test_type="lrt"
        ) as writer:
            ...
    """

    def __init__(self, path: Path, test_type: str = "wald"):
        """Initialize writer with output path.

        Args:
            path: Output file path. Parent directories created if needed.
            test_type: Test type for formatting ("wald", "score", or "lrt")
        """
        self.path = Path(path)
        self.test_type = test_type
        self._file = None
        self._count = 0

    def __enter__(self) -> "IncrementalAssocWriter":
        """Open file and write header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        if self.test_type == "score":
            header = HEADER_SCORE
        elif self.test_type == "lrt":
            header = HEADER_LRT
        else:
            header = HEADER_WALD
        self._file.write(header + "\n")
        return self

    def write(self, result: AssocResult) -> None:
        """Write single result immediately to disk.

        Args:
            result: AssocResult to write.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        if self.test_type == "score":
            line = format_assoc_line_score(result)
        elif self.test_type == "lrt":
            line = format_assoc_line_lrt(result)
        else:
            line = format_assoc_line(result)
        self._file.write(line + "\n")
        self._count += 1

    def write_batch(self, results: list[AssocResult]) -> None:
        """Write multiple results at once (convenience method).

        Args:
            results: List of AssocResult to write.
        """
        for result in results:
            self.write(result)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def count(self) -> int:
        """Number of results written."""
        return self._count
