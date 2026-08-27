"""AssocResult, the per-SNP association record every runner emits."""

from dataclasses import dataclass


@dataclass
class AssocResult:
    """Association test result for a single SNP.

    Matches GEMMA's output format. Fields present depend on test type:
    - Wald (-lmm 1): logl_H1, l_remle, p_wald
    - LRT (-lmm 2): l_mle, p_lrt (no beta/se in GEMMA output, but kept for consistency)
    - Score (-lmm 3): p_score only (no per-SNP logl_H1/l_remle)
    - All (-lmm 4): All fields
    """

    chr: str
    rs: str
    ps: int  # base position
    n_miss: int  # missing count for this SNP
    allele1: str  # minor allele
    allele0: str  # major allele
    af: float  # allele frequency
    beta: float
    se: float
    logl_H1: float | None = None  # Not present for Score-only
    l_remle: float | None = None  # Not present for Score-only
    p_wald: float | None = None  # Only for Wald/-lmm 1
    p_score: float | None = None  # Only for Score/-lmm 3
    l_mle: float | None = None  # MLE lambda (for LRT/-lmm 2)
    p_lrt: float | None = None  # LRT p-value (for LRT/-lmm 2)
