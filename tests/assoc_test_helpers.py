"""Association-result builders shared by validation tests."""

from __future__ import annotations

from jamma.lmm.stats import AssocResult


def make_assoc(
    rs: str = "rs1",
    *,
    beta: float = 0.5,
    se: float = 0.1,
    af: float = 0.3,
    p_wald: float | None = 0.01,
    logl_H1: float | None = -100.0,
    l_remle: float | None = 0.5,
    p_score: float | None = None,
    p_lrt: float | None = None,
    l_mle: float | None = None,
) -> AssocResult:
    return AssocResult(
        chr="1",
        rs=rs,
        ps=1000,
        n_miss=0,
        allele1="A",
        allele0="G",
        af=af,
        beta=beta,
        se=se,
        logl_H1=logl_H1,
        l_remle=l_remle,
        p_wald=p_wald,
        p_score=p_score,
        p_lrt=p_lrt,
        l_mle=l_mle,
    )
