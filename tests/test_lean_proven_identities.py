"""Numerical checks of the LMM identities proved over the reals in jamma-lean.

The jamma-lean repository proves, in Lean 4 with Mathlib, the exact-arithmetic
identities behind the LMM formulas. A proof over the reals says the formula is
right. It does not say the code implements that formula, or that the float64
evaluation stays near it. These tests run the real JAMMA code on random,
well-conditioned inputs and hold it to each identity at a rounding-level
tolerance. Each identity is paired with the Lean theorem that states it:

* ``AbIndex.abIndex_image``, ``abIndex_comm``, ``n_index_eq``: packed index
  is a symmetric bijection onto ``range(n_index)``.
* ``Pab.pab_succ``, ``Pab.resid_spec``, ``Rotation.pab_row0_eq_dense``: level
  ``p`` of ``calc_pab`` is ``aᵀ P_p b`` with ``P_p`` the dense GLS projector.
* ``Rotation.logdet_hMat``, ``Logdet.logdetKernel_eq_sum_log``:
  ``Σ log(λ ev + 1) = log det(λK + I)``, in NumPy and in the C kernel.
* ``Profile.gaussLogL_le_profiled``, ``gaussLogL_at_argmax``: the profiled
  constant is the Gaussian log-likelihood maximised over σ².
* ``Stats.px_yy_eq``, ``waldF_eq_beta_sq_div_var``, ``waldF_eq_r2``: Wald.
* ``Stats.scoreF_eq_r2``, ``scoreF_le_n``: Score.
* ``Stats.pXY_sq_le`` (``r² ≤ 1``): the LRT alternative never loses likelihood.

The kinship PSD/row-sum test and the REML centring-invariance test cover
properties the Lean README lists as not yet proved (centring invariance) or
that follow from the Gram form of ``-gk 1``; they share the same dense set-up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bed_reader import to_bed
from scipy.stats import f as f_dist

from jamma.core.constants import n_index
from jamma.genotype.dataset import GenotypeDataset
from jamma.kinship import compute_kinship_streaming
from jamma.lmm import accel
from jamma.lmm.eigen import center_kinship, eigendecompose_kinship
from jamma.lmm.likelihood import mle_log_likelihood
from jamma.lmm.likelihood_numpy import (
    _batch_grid_pab_numpy,
    _batch_pab_at_lambda_numpy,
    _logdet_diag,
    _logl_const,
    _mle_logl,
    _reml_logl,
)
from jamma.lmm.pab import build_index_table, calc_pab, compute_Uab, get_ab_index
from jamma.lmm.stats import (
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_from_pab_numpy,
)
from jamma.lmm.uab import (
    batch_compute_iab_numpy,
    batch_compute_pab_numpy,
    batch_compute_uab_numpy,
    compute_uab_invariant_soa,
)
from tests.math_validation.dense_oracle import evaluate, projection_products
from tests.support import requires_c

pytestmark = pytest.mark.tier0

SEEDS = (11, 23, 47)
N_CVTS = (1, 2, 3)
LAMBDAS = (1e-3, 0.5, 20.0)

# Every relative tolerance below is 1e-10. The inputs keep every condition
# number under ~1e2 (K = XXᵀ/p with p = 2n has eigenvalues in about
# [0.09, 2.9], and λ ≤ 20), so a correct float64 evaluation lands within
# ~1e-13 relative of the dense solve; 1e-10 leaves three orders of headroom
# for n ≈ 250 while still catching any wrong term, which moves results at
# the 1e-3 level or more.
RTOL = 1e-10


class _Case:
    """A random kinship, its eigenbasis, covariates with an intercept, SNPs, y."""

    def __init__(self, seed: int, n: int, n_cvt: int, n_snps: int = 4) -> None:
        rng = np.random.default_rng(seed)
        markers = rng.standard_normal((n, 2 * n))
        self.kinship = markers @ markers.T / (2 * n)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.kinship)
        self.W = np.column_stack((np.ones(n), rng.standard_normal((n, n_cvt - 1))))
        self.G = rng.standard_normal((n, n_snps))
        self.y = (
            self.W @ rng.standard_normal(n_cvt)
            + 0.3 * self.G[:, 0]
            + rng.standard_normal(n)
        )
        self.n = n
        self.n_cvt = n_cvt

    def rotate(self, v: np.ndarray) -> np.ndarray:
        return self.eigenvectors.T @ v

    def uab_batch(self) -> np.ndarray:
        """Uab for every SNP, through JAMMA's batch builder."""
        return batch_compute_uab_numpy(
            self.n_cvt, self.rotate(self.W), self.rotate(self.y), self.rotate(self.G).T
        )


def _dense_projected(case: _Case, snp: int, lam: float) -> tuple[float, float, float]:
    """(P_XX, P_XY, P_YY) at level n_cvt from the dense GLS projector."""
    vectors = np.column_stack((case.W, case.G[:, snp], case.y))
    levels = projection_products(case.kinship, vectors, lam)
    level = levels[case.n_cvt]
    c = case.n_cvt
    return float(level[c, c]), float(level[c, c + 1]), float(level[c + 1, c + 1])


# ---------------------------------------------------------------------------
# 1. AbIndex.abIndex_image, abIndex_comm, n_index_eq
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cvt", range(1, 21))
def test_ab_index_is_a_symmetric_bijection_onto_n_index(n_cvt: int) -> None:
    cols = n_cvt + 2
    images = [
        get_ab_index(a, b, n_cvt)
        for a in range(1, cols + 1)
        for b in range(a, cols + 1)
    ]
    assert sorted(images) == list(range(n_index(n_cvt)))
    for a in range(1, cols + 1):
        for b in range(1, cols + 1):
            assert get_ab_index(a, b, n_cvt) == get_ab_index(b, a, n_cvt)


# ---------------------------------------------------------------------------
# 2. Pab.pab_succ + resid_spec + Rotation.pab_row0_eq_dense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_cvt", N_CVTS)
@pytest.mark.parametrize("lam", LAMBDAS)
def test_calc_pab_equals_dense_projector_at_every_level(
    seed: int, n_cvt: int, lam: float
) -> None:
    """Pab[p, (a, b)] = v_aᵀ P_p v_b for every level p and every a, b > p.

    P_p = H⁻¹ - H⁻¹Z(ZᵀH⁻¹Z)⁻¹ZᵀH⁻¹ with Z the first p columns of [W, x, y] and
    H = λK + I on the unrotated data. calc_pab fills level p only for a > p
    (lower pairs are projected out), so those are the entries compared.
    """
    case = _Case(seed, 40, n_cvt)
    snp = 0
    vectors = np.column_stack((case.W, case.G[:, snp], case.y))
    dense = projection_products(case.kinship, vectors, lam)

    hi_eval = 1.0 / (lam * case.eigenvalues + 1.0)
    uab = compute_Uab(
        case.rotate(case.W), case.rotate(case.y), case.rotate(case.G[:, snp])
    )
    scalar = calc_pab(n_cvt, hi_eval, uab)
    batch = batch_compute_pab_numpy(n_cvt, hi_eval, case.uab_batch())[snp]

    cols = n_cvt + 2
    for p in range(cols):
        for a in range(p + 1, cols + 1):
            for b in range(a, cols + 1):
                expected = dense[p][a - 1, b - 1]
                # Off-diagonal entries can sit near zero; the floor scales with
                # the Cauchy-Schwarz bound sqrt(P_aa P_bb) at the same level.
                floor = RTOL * np.sqrt(dense[p][a - 1, a - 1] * dense[p][b - 1, b - 1])
                idx = get_ab_index(a, b, n_cvt)
                for name, pab in (("calc_pab", scalar), ("batch", batch)):
                    assert pab[p, idx] == pytest.approx(
                        expected, rel=RTOL, abs=floor
                    ), f"{name} level {p} pair ({a}, {b})"


# ---------------------------------------------------------------------------
# 3. Rotation.logdet_hMat (NumPy) and Logdet.logdetKernel_eq_sum_log (C)
# ---------------------------------------------------------------------------

# n > 64 so the C kernel's `(i & 60) == 60` renormalisation runs, and
# n % 4 != 0 so its scalar tail loop runs too.
LOGDET_SIZES = (67, 130, 257)


@pytest.mark.parametrize("n", LOGDET_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_numpy_logdet_h_equals_slogdet(n: int, seed: int) -> None:
    case = _Case(seed, n, 1)
    lambdas = np.array([1e-3, 0.5, 20.0, 3e3])
    uab = case.uab_batch()[:1].repeat(len(lambdas), axis=0)
    # NumPy's slogdet raises spurious divide/overflow FP flags on some LAPACK
    # builds for well-conditioned input; the sign and finiteness checks below
    # are what guard the reference value.
    with np.errstate(all="ignore"):
        signs, expected = np.array(
            [np.linalg.slogdet(lam * case.kinship + np.eye(n)) for lam in lambdas]
        ).T
    assert np.all(signs == 1.0)
    assert np.all(np.isfinite(expected))

    _pab, per_snp = _batch_pab_at_lambda_numpy(1, lambdas, case.eigenvalues, uab)
    _pab, grid = _batch_grid_pab_numpy(1, lambdas, case.eigenvalues, uab[:1])

    np.testing.assert_allclose(per_snp, expected, rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(grid[:, 0], expected, rtol=RTOL, atol=RTOL)


def _native_run(
    case: _Case, lmm_mode: int, hi_eval_null: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    utw = np.ascontiguousarray(case.rotate(case.W))
    uty = np.ascontiguousarray(case.rotate(case.y))
    kwargs: dict[str, object] = {}
    if hi_eval_null is not None:
        kwargs["hi_eval_null"] = hi_eval_null
    if lmm_mode == 2:
        # Feeds only p_lrt; the logls checked here do not read it.
        kwargs["logl_H0"] = 0.0
    workspace = accel.require().create_workspace_c(
        np.ascontiguousarray(case.eigenvalues),
        compute_uab_invariant_soa(utw, uty, n_cvt=case.n_cvt),
        utw,
        uty,
        case.n,
        1e-5,
        1e5,
        50,
        20,
        1,
        case.n_cvt,
        lmm_mode=lmm_mode,
        **kwargs,
    )
    return accel.require().compute_lmm_chunk_c(
        workspace, np.ascontiguousarray(case.rotate(case.G).T), 1
    )


@requires_c
@pytest.mark.parametrize("n", LOGDET_SIZES)
@pytest.mark.parametrize("n_cvt", N_CVTS)
def test_native_likelihoods_match_dense_logdet_and_projector(
    n: int, n_cvt: int
) -> None:
    """The C kernel's logls equal dense log-likelihoods at the λ it reports.

    logdet_h_lambda is reachable only inside the fused chunk kernel, so it is
    checked through the MLE log-likelihood (mode 2: const - ½ logdet H -
    ½ n log P_yy) and the REML one (mode 1), whose dense twins take log det H
    from a Cholesky factor. Mode 1 also checks the native Pab through β and SE.
    """
    case = _Case(1000 + n + n_cvt, n, n_cvt)

    mle = _native_run(case, lmm_mode=2)
    for snp, lam in enumerate(mle["lambdas_mle"]):
        oracle = evaluate(case.kinship, case.W, case.G[:, snp], case.y, lam)
        assert mle["logls"][snp] == pytest.approx(oracle["mle"], rel=RTOL)

    reml = _native_run(case, lmm_mode=1)
    for snp, lam in enumerate(reml["lambdas"]):
        oracle = evaluate(case.kinship, case.W, case.G[:, snp], case.y, lam)
        assert reml["logls"][snp] == pytest.approx(
            oracle["reml"], rel=RTOL, abs=RTOL * oracle["reml_term_scale"]
        )
        assert reml["betas"][snp] == pytest.approx(
            oracle["beta"], rel=RTOL, abs=RTOL * oracle["se"]
        )
        assert reml["ses"][snp] == pytest.approx(oracle["se"], rel=RTOL)


# ---------------------------------------------------------------------------
# 4. Profile.gaussLogL_at_argmax, gaussLogL_le_profiled
# ---------------------------------------------------------------------------


def _gauss_logl(m: int, q: float, sigma2: float) -> float:
    """Gaussian log-likelihood of m residuals with sum of squares q."""
    return -0.5 * m * np.log(2.0 * np.pi * sigma2) - q / (2.0 * sigma2)


@pytest.mark.parametrize("m", [3, 57, 1000, 150_000])
@pytest.mark.parametrize("q", [1e-6, 0.37, 52.0, 4.1e5])
def test_profiled_logl_is_the_gaussian_maximum_over_sigma2(m: int, q: float) -> None:
    profiled = _logl_const(m) - 0.5 * m * np.log(q)
    argmax = q / m
    # Both sides are sums of O(m log m) terms; 1e-12 relative of the largest
    # term covers the rounding in either evaluation.
    scale = 0.5 * m * (abs(np.log(m)) + np.log(2 * np.pi) + 1 + abs(np.log(q)))
    assert profiled == pytest.approx(_gauss_logl(m, q, argmax), abs=1e-12 * scale)
    for factor in (0.5, 0.9, 1.1, 2.0, 10.0):
        assert _gauss_logl(m, q, argmax * factor) < profiled


# ---------------------------------------------------------------------------
# 5. Stats.px_yy_eq, waldF_eq_beta_sq_div_var, waldF_eq_r2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_cvt", N_CVTS)
@pytest.mark.parametrize("lam", LAMBDAS)
def test_wald_statistics_satisfy_the_r2_identities(
    seed: int, n_cvt: int, lam: float
) -> None:
    case = _Case(seed, 60, n_cvt)
    table = build_index_table(n_cvt)
    hi_eval = 1.0 / (lam * case.eigenvalues + 1.0)
    pab = batch_compute_pab_numpy(n_cvt, hi_eval, case.uab_batch())
    betas, ses, _p = batch_calc_wald_stats_from_pab_numpy(n_cvt, pab, case.n)
    df = case.n - n_cvt - 1

    p_xx = pab[:, n_cvt, table.idx_xx]
    p_xy = pab[:, n_cvt, table.idx_xy]
    p_yy = pab[:, n_cvt, table.idx_yy]
    px_yy = pab[:, n_cvt + 1, table.idx_yy]
    r2 = p_xy**2 / (p_xx * p_yy)

    # r² stays well below 1 here, so P_YY - P_XY²/P_XX has no cancellation.
    np.testing.assert_allclose(px_yy, p_yy - p_xy**2 / p_xx, rtol=RTOL)
    # safe_sqrt only takes |var| for |var| < 1e-3, which is the identity on a
    # positive variance, so SE is sqrt(Px_YY / (df P_XX)) on every SNP.
    np.testing.assert_allclose((betas / ses) ** 2, df * r2 / (1.0 - r2), rtol=RTOL)
    # And the Pab the function reads matches the dense projector.
    for snp in range(case.G.shape[1]):
        dense_xx, dense_xy, dense_yy = _dense_projected(case, snp, lam)
        dense_r2 = dense_xy**2 / (dense_xx * dense_yy)
        assert r2[snp] == pytest.approx(dense_r2, rel=RTOL, abs=1e-14)


# ---------------------------------------------------------------------------
# 6. Stats.scoreF_eq_r2, scoreF_le_n
# ---------------------------------------------------------------------------


def _score_case(seed: int, n_cvt: int) -> _Case:
    """A case whose last SNP explains y almost perfectly, so F approaches n."""
    case = _Case(seed, 60, n_cvt)
    rng = np.random.default_rng(seed + 1)
    case.G[:, -1] = case.y + 1e-3 * rng.standard_normal(case.n)
    return case


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_cvt", N_CVTS)
def test_score_f_is_n_r2_and_at_most_n(seed: int, n_cvt: int) -> None:
    lam = 0.5
    case = _score_case(seed, n_cvt)
    df = case.n - n_cvt - 1
    hi_eval = 1.0 / (lam * case.eigenvalues + 1.0)
    _b, _se, p_score = batch_calc_score_stats_numpy(
        n_cvt, hi_eval, case.uab_batch(), case.n
    )

    f_dense = np.empty(case.G.shape[1])
    for snp in range(case.G.shape[1]):
        p_xx, p_xy, p_yy = _dense_projected(case, snp, lam)
        f_dense[snp] = case.n * p_xy**2 / (p_xx * p_yy)
    assert np.all(f_dense <= case.n)
    assert f_dense[-1] > 0.99 * case.n  # the near-collinear SNP is at the bound

    # Cephes betainc against scipy's F survival function: 1e-8 is the
    # documented p-value agreement (tests/test_likelihood_numpy.py header).
    np.testing.assert_allclose(p_score, f_dist.sf(f_dense, 1, df), rtol=1e-8)
    # F ≤ n, so no p-value can drop below the one at F = n.
    assert np.all(p_score >= f_dist.sf(case.n, 1, df) * (1 - 1e-8))


@requires_c
@pytest.mark.parametrize("n_cvt", N_CVTS)
def test_native_score_f_is_n_r2(n_cvt: int) -> None:
    lam = 0.5
    case = _score_case(7, n_cvt)
    df = case.n - n_cvt - 1
    hi_eval = 1.0 / (lam * case.eigenvalues + 1.0)
    result = _native_run(case, lmm_mode=3, hi_eval_null=hi_eval)

    f_dense = np.array(
        [
            case.n * xy**2 / (xx * yy)
            for xx, xy, yy in (
                _dense_projected(case, snp, lam) for snp in range(case.G.shape[1])
            )
        ]
    )
    np.testing.assert_allclose(result["p_scores"], f_dist.sf(f_dense, 1, df), rtol=1e-8)


# ---------------------------------------------------------------------------
# 7. LRT: logl_H1 - logl_H0 = -½ n log(1 - r²) ≥ 0 at a shared λ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_cvt", N_CVTS)
@pytest.mark.parametrize("lam", LAMBDAS)
def test_mle_with_genotype_never_below_null_at_shared_lambda(
    seed: int, n_cvt: int, lam: float
) -> None:
    case = _Case(seed, 60, n_cvt)
    uab = case.uab_batch()
    null_uab = compute_Uab(case.rotate(case.W), case.rotate(case.y))
    logl_h0 = mle_log_likelihood(lam, case.eigenvalues, null_uab, n_cvt)

    lambdas = np.full(uab.shape[0], lam)
    pab, logdet_h = _batch_pab_at_lambda_numpy(n_cvt, lambdas, case.eigenvalues, uab)
    logl_h1 = _mle_logl(pab, logdet_h, case.n)

    assert np.all(logl_h1 >= logl_h0)
    table = build_index_table(n_cvt)
    p_xx = pab[:, n_cvt, table.idx_xx]
    p_xy = pab[:, n_cvt, table.idx_xy]
    p_yy = pab[:, n_cvt, table.idx_yy]
    gain = -0.5 * case.n * np.log1p(-(p_xy**2) / (p_xx * p_yy))
    # The difference of two O(n) log-likelihoods: absolute rounding floor.
    np.testing.assert_allclose(logl_h1 - logl_h0, gain, rtol=RTOL, atol=1e-11)


# ---------------------------------------------------------------------------
# 8. Centred (-gk 1) kinship: symmetric PSD Gram matrix with K·1 = 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_centered_kinship_is_psd_with_zero_row_sums(seed: int, tmp_path: Path) -> None:
    rng = np.random.default_rng(seed)
    n_samples, n_snps = 50, 120
    genotypes = rng.binomial(2, rng.uniform(0.1, 0.5, n_snps), (n_samples, n_snps))
    genotypes = genotypes.astype(np.float64)
    genotypes[rng.random(genotypes.shape) < 0.05] = np.nan
    bfile = tmp_path / "random"
    to_bed(bfile.with_suffix(".bed"), genotypes)

    K = compute_kinship_streaming(
        GenotypeDataset.open_plink(bfile), check_memory=False, show_progress=False
    )

    scale = np.abs(K).max()
    np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-15 * scale)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() >= -1e-12 * eigenvalues.max()
    # Each centred, mean-imputed column sums to zero, so K·1 = X_c(X_cᵀ1)/p = 0
    # up to n additions of O(scale) terms.
    np.testing.assert_allclose(K @ np.ones(n_samples), 0.0, atol=1e-12 * scale)


# ---------------------------------------------------------------------------
# 9. REML is invariant to centring K when W holds an intercept; MLE is not
# ---------------------------------------------------------------------------


def _logls_through_jamma(
    case: _Case, kinship: np.ndarray, lambdas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """REML and MLE logls of SNP 0 at each λ, via JAMMA's eigen and Pab path."""
    eigenvalues, eigenvectors = eigendecompose_kinship(
        kinship.copy(), check_memory=False, show_progress=False
    )
    uab = batch_compute_uab_numpy(
        case.n_cvt,
        eigenvectors.T @ case.W,
        eigenvectors.T @ case.y,
        (eigenvectors.T @ case.G[:, :1]).T,
    )
    logdet_iab = _logdet_diag(batch_compute_iab_numpy(case.n_cvt, uab))
    uab = uab.repeat(len(lambdas), axis=0)
    pab, logdet_h = _batch_pab_at_lambda_numpy(case.n_cvt, lambdas, eigenvalues, uab)
    df = case.n - case.n_cvt - 1
    return _reml_logl(pab, logdet_h, logdet_iab, df), _mle_logl(pab, logdet_h, case.n)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_cvt", N_CVTS)
def test_reml_is_invariant_to_centring_kinship_and_mle_is_not(
    seed: int, n_cvt: int
) -> None:
    case = _Case(seed, 60, n_cvt)
    lambdas = np.array([1e-3, 0.1, 1.0, 10.0, 100.0])
    centred = case.kinship.copy()
    center_kinship(centred)

    reml_raw, mle_raw = _logls_through_jamma(case, case.kinship, lambdas)
    reml_centred, mle_centred = _logls_through_jamma(case, centred, lambdas)

    # Two different eigenbases: each side carries ~n·eps·cond rounding.
    np.testing.assert_allclose(reml_centred, reml_raw, rtol=RTOL)
    # MLE has no log det(WᵀH⁻¹W) term to absorb the change in log det H, so the
    # same inputs must separate it by far more than the REML tolerance; this is
    # what shows the REML check can fail. The gap grows with λ (6e-4 at 1e-3).
    assert np.all(np.abs(mle_centred - mle_raw) > 1e3 * RTOL * np.abs(mle_raw))
