"""Tests for jlinalg QR factorization and SVD.

Run:
    uv run pytest tests/test_jlinalg_lapack.py -x -v
"""

import numpy as np
import pytest

from jamma import jlinalg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _random_matrix(shape: tuple[int, ...]) -> np.ndarray:
    """Return a fresh random float64 matrix (deterministic seed)."""
    return _RNG.standard_normal(shape).astype(np.float64)


# ---------------------------------------------------------------------------
# TestQR
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not jlinalg.blas_has_dgeqrf,
    reason="vendor LAPACK dgeqrf not available (jlinalg-own / BLIS-ILP64 build)",
)
class TestQR:
    """Tests for jlinalg.qr() — reduced QR factorization."""

    @pytest.mark.parametrize(
        "shape",
        [(10, 5), (100, 20), (2, 1), (1, 1)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_reconstruction_accuracy(self, shape: tuple[int, int]) -> None:
        A = np.random.default_rng(42).standard_normal(shape)
        Q, R = jlinalg.qr(A)
        rel_err = np.linalg.norm(A - Q @ R) / np.linalg.norm(A)
        assert rel_err < 1e-14, f"QR reconstruction error {rel_err:.2e} for {shape}"

    @pytest.mark.slow
    def test_reconstruction_accuracy_large(self) -> None:
        A = np.random.default_rng(42).standard_normal((5000, 200))
        Q, R = jlinalg.qr(A)
        rel_err = np.linalg.norm(A - Q @ R) / np.linalg.norm(A)
        assert rel_err < 1e-14, f"QR reconstruction error {rel_err:.2e} for (5000, 200)"

    @pytest.mark.parametrize(
        "shape",
        [(10, 5), (100, 20), (2, 1), (1, 1)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_orthogonality(self, shape: tuple[int, int]) -> None:
        A = np.random.default_rng(42).standard_normal(shape)
        Q, _ = jlinalg.qr(A)
        n = shape[1]
        orth_err = np.linalg.norm(Q.T @ Q - np.eye(n))
        assert orth_err < 1e-14, f"Orthogonality error {orth_err:.2e} for {shape}"

    @pytest.mark.slow
    def test_orthogonality_large(self) -> None:
        A = np.random.default_rng(42).standard_normal((5000, 200))
        Q, _ = jlinalg.qr(A)
        orth_err = np.linalg.norm(Q.T @ Q - np.eye(200))
        assert orth_err < 1e-14, f"Orthogonality error {orth_err:.2e} for (5000, 200)"

    def test_shapes(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        Q, R = jlinalg.qr(A)
        assert Q.shape == (100, 20)
        assert R.shape == (20, 20)

    def test_r_upper_triangular(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        _, R = jlinalg.qr(A)
        assert np.allclose(R, np.triu(R)), "R is not upper triangular"

    def test_square_matrix(self) -> None:
        A = np.random.default_rng(42).standard_normal((50, 50))
        Q, R = jlinalg.qr(A)
        rel_err = np.linalg.norm(A - Q @ R) / np.linalg.norm(A)
        assert rel_err < 1e-14
        orth_err = np.linalg.norm(Q.T @ Q - np.eye(50))
        assert orth_err < 1e-14
        assert Q.shape == (50, 50)
        assert R.shape == (50, 50)

    def test_ill_conditioned(self) -> None:
        rng = np.random.default_rng(42)
        m, n = 100, 20
        U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
        V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
        s = 10.0 ** (-np.arange(n) * 14.0 / n)
        A = U @ np.diag(s) @ V.T
        Q, R = jlinalg.qr(A)
        rel_err = np.linalg.norm(A - Q @ R) / np.linalg.norm(A)
        # Slightly relaxed for ill-conditioning.
        assert rel_err < 1e-12, f"Ill-conditioned QR reconstruction error {rel_err:.2e}"

    def test_2d_validation(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            jlinalg.qr(np.array([1.0, 2.0, 3.0]))

    def test_matches_numpy(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        Q_jl, R_jl = jlinalg.qr(A)
        Q_np, R_np = np.linalg.qr(A, mode="reduced")
        # Sign flips are allowed per column.
        assert np.allclose(np.abs(Q_jl), np.abs(Q_np), atol=1e-13), (
            "Q columns differ beyond sign flips"
        )
        assert np.allclose(np.abs(R_jl), np.abs(R_np), atol=1e-13), (
            "R differs beyond sign flips"
        )


# ---------------------------------------------------------------------------
# TestSVD
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not jlinalg.blas_has_dgesvd,
    reason="vendor LAPACK dgesvd not available (jlinalg-own / BLIS-ILP64 build)",
)
class TestSVD:
    """Tests for jlinalg.svd() — reduced SVD of tall-skinny matrices."""

    @pytest.mark.parametrize(
        "shape",
        [(10, 5), (100, 20), (2, 1)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_reconstruction_accuracy(self, shape: tuple[int, int]) -> None:
        A = np.random.default_rng(42).standard_normal(shape)
        U, s, Vh = jlinalg.svd(A)
        rel_err = np.linalg.norm(A - U @ np.diag(s) @ Vh) / np.linalg.norm(A)
        assert rel_err < 1e-14, f"SVD reconstruction error {rel_err:.2e} for {shape}"

    @pytest.mark.slow
    def test_reconstruction_accuracy_large(self) -> None:
        A = np.random.default_rng(42).standard_normal((5000, 200))
        U, s, Vh = jlinalg.svd(A)
        rel_err = np.linalg.norm(A - U @ np.diag(s) @ Vh) / np.linalg.norm(A)
        assert rel_err < 1e-14, (
            f"SVD reconstruction error {rel_err:.2e} for (5000, 200)"
        )

    @pytest.mark.parametrize(
        "shape",
        [(10, 5), (100, 20), (2, 1)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_singular_values_ordering(self, shape: tuple[int, int]) -> None:
        A = np.random.default_rng(42).standard_normal(shape)
        _, s, _ = jlinalg.svd(A)
        assert np.all(s >= 0), "Singular values contain negative entries"
        assert np.all(np.diff(s) <= 0), "Singular values not in descending order"

    def test_shapes(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        U, s, Vh = jlinalg.svd(A)
        assert U.shape == (100, 20)
        assert s.shape == (20,)
        assert Vh.shape == (20, 20)

    def test_compute_uv_false(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        s_only = jlinalg.svd(A, compute_uv=False)
        assert isinstance(s_only, np.ndarray)
        assert s_only.ndim == 1
        assert s_only.shape == (20,)
        # Values should match the full SVD.
        _, s_full, _ = jlinalg.svd(A)
        np.testing.assert_allclose(s_only, s_full, rtol=1e-14)

    def test_compute_uv_false_return_type(self) -> None:
        """Verify the behavioral contract of compute_uv=False.

        tracemalloc cannot observe C-level malloc savings (U/Vh allocation
        happens in the C extension), so we verify the behavioral contract
        instead of comparing peak memory.
        """
        A = np.random.default_rng(42).standard_normal((2000, 100))
        s = jlinalg.svd(A, compute_uv=False)
        assert isinstance(s, np.ndarray), (
            f"compute_uv=False should return ndarray, got {type(s)}"
        )
        assert s.ndim == 1, f"expected 1-D singular values, got ndim={s.ndim}"
        result = jlinalg.svd(A, compute_uv=True)
        assert isinstance(result, tuple), (
            f"compute_uv=True should return tuple, got {type(result)}"
        )
        assert len(result) == 3, f"expected 3-tuple, got len={len(result)}"

    def test_tall_skinny_only(self) -> None:
        with pytest.raises(ValueError, match="m >= n"):
            jlinalg.svd(np.random.default_rng(42).standard_normal((5, 10)))

    def test_2d_validation(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            jlinalg.svd(np.array([1.0, 2.0, 3.0]))

    def test_rank_deficient(self) -> None:
        rng = np.random.default_rng(42)
        m, n, r = 100, 20, 10
        U_full, _ = np.linalg.qr(rng.standard_normal((m, n)), mode="reduced")
        Vh_full, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
        s_true = np.zeros(n)
        s_true[:r] = np.sort(rng.uniform(1.0, 10.0, size=r))[::-1]
        A = U_full @ np.diag(s_true) @ Vh_full.T
        _, s, _ = jlinalg.svd(A)
        # Last n-r singular values should be near-zero.
        assert np.all(s[r:] < 1e-14), f"Rank-deficient tail not zero: {s[r:]}"

    def test_matches_numpy(self) -> None:
        A = np.random.default_rng(42).standard_normal((100, 20))
        U_jl, s_jl, Vh_jl = jlinalg.svd(A)
        U_np, s_np, Vh_np = np.linalg.svd(A, full_matrices=False)
        np.testing.assert_allclose(s_jl, s_np, rtol=1e-14)
        # U and Vh may differ by column sign.
        assert np.allclose(np.abs(U_jl), np.abs(U_np), atol=1e-13), (
            "U columns differ beyond sign flips"
        )
        assert np.allclose(np.abs(Vh_jl), np.abs(Vh_np), atol=1e-13), (
            "Vh rows differ beyond sign flips"
        )


# ---------------------------------------------------------------------------
# TestCapabilityFlags
# ---------------------------------------------------------------------------


class TestCapabilityFlags:
    """Tests for jlinalg LAPACK capability flags."""

    def test_flags_are_int(self) -> None:
        assert isinstance(jlinalg.blas_has_dgeqrf, int)
        assert isinstance(jlinalg.blas_has_dgesvd, int)

    def test_flags_consistent(self) -> None:
        if jlinalg.HAS_C_EXTENSION:
            assert jlinalg.blas_has_dgeqrf in (0, 1)
            assert jlinalg.blas_has_dgesvd in (0, 1)
        else:
            assert jlinalg.blas_has_dgeqrf == 0
            assert jlinalg.blas_has_dgesvd == 0
