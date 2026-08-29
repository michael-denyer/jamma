"""_lmm_accel C extension tests: Pab table construction and kernel performance.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest


@pytest.mark.benchmark
class TestCExtensionPerformance:
    """Benchmark C extension vs Python on realistic data.

    Hardware-sensitive — `2x` speedup is not a correctness invariant. Runs
    only under `--benchmark-only`; on machines with <4 physical cores it
    skips. Numerical parity between C and Python paths IS a correctness
    invariant and is checked unconditionally.
    """

    def test_c_faster_than_python(self, benchmark):
        """Benchmark C-accelerated Wald; verify numerical parity vs Python."""
        from jamma.core.threading import get_physical_core_count
        from jamma.lmm import compute_numpy
        from jamma.lmm.compute_numpy import _compute_wald_numpy
        from jamma.lmm.likelihood_numpy import golden_section_optimize_lambda_numpy
        from jamma.lmm.stats import batch_calc_wald_stats_from_pab_numpy
        from jamma.lmm.uab import batch_compute_iab_numpy

        if compute_numpy._accel is None:
            pytest.skip("C extension not compiled")

        n_threads = get_physical_core_count()
        if n_threads < 4:
            pytest.skip(f"Benchmark needs >=4 physical cores; found {n_threads}")

        rng = np.random.default_rng(42)
        n_samples, n_snps = 500, 2000
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))

        # Uab columns are cross-products of (w, x, y), so each SNP's row is a
        # Gram matrix. Drawing the six columns independently instead breaks
        # Cauchy-Schwarz on about half the SNPs, driving P_xx/P_yy negative;
        # the two paths then disagree about which SNPs are degenerate and
        # return different NaN sets. Same construction as the shared fixtures.
        w = np.abs(rng.standard_normal((n_snps, n_samples))) + 1.0
        x = np.abs(rng.standard_normal((n_snps, n_samples))) + 0.5
        y = rng.standard_normal((n_snps, n_samples))
        Uab_batch = np.stack([w * w, w * x, w * y, x * x, x * y, y * y], axis=2)
        Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

        wald_kwargs = {"l_min": 1e-5, "l_max": 1e5, "n_grid": 50, "n_refine": 20}

        # Warmup: amortise OpenMP thread-pool startup before timing
        _compute_wald_numpy(
            1,
            eigenvalues,
            Uab_batch[:50],
            n_samples,
            **wald_kwargs,
            Iab_batch=Iab_batch[:50],
            n_threads=n_threads,
        )

        # pytest-benchmark times the C path; speedup vs the Python path is
        # tracked over time as benchmark history rather than asserted.
        result_c = benchmark(
            _compute_wald_numpy,
            1,
            eigenvalues,
            Uab_batch,
            n_samples,
            **wald_kwargs,
            Iab_batch=Iab_batch,
            n_threads=n_threads,
        )

        # Numerical parity is the actual correctness invariant. Build the
        # reference from the generic optimizer rather than by disabling
        # compute_numpy._accel is not None: that flag routes n_cvt=1 to the split-Uab
        # optimizer, a different algorithm from the C batch path, so the
        # comparison would not be like-for-like. Same approach and same
        # calibrated tolerances as test_c_vs_python_parity_synthetic, where
        # atol carries lambdas that converge to the l_min boundary from
        # opposite sides.
        lambdas_py, logls_py, pab_py = golden_section_optimize_lambda_numpy(
            1,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min=1e-5,
            l_max=1e5,
            n_grid=50,
            n_iter=20,
        )
        betas_py, ses_py, pwalds_py = batch_calc_wald_stats_from_pab_numpy(
            1, pab_py, n_samples
        )
        expected = {
            "lambdas": lambdas_py,
            "logls": logls_py,
            "betas": betas_py,
            "ses": ses_py,
            "pwalds": pwalds_py,
        }
        for field, want in expected.items():
            np.testing.assert_allclose(
                result_c[field],
                want,
                rtol=1e-6,
                atol=1e-4,
                equal_nan=True,
                err_msg=f"{field}: C vs Python mismatch",
            )


@pytest.mark.tier0
class TestBuildPabTableForC:
    """Verify build_pab_table_for_c produces correct flat arrays for C extension."""

    def test_ncvt1_basic_structure(self):
        """n_cvt=1: returns dict with all expected keys and correct scalar values."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(1)

        assert t.n_cvt == 1
        assert t.n_index == 6  # (1+3)*(1+2)//2 = 6
        assert t.n_rows == 3  # n_cvt + 2
        # idx_yy, idx_xx, idx_xy from build_index_table
        from jamma.lmm.likelihood import build_index_table

        ref = build_index_table(1)
        assert t.idx_yy == ref.idx_yy
        assert t.idx_xx == ref.idx_xx
        assert t.idx_xy == ref.idx_xy

    def test_ncvt2_dimensions(self):
        """n_cvt=2: n_index=10, n_rows=4, correct inv/var counts."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(2)

        assert t.n_index == 10
        assert t.n_rows == 4  # n_cvt + 2
        assert t.n_inv == 6
        assert t.n_var == 4
        assert t.n_inv + t.n_var == t.n_index

    def test_ncvt4_dimensions(self):
        """n_cvt=4: n_index=21, n_rows=6, correct inv/var counts."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(4)

        assert t.n_index == 21
        assert t.n_rows == 6  # n_cvt + 2
        assert t.n_inv == 15
        assert t.n_var == 6
        assert t.n_inv + t.n_var == t.n_index

    def test_invariant_varying_partition(self):
        """invariant + varying indices partition range(n_index) for n_cvt=1,2,4."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            inv = set(t.invariant_indices.tolist())
            var = set(t.varying_indices.tolist())
            assert inv & var == set(), f"n_cvt={n_cvt}: overlap in inv/var"
            assert inv | var == set(range(t.n_index)), (
                f"n_cvt={n_cvt}: inv+var doesn't cover range(n_index)"
            )

    def test_all_arrays_are_int32(self):
        """All index arrays must be int32 for C extension compatibility."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t = build_pab_table_for_c(2)
        array_keys = [
            "invariant_indices",
            "varying_indices",
            "logdet_diag_rows",
            "logdet_diag_cols",
            "level_offsets",
            "level_counts",
            "entries",
        ]
        for key in array_keys:
            assert getattr(t, key).dtype == np.int32, (
                f"{key} has dtype {getattr(t, key).dtype}, expected int32"
            )

    def test_level_offsets_index_entries(self):
        """level_offsets and level_counts correctly index into flat entries array."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            offsets = t.level_offsets
            counts = t.level_counts
            entries = t.entries

            # n_cvt+2 levels (0..n_cvt+1)
            assert len(offsets) == n_cvt + 2
            assert len(counts) == n_cvt + 2

            # Level 0 has no entries (row 0 comes from dot products)
            assert counts[0] == 0

            # Total entries must equal entries array length / 4 (stride-4)
            total_entries = sum(counts)
            assert len(entries) == total_entries * 4, (
                f"n_cvt={n_cvt}: entries length {len(entries)} != {total_entries * 4}"
            )

            # Each level's offset must be consistent
            running_offset = 0
            for level in range(n_cvt + 2):
                assert offsets[level] == running_offset, (
                    f"n_cvt={n_cvt}, level={level}: "
                    f"offset {offsets[level]} != {running_offset}"
                )
                running_offset += counts[level]

    def test_entries_match_pab_recursion(self):
        """Flat entries array matches build_index_table pab_recursion content."""
        from jamma.lmm.likelihood import build_index_table, build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            ref = build_index_table(n_cvt)
            entries = t.entries
            offsets = t.level_offsets
            counts = t.level_counts

            for level in range(1, n_cvt + 2):
                ref_entries = ref.pab_recursion[level]
                start = offsets[level] * 4
                count = counts[level]
                assert count == len(ref_entries), (
                    f"n_cvt={n_cvt}, level={level}: count mismatch"
                )
                for j, (_, _, idx_ab, idx_aw, idx_bw, idx_ww) in enumerate(ref_entries):
                    base = start + j * 4
                    assert entries[base] == idx_ab
                    assert entries[base + 1] == idx_aw
                    assert entries[base + 2] == idx_bw
                    assert entries[base + 3] == idx_ww

    def test_logdet_diag_matches_build_index_table(self):
        """logdet_diag_rows/cols match build_index_table logdet_diag_indices."""
        from jamma.lmm.likelihood import build_index_table, build_pab_table_for_c

        for n_cvt in (1, 2, 4):
            t = build_pab_table_for_c(n_cvt)
            ref = build_index_table(n_cvt)

            rows = t.logdet_diag_rows.tolist()
            cols = t.logdet_diag_cols.tolist()
            ref_pairs = ref.logdet_diag_indices

            assert len(rows) == len(ref_pairs)
            for i, (ref_row, ref_col) in enumerate(ref_pairs):
                assert rows[i] == ref_row, f"n_cvt={n_cvt}, i={i}: row mismatch"
                assert cols[i] == ref_col, f"n_cvt={n_cvt}, i={i}: col mismatch"

    def test_lru_cached(self):
        """Same n_cvt returns same object (lru_cache)."""
        from jamma.lmm.likelihood import build_pab_table_for_c

        t1 = build_pab_table_for_c(2)
        t2 = build_pab_table_for_c(2)
        assert t1 is t2
