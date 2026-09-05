"""_lmm_accel C extension tests: Pab table construction and kernel performance.

Split from the original single test_lmm_accel module. Shared fixtures
live in tests/lmm_accel_helpers.py.
"""

import numpy as np
import pytest


@pytest.mark.benchmark
class TestCExtensionPerformance:
    """Benchmark the C extension on realistic valid data.

    Hardware-sensitive — `2x` speedup is not a correctness invariant. Runs
    only under `--benchmark-only`; on machines with <4 physical cores it
    skips. Numerical correctness lives in ordinary tier-0 tests.
    """

    def test_native_wald_latency(self, benchmark):
        """Time the actual native fused Wald kernel on valid shared inputs."""
        from jamma.core.threading import get_physical_core_count
        from jamma.lmm import accel

        if not accel.available():
            pytest.skip("C extension not compiled")

        n_threads = get_physical_core_count()
        if n_threads < 4:
            pytest.skip(f"Benchmark needs >=4 physical cores; found {n_threads}")

        rng = np.random.default_rng(42)
        n_samples, n_snps = 500, 2000
        eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))

        w = rng.standard_normal(n_samples)
        y = rng.standard_normal(n_samples)
        utg_t = np.ascontiguousarray(rng.standard_normal((n_snps, n_samples)))
        invariant = np.stack((w * w, w * y, y * y))
        workspace = accel.require().create_workspace_ncvt1_c(
            eigenvalues, invariant, w, y, n_samples, 1e-5, 1e5, 50, 20, lmm_mode=1
        )

        # Warmup: amortise OpenMP thread-pool startup before timing
        accel.require().compute_lmm_chunk_ncvt1_c(workspace, utg_t[:50], n_threads)

        # pytest-benchmark tracks native latency history; it asserts no speed ratio.
        benchmark(
            accel.require().compute_lmm_chunk_ncvt1_c,
            workspace,
            utg_t,
            n_threads,
        )


@pytest.mark.tier0
class TestBuildPabTableForC:
    """Verify build_pab_table_for_c produces correct flat arrays for C extension."""

    def test_ncvt1_basic_structure(self):
        """n_cvt=1: returns dict with all expected keys and correct scalar values."""
        from jamma.lmm.pab import build_pab_table_for_c

        t = build_pab_table_for_c(1)

        assert t.n_cvt == 1
        assert t.n_index == 6  # (1+3)*(1+2)//2 = 6
        assert t.n_rows == 3  # n_cvt + 2
        # idx_yy, idx_xx, idx_xy from build_index_table
        from jamma.lmm.pab import build_index_table

        ref = build_index_table(1)
        assert t.idx_yy == ref.idx_yy
        assert t.idx_xx == ref.idx_xx
        assert t.idx_xy == ref.idx_xy

    def test_ncvt2_dimensions(self):
        """n_cvt=2: n_index=10, n_rows=4, correct inv/var counts."""
        from jamma.lmm.pab import build_pab_table_for_c

        t = build_pab_table_for_c(2)

        assert t.n_index == 10
        assert t.n_rows == 4  # n_cvt + 2
        assert t.n_inv == 6
        assert t.n_var == 4
        assert t.n_inv + t.n_var == t.n_index

    def test_ncvt4_dimensions(self):
        """n_cvt=4: n_index=21, n_rows=6, correct inv/var counts."""
        from jamma.lmm.pab import build_pab_table_for_c

        t = build_pab_table_for_c(4)

        assert t.n_index == 21
        assert t.n_rows == 6  # n_cvt + 2
        assert t.n_inv == 15
        assert t.n_var == 6
        assert t.n_inv + t.n_var == t.n_index

    def test_invariant_varying_partition(self):
        """invariant + varying indices partition range(n_index) for n_cvt=1,2,4."""
        from jamma.lmm.pab import build_pab_table_for_c

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
        from jamma.lmm.pab import build_pab_table_for_c

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
        from jamma.lmm.pab import build_pab_table_for_c

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
        from jamma.lmm.pab import build_index_table, build_pab_table_for_c

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
        from jamma.lmm.pab import build_index_table, build_pab_table_for_c

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
        from jamma.lmm.pab import build_pab_table_for_c

        t1 = build_pab_table_for_c(2)
        t2 = build_pab_table_for_c(2)
        assert t1 is t2
