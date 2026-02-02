"""Tests for LMM memory estimation functions.

Uses mock psutil.virtual_memory() to test memory checks deterministically
without requiring specific machine memory sizes.
"""

from unittest.mock import patch

import pytest

from jamma.core.memory import LmmMemoryBreakdown, estimate_lmm_memory


class TestEstimateLmmMemory:
    """Tests for estimate_lmm_memory() standalone helper."""

    def test_returns_lmm_memory_breakdown(self):
        """Should return LmmMemoryBreakdown with all fields."""
        result = estimate_lmm_memory(1000, 5000)

        assert isinstance(result, LmmMemoryBreakdown)
        assert result.genotypes_gb > 0
        assert result.eigenvectors_gb > 0
        assert result.chunk_gb > 0
        assert result.total_peak_gb > 0
        assert result.available_gb >= 0
        assert isinstance(result.sufficient, bool)

    def test_has_kinship_false_includes_kinship_computation(self):
        """When has_kinship=False, should include kinship matrix memory."""
        result = estimate_lmm_memory(1000, 5000, has_kinship=False)

        # n^2 * 8 bytes = 1000^2 * 8 = 8MB = 0.008GB
        assert result.kinship_gb == pytest.approx(0.008, rel=0.01)

    def test_has_kinship_true_still_includes_kinship_memory(self):
        """When has_kinship=True, kinship_gb reflects memory to hold loaded matrix."""
        result = estimate_lmm_memory(1000, 5000, has_kinship=True)

        # Still need memory to hold loaded kinship: n^2 * 8 bytes = 0.008GB
        assert result.kinship_gb == pytest.approx(0.008, rel=0.01)

    def test_genotype_memory_scales_with_snps(self):
        """Genotype memory should scale linearly with n_snps."""
        result_small = estimate_lmm_memory(1000, 5000)
        result_large = estimate_lmm_memory(1000, 10000)

        # Double SNPs should ~double genotype memory
        assert result_large.genotypes_gb == pytest.approx(
            result_small.genotypes_gb * 2, rel=0.01
        )

    def test_eigendecomp_workspace_included(self):
        """Should include DSYEVR workspace in estimates."""
        result = estimate_lmm_memory(1000, 5000)

        # workspace = 26*n*8 + 10*n*4 bytes
        expected_workspace = (26 * 1000 * 8 + 10 * 1000 * 4) / 1e9
        assert result.eigendecomp_workspace_gb == pytest.approx(
            expected_workspace, rel=0.01
        )

    def test_chunk_size_affects_chunk_memory(self):
        """Chunk memory should scale with chunk_size parameter."""
        result_small = estimate_lmm_memory(1000, 10000, chunk_size=1000)
        result_large = estimate_lmm_memory(1000, 10000, chunk_size=5000)

        assert result_large.chunk_gb > result_small.chunk_gb

    @patch("jamma.core.memory.psutil.virtual_memory")
    def test_sufficient_true_when_memory_available(self, mock_vm):
        """Should mark sufficient=True when enough memory available."""
        mock_vm.return_value.available = 100 * 1e9  # 100GB available

        result = estimate_lmm_memory(1000, 5000)  # Needs ~0.02GB

        assert result.sufficient is True

    @patch("jamma.core.memory.psutil.virtual_memory")
    def test_sufficient_false_when_memory_insufficient(self, mock_vm):
        """Should mark sufficient=False when not enough memory."""
        mock_vm.return_value.available = 0.001 * 1e9  # 1MB available

        result = estimate_lmm_memory(1000, 5000)  # Needs much more

        assert result.sufficient is False

    def test_safety_margin_applied(self):
        """Should use 10% safety margin in sufficient calculation."""
        # First get total_peak_gb without mocking
        result_first = estimate_lmm_memory(1000, 5000)
        peak_gb = result_first.total_peak_gb

        # Set available memory to exactly 1.05x of required (less than 1.1x margin)
        with patch("jamma.core.memory.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.available = peak_gb * 1.05 * 1e9
            result = estimate_lmm_memory(1000, 5000)

        # 1.05x is less than 1.1x safety margin, so should be insufficient
        assert result.sufficient is False

    def test_200k_samples_estimate(self):
        """Smoke test: 200k samples should estimate ~640GB peak (eigendecomp)."""
        result = estimate_lmm_memory(200_000, 95_000)

        # Peak should be eigendecomp: ~320GB kinship + ~320GB eigenvectors + workspace
        # = ~640GB
        assert result.total_peak_gb == pytest.approx(640, rel=0.05)
