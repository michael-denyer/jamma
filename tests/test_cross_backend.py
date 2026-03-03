"""Cross-backend comparison: JAX vs NumPy runners on same data.

Validates that both backends produce numerically equivalent results
within documented tolerance bounds. Only runs when JAX is installed.
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.runner_numpy import run_lmm_association_numpy
from tests.conftest import load_phenotypes_from_fam

pytestmark = pytest.mark.requires_jax

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_synthetic"


def _build_snp_info(plink_data) -> list[dict]:
    """Build snp_info list from plink data."""
    return [
        {
            "chr": str(plink_data.chromosome[i]),
            "rs": plink_data.sid[i],
            "pos": plink_data.bp_position[i],
            "a1": plink_data.allele_1[i],
            "a0": plink_data.allele_2[i],
            "maf": 0.0,
            "n_miss": 0,
        }
        for i in range(plink_data.n_snps)
    ]


@pytest.fixture(scope="module")
def cross_backend_data():
    """Load gemma_synthetic data for both runners."""
    plink_data = load_plink_binary(_FIXTURE_ROOT / "test")
    kinship = read_kinship_matrix(_FIXTURE_ROOT / "gemma_kinship.cXX.txt")
    phenotypes = load_phenotypes_from_fam((_FIXTURE_ROOT / "test").with_suffix(".fam"))
    snp_info = _build_snp_info(plink_data)

    return {
        "genotypes": plink_data.genotypes,
        "phenotypes": phenotypes,
        "kinship": kinship,
        "snp_info": snp_info,
    }


@pytest.mark.tier1
class TestCrossBackend:
    """Compare JAX and NumPy runner outputs.

    Both backends implement identical algorithms (same Pab, REML, golden
    section optimization). Numerical differences arise only from XLA
    compilation vs NumPy's BLAS, and from float64 accumulation order.
    These tolerances are tight because the code paths are near-identical.
    """

    # JAX uses generic golden section; NumPy n_cvt=1 uses split-Uab optimizer.
    # Different FP accumulation order causes divergence on weak-signal SNPs,
    # particularly for LRT/MLE statistics on flat optimization landscapes.
    TOLERANCES = {
        "beta": 1e-10,
        "se": 1e-10,
        "p_wald": 1e-10,
        "p_lrt": 5e-3,  # split-Uab vs generic optimizer divergence on weak signals
        "p_score": 1e-8,
        "logl_H1": 5e-3,  # same optimizer divergence affects per-SNP MLE logl
        "l_remle": 1e-10,
    }

    def _run_both(self, data, lmm_mode):
        from jamma.lmm.runner_jax import run_lmm_association_jax

        jax_results = run_lmm_association_jax(
            genotypes=data["genotypes"],
            phenotypes=data["phenotypes"],
            kinship=data["kinship"].copy(),
            snp_info=data["snp_info"],
            lmm_mode=lmm_mode,
            show_progress=False,
            check_memory=False,
        )

        numpy_results = run_lmm_association_numpy(
            genotypes=data["genotypes"],
            phenotypes=data["phenotypes"],
            kinship=data["kinship"].copy(),
            snp_info=data["snp_info"],
            lmm_mode=lmm_mode,
            show_progress=False,
            check_memory=False,
        )

        return jax_results, numpy_results

    def _compare(self, jax_results, numpy_results, fields):
        """Compare JAX and NumPy results field by field."""
        assert len(jax_results) == len(numpy_results), (
            f"Result count mismatch: JAX={len(jax_results)}, NumPy={len(numpy_results)}"
        )
        for field in fields:
            # Collect values, converting None to NaN for uniform handling
            jax_vals = np.array(
                [
                    v if (v := getattr(r, field)) is not None else float("nan")
                    for r in jax_results
                ],
                dtype=np.float64,
            )
            np_vals = np.array(
                [
                    v if (v := getattr(r, field)) is not None else float("nan")
                    for r in numpy_results
                ],
                dtype=np.float64,
            )

            # Skip NaN/None positions (degenerate SNPs or mode-specific fields)
            valid = ~np.isnan(jax_vals) & ~np.isnan(np_vals)
            if not np.any(valid):
                continue

            rtol = self.TOLERANCES[field]
            np.testing.assert_allclose(
                np_vals[valid],
                jax_vals[valid],
                rtol=rtol,
                atol=1e-14,
                err_msg=f"Cross-backend mismatch on {field}",
            )

    def test_wald_cross_backend(self, cross_backend_data):
        """Wald test: JAX vs NumPy produce equivalent results."""
        jax_r, np_r = self._run_both(cross_backend_data, 1)
        self._compare(jax_r, np_r, ["beta", "se", "p_wald", "l_remle"])

    def test_lrt_cross_backend(self, cross_backend_data):
        """LRT test: JAX vs NumPy produce equivalent results."""
        jax_r, np_r = self._run_both(cross_backend_data, 2)
        self._compare(jax_r, np_r, ["p_lrt", "logl_H1", "l_remle"])

    def test_score_cross_backend(self, cross_backend_data):
        """Score test: JAX vs NumPy produce equivalent results."""
        jax_r, np_r = self._run_both(cross_backend_data, 3)
        self._compare(jax_r, np_r, ["p_score", "l_remle"])

    def test_all_cross_backend(self, cross_backend_data):
        """Mode 4 (All): JAX vs NumPy produce equivalent Wald+LRT+Score results."""
        jax_r, np_r = self._run_both(cross_backend_data, 4)
        self._compare(
            jax_r,
            np_r,
            ["beta", "se", "p_wald", "p_lrt", "p_score", "logl_H1", "l_remle"],
        )
