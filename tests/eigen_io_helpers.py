"""Shared Hypothesis strategies for eigendecomposition I/O tests."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st


@st.composite
def genotype_matrix(draw, min_samples=10, max_samples=100, min_snps=5, max_snps=50):
    """Generate genotype matrices with realistic allele frequencies."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_snps = draw(st.integers(min_value=min_snps, max_value=max_snps))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    mafs = rng.uniform(0.1, 0.5, n_snps)
    genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
    for column, maf in enumerate(mafs):
        probabilities = [(1 - maf) ** 2, 2 * maf * (1 - maf), maf**2]
        genotypes[:, column] = rng.choice(
            [0.0, 1.0, 2.0], size=n_samples, p=probabilities
        )
    return genotypes
