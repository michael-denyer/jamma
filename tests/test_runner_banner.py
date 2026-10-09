"""The association run reports its analysed counts once."""

import pytest
from loguru import logger

from jamma.lmm import run_lmm_association_numpy
from tests.builders import make_runner_synthetic_data

pytestmark = pytest.mark.tier0


def test_batch_run_logs_each_analysed_count_once():
    genotypes, phenotypes, kinship, snp_info = make_runner_synthetic_data()
    n_samples, n_snps = genotypes.shape

    messages: list[str] = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        result = run_lmm_association_numpy(genotypes, phenotypes, kinship, snp_info)
    finally:
        logger.remove(sink)

    assert result.n_tested == n_snps
    lines = [message.strip() for message in messages]
    assert lines.count(f"Analyzed individuals: {n_samples:,}") == 1
    assert lines.count(f"Analyzed SNPs: {n_snps:,}") == 1
