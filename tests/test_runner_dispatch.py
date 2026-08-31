"""Tests for the small-sample warning emitted by the pipeline.

The run_lmm() dispatcher these tests were originally written around was
removed in 7.0.0; mode selection (plan_association) is covered by
test_backend_detection.py and test_pipeline.py.
"""

import pytest

from jamma.pipeline import (
    SMALL_SAMPLE_WARNING_THRESHOLD,
    warn_if_small_sample,
)

pytestmark = pytest.mark.tier0


class TestSmallSampleWarning:
    """Tests for the n<50 small-sample warning.

    See docs/GEMMA_DIVERGENCES.md §6 for rationale: golden section lambda
    optimization assumes unimodality, which can break at very small n, and
    LMM-based GWAS has insufficient power below ~50 samples regardless.
    """

    def test_threshold_is_fifty(self):
        """Threshold constant is 50 — document via test."""
        assert SMALL_SAMPLE_WARNING_THRESHOLD == 50

    def test_below_threshold_warns(self, caplog):
        """n=30 emits a loguru warning mentioning the threshold."""
        from loguru import logger

        # loguru isn't captured by caplog by default — bridge to stdlib logging
        handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
        try:
            with caplog.at_level("WARNING"):
                warn_if_small_sample(30)
        finally:
            logger.remove(handler_id)

        assert any("Small sample size" in rec.message for rec in caplog.records)
        assert any("30" in rec.message for rec in caplog.records)

    def test_at_threshold_does_not_warn(self, caplog):
        """n=50 is the threshold boundary — no warning (strict <)."""
        from loguru import logger

        handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
        try:
            with caplog.at_level("WARNING"):
                warn_if_small_sample(50)
        finally:
            logger.remove(handler_id)

        assert not any("Small sample size" in rec.message for rec in caplog.records)

    def test_large_n_does_not_warn(self, caplog):
        """Typical GWAS n=1940 produces no small-sample warning."""
        from loguru import logger

        handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
        try:
            with caplog.at_level("WARNING"):
                warn_if_small_sample(1940)
        finally:
            logger.remove(handler_id)

        assert not any("Small sample size" in rec.message for rec in caplog.records)
