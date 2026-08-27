"""compute_kinship validates its flag combinations before touching disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.pipeline import PipelineConfig
from jamma.pipeline_kinship import compute_kinship

pytestmark = pytest.mark.tier0

# A bfile that does not exist: every case below must fail on the guard, not on
# the missing .bed, which is what proves the guard runs first.
MISSING = Path("/nonexistent/p15/study")


def test_mode_outside_1_2_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid kinship mode 3"):
        compute_kinship(PipelineConfig(bfile=MISSING), 3)  # type: ignore[arg-type]


def test_loco_with_write_eigen_is_rejected() -> None:
    config = PipelineConfig(bfile=MISSING, loco=True, write_eigen=True)
    with pytest.raises(ValueError, match="-eigen not supported with -gk -loco"):
        compute_kinship(config, 1)


def test_loco_with_standardized_mode_is_rejected() -> None:
    config = PipelineConfig(bfile=MISSING, loco=True)
    with pytest.raises(ValueError, match=r"-gk 2 .* not supported with -loco"):
        compute_kinship(config, 2)


def test_missing_bed_is_reported_once_the_guards_pass() -> None:
    with pytest.raises(FileNotFoundError, match=r"\.bed file not found"):
        compute_kinship(PipelineConfig(bfile=MISSING), 1)
