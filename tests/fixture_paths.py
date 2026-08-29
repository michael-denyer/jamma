"""The committed fixture datasets, named once.

Every path under ``tests/fixtures`` that a test reads is a field here.
Twenty-two files had each derived their own ``fixtures`` root and their
own spelling of the mouse and synthetic paths; the manifest gate checks
the bytes, and this module is where the names live.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jamma.validation import ToleranceConfig

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class FixtureDataset:
    """One PLINK dataset and the GEMMA association outputs recorded for it.

    Attributes:
        bfile: PLINK prefix; ``.bed``, ``.bim`` and ``.fam`` sit beside it.
        assoc: GEMMA ``.assoc.txt`` outputs keyed by the run that made them
            (``wald``, ``lrt``, ``score``, ``all``, ``covar_*``, ``chrN``).
    """

    bfile: Path
    assoc: Mapping[str, Path]

    @property
    def dir(self) -> Path:
        return self.bfile.parent

    @property
    def bed(self) -> Path:
        return self.bfile.with_suffix(".bed")

    @property
    def bim(self) -> Path:
        return self.bfile.with_suffix(".bim")

    @property
    def fam(self) -> Path:
        return self.bfile.with_suffix(".fam")

    def ref(self, run: str) -> Path:
        """The recorded GEMMA output for ``run``; KeyError names the dataset."""
        if run not in self.assoc:
            raise KeyError(f"{self.dir.name} has no recorded {run!r} output")
        return self.assoc[run]


@dataclass(frozen=True)
class KinshipDataset(FixtureDataset):
    """A dataset GEMMA also ran ``-gk 1`` and ``-c`` on.

    Attributes:
        kinship: GEMMA centred kinship, ``.cXX.txt``.
        covariates: GEMMA ``-c`` covariate file.
    """

    kinship: Path
    covariates: Path


_SYN = FIXTURES / "gemma_synthetic"
_MOUSE = FIXTURES / "mouse_hs1940"
_COV = FIXTURES / "gemma_covariate"
_ALL = FIXTURES / "gemma_all_tests"

SYNTHETIC = KinshipDataset(
    bfile=_SYN / "test",
    kinship=_SYN / "gemma_kinship.cXX.txt",
    covariates=_COV / "covariates.txt",
    assoc={
        "wald": _SYN / "gemma_assoc.assoc.txt",
        "lrt": _SYN / "gemma_lrt.assoc.txt",
        "score": FIXTURES / "gemma_score" / "gemma_score.assoc.txt",
        "all": _ALL / "gemma_all.assoc.txt",
        "covar_wald": _COV / "gemma_covariate.assoc.txt",
        "covar_lrt": _COV / "gemma_covariate_lrt.assoc.txt",
        "covar_score": _COV / "gemma_covariate_score.assoc.txt",
        "covar_all": _ALL / "gemma_all_covar.assoc.txt",
    },
)

MOUSE = KinshipDataset(
    bfile=_MOUSE / "mouse_hs1940",
    kinship=_MOUSE / "mouse_hs1940_kinship.cXX.txt",
    covariates=_MOUSE / "covariates.txt",
    assoc={
        "all": _MOUSE / "mouse_hs1940_all.assoc.txt",
        "lrt": _MOUSE / "mouse_hs1940_lrt.assoc.txt",
        "score": _MOUSE / "mouse_hs1940_score.assoc.txt",
        "covar_wald": _MOUSE / "mouse_hs1940_covar_wald.assoc.txt",
        "covar_lrt": _MOUSE / "mouse_hs1940_covar_lrt.assoc.txt",
        "covar_score": _MOUSE / "mouse_hs1940_covar_score.assoc.txt",
        "covar_all": _MOUSE / "mouse_hs1940_covar_all.assoc.txt",
    },
)
MOUSE_COVARIATES_4 = _MOUSE / "covariates_4.txt"

LOCO = FixtureDataset(
    bfile=FIXTURES / "gemma_loco" / "test",
    assoc={
        f"chr{c}": FIXTURES / "gemma_loco" / f"gemma_loco_chr{c}.assoc.txt"
        for c in (1, 2, 3)
    },
)
LOCO_SNPS = FIXTURES / "gemma_loco" / "test_snps.txt"

KINSHIP_DIR = FIXTURES / "kinship"

# NumPy backend versus GEMMA on mouse_hs1940. Cephes betainc is close to GSL
# betainc for large a (n_samples > 1000); lambda optimisation is golden
# section against GEMMA's Brent. See docs/GEMMA_EQUIVALENCE.md.
NUMPY_GEMMA_TOLERANCES = ToleranceConfig(
    lambda_rtol=1e-3,
    pvalue_rtol=1e-2,
    se_rtol=5e-4,
    logl_rtol=5e-3,
    atol=1e-4,
)
