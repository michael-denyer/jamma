"""Write BGEN v1.2 fixtures at test time with the ``bgen`` package's writer.

``bgen.BgenWriter`` (a dev-only dependency) writes the ``.bgen`` and its
bgenix-schema ``.bgi`` together; the Oxford ``.sample`` is written here by
hand. The one committed BGEN, ``tests/fixtures/bgen_parity``, is the GEMMA
parity fixture that ``scripts/generate_bgen_parity_inputs.py`` builds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bed_reader import open_bed
from bgen import BgenWriter


@dataclass(frozen=True)
class BgenFiles:
    bgen: Path
    sample: Path
    bgi: Path


def write_sample(path: Path, fid: Sequence[str], iid: Sequence[str]) -> Path:
    """Write an Oxford ``.sample`` with ID_1 = ``fid`` and ID_2 = ``iid``."""
    rows = "".join(f"{f} {i} 0\n" for f, i in zip(fid, iid, strict=True))
    path.write_text("ID_1 ID_2 missing\n0 0 0\n" + rows)
    return path


def random_probabilities(
    rng: np.random.Generator, n_variants: int, n_samples: int, missing_rate: float
) -> np.ndarray:
    """``(n_variants, n_samples, 3)`` genotype probabilities, NaN rows missing."""
    alphas = rng.uniform(0.2, 3.0, size=(n_variants, 3))
    probs = np.stack([rng.dirichlet(a, size=n_samples) for a in alphas])
    probs[rng.random((n_variants, n_samples)) < missing_rate] = np.nan
    return probs


def write_bgen(
    path: Path,
    probabilities: np.ndarray,
    *,
    bit_depth: int = 8,
    compression: str | None = "zlib",
    alleles: Sequence[Sequence[str]] | None = None,
    chromosomes: Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    rsids: Sequence[str] | None = None,
    varids: Sequence[str] | None = None,
    fid: Sequence[str] | None = None,
    iid: Sequence[str] | None = None,
    embed_ids: bool = True,
) -> BgenFiles:
    """Write ``probabilities`` ``(m, n, 3)`` as ``path`` + ``.bgi`` + ``.sample``.

    Samples default to ``f0.. / s0..``; ``iid`` is also embedded in the
    ``.bgen`` when ``embed_ids``.
    """
    m, n, _ = probabilities.shape
    iid = list(iid) if iid is not None else [f"s{i}" for i in range(n)]
    fid = list(fid) if fid is not None else [f"f{i}" for i in range(n)]
    with BgenWriter(
        path, n, samples=iid if embed_ids else None, compression=compression
    ) as writer:
        for j in range(m):
            writer.add_variant(
                varids[j] if varids is not None else f"v{j}",
                rsids[j] if rsids is not None else f"rs{j}",
                chromosomes[j] if chromosomes is not None else "1",
                positions[j] if positions is not None else j + 1,
                list(alleles[j]) if alleles is not None else ["A", "G"],
                probabilities[j],
                bit_depth=bit_depth,
            )
    sample = write_sample(path.with_suffix(".sample"), fid, iid)
    return BgenFiles(bgen=path, sample=sample, bgi=Path(f"{path}.bgi"))


def one_hot_bgen_from_plink(
    bfile: Path, path: Path, *, bit_depth: int = 8
) -> BgenFiles:
    """Encode a PLINK fileset as BGEN with probabilities exactly 0 or 1.

    The ``.bim`` allele 1, which bed-reader counts, is written as the first
    BGEN allele, so both files give the same counted-allele dosage.
    """
    with open_bed(Path(f"{bfile}.bed")) as bed:
        genotypes = bed.read(dtype=np.float64)
        meta = (
            np.asarray(bed.chromosome).astype(str),
            bed.bp_position,
            bed.sid,
            bed.allele_1,
            bed.allele_2,
            bed.fid,
            bed.iid,
        )
    chrom, pos, sid, allele_1, allele_2, fid, iid = meta
    probs = np.zeros((genotypes.shape[1], genotypes.shape[0], 3))
    g = genotypes.T
    probs[g == 2, 0] = 1.0  # homozygous first allele
    probs[g == 1, 1] = 1.0
    probs[g == 0, 2] = 1.0
    probs[np.isnan(g)] = np.nan
    return write_bgen(
        path,
        probs,
        bit_depth=bit_depth,
        alleles=list(zip(allele_1, allele_2, strict=True)),
        chromosomes=list(chrom),
        positions=[int(p) for p in pos],
        rsids=list(sid),
        fid=list(fid),
        iid=list(iid),
    )
