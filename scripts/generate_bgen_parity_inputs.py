#!/usr/bin/env python3
"""Build the BGEN parity inputs under tests/fixtures/bgen_parity/.

MANUAL EXECUTION ONLY. Reads GEMMA 0.98.5's own mouse_hs1940 example (the
``example/`` directory of the GEMMA source tree) and writes:

- ``mouse_bgen.bgen`` (+ ``.bgi``, ``.sample``): the first ``N_SAMPLES``
  samples and the placed, biallelic chromosome 1 and 2 variants, 8-bit,
  zlib. The example's genotypes are almost all hard calls, so a seeded
  ``PERTURBED`` fraction of variants blend each call with Dirichlet noise,
  and a ``WITH_MISSING`` fraction lose ``MISSING_RATE`` of their samples.
  ``N_SAMPLES`` keeps GEMMA's text kinship under the 500 KB added-file cap.
- ``mouse_bgen.geno.txt.gz``: BIMBAM mean genotypes holding the dosages
  JAMMA decodes from that BGEN, written in shortest round-trip form so GEMMA
  parses the same doubles. BIMBAM column 2 is the first BGEN allele, the one
  both tools count.
- ``mouse_bgen.anno.txt`` and ``mouse_bgen.pheno.txt``: the matching rows of
  the example's annotation and phenotype files.

GEMMA then runs on the BIMBAM files through generate_gemma_fixtures.sh, whose
cells come from the ``generation_cmd`` MANIFEST.toml records for each output.

Usage:
    uv run python scripts/generate_bgen_parity_inputs.py /path/to/GEMMA/example
"""

from __future__ import annotations

import gzip
import sqlite3
import sys
from pathlib import Path

import numpy as np
from bgen import BgenWriter

from jamma.genotype.dataset import GenotypeDataset

N_SAMPLES = 160
CHROMOSOMES = ("1", "2")
PERTURBED = 0.2
WITH_MISSING = 0.1
MISSING_RATE = 0.01
SEED = 20260924

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "bgen_parity"
PREFIX = "mouse_bgen"


def _read_example(example: Path):
    anno = [line.split() for line in (example / "mouse_hs1940.anno.txt").open()]
    with gzip.open(example / "mouse_hs1940.geno.txt.gz", "rt") as f:
        geno = [[x.strip() for x in line.split(",")] for line in f]
    # Keep placed variants with two distinct alleles; the example also lists
    # SNPs with an NA position or a repeated allele.
    keep = [
        i
        for i, (a, g) in enumerate(zip(anno, geno, strict=True))
        if a[2] in CHROMOSOMES and a[1] != "NA" and g[1] != g[2]
    ]
    anno = [anno[i] for i in keep]
    rows = [geno[i] for i in keep]
    rs = [r[0] for r in rows]
    alleles = [(r[1], r[2]) for r in rows]
    calls = np.array([[float(x) for x in r[3 : 3 + N_SAMPLES]] for r in rows])
    fam = [line.split() for line in (example / "mouse_hs1940.fam").open()]
    pheno = (example / "mouse_hs1940.pheno.txt").read_text().splitlines()
    return anno, rs, alleles, calls, fam[:N_SAMPLES], pheno[:N_SAMPLES]


def _probabilities(calls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``(m, n, 3)`` as (P11, P12, P22), P11 homozygous for the counted allele."""
    m, n = calls.shape
    hard = np.rint(calls).astype(int)
    one_hot = np.eye(3)[2 - hard]
    probs = one_hot.copy()
    for j in np.flatnonzero(rng.random(m) < PERTURBED):
        certainty = rng.uniform(0.3, 0.95)
        noise = rng.dirichlet(rng.uniform(0.3, 2.0, 3), size=n)
        probs[j] = certainty * one_hot[j] + (1 - certainty) * noise
    for j in np.flatnonzero(rng.random(m) < WITH_MISSING):
        probs[j, rng.random(n) < MISSING_RATE] = np.nan
    return probs


def _pin_index_metadata(bgi: Path) -> None:
    """Replace the writer's path, mtime and clock in the bgenix Metadata row.

    JAMMA reads no Metadata; pinning it keeps a local path out of the
    committed index and makes the file reproducible.
    """
    with sqlite3.connect(bgi) as db:
        db.execute(
            "UPDATE Metadata SET filename = ?, last_write_time = 0, "
            "index_creation_time = ''",
            (bgi.name.removesuffix(".bgi"),),
        )
    with sqlite3.connect(bgi) as db:
        db.execute("VACUUM")


def main(example: Path) -> int:
    anno, rs, alleles, calls, fam, pheno = _read_example(example)
    probs = _probabilities(calls, np.random.default_rng(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    bgen = OUT / f"{PREFIX}.bgen"
    iid = [row[1] for row in fam]
    with BgenWriter(bgen, N_SAMPLES, samples=iid, compression="zlib") as writer:
        for j, row in enumerate(anno):
            writer.add_variant(
                rs[j],
                rs[j],
                row[2],
                int(row[1]),
                list(alleles[j]),
                probs[j],
                bit_depth=8,
            )
    _pin_index_metadata(Path(f"{bgen}.bgi"))
    rows = "".join(f"{row[0]} {row[1]} 0\n" for row in fam)
    (OUT / f"{PREFIX}.sample").write_text("ID_1 ID_2 missing\n0 0 0\n" + rows)

    dataset = GenotypeDataset.open_bgen(
        bgen, OUT / f"{PREFIX}.sample", Path(f"{bgen}.bgi")
    )
    dosages = np.hstack(
        [block.dosages() for block in dataset.blocks(dataset.n_variants)]
    )
    with gzip.GzipFile(OUT / f"{PREFIX}.geno.txt.gz", "wb", mtime=0) as raw:
        for j in range(dataset.n_variants):
            values = ", ".join(
                "NA" if np.isnan(x) else repr(float(x)) for x in dosages[:, j]
            )
            raw.write(f"{rs[j]}, {alleles[j][0]}, {alleles[j][1]}, {values}\n".encode())
    (OUT / f"{PREFIX}.anno.txt").write_text(
        "".join("\t".join(row) + "\n" for row in anno)
    )
    (OUT / f"{PREFIX}.pheno.txt").write_text("\n".join(pheno) + "\n")
    fractional = np.nansum(dosages != np.rint(dosages), axis=0) > 0
    print(
        f"{dataset.n_variants} variants x {N_SAMPLES} samples; "
        f"{fractional.sum()} carry fractional dosages, "
        f"{np.isnan(dosages).any(axis=0).sum()} carry missing samples"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
