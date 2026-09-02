"""Bit-exactness digest lever for the association run path.

``fingerprint.yml`` pins the C kernels and ``kinship_digest.py`` pins
kinship and eigendecomposition; nothing pinned the path from the pipeline's
phenotype loop down to the chunk engine. This script does, self-baselined the
same way: there is no committed expected digest, because the values depend on
the BLAS backend and the CPU. ``--out`` records ``key -> sha256`` for the
current checkout, and ``--diff`` compares a base file against a head file
built on the same machine.

Key axes:

- ``pipeline/<backend>/lmm<mode>/<covar>/<snps>``: a full ``PipelineRunner``
  run over ``gemma_synthetic`` with the fixture kinship, for both backends,
  all four ``-lmm`` modes, with and without covariates, and with and without
  a ``-snps`` list (every third SNP). The digest is the ``.assoc.txt`` bytes,
  ``n_snps_tested`` and the PVE pair, so it pins the product the user reads
  and the layer at which the SNP restriction is applied.
- ``pipeline/<backend>/multi``: the same with two phenotype columns, which
  is the branch that preloads genotypes once for the batch runner.
- ``pipeline/loco/<snps>``: the LOCO pipeline over ``gemma_loco``.
- ``api/batch/lmm<mode>/<covar>/<chunk>``: ``run_lmm_association_numpy`` on
  in-memory genotypes, with results kept in memory so the digest covers the
  raw float64 bytes of every statistic rather than the ``.6e`` text.
- ``api/streaming/lmm<mode>/<snps>/<chunk>``: the same for
  ``run_lmm_association_numpy_streaming``.

Usage::

    uv run python scripts/assoc_digest.py --out /tmp/a.json
    uv run python scripts/assoc_digest.py --diff base.json head.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamma import jlinalg  # noqa: E402
from jamma.io import load_plink_binary  # noqa: E402
from jamma.io.plink import get_plink_metadata, read_fam_phenotypes  # noqa: E402
from jamma.kinship.io import read_kinship_matrix  # noqa: E402
from jamma.lmm import (  # noqa: E402
    LmmConfig,
    run_lmm_association_numpy,
    run_lmm_association_numpy_streaming,
)
from jamma.lmm.schema import SnpMeta  # noqa: E402
from jamma.lmm.stats import AssocResult  # noqa: E402
from jamma.pipeline import PipelineRunner  # noqa: E402
from jamma.pipeline_config import PipelineConfig  # noqa: E402
from tests.fixture_paths import LOCO, SYNTHETIC  # noqa: E402

BACKENDS = ("numpy", "numpy-streaming")
MODES = (1, 2, 3, 4)


def _feed(h: Any, value: Any) -> None:
    if value is None:
        h.update(b"N|")
    elif isinstance(value, bool):
        h.update(b"b|" + repr(value).encode())
    elif isinstance(value, int):
        h.update(b"i|" + repr(value).encode())
    elif isinstance(value, float):
        h.update(b"f|" + struct.pack("<d", value))
    elif isinstance(value, str):
        h.update(b"s|" + value.encode() + b"|")
    elif isinstance(value, bytes):
        h.update(b"B|" + value)
    else:
        raise TypeError(f"cannot digest {type(value).__name__}")


def digest_values(*values: Any) -> str:
    h = hashlib.sha256()
    for value in values:
        _feed(h, value)
    return h.hexdigest()


def digest_results(results: list[AssocResult], n_tested: int, pve, pve_se) -> str:
    h = hashlib.sha256()
    _feed(h, n_tested)
    _feed(h, pve)
    _feed(h, pve_se)
    for r in results:
        for field in (
            r.chr,
            r.rs,
            r.ps,
            r.n_miss,
            r.allele1,
            r.allele0,
            r.af,
            r.beta,
            r.se,
            r.logl_H1,
            r.l_remle,
            r.p_wald,
            r.p_score,
            r.l_mle,
            r.p_lrt,
        ):
            _feed(h, None if field is None else field)
    return h.hexdigest()


def _every_third_snp_file(bfile: Path, dest: Path) -> tuple[Path, np.ndarray]:
    meta = get_plink_metadata(bfile)
    indices = np.arange(0, meta.n_snps, 3, dtype=np.intp)
    path = dest / "snps.txt"
    path.write_text("".join(f"{meta.sid[i]}\n" for i in indices))
    return path, indices


def _two_phenotype_copy(bfile: Path, dest: Path) -> Path:
    for ext in (".bed", ".bim"):
        shutil.copy(bfile.with_suffix(ext), dest / f"copy{ext}")
    rows = bfile.with_suffix(".fam").read_text().splitlines()
    rng = np.random.default_rng(20260902)
    out = []
    for row in rows:
        cols = row.split()
        second = float(cols[5]) * 0.5 + float(rng.normal())
        out.append("\t".join([*cols[:6], f"{second:.6f}"]))
    (dest / "copy.fam").write_text("\n".join(out) + "\n")
    return dest / "copy"


def _run_pipeline(config: PipelineConfig) -> str:
    result = PipelineRunner(config).run()
    payload: list[Any] = [result.n_snps_tested, result.pve_estimate, result.pve_se]
    for path in result.assoc_paths:
        payload.append(path.read_bytes())
    return digest_values(*payload)


def _pipeline_keys(work: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    snps_file, _ = _every_third_snp_file(SYNTHETIC.bfile, work)
    for backend in BACKENDS:
        for mode in MODES:
            for covar_label, covar in (
                ("nocovar", None),
                ("covar", SYNTHETIC.covariates),
            ):
                for snps_label, snps in (("allsnps", None), ("snps3", snps_file)):
                    out = work / f"{backend}-{mode}-{covar_label}-{snps_label}"
                    digests[
                        f"pipeline/{backend}/lmm{mode}/{covar_label}/{snps_label}"
                    ] = _run_pipeline(
                        PipelineConfig(
                            bfile=SYNTHETIC.bfile,
                            kinship_file=SYNTHETIC.kinship,
                            covariate_file=covar,
                            lmm_mode=mode,
                            output_dir=out,
                            snps_file=snps,
                            backend=backend,
                            check_memory=False,
                            show_progress=False,
                            no_telemetry=True,
                        )
                    )

    multi_bfile = _two_phenotype_copy(SYNTHETIC.bfile, work)
    for backend in BACKENDS:
        digests[f"pipeline/{backend}/multi"] = _run_pipeline(
            PipelineConfig(
                bfile=multi_bfile,
                kinship_file=SYNTHETIC.kinship,
                lmm_mode=1,
                output_dir=work / f"{backend}-multi",
                snps_file=snps_file,
                backend=backend,
                phenotype_columns=(1, 2),
                check_memory=False,
                show_progress=False,
                no_telemetry=True,
            )
        )

    loco_snps, _ = _every_third_snp_file(LOCO.bfile, work / "loco")
    for snps_label, snps in (("allsnps", None), ("snps3", loco_snps)):
        digests[f"pipeline/loco/{snps_label}"] = _run_pipeline(
            PipelineConfig(
                bfile=LOCO.bfile,
                lmm_mode=1,
                output_dir=work / f"loco-{snps_label}",
                snps_file=snps,
                loco=True,
                backend="numpy",
                check_memory=False,
                show_progress=False,
                no_telemetry=True,
            )
        )
    return digests


def _api_keys(work: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    data = load_plink_binary(SYNTHETIC.bfile)
    kinship = read_kinship_matrix(SYNTHETIC.kinship, data.n_samples)
    phenotypes = read_fam_phenotypes(SYNTHETIC.bfile.with_suffix(".fam"))
    covariates = np.loadtxt(SYNTHETIC.covariates, dtype=np.float64)
    snp_meta = SnpMeta.from_plink_meta(data.meta)
    _, snps_indices = _every_third_snp_file(SYNTHETIC.bfile, work / "api")

    for mode in MODES:
        config = LmmConfig(lmm_mode=mode, check_memory=False, show_progress=False)
        for covar_label, covar in (("nocovar", None), ("covar", covariates)):
            for chunk_label, max_chunk in (("onechunk", None), ("chunk64", 64)):
                run = run_lmm_association_numpy(
                    data.genotypes,
                    phenotypes,
                    kinship.copy(),
                    snp_meta,
                    covariates=covar,
                    config=config,
                    max_chunk_size=max_chunk,
                )
                digests[f"api/batch/lmm{mode}/{covar_label}/{chunk_label}"] = (
                    digest_results(run.associations, run.n_tested, run.pve, run.pve_se)
                )
        for snps_label, snps in (("allsnps", None), ("snps3", snps_indices)):
            for chunk_label, chunk in (("default", None), ("chunk64", 64)):
                run = run_lmm_association_numpy_streaming(
                    SYNTHETIC.bfile,
                    phenotypes,
                    kinship.copy(),
                    chunk_size=chunk,
                    snps_indices=snps,
                    config=config,
                )
                digests[f"api/streaming/lmm{mode}/{snps_label}/{chunk_label}"] = (
                    digest_results(run.associations, run.n_tested, run.pve, run.pve_se)
                )
    return digests


def compute_all_digests() -> dict[str, str]:
    logger.remove()
    with tempfile.TemporaryDirectory(prefix="assoc-digest-") as tmp:
        work = Path(tmp)
        (work / "loco").mkdir()
        (work / "api").mkdir()
        digests = _pipeline_keys(work)
        digests.update(_api_keys(work))
    return digests


def _header() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = None
    return {
        "blas_backend": jlinalg.blas_backend,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sha": sha,
    }


def cmd_out(path: Path) -> int:
    payload = {"header": _header(), "digests": compute_all_digests()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"assoc_digest: {len(payload['digests'])} keys -> {path}")
    return 0


def _load(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.loads(path.read_text())
    return payload["header"], payload["digests"]


def cmd_diff(path_a: Path, path_b: Path) -> int:
    header_a, digests_a = _load(path_a)
    header_b, digests_b = _load(path_b)

    for field in ("blas_backend", "platform"):
        if header_a.get(field) != header_b.get(field):
            print(
                f"ERROR: {field} differs between runs "
                f"({header_a.get(field)!r} vs {header_b.get(field)!r}); "
                "a digest comparison across backends or platforms is meaningless.",
                file=sys.stderr,
            )
            return 2

    keys_a, keys_b = set(digests_a), set(digests_b)
    shared = keys_a & keys_b
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    differing = sorted(k for k in shared if digests_a[k] != digests_b[k])

    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        if keys:
            print(f"{len(keys)} key(s) {label} (coverage change, not compared):")
            for key in keys:
                print(f"  {key}")

    if differing:
        print(f"{len(differing)} keys differ:", file=sys.stderr)
        for key in differing:
            print(f"  {key}  A={digests_a[key]}  B={digests_b[key]}", file=sys.stderr)
        return 1

    print(f"0 keys differ ({len(shared)} shared, {len(shared)} identical)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, metavar="FILE", help="write digests to FILE")
    group.add_argument(
        "--diff",
        nargs=2,
        type=Path,
        metavar=("A", "B"),
        help="compare two digest files",
    )
    args = parser.parse_args(argv)

    if args.out is not None:
        return cmd_out(args.out)
    return cmd_diff(*args.diff)


if __name__ == "__main__":
    sys.exit(main())
