"""Bit-exactness digest lever for kinship and eigendecomposition.

``fingerprint.yml`` does not path-match ``src/jamma/kinship/compute.py``, so
this script is the only bit-exactness gate a kinship change gets. It
self-baselines the same way ``scripts/run-fingerprint.sh`` does for the C
accelerator: there is no committed expected-digest file, because digests
depend on the BLAS backend and the CPU. ``--out`` records ``key -> sha256``
for the current checkout, and ``--diff`` compares two such files, one from
the base and one from the head, built on the same machine.

Key axes (the committed coverage matrix):

- fixture: ``mouse_hs1940``, ``gemma_synthetic``, ``gemma_loco``
- gk mode: ``gk1`` (centered, GEMMA -gk 1), ``gk2`` (standardized, GEMMA -gk 2)
- path: ``inmemory``, ``streaming``, and for LOCO ``loco-single``/``loco-multi``
  (``loco-multi`` forces ``_max_batch_chrs=1`` so both LOCO passes get covered
  without depending on how much RAM the runner machine has)
- sample set: ``all`` (every sample) or ``valid`` (a deterministic subset via
  ``valid_indices``, dropping the last 10% of samples)
- filter: ``unfiltered`` or ``maf0.05``

LOCO keys carry one extra segment, the chromosome name, since
``compute_loco_kinship_streaming`` yields one matrix per chromosome rather than
one matrix for the whole run.

The digest is ``sha256(shape + arr.tobytes())``, so a reshape cannot collide
with a same-byte-count different-shape array.

Usage::

    uv run python scripts/kinship_digest.py --out /tmp/k.json
    uv run python scripts/kinship_digest.py --diff base.json head.json

``--diff`` refuses to compare two files whose header disagrees on BLAS
backend or platform, since that comparison would not mean anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamma import jlinalg  # noqa: E402
from jamma.io import load_plink_binary  # noqa: E402
from jamma.kinship import (  # noqa: E402
    compute_kinship_streaming,
    compute_loco_kinship_streaming,
    compute_standardized_kinship_streaming,
)
from tests.fixture_paths import LOCO, MOUSE, SYNTHETIC  # noqa: E402
from tests.reference.kinship import (  # noqa: E402
    compute_centered_kinship,
    compute_standardized_kinship,
)

FIXTURES = {
    "mouse_hs1940": MOUSE.bfile,
    "gemma_synthetic": SYNTHETIC.bfile,
    "gemma_loco": LOCO.bfile,
}
LOCO_FIXTURES = {"gemma_loco"}
FILTERS = {"unfiltered": 0.0, "maf0.05": 0.05}


def digest_array(arr: np.ndarray) -> str:
    """Shape-prefixed so a reshape cannot collide with a same-byte-count array."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _valid_indices(n_samples: int) -> np.ndarray:
    n_valid = max(1, n_samples - max(1, n_samples // 10))
    return np.arange(n_valid)


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


def _kinship_keys(fixture: str, bfile: Path) -> dict[str, str]:
    data = load_plink_binary(bfile)
    n_samples = data.n_samples
    digests: dict[str, str] = {}

    for sample_label, valid_indices in (
        ("all", None),
        ("valid", _valid_indices(n_samples)),
    ):
        genotypes = (
            data.genotypes
            if valid_indices is None
            else data.genotypes[valid_indices, :]
        )
        for filter_label, maf in FILTERS.items():
            key_prefix = f"{fixture}/{sample_label}/{filter_label}"

            k_gk1_mem = compute_centered_kinship(
                genotypes.copy(), maf_threshold=maf, check_memory=False
            )
            digests[f"{key_prefix}/gk1/inmemory"] = digest_array(k_gk1_mem)

            k_gk2_mem = compute_standardized_kinship(
                genotypes.copy(), maf_threshold=maf, check_memory=False
            )
            digests[f"{key_prefix}/gk2/inmemory"] = digest_array(k_gk2_mem)

            k_gk1_stream = compute_kinship_streaming(
                bfile,
                maf_threshold=maf,
                check_memory=False,
                show_progress=False,
                valid_indices=valid_indices,
            )
            digests[f"{key_prefix}/gk1/streaming"] = digest_array(k_gk1_stream)

            k_gk2_stream = compute_standardized_kinship_streaming(
                bfile,
                maf_threshold=maf,
                check_memory=False,
                show_progress=False,
                valid_indices=valid_indices,
            )
            digests[f"{key_prefix}/gk2/streaming"] = digest_array(k_gk2_stream)

    return digests


def _loco_keys(fixture: str, bfile: Path) -> dict[str, str]:
    data = load_plink_binary(bfile)
    n_samples = data.n_samples
    digests: dict[str, str] = {}

    for sample_label, valid_indices in (
        ("all", None),
        ("valid", _valid_indices(n_samples)),
    ):
        for filter_label, maf in FILTERS.items():
            key_prefix = f"{fixture}/{sample_label}/{filter_label}"

            for path_label, max_batch_chrs in (
                ("loco-single", None),
                ("loco-multi", 1),
            ):
                stream = compute_loco_kinship_streaming(
                    bfile,
                    maf_threshold=maf,
                    check_memory=False,
                    show_progress=False,
                    valid_indices=valid_indices,
                    _max_batch_chrs=max_batch_chrs,
                )
                for chr_name, k_loco in stream:
                    digests[f"{key_prefix}/{path_label}/chr{chr_name}"] = digest_array(
                        k_loco
                    )

    return digests


def _eigen_keys(fixture: str, bfile: Path) -> dict[str, str]:
    data = load_plink_binary(bfile)
    k = compute_centered_kinship(data.genotypes.copy(), check_memory=False)
    eigenvalues, eigenvectors, _ = jlinalg.eigh(k)
    return {
        f"{fixture}/eigen/eigenvalues": digest_array(eigenvalues),
        f"{fixture}/eigen/eigenvector0": digest_array(eigenvectors[:, 0]),
    }


def compute_all_digests() -> dict[str, str]:
    digests: dict[str, str] = {}
    for fixture, bfile in FIXTURES.items():
        digests.update(_kinship_keys(fixture, bfile))
        if fixture in LOCO_FIXTURES:
            digests.update(_loco_keys(fixture, bfile))
        digests.update(_eigen_keys(fixture, bfile))
    return digests


def cmd_out(path: Path) -> int:
    payload = {"header": _header(), "digests": compute_all_digests()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"kinship_digest: {len(payload['digests'])} keys -> {path}")
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
