"""Content + parameter cache key for LOCO per-chromosome eigendecomposition.

A manifest file written alongside eigen files records a SHA-256 digest of
all inputs that determine the eigendecomposition (file identity, filter
thresholds, sample mask, SNP restriction).  On the next run the digest is
recomputed and compared; a mismatch forces a full recompute rather than
silently reusing stale eigen files. The genotype `.bed` is fingerprinted by
size + mtime while the `.bim` is fingerprinted by content hash, since a
re-annotated `.bim` can change the LOCO partition without changing `.bed`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np
from loguru import logger

from jamma.genotype.dataset import GenotypeDataset
from jamma.lmm.eigen_io import (
    EigenGeneration,
    Partition,
    load_manifest,
    member_record,
    publish_manifest,
)

# Version 3 adds one immutable generation and a complete chromosome member map.
# Version 2 separated analysed-sample filtering from full-population centering.
EIGEN_CACHE_SCHEMA_VERSION: int = 3


class EigenCacheComponents(TypedDict):
    """Canonical, JSON-serialisable payload hashed into the cache key.

    All values are plain JSON scalars: a TypedDict gives static shape
    checking without the runtime cost or rigidity of a dataclass. JSON
    decode yields a plain dict at runtime, so consumers that read this back
    off disk must still guard field access defensively.
    """

    schema_version: int
    bed_fingerprint: str
    bim_sha256: str
    maf_threshold: float
    miss_threshold: float
    valid_mask_sha256: str
    ksnps: str


class EigenCacheManifest(TypedDict):
    """On-disk manifest wrapping the cache key and its hashed components."""

    schema_version: int
    cache_key: str
    components: EigenCacheComponents
    generation: str
    artifacts: dict[str, dict[str, str]]


def _build_components(
    dataset: GenotypeDataset,
    *,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    ksnps_indices: np.ndarray | None,
) -> EigenCacheComponents:
    """Assemble the canonical dict of cache-key components.

    Args:
        dataset: The genotypes; ``dataset.fingerprint()`` supplies the file
            components (for PLINK, ``bed_fingerprint`` and ``bim_sha256``).
        maf_threshold: Minimum MAF used for SNP filtering.
        miss_threshold: Maximum missing rate used for SNP filtering.
        valid_mask: Boolean array of shape (n_samples_total,); True = included.
        ksnps_indices: Column indices for -ksnps restriction, or None.

    Returns:
        Dict ready for JSON serialisation as the key payload.
    """
    fingerprint = dataset.fingerprint()

    mask_bytes = np.ascontiguousarray(valid_mask, dtype=bool).tobytes()
    valid_mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()

    if ksnps_indices is None:
        ksnps_val: str = "none"
    else:
        arr = np.sort(np.unique(np.asarray(ksnps_indices, dtype=np.int64)))
        ksnps_val = hashlib.sha256(arr.tobytes()).hexdigest()

    return {
        "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
        "bed_fingerprint": fingerprint["bed_fingerprint"],
        "bim_sha256": fingerprint["bim_sha256"],
        "maf_threshold": maf_threshold,
        "miss_threshold": miss_threshold,
        "valid_mask_sha256": valid_mask_sha256,
        "ksnps": ksnps_val,
    }


def compute_eigen_cache_key(
    dataset: GenotypeDataset,
    *,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    ksnps_indices: np.ndarray | None = None,
) -> tuple[str, EigenCacheComponents]:
    """Compute a SHA-256 cache key over all eigendecomposition determinants.

    Args:
        dataset: The genotypes whose file identity the key covers.
        maf_threshold: Minimum MAF used for SNP filtering.
        miss_threshold: Maximum missing rate used for SNP filtering.
        valid_mask: Boolean array of shape (n_samples_total,); True = included.
            Its length encodes total sample count, so different sample sets
            (different length OR different True/False positions) yield distinct keys.
        ksnps_indices: Column indices for -ksnps restriction, or None.
            The SNP set is the determinant, so indices are sorted + de-duped
            before hashing.

    Returns:
        Tuple of (key, components). key is the hex SHA-256 digest. components is
        the exact canonical payload that was hashed, returned so the caller can
        persist it in the manifest and diff it against a future mismatch.
    """
    components = _build_components(
        dataset,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        valid_mask=valid_mask,
        ksnps_indices=ksnps_indices,
    )
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return key, components


def eigen_cache_manifest_path(eigen_dir: Path, prefix: str) -> Path:
    """Return the manifest path for a given eigen directory and prefix.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix (e.g. "result").

    Returns:
        Path to the manifest JSON file.
    """
    return eigen_dir / f"{prefix}.loco.cache_manifest.json"


def write_eigen_cache_manifest(
    generation: EigenGeneration,
    key: str,
    *,
    components: EigenCacheComponents,
    members: Mapping[str, tuple[Path, Path]],
) -> Path:
    """Commit a complete LOCO generation under its cache key.

    Args:
        generation: The generation whose members were all written.
        key: Hex SHA-256 cache key string.
        components: Key components (the payload that was hashed) for
            debuggability.
        members: Chromosome -> ``(eigenD, eigenU)`` written by
            ``generation.write_member``.

    Returns:
        Path to the written manifest file.

    Raises:
        ValueError: If ``members`` is empty or names a missing file.
    """
    if not members or not all(
        path.is_file() for pair in members.values() for path in pair
    ):
        raise ValueError("LOCO eigen manifest must name a complete existing generation")
    manifest: EigenCacheManifest = {
        "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "components": components,
        "generation": generation.generation,
        "artifacts": {
            chromosome: member_record(*pair) for chromosome, pair in members.items()
        },
    }
    target = eigen_cache_manifest_path(generation.directory, generation.prefix)
    publish_manifest(target, json.dumps(manifest))
    return target


def resolve_eigen_cache(
    manifest: Mapping[str, object], eigen_dir: Path, prefix: str, chr_names: list[str]
) -> dict[str, tuple[Path, Path]] | None:
    """Each chromosome's committed members, or None if any is unsafe or missing."""
    source = eigen_cache_manifest_path(eigen_dir, prefix)
    artifacts = manifest.get("artifacts")
    records: dict[Partition, object] = {
        chromosome: artifacts.get(chromosome) if isinstance(artifacts, dict) else None
        for chromosome in chr_names
    }
    try:
        generation = EigenGeneration.committed(manifest, eigen_dir, prefix, source)
        resolved = generation.resolve(records, source)
    except ValueError:
        return None
    if not all(path.is_file() for pair in resolved.values() for path in pair):
        return None
    return {chromosome: resolved[chromosome] for chromosome in chr_names}


def read_eigen_cache_manifest(eigen_dir: Path, prefix: str) -> dict[str, object] | None:
    """Read and parse the cache manifest.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix.

    Returns:
        Parsed manifest dict, or None if absent, corrupt, or unreadable.
    """
    path = eigen_cache_manifest_path(eigen_dir, prefix)
    try:
        return load_manifest(path)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        logger.warning(f"Corrupt eigen cache manifest {path}: {exc}")
        return None
    except OSError as exc:
        logger.warning(f"Could not read eigen cache manifest {path}: {exc}")
        return None


def eigen_cache_manifest_is_valid(
    manifest: Mapping[str, object], path: Path, current_key: str
) -> tuple[bool, str]:
    """Validate an already-read manifest so transaction readers read it once."""
    got_version = manifest.get("schema_version")
    if got_version != EIGEN_CACHE_SCHEMA_VERSION:
        return (
            False,
            f"manifest schema_version {got_version} != current "
            f"{EIGEN_CACHE_SCHEMA_VERSION}; recomputing",
        )
    if "cache_key" not in manifest:
        return (
            False,
            f"malformed eigen cache manifest at {path}: no cache_key "
            f"(old-schema or truncated manifest)",
        )
    if manifest["cache_key"] == current_key:
        return True, "cache key matches"
    return False, "cache key mismatch: inputs changed since the eigen cache was written"
