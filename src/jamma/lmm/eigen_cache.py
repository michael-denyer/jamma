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
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import TypedDict

import numpy as np
from loguru import logger

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


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file's bytes, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_components(
    bed_path: Path,
    *,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    ksnps_indices: np.ndarray | None,
) -> EigenCacheComponents:
    """Assemble the canonical dict of cache-key components.

    Args:
        bed_path: PLINK prefix (without extension); .bed and .bim are derived
            by appending the appropriate suffix.
        maf_threshold: Minimum MAF used for SNP filtering.
        miss_threshold: Maximum missing rate used for SNP filtering.
        valid_mask: Boolean array of shape (n_samples_total,); True = included.
        ksnps_indices: Column indices for -ksnps restriction, or None.

    Returns:
        Dict ready for JSON serialisation as the key payload.
    """
    bed_file = Path(str(bed_path) + ".bed")
    bim_file = Path(str(bed_path) + ".bim")

    st = bed_file.stat()
    bed_fingerprint = f"{bed_file.name}:{st.st_size}:{st.st_mtime_ns}"
    bim_sha256 = _sha256_file(bim_file)

    mask_bytes = np.ascontiguousarray(valid_mask, dtype=bool).tobytes()
    valid_mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()

    if ksnps_indices is None:
        ksnps_val: str = "none"
    else:
        arr = np.sort(np.unique(np.asarray(ksnps_indices, dtype=np.int64)))
        ksnps_val = hashlib.sha256(arr.tobytes()).hexdigest()

    return {
        "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
        "bed_fingerprint": bed_fingerprint,
        "bim_sha256": bim_sha256,
        "maf_threshold": maf_threshold,
        "miss_threshold": miss_threshold,
        "valid_mask_sha256": valid_mask_sha256,
        "ksnps": ksnps_val,
    }


def compute_eigen_cache_key(
    bed_path: Path,
    *,
    maf_threshold: float,
    miss_threshold: float,
    valid_mask: np.ndarray,
    ksnps_indices: np.ndarray | None = None,
) -> tuple[str, EigenCacheComponents]:
    """Compute a SHA-256 cache key over all eigendecomposition determinants.

    Args:
        bed_path: PLINK prefix (without extension).
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
        bed_path,
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
    eigen_dir: Path,
    prefix: str,
    key: str,
    *,
    components: EigenCacheComponents,
    generation: str,
    artifacts: dict[str, dict[str, str]],
) -> Path:
    """Write a cache manifest JSON atomically.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix (e.g. "result").
        key: Hex SHA-256 cache key string.
        components: Key components (the payload that was hashed) for
            debuggability.

    Returns:
        Path to the written manifest file.
    """
    manifest: EigenCacheManifest = {
        "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "components": components,
        "generation": generation,
        "artifacts": artifacts,
    }
    if (
        not artifacts
        or loco_eigen_paths_from_manifest(eigen_dir, prefix, list(artifacts), manifest)
        is None
    ):
        raise ValueError("LOCO eigen manifest must name a complete existing generation")
    target = eigen_cache_manifest_path(eigen_dir, prefix)
    fd, tmp_name = tempfile.mkstemp(dir=eigen_dir, suffix=".json")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(target)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise
    return target


def loco_eigen_paths_from_manifest(
    eigen_dir: Path,
    prefix: str,
    chr_names: list[str],
    manifest: Mapping[str, object],
) -> dict[str, tuple[Path, Path]] | None:
    """Validate and resolve one LOCO generation without rereading its manifest."""
    generation = manifest.get("generation")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(generation, str)
        or not generation
        or not isinstance(artifacts, dict)
    ):
        return None
    generation_prefix = f"{prefix}.generation.{generation}.loco.chr"
    resolved: dict[str, tuple[Path, Path]] = {}
    for chromosome in chr_names:
        members = artifacts.get(chromosome)
        if not isinstance(members, dict):
            return None
        paths: list[Path] = []
        for kind in ("eigenD", "eigenU"):
            name = members.get(kind)
            expected = f"{generation_prefix}{chromosome}.{kind}."
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith(expected)
            ):
                return None
            path = eigen_dir / name
            if not path.is_file():
                return None
            paths.append(path)
        resolved[chromosome] = (paths[0], paths[1])
    return resolved


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
        with open(path) as fh:
            value = json.load(fh)
            if not isinstance(value, dict):
                logger.warning(
                    f"Malformed eigen cache manifest {path}: expected object"
                )
                return None
            return value
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning(f"Corrupt eigen cache manifest {path}: {exc}")
        return None
    except OSError as exc:
        logger.warning(f"Could not read eigen cache manifest {path}: {exc}")
        return None


def eigen_cache_is_valid(
    eigen_dir: Path, prefix: str, current_key: str
) -> tuple[bool, str]:
    """Check whether the on-disk eigen cache matches the current inputs.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix.
        current_key: Hex SHA-256 digest computed from the current inputs.

    Returns:
        Tuple of (is_valid, reason). reason is always a non-empty string.
    """
    manifest = read_eigen_cache_manifest(eigen_dir, prefix)
    path = eigen_cache_manifest_path(eigen_dir, prefix)
    if manifest is None:
        return False, f"no valid cache manifest found at {path}"
    return eigen_cache_manifest_is_valid(manifest, path, current_key)


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
