"""Content + parameter cache key for LOCO per-chromosome eigendecomposition.

A manifest file written alongside eigen files records a SHA-256 digest of
all inputs that determine the eigendecomposition (file identity, filter
thresholds, sample mask, SNP restriction).  On the next run the digest is
recomputed and compared; a mismatch forces a full recompute rather than
silently reusing stale eigen files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
from loguru import logger

EIGEN_CACHE_SCHEMA_VERSION: int = 1


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
) -> dict:
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
) -> tuple[str, dict]:
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


def invalidate_eigen_cache_manifest(eigen_dir: Path, prefix: str) -> None:
    """Remove the cache manifest if present; no-op if absent.

    Called before a write_eigen rewrite so a stale manifest cannot validate a
    half-rewritten eigen cache: the fresh manifest is written only after all
    per-chromosome eigen files succeed, so an interrupted rewrite leaves no
    manifest and the next read recomputes.
    """
    with suppress(FileNotFoundError):
        eigen_cache_manifest_path(eigen_dir, prefix).unlink()


def write_eigen_cache_manifest(
    eigen_dir: Path,
    prefix: str,
    key: str,
    *,
    components: dict,
) -> Path:
    """Write a cache manifest JSON atomically.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix (e.g. "result").
        key: Hex SHA-256 cache key string.
        components: Dict of key components (the payload that was hashed) for
            debuggability.

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "components": components,
    }
    target = eigen_cache_manifest_path(eigen_dir, prefix)
    fd, tmp_name = tempfile.mkstemp(dir=eigen_dir, suffix=".json")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(manifest, fh)
        tmp_path.replace(target)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise
    return target


def read_eigen_cache_manifest(eigen_dir: Path, prefix: str) -> dict | None:
    """Read and parse the cache manifest.

    Args:
        eigen_dir: Directory containing eigen files.
        prefix: Filename prefix.

    Returns:
        Parsed manifest dict, or None if absent or corrupt.
    """
    path = eigen_cache_manifest_path(eigen_dir, prefix)
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Corrupt eigen cache manifest {path}: {exc}")
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
    if manifest is None:
        path = eigen_cache_manifest_path(eigen_dir, prefix)
        return False, f"no valid cache manifest found at {path}"
    if manifest.get("cache_key") == current_key:
        return True, "cache key matches"
    return False, "cache key mismatch: inputs changed since the eigen cache was written"
