"""Eigendecomposition file I/O — binary .npy default, GEMMA text legacy option.

Binary .npy is the default write format for performance at scale.
At 50k samples, .npy reads take ~3s vs ~4 min for text parsing.

GEMMA text format (.eigenD.txt / .eigenU.txt) remains available via
legacy_text=True for interoperability with external tools.

Read behaviour:
- .npy suffix: loads read-only through NumPy's memory-mapped path.
- .txt suffix: checks for .npy sidecar cache (memory-mapped, demand-paged via
  mmap_mode='r') when available. Falls back to text parsing.

Managed JAMMA writes use immutable generation filenames plus one atomic manifest.
Explicit external GEMMA pairs remain supported without a manifest.

Format follows GEMMA param.cpp WriteVector/WriteMatrix:
- eigenD: one value per line, 10 significant digits (.10g format)
- eigenU: tab-separated rows, 10 significant digits per value
- No headers in either file
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_reader import read_matrix_parallel
from jamma.io.matrix_writer import write_matrix_parallel
from jamma.utils.atomic_publish import AtomicOutput
from jamma.utils.npy_cache import (
    read_array_artifact,
    save_npy_atomic,
    write_npy_cache,
)

EIGEN_MANIFEST_SCHEMA_VERSION = 1

_GENERATION_MARK = ".generation."

# ---------------------------------------------------------------------------
# .npy sidecar cache helpers (used for text-format files only)
# ---------------------------------------------------------------------------


def _npy_cache_path(txt_path: Path) -> Path:
    """Derive .npy sidecar path from a text file path.

    .eigenU.txt → .eigenU.npy, .eigenD.txt → .eigenD.npy
    """
    return txt_path.with_suffix(".npy")


def _checked_shape(data: np.ndarray, *, what: str, ndim: int, path: Path) -> np.ndarray:
    """Promote to ``ndim`` and reject anything that is not that shape.

    Eigenvalues must be a vector; eigenvectors a square matrix. ``np.loadtxt``
    returns a 0-D scalar for a one-line file, which ``atleast_*d`` repairs.
    """
    data = np.atleast_1d(data) if ndim == 1 else np.atleast_2d(data)
    ok = data.ndim == ndim and (ndim == 1 or data.shape[0] == data.shape[1])
    if not ok:
        expected = "a vector" if ndim == 1 else "a square matrix"
        raise ValueError(
            f"{what.capitalize()} file has wrong shape {data.shape}, "
            f"expected {expected}: {path}"
        )
    return data


def _read_array(
    path: Path,
    *,
    what: str,
    ndim: int,
    parse_text: Callable[[Path], np.ndarray],
) -> np.ndarray:
    """Read one eigen array from .npy, its .npy sidecar, or GEMMA text.

    Every binary load is a read-only memory map, direct .npy path and sidecar
    alike; see ``read_array_artifact``.
    """
    return read_array_artifact(
        path,
        what=what,
        parse_text=parse_text,
        check=lambda data, p: _checked_shape(data, what=what, ndim=ndim, path=p),
        mmap_mode="r",
    )


def _read_eigenvalues(path: Path) -> np.ndarray:
    """1-D float64 eigenvalues from .eigenD.npy or GEMMA .eigenD.txt.

    A read-only ``np.memmap`` whenever the bytes come from a .npy file,
    whether that is the given path or the sidecar beside a text path.
    Callers must not mutate it in place.

    Raises:
        ValueError: If the file is empty, non-numeric, or not a vector.
    """
    return _read_array(
        path,
        what="eigenvalue",
        ndim=1,
        parse_text=lambda p: np.loadtxt(p, dtype=np.float64),
    )


def _read_eigenvectors(path: Path) -> np.ndarray:
    """2-D float64 eigenvectors from .eigenU.npy or GEMMA .eigenU.txt.

    Text parsing goes through ``read_matrix_parallel``. A read-only
    ``np.memmap`` whenever the bytes come from a .npy file, whether that is
    the given path or the sidecar beside a text path. Callers must not mutate
    it in place.

    Raises:
        ValueError: If the file is empty, non-numeric, or not a square matrix.
    """
    return _read_array(
        path,
        what="eigenvector",
        ndim=2,
        parse_text=lambda p: read_matrix_parallel(p, delimiter=None),
    )


def read_eigen_files(
    eigenD_path: Path,
    eigenU_path: Path,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read both eigenvalue and eigenvector files with cross-validation.

    Handles both .npy paths (binary, default) and .txt paths (legacy text).
    Validates that eigenvalue count matches eigenvector dimensions.
    Optionally validates against expected sample count.

    Args:
        eigenD_path: Path to eigenvalue file (.eigenD.npy or .eigenD.txt).
        eigenU_path: Path to eigenvector file (.eigenU.npy or .eigenU.txt).
        n_samples: Expected number of samples (optional validation).

    Returns:
        Tuple of (eigenvalues, eigenvectors).

    Raises:
        ValueError: If dimensions are inconsistent or do not match
            n_samples.
    """
    eigenD_path = Path(eigenD_path)
    eigenU_path = Path(eigenU_path)
    d_generation = _member_generation(eigenD_path)
    u_generation = _member_generation(eigenU_path)
    if (d_generation is None) != (u_generation is None) or (
        d_generation is not None and d_generation != u_generation
    ):
        raise ValueError(
            "Eigen files name different managed generations; use both paths "
            "returned by one write_eigen_files call"
        )
    stable = _stable_pair_identity(eigenD_path, eigenU_path)
    if stable is not None:
        directory, prefix = stable
        manifest = eigen_manifest_path(directory, prefix)
        if manifest.is_file():
            eigenD_path, eigenU_path = resolve_eigen_generation(directory, prefix)

    eigenvalues = _read_eigenvalues(eigenD_path)
    eigenvectors = _read_eigenvectors(eigenU_path)

    n_eval = eigenvalues.shape[0]
    n_rows = eigenvectors.shape[0]

    if n_eval != n_rows:
        raise ValueError(
            f"Eigenvalue count ({n_eval}) does not match eigenvector "
            f"dimensions ({n_rows} x {n_rows}). Files may be mismatched: "
            f"{eigenD_path}, {eigenU_path}"
        )

    if n_samples is not None and n_eval != n_samples:
        raise ValueError(
            f"Eigen files have {n_eval} samples but pipeline expects "
            f"{n_samples} after phenotype/covariate filtering. "
            f"Re-run with -eigen to regenerate eigen files matching "
            f"the current filtering."
        )

    n_negative = int(np.sum(eigenvalues < 0))
    if n_negative > 0:
        raise ValueError(
            f"Eigenvalue file contains {n_negative} negative value(s). "
            f"Kinship eigenvalues must be non-negative. "
            f"File: {eigenD_path}"
        )

    return eigenvalues, eigenvectors


def _member_generation(path: Path) -> str | None:
    """Return the full managed generation identity embedded in a member name."""
    if _GENERATION_MARK not in path.name:
        return None
    for kind in (".eigenD.", ".eigenU."):
        if kind in path.name:
            identity, _separator, _suffix = path.name.partition(kind)
            return identity
    return None


def _stable_pair_identity(
    eigenD_path: Path, eigenU_path: Path
) -> tuple[Path, str] | None:
    """Return the managed stable pair identity, or None for explicit inputs."""
    if eigenD_path.parent != eigenU_path.parent:
        return None
    for suffix in (".npy", ".txt"):
        d_tail = f".eigenD{suffix}"
        u_tail = f".eigenU{suffix}"
        if eigenD_path.name.endswith(d_tail) and eigenU_path.name.endswith(u_tail):
            d_prefix = eigenD_path.name[: -len(d_tail)]
            u_prefix = eigenU_path.name[: -len(u_tail)]
            if d_prefix == u_prefix and _GENERATION_MARK not in d_prefix:
                return eigenD_path.parent, d_prefix
    return None


def managed_eigen_pair_exists(eigenD_path: Path, eigenU_path: Path) -> bool:
    """Whether a stable managed pair resolves to two committed members."""
    stable = _stable_pair_identity(Path(eigenD_path), Path(eigenU_path))
    if stable is None:
        return False
    try:
        members = resolve_eigen_generation(*stable)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        return False
    return all(path.is_file() for path in members)


def _write_array(
    array: np.ndarray,
    path: Path,
    *,
    what: str,
    legacy_text: bool,
    save_text: Callable[[np.ndarray, Path], None],
) -> None:
    """Write one eigen array as .npy, or as GEMMA text plus a .npy sidecar.

    With ``legacy_text`` the array goes to ``path`` as-is (typically .txt)
    through ``save_text``, then to the sidecar for fast re-reads. Otherwise
    only ``path`` with its suffix swapped to .npy is written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_text:
        logger.info(f"Writing {what} to {path}")
        save_text(array, path)
        write_npy_cache(
            array, _npy_cache_path(path), source_mtime_ns=path.stat().st_mtime_ns
        )
    else:
        npy_path = path.with_suffix(".npy")
        logger.info(f"Writing {what} to {npy_path}")
        save_npy_atomic(array, npy_path)


def _write_eigenvalues(
    eigenvalues: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """One eigenvalue per line at 10 significant digits, GEMMA's precision(10)."""
    _write_array(
        eigenvalues,
        path,
        what="eigenvalues",
        legacy_text=legacy_text,
        save_text=_save_eigenvalues_text,
    )


def _save_eigenvalues_text(eigenvalues: np.ndarray, path: Path) -> None:
    """Write legacy eigenvalues without exposing a partial destination."""
    with AtomicOutput(path) as temporary:
        np.savetxt(temporary, eigenvalues, fmt="%.10g")


def _write_eigenvectors(
    eigenvectors: np.ndarray, path: Path, *, legacy_text: bool = False
) -> None:
    """Tab-separated rows at 10 significant digits, GEMMA's precision(10)."""
    _write_array(
        eigenvectors,
        path,
        what="eigenvectors",
        legacy_text=legacy_text,
        save_text=lambda a, p: write_matrix_parallel(a, p, fmt="%.10g", delimiter="\t"),
    )


WHOLE_GENOME = None
"""The partition of a whole-genome generation; LOCO partitions are chromosomes."""

Partition = str | None


@dataclass(frozen=True)
class EigenGeneration:
    """One immutable set of eigenpair members, one pair per partition.

    A generation is written member by member and becomes visible only when a
    manifest naming it is published with :func:`publish_manifest`, so a reader
    sees the previous generation until the new one is complete. Whole-genome
    and LOCO manifests differ in schema, not in how members are named, checked
    or committed.
    """

    directory: Path
    prefix: str
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def committed(
        cls, manifest: Mapping[str, object], directory: Path, prefix: str, source: Path
    ) -> EigenGeneration:
        """The generation a parsed manifest commits to.

        Raises:
            ValueError: If the manifest names no generation.
        """
        generation = manifest.get("generation")
        if not isinstance(generation, str) or not generation:
            raise ValueError(f"Malformed eigen manifest: {source}")
        return cls(Path(directory), prefix, generation)

    def member_paths(self, partition: Partition, suffix: str) -> tuple[Path, Path]:
        """``(eigenD, eigenU)`` paths of one partition's member pair."""
        stem = f"{self.prefix}{_GENERATION_MARK}{self.generation}"
        if partition is not WHOLE_GENOME:
            stem = f"{stem}.loco.chr{partition}"
        return (
            self.directory / f"{stem}.eigenD{suffix}",
            self.directory / f"{stem}.eigenU{suffix}",
        )

    def write_member(
        self,
        partition: Partition,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        *,
        legacy_text: bool = False,
    ) -> tuple[Path, Path]:
        """Write one partition's pair without publishing a commit record."""
        eigenD_path, eigenU_path = self.member_paths(
            partition, ".txt" if legacy_text else ".npy"
        )
        _write_eigenvalues(eigenvalues, eigenD_path, legacy_text=legacy_text)
        _write_eigenvectors(eigenvectors, eigenU_path, legacy_text=legacy_text)
        return eigenD_path, eigenU_path

    def resolve(
        self, records: Mapping[Partition, object], source: Path
    ) -> dict[Partition, tuple[Path, Path]]:
        """Member paths for each partition's manifest record.

        A record is ``{"eigenD": name, "eigenU": name}`` and must name exactly
        this generation's pair for its partition, both in one format.

        Raises:
            ValueError: If any record is missing, malformed or names a file
                this generation would not have written.
        """
        resolved: dict[Partition, tuple[Path, Path]] = {}
        for partition, record in records.items():
            if not isinstance(record, dict):
                raise ValueError(f"Unsafe or malformed member record in {source}")
            names = (record.get("eigenD"), record.get("eigenU"))
            suffix = Path(names[0]).suffix if isinstance(names[0], str) else ""
            expected = self.member_paths(partition, suffix)
            if suffix not in {".npy", ".txt"} or names != tuple(
                p.name for p in expected
            ):
                raise ValueError(f"Unsafe or malformed member record in {source}")
            resolved[partition] = expected
        return resolved


def member_record(eigenD_path: Path, eigenU_path: Path) -> dict[str, str]:
    """The manifest record naming one partition's member pair."""
    return {"eigenD": eigenD_path.name, "eigenU": eigenU_path.name}


def load_manifest(path: Path) -> dict[str, object]:
    """Parse a manifest JSON object.

    Raises:
        FileNotFoundError: If no manifest is committed.
        json.JSONDecodeError: If the manifest is not JSON.
        ValueError: If it is not a JSON object.
    """
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"Malformed eigen manifest: {path}")
    return manifest


def publish_manifest(path: Path, text: str) -> None:
    """Commit a generation by durably replacing its manifest with ``text``.

    The members it names must already exist. The fsync makes the commit survive
    a power cut; the manifest is a few hundred bytes, so it costs nothing.
    """
    with AtomicOutput(path) as temporary, open(temporary, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def write_eigen_files(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    output_dir: Path,
    prefix: str = "result",
    *,
    legacy_text: bool = False,
) -> tuple[Path, Path]:
    """Write both eigenvalue and eigenvector files.

    Binary .npy is the default format (no text files written). Use
    legacy_text=True for GEMMA-compatible .txt + .npy sidecar output.

    Members include an immutable generation ID. A small manifest at
    ``{prefix}.eigen_manifest.json`` selects the current complete pair.

    Args:
        eigenvalues: 1D array of eigenvalues.
        eigenvectors: 2D array of eigenvectors.
        output_dir: Directory for output files.
        prefix: Filename prefix (default "result").
        legacy_text: If True, write GEMMA text format + sidecars. Default False
            writes only binary .npy.

    Returns:
        Paths to the committed immutable members. JAMMA also accepts the stable
        ``{prefix}.eigenD/eigenU`` pair and resolves it through the manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = EigenGeneration(output_dir, prefix)
    eigenD_path, eigenU_path = generation.write_member(
        WHOLE_GENOME, eigenvalues, eigenvectors, legacy_text=legacy_text
    )
    manifest = {
        "schema_version": EIGEN_MANIFEST_SCHEMA_VERSION,
        "generation": generation.generation,
        "members": member_record(eigenD_path, eigenU_path),
    }
    publish_manifest(
        eigen_manifest_path(output_dir, prefix),
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    return eigenD_path, eigenU_path


def eigen_manifest_path(output_dir: Path, prefix: str) -> Path:
    """Stable commit record for the latest managed eigenpair generation."""
    return Path(output_dir) / f"{prefix}.eigen_manifest.json"


def resolve_eigen_generation(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    """Resolve the latest managed eigenpair from one manifest read."""
    manifest_path = eigen_manifest_path(output_dir, prefix)
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != EIGEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported eigen manifest: {manifest_path}")
    generation = EigenGeneration.committed(manifest, output_dir, prefix, manifest_path)
    return generation.resolve({WHOLE_GENOME: manifest.get("members")}, manifest_path)[
        WHOLE_GENOME
    ]
