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

import json
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.matrix_reader import read_matrix_parallel
from jamma.io.matrix_writer import write_matrix_parallel
from jamma.utils.atomic_publish import atomic_output
from jamma.utils.npy_cache import (
    read_array_artifact,
    save_npy_atomic,
    write_npy_cache,
)

EIGEN_MANIFEST_SCHEMA_VERSION = 1

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

    Sidecar loads are read-only memory maps; see ``read_array_artifact``.
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

    May be a read-only ``np.memmap`` when loaded from the .npy sidecar;
    callers must not mutate it in place.

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

    Text parsing goes through ``read_matrix_parallel``. May be a read-only
    ``np.memmap`` when loaded from the .npy sidecar; callers must not mutate
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
    marker = ".generation."
    if marker not in path.name:
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
            if d_prefix == u_prefix and ".generation." not in d_prefix:
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
        write_npy_cache(array, _npy_cache_path(path))
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
    with atomic_output(path) as temporary:
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
    generation = uuid.uuid4().hex
    eigenD_path, eigenU_path = write_eigen_generation_members(
        eigenvalues,
        eigenvectors,
        output_dir,
        prefix,
        generation,
        legacy_text=legacy_text,
    )
    _write_manifest(
        eigen_manifest_path(output_dir, prefix),
        {
            "schema_version": EIGEN_MANIFEST_SCHEMA_VERSION,
            "generation": generation,
            "members": {
                "eigenD": eigenD_path.name,
                "eigenU": eigenU_path.name,
            },
        },
    )
    return eigenD_path, eigenU_path


def eigen_manifest_path(output_dir: Path, prefix: str) -> Path:
    """Stable commit record for the latest managed eigenpair generation."""
    return Path(output_dir) / f"{prefix}.eigen_manifest.json"


def _generation_prefix(prefix: str, generation: str) -> str:
    return f"{prefix}.generation.{generation}"


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Publish a small JSON commit record after all referenced files exist."""
    with atomic_output(path) as temporary, open(temporary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
        fh.write("\n")


def resolve_eigen_generation(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    """Resolve the latest managed eigenpair from one manifest read."""
    manifest_path = eigen_manifest_path(output_dir, prefix)
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported eigen manifest: {manifest_path}")
    generation = manifest.get("generation")
    members = manifest.get("members")
    if not isinstance(generation, str) or not isinstance(members, dict):
        raise ValueError(f"Malformed eigen manifest: {manifest_path}")
    expected_prefix = _generation_prefix(prefix, generation)
    resolved: list[Path] = []
    formats: list[str] = []
    for kind in ("eigenD", "eigenU"):
        name = members.get(kind)
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith(f"{expected_prefix}.{kind}.")
        ):
            raise ValueError(f"Unsafe or malformed {kind} member in {manifest_path}")
        artifact_format = Path(name).suffix
        if artifact_format not in {".npy", ".txt"}:
            raise ValueError(f"Unsupported {kind} member format in {manifest_path}")
        formats.append(artifact_format)
        resolved.append(Path(output_dir) / name)
    if formats[0] != formats[1]:
        raise ValueError(f"Mixed member formats in eigen manifest: {manifest_path}")
    return resolved[0], resolved[1]


def write_eigen_generation_members(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    output_dir: Path,
    prefix: str,
    generation: str,
    *,
    legacy_text: bool = False,
    label: str | None = None,
) -> tuple[Path, Path]:
    """Write immutable pair members without publishing a commit record."""
    output_dir = Path(output_dir)
    suffix = ".txt" if legacy_text else ".npy"
    member_prefix = _generation_prefix(prefix, generation)
    if label is not None:
        member_prefix = f"{member_prefix}.{label}"
    eigenD_path = output_dir / f"{member_prefix}.eigenD{suffix}"
    eigenU_path = output_dir / f"{member_prefix}.eigenU{suffix}"
    _write_eigenvalues(eigenvalues, eigenD_path, legacy_text=legacy_text)
    _write_eigenvectors(eigenvectors, eigenU_path, legacy_text=legacy_text)
    return eigenD_path, eigenU_path
