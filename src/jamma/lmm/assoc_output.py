"""The GEMMA ``.assoc.txt`` output row.

Turns a chunk's stat arrays into output rows, either as ``AssocResult``
records (``build_results``) or as TSV lines on disk
(``IncrementalAssocWriter``), and exposes the per-chunk sink factories the
batch, streaming and LOCO NumPy runners share. A ``ModeSpec`` names the
columns in both forms.
"""

from __future__ import annotations

import errno
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from jamma.lmm.schema import ModeSpec, SnpMeta
from jamma.lmm.stats import AssocResult
from jamma.utils.atomic_publish import AtomicOutput

if TYPE_CHECKING:
    from jamma.lmm.genotype_source import PreparedGenotypes

# Retry backoff schedule (seconds) for transient write failures
_RETRY_BACKOFF = (0.1, 0.5, 2.0)

# Errno values worth retrying (transient I/O conditions)
_RETRYABLE_ERRNOS = frozenset(
    {
        errno.ENOSPC,  # No space left on device (may free up)
        errno.EIO,  # I/O error (transient on network filesystems)
        errno.EAGAIN,  # Resource temporarily unavailable
        errno.EBUSY,  # Device or resource busy
    }
)


class IncrementalAssocWriter:
    """Write association results incrementally to disk.

    Context manager that writes results immediately as they are produced,
    avoiding memory accumulation for large GWAS.

    Writes via ``write_arrays_batch(...)``, directly from numpy arrays.

    Results go to the sibling temp file ``AtomicOutput`` owns and are
    published onto ``path`` only when the context exits cleanly. An ordinary
    exception discards them. An interrupt or an out-of-memory keeps what was
    written at ``partial_path``, beside the destination.

    Example:
        with IncrementalAssocWriter(Path("output.assoc.txt"), mode) as writer:
            writer.write_arrays_batch(snp_indices, snp_info, afs, miss_counts,
                                      arrays)
        print(f"Wrote {writer.count} results")
    """

    def __init__(self, path: Path, mode: ModeSpec):
        """Initialize writer with output path.

        Args:
            path: Output file path. Parent directories created if needed.
            mode: The LMM mode whose columns every row carries.
        """
        self.path = Path(path)
        self.partial_path = self.path.with_name(f"{self.path.name}.partial")
        self.mode = mode
        self._file = None
        self._count = 0

    def __enter__(self) -> IncrementalAssocWriter:
        """Open the temp file and write the header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._publish = AtomicOutput(self.path)
        self._temp_path = self._publish.__enter__()
        try:
            self._file = open(self._temp_path, "w")
            self._file.write(self.mode.header + "\n")
        except BaseException as error:
            self._close_file()
            self._publish.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _close_file(self) -> None:
        """Close file handle (best-effort), set to None."""
        if self._file is not None:
            try:
                self._file.close()
            except OSError as e:
                logger.warning(f"Failed to close output file {self._temp_path}: {e}")
            finally:
                self._file = None

    def _discard_temp(self) -> None:
        """Close the file and delete the temp (best-effort), for the retry path."""
        self._close_file()
        self._publish.discard()

    def _write_buf(self, buf: str, count: int) -> None:
        """Write pre-formatted buffer with retry logic.

        Args:
            buf: Pre-formatted string to write (must end with newline).
            count: Number of logical results in the buffer.

        Raises:
            RuntimeError: If the writer is not open.
            OSError: After exhausting retries on write failure, or immediately
                if the rollback position cannot be read.
        """
        handle = self._file
        if handle is None:
            raise RuntimeError("Writer not opened. Use as context manager.")

        # Outside the try: a tell() failure must surface as the OSError itself,
        # not as an unbound `pos` in the rollback below.
        pos = handle.tell()

        last_error: OSError | None = None
        for attempt in range(1 + len(_RETRY_BACKOFF)):
            try:
                handle.write(buf)
                handle.flush()
                self._count += count
                return
            except OSError as e:
                last_error = e
                try:
                    handle.seek(pos)
                    handle.truncate()
                except OSError as seek_err:
                    logger.warning(
                        f"Failed to rollback partial write at position "
                        f"{pos} ({seek_err}); file may be inconsistent"
                    )
                    self._discard_temp()
                    raise last_error from None
                if attempt < len(_RETRY_BACKOFF):
                    err_code = getattr(e, "errno", None)
                    if err_code is not None and err_code not in _RETRYABLE_ERRNOS:
                        break
                    logger.warning(
                        f"Write attempt {attempt + 1} failed ({e}), "
                        f"retrying in {_RETRY_BACKOFF[attempt]}s..."
                    )
                    time.sleep(_RETRY_BACKOFF[attempt])

        self._discard_temp()
        err_code = getattr(last_error, "errno", None)
        if attempt == 0 and err_code is not None and err_code not in _RETRYABLE_ERRNOS:
            logger.error(
                f"Write failed immediately "
                f"(non-retryable errno={err_code}): {self._temp_path}"
            )
        else:
            logger.error(f"Write failed after {attempt + 1} retries: {self._temp_path}")
        raise last_error  # type: ignore[misc]

    def write_arrays_batch(
        self,
        snp_indices: np.ndarray,
        snp_info: SnpMeta,
        afs: np.ndarray,
        miss_counts: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Format and write results directly from numpy arrays.

        Bypasses AssocResult construction. Formats each SNP's metadata
        and the mode's stat columns into TSV.
        ``afs``, ``miss_counts``, and each array in ``arrays`` must have
        the same length as ``snp_indices`` (one value per SNP in the batch).

        Args:
            snp_indices: Global SNP indices into snp_info's arrays.
            snp_info: SNP metadata columns.
            afs: Allele frequencies, one per SNP in this batch.
            miss_counts: Missing counts, one per SNP in this batch.
            arrays: Stat arrays keyed by array_key names, same length.

        Raises:
            RuntimeError: If writer is not opened as context manager.
            ValueError: If arrays keys are missing or lengths disagree.
            OSError: After exhausting retries on write failure.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        n = len(snp_indices)
        if n == 0:
            return

        spec = self.mode
        expected_keys = {c.array_key for c in spec.stat_columns}
        missing_keys = expected_keys - set(arrays.keys())
        if missing_keys:
            raise ValueError(
                f"write_arrays_batch: missing arrays for mode {spec.test_type!r}: "
                f"{missing_keys}. Expected: {expected_keys}, got: {set(arrays.keys())}"
            )

        for name, arr in (("afs", afs), ("miss_counts", miss_counts)):
            if len(arr) != n:
                raise ValueError(
                    f"write_arrays_batch: {name} has length {len(arr)}, "
                    f"expected {n} (matching snp_indices)"
                )

        for col in spec.stat_columns:
            arr = arrays[col.array_key]
            if len(arr) != n:
                raise ValueError(
                    f"write_arrays_batch: stat array {col.array_key!r} has length "
                    f"{len(arr)}, expected {n} (matching snp_indices)"
                )

        col_arrays = [arrays[c.array_key] for c in spec.stat_columns]

        chrs = snp_info.chr[snp_indices]
        rss = snp_info.rs[snp_indices]
        poss = snp_info.pos[snp_indices]
        a1s = snp_info.a1[snp_indices]
        a0s = snp_info.a0[snp_indices]

        lines: list[str] = []
        for j in range(n):
            prefix = (
                f"{chrs[j]}\t"
                f"{rss[j]}\t"
                f"{poss[j]}\t"
                f"{int(miss_counts[j])}\t"
                f"{a1s[j]}\t"
                f"{a0s[j]}\t"
                f"{float(afs[j]):.3f}"
            )

            stats = "\t".join(f"{float(arr[j]):.6e}" for arr in col_arrays)
            lines.append(f"{prefix}\t{stats}")

        self._write_buf("\n".join(lines) + "\n", n)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Publish on success, discard on error, retain a partial on interrupt."""
        if exc_type is None:
            if self._file is None:
                return  # _write_buf hit an error the caller swallowed
            try:
                self._file.close()  # close() flushes, so a bad flush raises here
                self._file = None
            except BaseException as error:
                self._close_file()
                self._publish.__exit__(type(error), error, error.__traceback__)
                raise
            self._publish.__exit__(None, None, None)
            return

        self._close_file()
        if not issubclass(exc_type, Exception) or issubclass(exc_type, MemoryError):
            retained_path = self._publish.retain(self.partial_path)
            logger.warning(
                f"{exc_type.__name__} after {self._count} results written; "
                f"partial output retained at {retained_path}"
            )
        else:
            logger.warning(
                f"{exc_type.__name__}: {exc_val}; discarding partial output "
                f"for {self.path} ({self._count} results written)"
            )
        self._publish.__exit__(exc_type, exc_val, exc_tb)

    @property
    def count(self) -> int:
        """Number of results written."""
        return self._count


# Per-chunk result sink handed to the shared NumPy LMM chunk runner:
# (chunk_arrays, filtered_start, filtered_end) -> None.
ChunkSink = Callable[[dict[str, np.ndarray], int, int], None]


def build_results(
    mode: ModeSpec,
    snp_indices: np.ndarray,
    filtered_afs: np.ndarray,
    filtered_miss: np.ndarray,
    snp_info: SnpMeta,
    arrays: dict[str, np.ndarray],
) -> list[AssocResult]:
    """Build AssocResult objects for any LMM test mode.

    Args:
        mode: The LMM mode whose columns ``arrays`` carries.
        snp_indices: Indices of SNPs that passed filtering.
        filtered_afs: Allele frequencies for filtered SNPs.
        filtered_miss: Missing counts for filtered SNPs.
        snp_info: SNP metadata columns, indexed by global SNP index.
        arrays: Dict mapping stat name -> numpy array of values.

    Returns:
        List of AssocResult objects.
    """
    field_map = {c.array_key: c.field_name for c in mode.stat_columns}
    missing_keys = set(field_map.keys()) - set(arrays.keys())
    if missing_keys:
        raise ValueError(
            f"Missing arrays for mode {mode.test_type!r}: {missing_keys}. "
            f"Expected keys: {set(field_map.keys())}, got: {set(arrays.keys())}"
        )
    # Convert stat arrays to Python lists in one C call each
    # (avoids per-element float() conversion overhead)
    stat_lists = {
        field_name: arrays[array_key].tolist()
        for array_key, field_name in field_map.items()
    }
    af_list = filtered_afs.tolist()
    miss_list = filtered_miss.tolist()
    chr_list = snp_info.chr[snp_indices].tolist()
    rs_list = snp_info.rs[snp_indices].tolist()
    pos_list = snp_info.pos[snp_indices].tolist()
    a1_list = snp_info.a1[snp_indices].tolist()
    a0_list = snp_info.a0[snp_indices].tolist()

    results = []
    for j in range(len(snp_indices)):
        meta: dict[str, Any] = {
            "chr": chr_list[j],
            "rs": rs_list[j],
            "ps": pos_list[j],
            "n_miss": int(miss_list[j]),
            "allele1": a1_list[j],
            "allele0": a0_list[j],
            "af": af_list[j],
        }
        for field_name, vals in stat_lists.items():
            meta[field_name] = vals[j]

        results.append(AssocResult(**meta))
    return results


def make_writer_sink(
    writer: IncrementalAssocWriter,
    genotypes: PreparedGenotypes,
) -> ChunkSink:
    """Build a chunk sink that streams each result chunk to disk.

    The returned sink slices the prepared source's bound selection by the
    ``[filtered_start, filtered_end)`` range it receives per chunk.
    """
    selection = genotypes.selection

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        writer.write_arrays_batch(
            selection.indices[filtered_start:filtered_end],
            genotypes.snp_meta,
            selection.filtered_afs[filtered_start:filtered_end],
            selection.filtered_miss[filtered_start:filtered_end],
            chunk_arrays,
        )

    return _sink


def make_result_list_sink(
    results: list[AssocResult],
    mode: ModeSpec,
    genotypes: PreparedGenotypes,
) -> ChunkSink:
    """Build a chunk sink that appends built ``AssocResult`` objects to ``results``.

    Shared by the streaming and LOCO NumPy runners on their in-memory
    (no ``output_path``) path.
    """
    selection = genotypes.selection

    def _sink(
        chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
    ) -> None:
        results.extend(
            build_results(
                mode,
                selection.indices[filtered_start:filtered_end],
                selection.filtered_afs[filtered_start:filtered_end],
                selection.filtered_miss[filtered_start:filtered_end],
                genotypes.snp_meta,
                chunk_arrays,
            )
        )

    return _sink
