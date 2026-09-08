"""I/O module for LMM association results.

Writes association results in GEMMA .assoc.txt format for
byte-identical output compatibility.
"""

import errno
import time
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.lmm.schema import FORMAT_COLUMNS, HEADERS, SnpMeta, get_spec
from jamma.lmm.stats import AssocResult
from jamma.utils.atomic_publish import AtomicOutput

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


# Convenience alias for direct import
HEADER_WALD = HEADERS["wald"]


def format_assoc_line(result: AssocResult, test_type: str = "wald") -> str:
    """Format a single association result as tab-separated line.

    Matches GEMMA's WriteFiles formatting exactly:
    - af: .3f (3 decimal places, fixed)
    - All stat columns: .6e (scientific notation, 6 decimal places)
    - chr, rs: string as-is
    - ps, n_miss: integer as-is

    The 7-column prefix (chr, rs, ps, n_miss, allele1, allele0, af) is
    shared across all test types. Only the stat columns differ.

    Args:
        result: AssocResult dataclass instance.
        test_type: One of "wald", "score", "lrt", "all".

    Returns:
        Tab-separated string (no newline).

    Raises:
        ValueError: If test_type is not recognized.
    """
    if test_type not in FORMAT_COLUMNS:
        raise ValueError(
            f"Unknown test_type={test_type!r}; expected one of {list(FORMAT_COLUMNS)}"
        )
    prefix = [
        result.chr,
        result.rs,
        str(result.ps),
        str(result.n_miss),
        result.allele1,
        result.allele0,
        f"{result.af:.3f}",
    ]
    stat_cols = FORMAT_COLUMNS[test_type]
    stats = [f"{getattr(result, col):.6e}" for col in stat_cols]
    return "\t".join(prefix + stats)


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
        with IncrementalAssocWriter(Path("output.assoc.txt")) as writer:
            writer.write_arrays_batch(lmm_mode, snp_indices, snp_info,
                                       afs, miss_counts, arrays)
        print(f"Wrote {writer.count} results")
    """

    def __init__(self, path: Path, test_type: str = "wald"):
        """Initialize writer with output path.

        Args:
            path: Output file path. Parent directories created if needed.
            test_type: One of "wald", "score", "lrt", "all".

        Raises:
            ValueError: If test_type is not recognized.
        """
        if test_type not in FORMAT_COLUMNS:
            raise ValueError(
                f"Unknown test_type={test_type!r}; "
                f"expected one of {list(FORMAT_COLUMNS)}"
            )
        self.path = Path(path)
        self.partial_path = self.path.with_name(f"{self.path.name}.partial")
        self.test_type = test_type
        self._file = None
        self._count = 0

    def __enter__(self) -> "IncrementalAssocWriter":
        """Open the temp file and write the header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._publish = AtomicOutput(self.path)
        self._temp_path = self._publish.__enter__()
        try:
            self._file = open(self._temp_path, "w")
            self._file.write(HEADERS[self.test_type] + "\n")
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

        Called by ``write_arrays_batch``.

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

        # Read the rollback position once, outside the retry loop. It is the
        # same on every attempt, since a failed write is truncated back to it.
        # Taking it inside the try left `pos` unbound whenever tell() itself
        # raised, so the rollback below raised UnboundLocalError instead --
        # not an OSError, so it bypassed the retry, the rollback and the
        # partial-file cleanup, and escaped this method's documented contract.
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
        lmm_mode: int,
        snp_indices: np.ndarray,
        snp_info: SnpMeta,
        afs: np.ndarray,
        miss_counts: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Format and write results directly from numpy arrays.

        Bypasses AssocResult construction. Formats each SNP's metadata
        and stat columns into TSV using the mode's StatColumn descriptors.
        ``afs``, ``miss_counts``, and each array in ``arrays`` must have
        the same length as ``snp_indices`` (one value per SNP in the batch).

        Args:
            lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
            snp_indices: Global SNP indices into snp_info's arrays.
            snp_info: SNP metadata columns.
            afs: Allele frequencies, one per SNP in this batch.
            miss_counts: Missing counts, one per SNP in this batch.
            arrays: Stat arrays keyed by array_key names, same length.

        Raises:
            RuntimeError: If writer is not opened as context manager.
            ValueError: If lmm_mode/test_type mismatch or arrays keys missing.
            OSError: After exhausting retries on write failure.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        n = len(snp_indices)
        if n == 0:
            return

        spec = get_spec(lmm_mode)
        if spec.test_type != self.test_type:
            raise ValueError(
                f"lmm_mode={lmm_mode} (test_type={spec.test_type!r}) does not "
                f"match writer's test_type={self.test_type!r}"
            )

        expected_keys = {c.array_key for c in spec.stat_columns}
        missing_keys = expected_keys - set(arrays.keys())
        if missing_keys:
            raise ValueError(
                f"write_arrays_batch: missing arrays for lmm_mode={lmm_mode}: "
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
        col_fmts = [c.fmt for c in spec.stat_columns]

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

            stats = "\t".join(
                f"{float(arr[j]):{fmt}}"
                for arr, fmt in zip(col_arrays, col_fmts, strict=True)
            )
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
