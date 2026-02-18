"""I/O module for LMM association results.

Writes association results in GEMMA .assoc.txt format for
byte-identical output compatibility.
"""

import errno
import time
from pathlib import Path

from loguru import logger

from jamma.lmm.stats import AssocResult

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


# Column spec per test type: which fields follow the 7-col prefix
_FORMAT_COLUMNS: dict[str, list[str]] = {
    "wald": ["beta", "se", "logl_H1", "l_remle", "p_wald"],
    "score": ["beta", "se", "p_score"],
    "lrt": ["l_mle", "p_lrt"],
    "all": ["beta", "se", "logl_H1", "l_remle", "l_mle", "p_wald", "p_lrt", "p_score"],
}


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
    if test_type not in _FORMAT_COLUMNS:
        raise ValueError(
            f"Unknown test_type={test_type!r}; expected one of {list(_FORMAT_COLUMNS)}"
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
    stat_cols = _FORMAT_COLUMNS[test_type]
    stats = [f"{getattr(result, col):.6e}" for col in stat_cols]
    return "\t".join(prefix + stats)


# GEMMA headers — keyed by test_type, matching _FORMAT_COLUMNS
_HEADER_PREFIX = "chr\trs\tps\tn_miss\tallele1\tallele0\taf"
_HEADERS: dict[str, str] = {
    tt: _HEADER_PREFIX + "\t" + "\t".join(cols) for tt, cols in _FORMAT_COLUMNS.items()
}

# Keep named constants for backward compatibility / direct import
HEADER_WALD = _HEADERS["wald"]
HEADER_SCORE = _HEADERS["score"]
HEADER_LRT = _HEADERS["lrt"]
HEADER_ALL = _HEADERS["all"]


def write_assoc_results(results: list[AssocResult], path: Path) -> None:
    """Write association results in GEMMA .assoc.txt format.

    Output format matches GEMMA exactly:
    - Tab-separated columns
    - Header: chr, rs, ps, n_miss, allele1, allele0, af, beta, se, ...
    - Scientific notation for statistics (6 significant digits)

    Args:
        results: List of AssocResult dataclass instances
        path: Output file path (parent directories created if needed)
    """
    # Ensure parent directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(HEADER_WALD + "\n")
        for result in results:
            f.write(format_assoc_line(result) + "\n")


class IncrementalAssocWriter:
    """Write association results incrementally to disk.

    Context manager that writes results immediately as they are produced,
    avoiding memory accumulation for large GWAS. Output format matches
    write_assoc_results exactly for byte-identical output.

    Example:
        with IncrementalAssocWriter(Path("output.assoc.txt")) as writer:
            for result in compute_results():
                writer.write(result)
        print(f"Wrote {writer.count} results")

        # For Score test:
        with IncrementalAssocWriter(
            Path("output.assoc.txt"), test_type="score"
        ) as writer:
            ...

        # For LRT:
        with IncrementalAssocWriter(
            Path("output.assoc.txt"), test_type="lrt"
        ) as writer:
            ...
    """

    def __init__(self, path: Path, test_type: str = "wald"):
        """Initialize writer with output path.

        Args:
            path: Output file path. Parent directories created if needed.
            test_type: Test type for formatting ("wald", "score", "lrt", or "all")

        Raises:
            ValueError: If test_type is not recognized.
        """
        if test_type not in _FORMAT_COLUMNS:
            raise ValueError(
                f"Unknown test_type={test_type!r}; "
                f"expected one of {list(_FORMAT_COLUMNS)}"
            )
        self.path = Path(path)
        self.test_type = test_type
        self._file = None
        self._count = 0

    def __enter__(self) -> "IncrementalAssocWriter":
        """Open file and write header."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._file.write(_HEADERS[self.test_type] + "\n")
        return self

    def _cleanup_partial(self) -> None:
        """Close file and delete partial output (best-effort).

        Called after final write failure to avoid leaving corrupt partial
        files on disk.
        """
        if self._file is not None:
            try:
                self._file.close()
            except OSError as e:
                logger.warning(f"Failed to close partial output file: {e}")
            self._file = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to delete partial output {self.path}: {e}")

    def write(self, result: AssocResult) -> None:
        """Write single result immediately to disk.

        Retries up to 3 times with increasing backoff on OSError.
        After final failure, cleans up partial output and re-raises.

        Args:
            result: AssocResult to write.

        Raises:
            RuntimeError: If writer is not opened as context manager.
            OSError: After exhausting retries on write failure.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        line = format_assoc_line(result, self.test_type)

        last_error: OSError | None = None
        for attempt in range(1 + len(_RETRY_BACKOFF)):
            try:
                pos = self._file.tell()
                self._file.write(line + "\n")
                self._file.flush()
                self._count += 1
                return
            except OSError as e:
                last_error = e
                # Truncate any partial write back to pre-write position
                try:
                    self._file.seek(pos)
                    self._file.truncate()
                except OSError as seek_err:
                    logger.warning(
                        f"Failed to rollback partial write at position {pos} "
                        f"({seek_err}); file may be in inconsistent state"
                    )
                    self._cleanup_partial()
                    raise last_error from None
                if attempt < len(_RETRY_BACKOFF):
                    err_code = getattr(e, "errno", None)
                    if err_code is not None and err_code not in _RETRYABLE_ERRNOS:
                        # Non-retryable error — fail immediately
                        break
                    logger.warning(
                        f"Write attempt {attempt + 1} failed ({e}), "
                        f"retrying in {_RETRY_BACKOFF[attempt]}s..."
                    )
                    time.sleep(_RETRY_BACKOFF[attempt])

        # All retries exhausted (or non-retryable error)
        self._cleanup_partial()
        err_code = getattr(last_error, "errno", None)
        if attempt == 0 and err_code is not None and err_code not in _RETRYABLE_ERRNOS:
            logger.error(
                f"Write failed immediately (non-retryable errno={err_code}): "
                f"{self.path}"
            )
        else:
            logger.error(f"Write failed after {attempt} retries: {self.path}")
        raise last_error  # type: ignore[misc]

    def write_batch(self, results: list[AssocResult]) -> None:
        """Write multiple results as a single buffered write + flush.

        Formats all results into one string buffer, writes once, flushes once.
        This reduces flush syscalls from N to 1 per batch -- critical on
        network/cloud filesystems where each flush costs 5-50ms.

        Args:
            results: List of AssocResult to write.

        Raises:
            RuntimeError: If writer is not opened as context manager.
            OSError: After exhausting retries on write failure.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        if not results:
            return

        # Format entire batch into single buffer
        buf = "\n".join(format_assoc_line(r, self.test_type) for r in results) + "\n"

        last_error: OSError | None = None
        for attempt in range(1 + len(_RETRY_BACKOFF)):
            try:
                pos = self._file.tell()
                self._file.write(buf)
                self._file.flush()
                self._count += len(results)
                return
            except OSError as e:
                last_error = e
                try:
                    self._file.seek(pos)
                    self._file.truncate()
                except OSError as seek_err:
                    logger.warning(
                        f"Failed to rollback partial write at position {pos} "
                        f"({seek_err}); file may be in inconsistent state"
                    )
                    self._cleanup_partial()
                    raise last_error from None
                if attempt < len(_RETRY_BACKOFF):
                    err_code = getattr(e, "errno", None)
                    if err_code is not None and err_code not in _RETRYABLE_ERRNOS:
                        break
                    logger.warning(
                        f"Batch write attempt {attempt + 1} failed ({e}), "
                        f"retrying in {_RETRY_BACKOFF[attempt]}s..."
                    )
                    time.sleep(_RETRY_BACKOFF[attempt])

        self._cleanup_partial()
        err_code = getattr(last_error, "errno", None)
        if attempt == 0 and err_code is not None and err_code not in _RETRYABLE_ERRNOS:
            logger.error(
                f"Batch write failed immediately (non-retryable errno={err_code}): "
                f"{self.path}"
            )
        else:
            logger.error(f"Batch write failed after {attempt} retries: {self.path}")
        raise last_error  # type: ignore[misc]

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file, cleaning up partial output on OSError.

        On normal exit or non-OSError exceptions, closes the file normally.
        On OSError (write failure propagating up), deletes the partial file.
        """
        if exc_type is not None and issubclass(exc_type, OSError):
            self._cleanup_partial()
        elif self._file:
            try:
                self._file.flush()
            except OSError as e:
                logger.warning(f"Failed to flush output file on close: {e}")
                self._cleanup_partial()
                return
            try:
                self._file.close()
            except OSError as e:
                logger.warning(f"Failed to close output file (data was flushed): {e}")
            finally:
                self._file = None

    @property
    def count(self) -> int:
        """Number of results written."""
        return self._count
