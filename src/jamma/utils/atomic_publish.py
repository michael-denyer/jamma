"""Publish a file atomically: write a sibling temp, then rename onto the target.

Every writer that overwrites an existing artifact needs this. ``np.save`` and
``np.savetxt`` both truncate their target on open, so a write that fails partway
leaves a partial file where a valid one was. Writing to a sibling and renaming
means a reader sees either the old file or the new one, never a splice.

The guarantee is visibility, not durability: ``os.replace`` is atomic with
respect to concurrent readers, but nothing here fsyncs, so a power cut can still
land the rename without the data blocks behind it. That is a deliberate
trade — these artifacts reach 100k x 100k, and an fsync on every publish costs
more than the failure mode is worth. Callers that need power-loss durability
must fsync themselves.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from types import TracebackType

from loguru import logger


class AtomicOutput:
    """Own a sibling temp until publication, discard, or explicit retention.

    This is the ordinary non-durable publish protocol: callers own opening and
    writing the temp file, while this context owns replacement and cleanup. It
    deliberately does not fsync; durable commit markers such as LOCO eigen-cache
    manifests keep their own descriptor-based protocol.

    ``retain`` transfers the temp to a recovery artifact. If its rename fails,
    the original temp survives context exit at the path returned to the caller.
    """

    def __init__(self, path: Path, *, suffix: str = "") -> None:
        self.path = path
        self.temp_path = publish_temp_path(path, suffix=suffix)
        self._finished = False

    def __enter__(self) -> Path:
        return self.temp_path

    def retain(self, recovery_path: Path) -> Path:
        """Keep the output, returning its actual path even when rename fails."""
        self._finished = True
        try:
            return self.temp_path.replace(recovery_path)
        except OSError as error:
            logger.warning(f"Could not move partial output to {recovery_path}: {error}")
            return self.temp_path

    def discard(self) -> None:
        """Discard an unsuccessful write without permitting later publication."""
        self._finished = True
        unlink_quietly(self.temp_path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._finished:
            return
        try:
            if exc_type is None:
                self.temp_path.replace(self.path)
        finally:
            self.discard()


def publish_temp_path(path: Path, *, suffix: str = "") -> Path:
    """Build a unique sibling temp path for an atomic publish onto ``path``.

    A sibling is guaranteed to be on the same filesystem, which os.replace()
    needs to be atomic. The pid and uuid keep concurrent writers off each
    other's temp file. The leading dot hides it from a casual listing.

    Args:
        path: The destination the temp file will be renamed onto.
        suffix: Extension to append, for writers that add one themselves.
            ``np.save`` appends ``.npy`` when it is absent, so the .npy callers
            pass it explicitly and the stem is used in its place.

    Returns:
        A path beside ``path`` that no concurrent writer will pick.
    """
    stem = path.stem if suffix else path.name
    return path.parent / f".{stem}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}{suffix}"


def unlink_quietly(path: str | Path) -> None:
    """Unlink a file, ignoring absence and logging any other OS error.

    The cleanup idiom every temp file needs on a failure path: a missing file is
    fine (already cleaned), any other OSError is warned but not raised, so
    cleanup never masks the real error it is unwinding from. Logger calls are
    guarded because this can run from a finalizer during interpreter shutdown
    when loguru may already be torn down.
    """
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        # loguru may be torn down when this runs from a finalizer at shutdown.
        with contextlib.suppress(Exception):
            logger.warning(f"Failed to clean up temp file {path}: {e}")
