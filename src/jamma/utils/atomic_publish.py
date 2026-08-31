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

from loguru import logger


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
