"""Shared batches and I/O fault injection for the association writer tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import TextIOWrapper

import numpy as np
from loguru import logger

from jamma.genotype.variants import SnpMeta
from jamma.lmm.assoc_output import IncrementalAssocWriter


def open_handle(writer: IncrementalAssocWriter) -> TextIOWrapper:
    """Return an open writer's file handle, for injecting I/O failures.

    The tests below drive the retry and rollback paths by replacing methods on
    the live handle, which means reaching past the public API. ``_file`` is
    None until ``__enter__`` runs, so going through here asserts the writer is
    open once instead of at every call site.
    """
    handle = writer._file
    assert handle is not None, "writer must be open before patching its handle"
    return handle


@contextmanager
def _inject_write_failure(
    writer: IncrementalAssocWriter,
    *,
    message: str,
    fail_on: int | None = None,
    always: bool = False,
    tab_only: bool = False,
    errno_code: int | None = None,
):
    """Replace the writer's file handle ``.write`` with one that fails on cue.

    Exactly one of ``fail_on`` (fail on the Nth matching call, then delegate
    to the original for every other call) or ``always`` (fail on every
    matching call) selects when. ``tab_only`` restricts matching to data
    lines, which contain a tab, so the header write is never counted or
    failed. ``errno_code`` attaches an errno to the raised ``OSError`` so a
    retry-eligibility test can target a specific one.
    """
    assert fail_on is not None or always, "must set fail_on or always"
    handle = open_handle(writer)
    original_write = handle.write
    call_count = 0

    def patched_write(data):
        nonlocal call_count
        if tab_only and "\t" not in data:
            return original_write(data)
        call_count += 1
        if always or call_count == fail_on:
            err = OSError(message)
            if errno_code is not None:
                err.errno = errno_code
            raise err
        return original_write(data)

    handle.write = patched_write
    yield


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    """Collect the loguru warnings emitted inside the block."""
    messages: list[str] = []
    handler_id = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


@dataclass
class SampleBatch:
    """One or more SNPs as write_arrays_batch's array arguments (Wald mode).

    write_arrays_batch is the only writer entry point left, so tests drive
    it with a small batch of pre-built arrays rather than AssocResult
    objects. Wald mode (``MODE_SPECS[1]``) throughout; ``arrays`` keys match
    its stat_columns' array_key names (betas, ses, logls, lambdas, pwalds).
    """

    snp_indices: np.ndarray
    snp_info: SnpMeta
    afs: np.ndarray
    miss_counts: np.ndarray
    arrays: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.snp_indices)

    def as_call_args(self) -> tuple:
        """Positional args for writer.write_arrays_batch(*batch.as_call_args())."""
        return (
            self.snp_indices,
            self.snp_info,
            self.afs,
            self.miss_counts,
            self.arrays,
        )

    def slice_one(self, i: int) -> "SampleBatch":
        """Return a one-SNP batch holding position i's values, index reset to 0."""
        return SampleBatch(
            snp_indices=np.array([0]),
            snp_info=SnpMeta(
                chr=self.snp_info.chr[[i]],
                rs=self.snp_info.rs[[i]],
                pos=self.snp_info.pos[[i]],
                a1=self.snp_info.a1[[i]],
                a0=self.snp_info.a0[[i]],
            ),
            afs=self.afs[[i]],
            miss_counts=self.miss_counts[[i]],
            arrays={k: v[[i]] for k, v in self.arrays.items()},
        )


def single_snp_batch() -> SampleBatch:
    """A single-SNP batch for testing."""
    return SampleBatch(
        snp_indices=np.array([0]),
        snp_info=SnpMeta.from_dicts(
            [{"chr": "1", "rs": "rs12345", "pos": 100000, "a1": "A", "a0": "G"}]
        ),
        afs=np.array([0.25]),
        miss_counts=np.array([5]),
        arrays={
            "betas": np.array([0.123456]),
            "ses": np.array([0.0234567]),
            "logls": np.array([-1234.567]),
            "lambdas": np.array([0.456789]),
            "pwalds": np.array([0.00123456]),
        },
    )


def ten_snp_batch() -> SampleBatch:
    """A ten-SNP batch for testing."""
    n = 10
    return SampleBatch(
        snp_indices=np.arange(n),
        snp_info=SnpMeta.from_dicts(
            [
                {
                    "chr": str((i % 22) + 1),
                    "rs": f"rs{10000 + i}",
                    "pos": 100000 + i * 1000,
                    "a1": "A",
                    "a0": "G",
                }
                for i in range(n)
            ]
        ),
        afs=np.array([0.1 + i * 0.05 for i in range(n)]),
        miss_counts=np.arange(n),
        arrays={
            "betas": np.array([0.1 * (i + 1) for i in range(n)]),
            "ses": np.array([0.01 * (i + 1) for i in range(n)]),
            "logls": np.array([-1000.0 - i for i in range(n)]),
            "lambdas": np.array([0.5 + i * 0.1 for i in range(n)]),
            "pwalds": np.array([0.05 / (i + 1) for i in range(n)]),
        },
    )
