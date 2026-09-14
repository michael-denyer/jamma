"""Byte compatibility, buffer ownership, bounded work, and atomic failure paths."""

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from jamma.io._matrix_text import BYTES_PER_VALUE, format_into
from jamma.io._native_matrix_writer import _VALUES_PER_BLOCK, write_native_matrix
from jamma.io.matrix_writer import write_matrix_parallel
from scripts.smoke_test_matrix_text import verify

pytestmark = pytest.mark.tier0


def reference(matrix: np.ndarray) -> bytes:
    stream = io.StringIO()
    np.savetxt(stream, matrix, fmt="%.10g", delimiter="\t")
    return stream.getvalue().encode("ascii")


def test_adversarial_precision_corpus() -> None:
    assert "603,803 values byte-identical" in verify()


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize(
    "layout", ["C", "F", "reverse", "float32", "big_endian", "unaligned"]
)
def test_layouts(tmp_path: Path, layout: str, workers: int) -> None:
    matrix = np.random.default_rng(13).normal(size=(501, 401))
    if layout == "F":
        matrix = np.asfortranarray(matrix)
    elif layout == "reverse":
        matrix = matrix[::-1, ::-1]
    elif layout == "float32":
        matrix = matrix.astype(np.float32)
    elif layout == "big_endian":
        matrix = matrix.astype(">f8")
    elif layout == "unaligned":
        unaligned = np.ndarray(
            matrix.shape,
            dtype=np.float64,
            buffer=bytearray(matrix.nbytes + 1),
            offset=1,
        )
        unaligned[:] = matrix
        matrix = unaligned
    output = tmp_path / "matrix.txt"
    write_matrix_parallel(matrix, output, n_workers=workers)
    assert output.read_bytes() == reference(matrix)
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1), (2, 100_003)])
def test_empty_and_wide_matrices(tmp_path: Path, shape: tuple[int, int]) -> None:
    matrix = np.ones(shape)
    output = tmp_path / "matrix.txt"
    write_matrix_parallel(matrix, output, n_workers=2, min_rows_for_parallel=0)
    assert output.read_bytes() == reference(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones(3),
        np.ones((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=">f8"),
        np.ones((3, 4))[:, ::2],
    ],
)
def test_native_rejects_unsupported_buffers(matrix: np.ndarray) -> None:
    output = bytearray(1024)
    with pytest.raises((ValueError, BufferError)):
        format_into(matrix, output)
    output.extend(b"released")


def test_output_bounds_and_exports() -> None:
    matrix = np.array([[1.0, -0.0]])
    output = bytearray(matrix.size * BYTES_PER_VALUE - 1)
    with pytest.raises(ValueError, match="32 bytes"):
        format_into(matrix, output)
    output.extend(b"x")
    used = format_into(matrix, output)
    assert output[:used] == b"1\t-0\n"
    output.extend(b"released")
    with pytest.raises(TypeError, match="bytearray"):
        format_into(matrix, bytes(128))


def test_aliasing_rejected_before_write() -> None:
    output = bytearray(128)
    matrix = np.ndarray((1, 2), dtype=np.float64, buffer=output)
    matrix[:] = [1, 2]
    before = bytes(output)
    with pytest.raises(ValueError, match="overlap"):
        format_into(matrix, output)
    assert bytes(output) == before


def test_read_only_input_is_supported() -> None:
    matrix = np.array([[1.0, -0.0]])
    matrix.setflags(write=False)
    output = bytearray(64)
    used = format_into(matrix, output)
    assert output[:used] == reference(matrix)


@pytest.mark.parametrize("dtype", [np.uint8, object])
def test_output_rejects_array_storage(dtype) -> None:
    # An empty input proves rejection without risking writes over Python
    # object pointers if this boundary regresses.
    with pytest.raises(TypeError, match="bytearray"):
        format_into(np.empty((0, 1)), np.empty(32, dtype=dtype))


def test_ordered_queue_is_bounded(tmp_path: Path) -> None:
    columns = 257
    block_rows = _VALUES_PER_BLOCK // columns
    matrix = np.arange(block_rows * 12 * columns, dtype=np.float64).reshape(-1, columns)
    first_started = threading.Event()
    other_blocks_done = threading.Event()
    release_first = threading.Event()
    lock = threading.Lock()
    calls: list[tuple[int, int]] = []
    completed = 0

    def controlled_format(block: np.ndarray, buffer: bytearray) -> int:
        nonlocal completed
        with lock:
            calls.append((block.size, id(buffer)))
        if block[0, 0] == 0:
            first_started.set()
            assert release_first.wait(15)
        used = format_into(block, buffer)
        with lock:
            completed += 1
            if completed == 3:
                other_blocks_done.set()
        return used

    output = tmp_path / "matrix.txt"
    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(
            write_native_matrix, matrix, output, 2, controlled_format
        )
        try:
            assert first_started.wait(15)
            assert other_blocks_done.wait(15)
            with lock:
                assert len(calls) == 4
            assert not output.exists()
        finally:
            release_first.set()
        future.result(timeout=15)
    assert max(size for size, _ in calls) <= _VALUES_PER_BLOCK
    assert len({buffer_id for _, buffer_id in calls}) == 4
    assert output.read_bytes() == reference(matrix)


def test_failed_conversion_preserves_destination(tmp_path: Path) -> None:
    matrix = np.ones((1000, 257), dtype=object)
    matrix[900, 0] = "invalid float"
    output = tmp_path / "matrix.txt"
    output.write_bytes(b"prior output")
    with pytest.raises(ValueError, match="could not convert"):
        write_native_matrix(matrix, output, 2, format_into)
    assert output.read_bytes() == b"prior output"
    assert list(tmp_path.iterdir()) == [output]
    assert not any(t.name.startswith("matrix-text") for t in threading.enumerate())


@pytest.mark.parametrize("failure", [OSError("disk full"), KeyboardInterrupt()])
def test_write_failure_joins_workers_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    import builtins

    real_open = builtins.open

    class FailedOutput:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, _):
            raise failure

    def failing_open(path, mode="r", *args, **kwargs):
        if mode == "wb":
            return FailedOutput()
        return real_open(path, mode, *args, **kwargs)

    output = tmp_path / "matrix.txt"
    output.write_bytes(b"prior output")
    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(type(failure)):
        write_native_matrix(np.ones((1000, 257)), output, 2, format_into)
    assert output.read_bytes() == b"prior output"
    assert list(tmp_path.iterdir()) == [output]
    assert not any(t.name.startswith("matrix-text") for t in threading.enumerate())


def test_extension_unavailable_uses_existing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jamma.io.matrix_writer.native_formatter", lambda: None)
    matrix = np.array([[0.0, -0.0, np.nan, np.inf]] * 500)
    output = tmp_path / "matrix.txt"
    write_matrix_parallel(matrix, output, n_workers=2)
    assert output.read_bytes() == reference(matrix)
