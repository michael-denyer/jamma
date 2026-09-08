"""Association publication preserves completed artifacts on failure."""

import pytest

from jamma.lmm.io import IncrementalAssocWriter

pytestmark = pytest.mark.tier0


def test_computation_failure_preserves_completed_output(tmp_path):
    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")
    with pytest.raises(ValueError, match="computation failed"):
        with IncrementalAssocWriter(destination):
            raise ValueError("computation failed")
    assert destination.read_bytes() == b"previous completed run\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_close_failure_prevents_publication(tmp_path, monkeypatch):
    import builtins

    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")
    real_open = builtins.open

    def failing_close_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        close = handle.close

        def fail_close():
            close()
            raise OSError("close failed")

        handle.close = fail_close
        return handle

    monkeypatch.setattr(builtins, "open", failing_close_open)
    with pytest.raises(OSError, match="close failed"):
        with IncrementalAssocWriter(destination):
            pass
    assert destination.read_bytes() == b"previous completed run\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, MemoryError])
def test_interruption_retains_separate_partial(tmp_path, error):
    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")
    with pytest.raises(error):
        with IncrementalAssocWriter(destination):
            raise error()
    assert destination.read_bytes() == b"previous completed run\n"
    partials = list(tmp_path.glob("*.partial.*"))
    assert len(partials) == 1
    assert partials[0].read_text().startswith("chr\trs\t")


def test_header_failure_cleans_temporary_output(tmp_path, monkeypatch):
    import builtins

    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")
    real_open = builtins.open

    def failing_write_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)

        def fail_write(data):
            raise OSError("header failed")

        handle.write = fail_write
        return handle

    monkeypatch.setattr(builtins, "open", failing_write_open)
    with pytest.raises(OSError, match="header failed"):
        with IncrementalAssocWriter(destination):
            pass
    assert destination.read_bytes() == b"previous completed run\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("operation", ["flush", "replace"])
def test_publication_io_failure_preserves_destination(tmp_path, monkeypatch, operation):
    import builtins
    from pathlib import Path

    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")

    def fail(*args, **kwargs):
        raise OSError("publication failed")

    if operation == "replace":
        monkeypatch.setattr(Path, "replace", fail)
    else:
        real_open = builtins.open

        def failing_open(*args, **kwargs):
            handle = real_open(*args, **kwargs)
            handle.flush = fail
            return handle

        monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="publication failed"):
        with IncrementalAssocWriter(destination):
            pass
    assert destination.read_bytes() == b"previous completed run\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_success_replaces_completed_output(tmp_path):
    destination = tmp_path / "result.assoc.txt"
    destination.write_bytes(b"previous completed run\n")
    with IncrementalAssocWriter(destination):
        assert destination.read_bytes() == b"previous completed run\n"
    assert destination.read_text().startswith("chr\trs\t")
    assert list(tmp_path.iterdir()) == [destination]
