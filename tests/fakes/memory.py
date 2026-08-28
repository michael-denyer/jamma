"""Stand-ins for the two psutil reads JAMMA's memory code makes.

``jamma.core.memory`` calls ``psutil.virtual_memory()`` for ``available``
and ``total``, and ``jamma.core.memory_snapshot`` calls
``psutil.Process().memory_info()`` for ``rss`` and ``vms``. Tests that
pin the machine's memory had built those with MagicMock, which answers
any attribute; these fakes declare only the fields the code reads, and
``tests/fakes/test_fakes.py`` checks each name against psutil's own
result types.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import pytest


@dataclass(frozen=True)
class FakeVirtualMemory:
    """What ``psutil.virtual_memory()`` returns, in bytes."""

    available: float
    total: float


@dataclass(frozen=True)
class FakeMemoryInfo:
    """What ``psutil.Process().memory_info()`` returns, in bytes."""

    rss: float
    vms: float


@dataclass(frozen=True)
class FakeProcess:
    """A ``psutil.Process`` whose ``memory_info`` reports fixed numbers."""

    info: FakeMemoryInfo

    def memory_info(self) -> FakeMemoryInfo:
        return self.info


def use_fake_psutil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: float,
    total: float | None = None,
    rss: float = 0.0,
    vms: float = 0.0,
) -> FakeVirtualMemory:
    """Pin what every psutil read in JAMMA reports, for one test.

    ``total`` defaults to ``available``. Patches the ``psutil`` module's
    own attributes, which is the one object every ``import psutil`` shares.
    """
    memory = FakeVirtualMemory(
        available=available, total=available if total is None else total
    )
    process = FakeProcess(FakeMemoryInfo(rss=rss, vms=vms))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(psutil, "Process", lambda *_args, **_kwargs: process)
    return memory
