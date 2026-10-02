# Spawn text pool verification

The original ordered `Pool.imap` caller waited forever when a worker exited.
The pool replaced the worker but never supplied the lost task's result. A
`SystemExit(7)` reproduction remained blocked after four seconds. The same
failure occurs with `os._exit(0)` or `os._exit(7)`, including abrupt native or
OS process termination. This is reachable with the shipped worker counts.

The fixed caller retains the initial process handles and checks their exit
codes while polling `IMapIterator.next(timeout=0.1)`. A timeout does not fail a
slow task. An exited worker raises `RuntimeError` and the existing parent
exception path terminates and joins the pool before temporary files are
removed. Retaining handles detects a death even after Pool replaces a worker.
This uses Pool's private `_pool` member because Pool has no public health API;
that dependency is confined to this caller and tested with real spawn workers.

## Target and source mapping

| Target | Actors / wait | Ownership | Verdict |
|---|---|---|---|
| Spawn matrix I/O | Parent and 1, 2 or 3 processes; ordered result wait with timeout | Each task borrows a disjoint memmap slice; parent retains files until pool joins | PASS after the health-check fix |

`Acquire` maps to the library task dispatch from `_parallel_text.py:125`.
`Finish`, `Fail` and `Die` map to the worker functions passed by
`matrix_reader.py:231` and `matrix_writer.py:200`. `Receive` maps to
`_parallel_text.py:134`. `Poll` maps to the process checks at lines 127-132;
`Terminate` and `Join` map to lines 140-141. The original failing trace claims
items 1 and 2, completes item 1, loses item 2 when its worker exits, consumes
item 1, and then stutters waiting for item 2 forever. No other worker can
supply that result. The new timeout/health check escapes that wait.

## Checks and boundaries

The matrix checks `TypeOK`, `Own`, `Order`, `NoLeak` and eventual `Closed`.
It covers item counts 0, 1, 3 and 5 and worker counts 1, 2 and 3, with normal
completion, ordinary task errors and worker exit enabled at every item.
The reader/writer cap defaults at 32 workers; the three-worker model checks
the same protocol but does not exhaust a 32-process instance. Library queues,
Futures and process termination are abstract atomic operations. Parent
interruption can abort any result wait. There is no handwritten mutex or
condition variable in this caller.

## Reproduction and limits

`tests/test_spawn_pool_exit.py` isolates the call in a process, forces real
worker exits, bounds the wait, and verifies that no child survives error
reporting. The targeted spawn, temp-directory, reader and writer suite passed
59 tests. The original standalone `SystemExit` reproduction hung; the fixed
regressions cover exit codes zero and seven as well.

The model assumes finite task computation, functioning IPC for ordinary task
results, and reliable termination/join. It does not verify CPython's Pool
internals, task pickling, catastrophic allocation failures or repeated signals
inside the exception handler's terminate/join. It checks task and file lifetime
rather than operating-system file descriptor accounting. An earlier failing
task may be skipped when a later worker dies; no ordered error guarantee is
claimed for abrupt worker death. Numerical results and memory ordering below
library synchronization remain outside these models.

The commands that recheck this model are in [TESTING.md](TESTING.md#4-formal-models). The deliberate bugs its checks must catch are listed in `tla/SpawnPool.mutations`.
