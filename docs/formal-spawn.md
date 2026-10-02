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

12 PASS runs; 59,367 summed distinct states. Counts are per bounded model, not a count of real executions.

| Run | Distinct states | Verdict |
|---|---:|---|
| workers=1 items=0 fail=all | 5 | PASS |
| workers=1 items=1 fail=all | 26 | PASS |
| workers=1 items=3 fail=all | 196 | PASS |
| workers=1 items=5 fail=all | 942 | PASS |
| workers=2 items=0 fail=all | 7 | PASS |
| workers=2 items=1 fail=all | 48 | PASS |
| workers=2 items=3 fail=all | 900 | PASS |
| workers=2 items=5 fail=all | 7956 | PASS |
| workers=3 items=0 fail=all | 11 | PASS |
| workers=3 items=1 fail=all | 92 | PASS |
| workers=3 items=3 fail=all | 2987 | PASS |
| workers=3 items=5 fail=all | 46197 | PASS |

## Mutation sensitivity

Each mutation was tested independently, with only its named property enabled.
The unmutated matrix always sets `Mutation="none"` and `DetectDeath=TRUE`.

| Mutation | Detecting property | Verdict |
|---|---|---|
| Invalid worker terminal flag | TypeOK | Expected FAIL |
| Leave a completed task held by its idle worker | Own | Expected FAIL |
| Consume last item instead of the next FIFO item | Order | Expected FAIL |
| Close files before workers stop | NoLeak | Expected FAIL |
| Remove worker-death detection, preserving original caller | Closed | Expected FAIL, lost result wait |

## Reproduction and commands

`tests/test_spawn_pool_exit.py` isolates the call in a process, forces real
worker exits, bounds the wait, and verifies that no child survives error
reporting. The targeted spawn, temp-directory, reader and writer suite passed
59 tests. The original standalone `SystemExit` reproduction hung; the fixed
regressions cover exit codes zero and seven as well.

Run from the repository root, with the installed helper directory assigned to
`VERIFY_SKILL`:

```sh
VERIFY_SKILL=/Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify
export JAVA=/opt/homebrew/opt/openjdk@21/bin/java
export TLC_WORKERS=2
bash "$VERIFY_SKILL/scripts/setup.sh" tla
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/SpawnPool.matrix"
uv run pytest tests/test_spawn_pool_exit.py tests/test_parallel_text.py tests/test_matrix_reader.py tests/test_matrix_writer.py -x
```

For concurrent JVMs, give each a separate existing directory via
`JAVA_TOOL_OPTIONS=-Djava.io.tmpdir=/absolute/unique/directory`, because bundled
TLA+ modules otherwise share the JVM's extraction directory.

The model assumes finite task computation, functioning IPC for ordinary task
results, and reliable termination/join. It does not verify CPython's Pool
internals, task pickling, catastrophic allocation failures or repeated signals
inside the exception handler's terminate/join. It checks task and file lifetime
rather than operating-system file descriptor accounting. An earlier failing
task may be skipped when a later worker dies; no ordered error guarantee is
claimed for abrupt worker death. Numerical results and memory ordering below
library synchronization remain outside these models. No CI was added.
