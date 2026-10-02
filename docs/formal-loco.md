# LOCO worker protocol verification

The model checks the caller protocol of `solve_eigen_pairs` in
`src/jamma/lmm/loco_workers.py:86-158`, including its closing wrapper in
`src/jamma/lmm/loco_eigen.py:405-422`. The target was selected by inspecting
worker waits, pending Futures, shutdown and BLAS ownership before modelling.

| Target | Threads and shared state | Waits and wakeups | Terminal states and resources | Verdict |
| --- | --- | --- | --- | --- |
| Ordered LOCO eigen worker pool | One consumer and 1, 2 or 3 daemon workers; jobs, pending Futures, completed outcomes, started workers, BLAS scope | `SimpleQueue.get` waits for a job or sentinel; `Future.result` waits for success or error; `join` waits for worker termination | Normal close joins started workers and restores BLAS; KeyboardInterrupt deliberately abandons joins; matrices move between queue, worker, settled Future and caller | Startup cleanup bug confirmed and fixed; final matrix passes |

The job queue has no fixed slot array. Its submission window equals the worker
count. One worker is the default; larger requests are capped by chromosomes,
cores and available memory. The three-worker boundary matches the committed
three-chromosome LOCO fixture. The matrix checks empty input and 1, 3 and 5
items for each worker count, failures at every item, caller stopping or closing,
consumer interruption, and first-thread, later-thread and BLAS-entry failures.

## Confirmed startup cleanup bug

Before the fix, workers started at `loco_workers.py:132-133`, followed by BLAS
scope entry at line 136. Cleanup did not begin until the inner `try` at line
137. If the second `Thread.start()` raised `RuntimeError`, the first worker
remained blocked at `SimpleQueue.get`, line 117. Closing the already-failed
generator did not send the sentinel at lines 153-154 or join it at lines
156-157. BLAS scope entry failure had the same effect on every started worker.

The reduced partial-start counterexample has these code events:

1. The consumer begins its first pull and creates the worker objects.
2. `Thread.start()` starts the first daemon, which can enter `work.get()`.
3. The next `Thread.start()` raises; the generator exits before cleanup.
4. The live worker has no producer left to enqueue its sentinel.

TLC detects this with `SetupClean` in four explored states. The scope-entry
variant starts three workers before scope entry fails, and is detected in
21 states. The partial-start path requires at least two workers and is
reachable with supported concurrent LOCO settings. Scope-entry failure is
also reachable with the default one worker. Neither needs a reduced queue
size or a numerical solver failure.

The deterministic reproduction used real threads and queues and injected an
OS `Thread.start` failure. It printed `loco-eigen-0` alive after the first pull
raised and `pairs.close()` returned. The committed regression file
`tests/test_loco_worker_startup.py` checks first, second and third thread-start
failures and a hardware BLAS-controller entry failure. Before the fix it
reported **3 failed, 1 passed**. After the fix, the startup regressions and
existing worker/order/interruption/resource tests reported **31 passed**.
Its failure-only cleanup releases any leaked workers so a red test cannot
contaminate subsequent tests.

The fix enters BLAS before any worker starts, starts workers inside the
existing `try/finally`, and joins only threads with an assigned thread identity.
The current source locations are scope entry at line 134, startup at
lines 136-137 and guarded join at lines 156-158. Normal cleanup therefore
covers partial startup, while failure entering BLAS creates no live workers.

## Checked properties and assumptions

`TypeOK` checks state domains. `Ownership` checks queue and worker ownership,
including duplicate jobs. `DeliveryOrder` checks the input-order prefix.
`EndOrder` checks that end is not reported while submitted output remains.
`NoDeliveryAfterFailure` requires every delivered item to have a successful
outcome. `ErrorOrder` requires all preceding items before a solve error is
reported. Input-producer failures have no promise to drain earlier solves.
`WindowBound` limits pool-owned items to the worker count. `ClosedClean`
requires no live started workers or pool-owned resources after normal close.
`SetupClean` rejects startup escaping with live workers.
`ScopeCoversSolve` keeps BLAS active until normal worker completion.

`CallsReturn` and `CloseReturns` check termination of active pulls and normal
close. `DrainEnds` checks end or reported error when the caller keeps pulling.
Internal consumer steps and each worker's queue/solve steps have weak fairness.
Consumer close/interrupt choices and spurious wakes have no fairness. The
`Drain` mode explicitly requires the caller to keep pulling. Otherwise it
may remain idle after any yield for as long as it chooses. No mutex fairness
assumption is needed because application code owns no mutex here.

Queue/Future lock operations are atomic library contracts. The model has
explicit queue and Future wait states; enqueue/settlement wakes them, and the
woken participant rereads the predicate. Lost-wakeup mutations produce
liveness failures rather than a deadlock report. A sentinel follows all
submitted jobs; pending queued jobs cancel, and running solves finish before
normal join. Surplus sentinels for unstarted threads are harmless.

The liveness claim assumes finite input, terminating input pulls and numerical
solves, and successful queue/Future bookkeeping. It does not prove LAPACK
termination, library internals, Python's memory allocator, native memory
ordering or cancellation of a nonterminating solver. Thread object allocation
happens before startup; no application lock surrounds that allocation, Future
creation, queue submission or a failure exit. Out-of-memory failures inside
library bookkeeping are outside this model.

Ownership is logical pool ownership, not a claim that Python maintains only
one reference. Worker deletion of its settled Future and input at line 126
is represented in settlement. The brief reference-release interval, exception
traceback retention and caller-held output arrays are omitted; existing weak
reference tests check the actual idle-worker release behavior. Intentional
KeyboardInterrupt abandonment is a separate terminal state, so normal-close
leak claims do not apply to it. Floating-point and eigendecomposition results
are abstract successful outcomes. The model does not check numerical accuracy.

## Final matrix

All 15 runs pass, with 20,118 distinct states summed across runs. Separate runs
can contain equivalent states; the sum is a workload count, not one state graph.
`caller` permits early close, stop-after-yield and KeyboardInterrupt. Nonempty
failure runs allow solve and producer failure at every item independently.

| Label | Distinct states | Verdict |
| --- | ---: | --- |
| workers=1 items=0 drain | 12 | PASS |
| workers=1 items=1 drain | 26 | PASS |
| workers=1 items=3 caller | 218 | PASS |
| workers=1 items=5 drain | 152 | PASS |
| workers=2 items=0 caller | 31 | PASS |
| workers=2 items=1 drain | 92 | PASS |
| workers=2 items=3 caller | 1,178 | PASS |
| workers=2 items=5 drain | 856 | PASS |
| workers=3 items=0 drain | 42 | PASS |
| workers=3 items=1 caller | 398 | PASS |
| workers=3 items=3 drain | 2,318 | PASS |
| workers=3 items=5 caller | 14,744 | PASS |
| workers=2 partial-start-failure | 10 | PASS |
| workers=3 scope-entry-failure | 5 | PASS |
| workers=2 first-start-failure | 6 | PASS |

## Mutation checks

Each mutation ran separately on a temporary copy, with only the named property
checked so an earlier invariant could not hide its sensitivity. All 15 were
rejected. The repository matrix sets `Mutation="none"` and
`SetupCleanup=TRUE`. Diagnostic branches represent individual hypothetical
source changes; they are inactive during the final check.

Unless overridden below, mutation configurations use two workers, three items,
`FailAt={1,2,3}`, no producer failures, `Drain=TRUE`, no interruption and no
setup failure. Failure-state counts were read from TLC's full logs; the matrix
helper's reduced failure output omits them.

| Mutation | Property that rejects it | Distinct states | Verdict |
| --- | --- | ---: | --- |
| Drop ownership during dequeue, `orphan` | TypeOK | 18 | FAIL as expected |
| Take a job without removing it, `duplicate-job` | Ownership | 18 | FAIL as expected |
| Resolve pending Futures as a stack, `lifo` | DeliveryOrder | 95 | FAIL as expected |
| Yield a failed Future, `ignore-error` | NoDeliveryAfterFailure | 91 | FAIL as expected |
| Yield a failed Future, `ignore-error` | ErrorOrder | 91 | FAIL as expected |
| Allow one extra outstanding job, `oversubmit` | WindowBound | 25 | FAIL as expected |
| Retain a settled Future at close, `retain-future` | ClosedClean | 228 | FAIL as expected |
| Restore original startup ordering, SetupFailure=1, SetupCleanup=FALSE | SetupClean | 4 | FAIL as expected |
| Restore BLAS before joining, `early-restore` | ScopeCoversSolve | 122 | FAIL as expected |
| Remove settlement wakeup, `no-future-wake` | CallsReturn | 451 | FAIL as expected |
| Omit shutdown sentinels, `no-sentinels` | CloseReturns | 383 | FAIL as expected |
| Omit shutdown sentinels, `no-sentinels` | DrainEnds | 383 | FAIL as expected |
| Report end before pending Futures drain, `early-end`, FailAt={} | EndOrder | 140 | FAIL as expected |
| Remove enqueue wakeup, `no-queue-wake`, FailAt={} | CallsReturn | 259 | FAIL as expected |
| Restore original scope ordering, Workers=3, Items=0, SetupFailure=2, SetupCleanup=FALSE | SetupClean | 21 | FAIL as expected |

## Recheck

The helpers remain in the installed skill. From the repository root:

```sh
mkdir -p /tmp/loco-jvm
JAVA_TOOL_OPTIONS=-Djava.io.tmpdir=/tmp/loco-jvm \
JAVA=/opt/homebrew/opt/openjdk@21/bin/java TLC_WORKERS=2 \
bash /Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify/scripts/tlc-matrix.sh \
/Users/mdenyer/VSCode/jamma/tla/LocoWorkers.matrix

uv run pytest tests/test_loco_worker_startup.py tests/test_loco_workers.py tests/test_loco_worker_resources.py -q
```

The diagnostic mutation command was:

```sh
JAVA=/tmp/loco-java-capture TLC_WORKERS=2 \
bash /Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify/scripts/tlc-matrix.sh \
/tmp/loco-mutations/LocoWorkers.matrix
```

That wrapper forwarded arguments to the same Java binary and retained the full
TLC logs and configurations. The temporary mutation matrix and capture wrapper
are session diagnostics. To repeat a mutation, copy the model to a temporary
directory, select the constants/property from the table in a one-run matrix,
and run the same matrix helper. Keep the repository matrix unmutated.

TLC needs a local RMI listener. The restricted sandbox initially denied it;
these checks ran with approved local execution. Concurrent TLC processes can
extract their bundled standard modules to the same OS temporary directory;
the final matrix uses an isolated Java temporary directory to prevent partial
module reads. No model CI job was added.
