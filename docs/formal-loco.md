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

The two setup mutations in `tla/LocoWorkers.mutations` restore this order,
and TLC rejects both with `SetupClean`. The scope-entry variant starts three
workers before scope entry fails. The partial-start path requires at least two workers and is
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

The commands that recheck this model are in [TESTING.md](TESTING.md#4-formal-models). The deliberate bugs its checks must catch are listed in `tla/LocoWorkers.mutations`.
