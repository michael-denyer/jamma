# Joining thread pool formal verification

| Target | Threads and shared variables | Wait and wake contract | Terminal state and resources | Verdict |
| --- | --- | --- | --- | --- |
| `JoiningThreadPoolExecutor`, `src/jamma/core/thread_pool.py:21-92` | One submit/shutdown caller; one, two or three executor workers. `_completion` mutex, `_outstanding` dictionary mapping unique tokens to pending/running, executor queue and worker PCs. Token absence means settled. The model's new/settled distinction is ghost history. | Shutdown checks the dictionary under Condition, enters an explicit wait PC and releases the lock. A settlement notifies all waiters. Spurious wakeups and SIGINT can also leave wait; shutdown reacquires and checks again. | Closed scope. Each callable's arguments stay borrowed until its finally settlement; pending cancellations must prevent a wrapper borrowing after its token disappears. | 36/36 configurations pass, 13–262,483 states each. Directed mutations detect each checked property. |

## Source correspondence

Registration at lines 48-50 adds a pending token under Condition. Enqueue is an
unlocked library operation at line 51. A failed submit can happen after enqueue
and race a wrapper that starts running. The except path at lines 56-61 removes
only pending tokens. A running callable remains outstanding until its finally
block at lines 43-46 settles it.

The wrapper acquires Condition at lines 38-42. An absent token raises
CancelledError before borrowing arguments; otherwise it atomically changes the
dictionary value to running. Settlement at lines 31-36 removes the token and
notifies under one Condition critical section. Repeated pop of an absent token
is idempotent, so the counter's duplicate-decrement hazard is eliminated by the
dictionary representation.

The pending-cancellation barrier at lines 70-81 removes pending entries before
calling the library's cancel operation at line 83. A SIGINT inside Future.cancel
can skip its callbacks; because pending tokens were already removed, that cannot
strand shutdown's outstanding set. The model separates this Condition critical
section from the unlocked library cancel phase and permits interruption between
them. Claimed but still pending wrappers recheck token membership under Condition
before they run.

Shutdown's while-loop and wait map to lines 84-86. The interrupt handler at lines
89-90 retries cleanup, and lines 91-92 propagate the saved interrupt only after
all borrowed callables have completed. The model records one interrupt; finite
repeated interruptions have the same retry transition. Infinite interruptions
are outside its termination assumption.

## Commands and counts

```sh
JAVA=/opt/homebrew/opt/openjdk@21/bin/java TLC_WORKERS=2 bash /Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify/scripts/tlc-matrix.sh /Users/mdenyer/VSCode/jamma/tla/JoiningThreadPool.matrix
```

`submit-error` permits success or failure at every item. Failures can occur
between registration/enqueue and return, before or after a worker claims or
starts its wrapper. `cancel` exercises shutdown's pending-token barrier and
library Future cancellation. Every configuration permits early close, spurious
wakeups and interrupted shutdown.

| Run label | Distinct states | Verdict |
| --- | ---: | --- |
| workers=1 items=0 mode=normal | 13 | PASS |
| workers=1 items=0 mode=cancel | 17 | PASS |
| workers=1 items=0 mode=submit-error | 13 | PASS |
| workers=1 items=1 mode=normal | 97 | PASS |
| workers=1 items=1 mode=cancel | 123 | PASS |
| workers=1 items=1 mode=submit-error | 183 | PASS |
| workers=1 items=3 mode=normal | 499 | PASS |
| workers=1 items=3 mode=cancel | 677 | PASS |
| workers=1 items=3 mode=submit-error | 955 | PASS |
| workers=1 items=5 mode=normal | 1213 | PASS |
| workers=1 items=5 mode=cancel | 1687 | PASS |
| workers=1 items=5 mode=submit-error | 2303 | PASS |
| workers=2 items=0 mode=normal | 13 | PASS |
| workers=2 items=0 mode=cancel | 17 | PASS |
| workers=2 items=0 mode=submit-error | 13 | PASS |
| workers=2 items=1 mode=normal | 153 | PASS |
| workers=2 items=1 mode=cancel | 199 | PASS |
| workers=2 items=1 mode=submit-error | 297 | PASS |
| workers=2 items=3 mode=normal | 3207 | PASS |
| workers=2 items=3 mode=cancel | 4043 | PASS |
| workers=2 items=3 mode=submit-error | 6159 | PASS |
| workers=2 items=5 mode=normal | 14253 | PASS |
| workers=2 items=5 mode=cancel | 18527 | PASS |
| workers=2 items=5 mode=submit-error | 26973 | PASS |
| workers=3 items=0 mode=normal | 13 | PASS |
| workers=3 items=0 mode=cancel | 17 | PASS |
| workers=3 items=0 mode=submit-error | 13 | PASS |
| workers=3 items=1 mode=normal | 209 | PASS |
| workers=3 items=1 mode=cancel | 275 | PASS |
| workers=3 items=1 mode=submit-error | 411 | PASS |
| workers=3 items=3 mode=normal | 14119 | PASS |
| workers=3 items=3 mode=cancel | 17039 | PASS |
| workers=3 items=3 mode=submit-error | 27235 | PASS |
| workers=3 items=5 mode=normal | 138233 | PASS |
| workers=3 items=5 mode=cancel | 170927 | PASS |
| workers=3 items=5 mode=submit-error | 262483 | PASS |

The checked invariants are `TypeOK`, `TokensExact`, `Ownership`,
`BorrowedRunning`, `NoBorrowAfterRelease` and `NoLeak`. Liveness checks require
blocking shutdown to return, and require whole-drive termination when the caller
eventually closes. The token dictionary must exactly describe the pending and
running jobs, with at most one worker borrowing each job.

## Mutation sensitivity and fairness

Mutations ran on copies under `/tmp/thread-pool-mutations`, one change per copy.
Each mutation below failed the specified property.

| Mutation | Detecting property |
| --- | --- |
| no-notify | BlockingReturns |
| omit-pending-cancellation-barrier | BlockingReturns |
| if-wait | NoLeak |
| cancel-token-remains | TokensExact |
| clear-all-tokens | TokensExact |
| cancel-running | BorrowedRunning |
| ignore-cancelled-wrapper | TokensExact |
| counter-bound | TypeOK |
| keep-queue-item | Ownership |
| borrow-cancelled-after-release | NoBorrowAfterRelease |
| never-return | NoCloseTerminates |

`no-notify` strands the shutdown thread in its explicit wait PC after the final
callable settles; the failure is liveness, not deadlock. `if-wait` lets a
spurious wakeup return without reacquiring and checking. `cancel-running`
models treating a submit failure as permission to remove a token that a worker
already owns. `ignore-cancelled-wrapper` models running an enqueued wrapper
whose submit failed and whose pending token was removed.

A diagnostic `weak-mutex-starvation` copy changed strong acquisition fairness to
weak fairness. TLC emitted a pure starvation cycle: the main thread repeatedly
wakes spuriously, acquires Condition, checks, and waits again, while a worker in
settle-lock never acquires it. The source contains no missing notification in
that trace. Therefore `MutexFairness` explicitly gives each worker's mutex
acquisition strong fairness. Other real thread steps use weak fairness, and
spurious wakes, interrupts and caller close choices are not fair. The helper's
safety properties do not depend on this scheduling assumption.

## Runtime checks and limits

`tests/test_chunk_pipeline_interrupt.py` checks the caller's BLAS lifetime with a
real SIGINT delivered during cleanup. The writer interruption and cancellation
regressions cover the other caller and the interrupted Future.cancel path.
`tests/test_joined_pool_submit_failure.py` injects an OS worker-start failure
after enqueue, then starts a real subsequent worker. The abandoned wrapper
rejects its borrow when the old queue entry is eventually dequeued.

One caller owns submit and shutdown, as both production scopes do. Library
Future and executor queue internals are atomic abstractions. The model does not
verify Python's native mutex memory ordering, the GIL, native C call safety,
external threads cancelling Futures independently of shutdown, or arbitrary
concurrent submitters. Library cancellation callbacks are redundant for the two
callers' shutdown paths after the pending-token barrier.

The model checks `wait=True`, which both callers use when releasing borrowed
arguments. `wait=False` explicitly permits ongoing work and is outside that
resource-release guarantee. Callables eventually return or raise, and the OS
keeps the interpreter alive. Python exceptions leave a Condition context through
its normal unlock path; there is no longjmp or os._exit in the helper.

Allocations under Condition are the submit dictionary insertion at line 50 and
the cancellation dictionary comprehension at lines 76-80. Submit allocation or
thread-start failure is represented by the cancellation path. The proof assumes
shutdown's cleanup allocations succeed; a MemoryError in the dictionary
comprehension is not handled by the KeyboardInterrupt retry and is outside this
proof. Similar catastrophic failures during exception construction, Condition
machinery, and interpreter teardown are not claimed safe. No CI job was added.
