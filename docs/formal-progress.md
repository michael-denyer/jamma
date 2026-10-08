# Timed progress verification

The target was selected by reading `src/jamma/core/progress.py` in full,
`tests/test_progress.py`, `tests/fakes/progress.py`, and its eigendecomposition
caller at `src/jamma/lmm/eigen.py:269-275` before modelling. The model includes
the real progressbar2 `finish` behavior, which the original recording fake
omitted.

| Target | Threads and shared state | Waits and wakeups | Terminal states and resources | Verdict |
| --- | --- | --- | --- | --- |
| `timed_progress`, `core/progress.py:138-227` | One consumer and one daemon worker; result/error boxes, Event, progress display | Consumer `done.wait` has a positive timeout; worker publishes then calls `done.set`; normal join polls for worker exit | Normal result/error reports follow join; consumer KeyboardInterrupt abandons joins; output ownership moves from worker to box to caller | False completion confirmed and fixed; 8 runs pass |

## Confirmed false completion

There were two paths to an incorrect 100% display. The code then made an
explicit `bar.update(100)` call between the polling loop and `bar.finish()`:

- A polling `bar.update` raised `OSError` at lines 209-211. The consumer left
  the Event-wait loop and made the explicit `bar.update(100)` call before the
  still-running worker returned. It eventually joined, so the returned
  numerical value was unaffected.
- The unconditional `bar.finish()`, now at line 217, implicitly forced a full
  redraw on worker errors and consumer cancellation. The real library's
  `ProgressBar.finish(dirty=False)` invokes `update(max_value, force=True)`;
  its `dirty=True` option preserves the current percentage. The shared fake
  only marked itself finished, so the existing no-100-on-error test missed
  the implicit redraw.

The first counterexample maps to a consumer timeout at line 200, a polling
stdout failure at line 209, the break at line 211 and the premature explicit
full update, while the worker remains inside `fn()` at line 188. The mutation
"finish is clean without checking that the worker finished" in
`tla/TimedProgress.mutations` restores this behavior, and TLC rejects it.

For a failing worker, the trace is worker failure at line 188, publication in
`exception` at line 190, notification at line 192, consumer wake at line 200,
then implicit full redraw from line 217. Consumer KeyboardInterrupt takes the
same unclean-finish path at lines 212-217. The two "finish redraws 100%"
mutations restore that finish,
and TLC rejects both. All
three paths are reachable with the shipped single worker and normal 100-tick
bar. No special queue capacity or unsupported caller is needed.

`tests/test_timed_progress_completion.py` uses the real ProgressBar, with a
subclass that records its updates. Before the fix, all five initial
regressions failed. RuntimeError, MemoryError, SystemExit and KeyboardInterrupt
raised by the worker each recorded `[0, 100]`. A deterministic transient
stdout failure, with the worker blocked on an Event, recorded two premature
100 updates, one explicit and one from `finish`.

The fix calls `finish(dirty=cancelled or not done.is_set() or
bool(exception))` at line 217, so failed, interrupted or unfinished work
preserves its current percentage. Normal join and worker error propagation
remain intact. The fake's `finish` now accepts the real `dirty` argument.

`finish` is the only call that draws 100%. The explicit `bar.update(100)` is
gone: on success `finish` drew 100% a second time, which printed two 100%
lines in redirected output. A seventh regression captures that output and
counts one 100% line.

A sixth real-library regression raises KeyboardInterrupt on the consumer's
polling update. It proves that interruption returns while the worker is still
alive and the bar has not rendered 100%; teardown releases and joins the
worker. After the fix, this regression file and the existing progress tests
reported **21 passed**. The root agent also checked the progress/fake suite.
Ruff and targeted pyrefly passed for the regression file.

## Model and properties

`tla/TimedProgress.tla` has separate worker phases for executing `fn`,
publishing the result/error box, setting the Event and exiting the thread.
The consumer has Event-wait, timeout-poll, finish, join and report phases. Empty output, nonempty output, Exception and BaseException are
separate abstract outcomes. This preserves the distinction between callable
completion, notification and actual worker exit.

The library Event and Thread operations are atomic contracts. A wait returns
through timeout or notification and rereads the Event predicate. Spurious
wakes have no fairness. Worker and consumer internal actions have weak
fairness; interrupt choices do not. Normal execution assumes a terminating
callable and positive finite polling/join intervals. There is one worker in
the shipped protocol, so a 1/2/3-worker or slot matrix would change its API
rather than check a boundary.

The six checked properties are `TypeOK`, `HonestCompletion`,
`JoinedBeforeReport`, `OutcomeDelivered`, `Ownership` and `Terminates`.
They check state domains, truthful completion rendering, normal reports
following worker exit, correct success/error routing, output ownership
transfer and eventual report/cancellation. Consumer cancellation permits a
live daemon worker intentionally. Output/errors retain their source ownership
until publication, then belong to the box and finally to the caller.

The model does not prove that LAPACK returns, floating-point accuracy, Python
allocator success or library memory ordering. Output is an abstract value,
and caller-held arrays/exception tracebacks are allowed. No application
mutex surrounds callable execution, result/error list allocation or an exit;
Event and Thread implementation locks are outside the calling-code model.
Thread startup failure before polling is excluded because no already-started
worker is left by the source's single-thread startup.

The finish path assumes rendering succeeds or raises the supported OSError.
An arbitrary non-OSError exception from a custom widget or output stream can
escape finish before the source's join loop. No separate issue was claimed
without demonstrating such an exception from the configured widgets and
supported stdout contract. That exceptional join-bypass path is an explicit
limitation of this model. Thread-start/result-box allocator failures and
asynchronous interrupts during library internals are also outside it.

The commands that recheck this model are in [TESTING.md](TESTING.md#4-formal-models). The deliberate bugs its checks must catch are listed in `tla/TimedProgress.mutations`.
