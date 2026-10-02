# Chunk pipeline formal verification

| Target | Threads, shared state and waits | Terminal states and resources | Verdict |
| --- | --- | --- | --- |
| `_overlapped_chunks` / `_drive_pipeline`, `chunk_pipeline.py:98-186`; `_ChunkEngine`, `chunk_runner_numpy.py:266-339` | Foreground and one background worker; chunk counter, reusable rotation buffers, current chunk and future. Foreground waits on Future.result; executor close waits for background callable completion. Future completion wakes its waiter. | Completed, failed or interrupted driver. Background rotation must end before process-wide BLAS restoration. Buffer leases belong to the current compute, in-flight prepare or future result, or are reusable. | Original interrupted cleanup failed; with the completion barrier every two-buffer configuration passes. The one-buffer boundary fails as expected. |
| Buffer modulo and chunk geometry, `chunk_runner_numpy.py:304-305`, `chunk_sizing.py:256-267,288-293` | Sequential Python integer arithmetic; pipeline always receives two buffers from its plan. | Distinct slots for adjacent prepared chunks. | Arithmetic proof belongs to the separate Lean target; protocol model detects unsafe one-buffer use. |

The shipped pipeline has exactly one background worker and two rotation buffers.
Two or three pipeline workers are outside this protocol's callers. The helper's
separate model checks one, two and three executor workers.

## Checked cases

`FailAt` and `ComputeFailAt` allow a failure or success at every listed item;
`prepare-all` therefore explores failures at later items as well as the first.
The properties cover type bounds, ownership and item partition, exact delivery
order, duplicate writes, producer-error ordering, no live leases at closed,
BLAS lifetime, no output after end/error, return from each blocking call, close
termination, error cleanup and full-drive termination.

## Counterexamples and reproduction

The reachable original-code trace is initial prepare, successor submission,
background rotation, foreground compute failure, executor join, SIGINT during
join, then BLAS restoration while rotation is still active. The submit maps to
`chunk_pipeline.py:128`; rotation maps to `chunk_runner_numpy.py:307-313`;
compute maps to `chunk_pipeline.py:185`; executor scope and BLAS restoration
map to `chunk_pipeline.py:170-173`. Before the fix the executor at that site
was the standard library ThreadPoolExecutor.

The mutation "interrupted join returns while the worker still rotates" in
`tla/ChunkPipeline.mutations` restores that executor. With two buffers, three
chunks, a compute failure at chunk 1 and an interrupted join it violates
`BlasLifetime`. This failure does not require one-buffer geometry or a
non-shipped worker count.

`tests/test_chunk_pipeline_interrupt.py` delivers a real SIGINT in a child
process only after foreground failure enters executor cleanup. It holds the
background callable on an event. Originally it observed BLAS restore and driver
exit before the callable finished. The fixed helper keeps the limit active until
callable completion, then propagates KeyboardInterrupt. The numerical kernels
are outside this test; the fake BLAS controller models process-wide hardware
state and the engine models the driver's prepare/compute interface.

The fix replaces the executor with `JoiningThreadPoolExecutor`, whose separate
Condition model proves the completion barrier. Retrying Thread.join alone is
insufficient on affected CPython versions because an interrupted join can mark
the thread stopped while its target still runs.

The one-buffer trace prepares item 1 in slot 1, submits item 2, then item 2
writes slot 1 while item 1 still belongs to compute. The exact allocation is
`chunk_runner_numpy.py:304-313`. This is unreachable through `LmmChunkPlan`,
which selects two buffers whenever it selects pipelining. Changing that policy
would make the counterexample reachable. No production guard or fix is added
for this prohibited internal configuration.

## Assumptions and limits

Rotation, compute, raw-source iteration and sinks eventually return or raise.
Failures after a partial phenotype write are not transactional rollbacks. The
model has one abstract consumer; each real phenotype runs sequentially against
the same prepared buffer, so partial multi-phenotype output is outside its
order property. Floating-point values, GIL/native memory ordering, BLAS
implementation behavior and native crashes require runtime checks.

The public API drives the whole iterator. Arbitrary idle `next`/close choices
in the model are conservative wrapper states. If a consumer stops making
requests forever, end-of-input is not promised; `NoCloseReachesEnd` assumes
continued requests, as the production for-loop does. Future and executor
internals are abstract here; the helper's own Condition is modelled separately.
Buffer ownership means logical leases, not freeing engine-owned NumPy arrays.

The commands that recheck this model are in [TESTING.md](TESTING.md#4-formal-models). The deliberate bugs its checks must catch are listed in `tla/ChunkPipeline.mutations`.
