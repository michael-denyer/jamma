# Matrix writer formal verification

| Target | Threads and shared state | Wait and wake | Terminal ownership | Verdict |
| --- | --- | --- | --- | --- |
| Native writer, `_native_matrix_writer.py:65-99` | Foreground and 1/2/3 executor workers; pending Future/buffer deque, next block, output order | `future.result()` waits for completion; shutdown waits for borrowed arguments | Every submitted block is written or released, then foreground closes file | 57 PASS runs with the completion barrier; original executor violates `JoinedAtReturn` during shutdown interruption |
| Publication, `atomic_publish.py:68-80` | Foreground alone owns temporary file | No application wait | Publish complete success or discard failure | Included in writer model; premature publication mutation fails |

`tla/MatrixWriter.tla` follows the calling code. Library Future/executor operations are atomic transitions. `pc="wait"` represents Future blocking. Completion enables the next foreground step. The shared executor's completion Condition needs its own protocol model; this writer model assumes that contract. Pending references and worker references alias the same buffer, while each worker has exclusive mutation rights until completion.

The model checks types, buffer ownership, queue membership, FIFO output, formatting error order, no output after errors, complete publication, no outstanding borrows at return, and eventual result/shutdown return. Foreground and worker steps have weak fairness; interrupts have no fairness. Each submitted formatter eventually terminates. This is an eager function, so foreground fairness excludes the caller voluntarily abandoning its execution forever.

Boundary cases cover 0/1/3/5 blocks; 1/2/3 workers; 1/2 slots and the shipped `2*workers` slot count; no formatting failure; failure at every block; and a later failure at block 3. Submission/allocation failure, write failure and interrupts may occur at the corresponding foreground phases. Shutdown may be interrupted while workers run. Smaller slot counts are hypothetical reductions. Effective worker count 1 and empty inputs use serial code in production. These rows test the calling-protocol design rather than reachable executor configurations.

## Counterexample and reproduction

Before the fix, a worker starts after submission. A submission or formatting failure enters shutdown. SIGINT interrupts `ThreadPoolExecutor.shutdown(wait=True)` while that worker still formats. The exception exits the file scope and returns to the caller with the matrix still borrowed. The reduced model trace is `Submit`, `ExternalError`, `Start`, `Shutdown`, `AbortJoin`, `Close`. `JoinedAtReturn` fails. Current writer source events map to lines 75, 83-84, 34-35 and 99; context exit maps to line 54 and AtomicOutput lines 68-80. The original shutdown call was line 96.

This occurs at the shipped configuration, with 2 workers and 4 slots, and needs no reduced slot count. `tests/test_native_matrix_writer_interrupt.py` uses actual executor threads, a blocked formatter and actual SIGINT observed during shutdown. Before the fix it failed with `writer returned while formatter borrowed matrix`. After the fix it passed. The prior destination survives both paths.

The fix uses `JoiningThreadPoolExecutor` to record task completion separately from thread join, drain the borrows despite interrupts, then reraise the interrupt. Repeated joins alone cannot establish this on affected CPython versions, where an interrupted join can mark a still-running thread stopped.

## Commands

```sh
JAVA=/opt/homebrew/opt/openjdk@21/bin/java TLC_WORKERS=2 bash /Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify/scripts/tlc-matrix.sh /Users/mdenyer/VSCode/jamma/tla/MatrixWriter.matrix
.venv/bin/pytest tests/test_native_matrix_writer_interrupt.py -x
```

Use a working JDK on other hosts. Concurrent TLC processes should use different TMPDIR directories because they extract standard modules there.

## Mutation checks

Each mutation ran on a separate temporary copy. The checked-in model is unmutated. All these verdicts are expected failures.

| Mutation | Detected property |
| --- | --- |
| Pop pending as a stack | `DeliveryOrder` |
| Assign every live job buffer 1 | `Ownership` |
| Remove worker wait | `Ownership` |
| Publish despite errors | `PublishComplete` |
| Return from interrupted shutdown before the barrier | `JoinedAtReturn` |
| Keep a deque entry after handing it to foreground | `Ownership` |
| Observe failing block 3 before earlier blocks | `ErrorOrder` |
| Increment the next-item index by 2 | `TypeOK` |
| Enter write state after reporting formatting failure | `NoDeliveryAfterError` |
| Retain pending references after shutdown | `NoLeakAtReturn` |
| Remove worker completion | `Terminates`, `ResultReturns`, and `CloseReturns`, each checked independently |

## Assumptions and limits

This model checks block identities, not floating-point formatting, buffer capacities, short writes, disk durability or memory ordering below library synchronization. Input matrices remain stable. Ordinary cleanup succeeds, and no hostile process replaces sibling temporary paths. File close and publication errors collapse to unsuccessful publication. The model does not prove removal succeeds when the OS rejects cleanup. The process writer fallback and asynchronous exceptions inside shared-executor submission bookkeeping require separate checks.

## Full matrix results

57 PASS runs, 53,473 total distinct states across runs. Counts are per configuration, not a union.

| Run label | Distinct states | Verdict |
| --- | ---: | --- |
| `w=1 s=1 n=0 fail=none` | 14 | PASS |
| `w=1 s=1 n=1 fail=none` | 61 | PASS |
| `w=1 s=1 n=1 fail=all` | 46 | PASS |
| `w=1 s=1 n=3 fail=none` | 133 | PASS |
| `w=1 s=1 n=3 fail=all` | 46 | PASS |
| `w=1 s=1 n=5 fail=none` | 205 | PASS |
| `w=1 s=1 n=5 fail=all` | 46 | PASS |
| `w=1 s=2 n=0 fail=none` | 14 | PASS |
| `w=1 s=2 n=1 fail=none` | 61 | PASS |
| `w=1 s=2 n=1 fail=all` | 46 | PASS |
| `w=1 s=2 n=3 fail=none` | 273 | PASS |
| `w=1 s=2 n=3 fail=all` | 124 | PASS |
| `w=1 s=2 n=5 fail=none` | 457 | PASS |
| `w=1 s=2 n=5 fail=all` | 124 | PASS |
| `w=2 s=1 n=0 fail=none` | 14 | PASS |
| `w=2 s=1 n=1 fail=none` | 72 | PASS |
| `w=2 s=1 n=1 fail=all` | 57 | PASS |
| `w=2 s=1 n=3 fail=none` | 160 | PASS |
| `w=2 s=1 n=3 fail=all` | 57 | PASS |
| `w=2 s=1 n=5 fail=none` | 248 | PASS |
| `w=2 s=1 n=5 fail=all` | 57 | PASS |
| `w=2 s=2 n=0 fail=none` | 14 | PASS |
| `w=2 s=2 n=1 fail=none` | 72 | PASS |
| `w=2 s=2 n=1 fail=all` | 57 | PASS |
| `w=2 s=2 n=3 fail=none` | 413 | PASS |
| `w=2 s=2 n=3 fail=all` | 197 | PASS |
| `w=2 s=2 n=5 fail=none` | 707 | PASS |
| `w=2 s=2 n=5 fail=all` | 197 | PASS |
| `w=2 s=4 n=0 fail=none` | 14 | PASS |
| `w=2 s=4 n=1 fail=none` | 72 | PASS |
| `w=2 s=4 n=1 fail=all` | 57 | PASS |
| `w=2 s=4 n=3 fail=none` | 870 | PASS |
| `w=2 s=4 n=3 fail=all` | 616 | PASS |
| `w=2 s=4 n=5 fail=none` | 3,977 | PASS |
| `w=2 s=4 n=5 fail=all` | 1,809 | PASS |
| `w=3 s=1 n=0 fail=none` | 14 | PASS |
| `w=3 s=1 n=1 fail=none` | 83 | PASS |
| `w=3 s=1 n=1 fail=all` | 68 | PASS |
| `w=3 s=1 n=3 fail=none` | 187 | PASS |
| `w=3 s=1 n=3 fail=all` | 68 | PASS |
| `w=3 s=1 n=5 fail=none` | 291 | PASS |
| `w=3 s=1 n=5 fail=all` | 68 | PASS |
| `w=3 s=2 n=0 fail=none` | 14 | PASS |
| `w=3 s=2 n=1 fail=none` | 83 | PASS |
| `w=3 s=2 n=1 fail=all` | 68 | PASS |
| `w=3 s=2 n=3 fail=none` | 591 | PASS |
| `w=3 s=2 n=3 fail=all` | 292 | PASS |
| `w=3 s=2 n=5 fail=none` | 1,027 | PASS |
| `w=3 s=2 n=5 fail=all` | 292 | PASS |
| `w=3 s=6 n=0 fail=none` | 14 | PASS |
| `w=3 s=6 n=1 fail=none` | 83 | PASS |
| `w=3 s=6 n=1 fail=all` | 68 | PASS |
| `w=3 s=6 n=3 fail=none` | 1,503 | PASS |
| `w=3 s=6 n=3 fail=all` | 1,138 | PASS |
| `w=3 s=6 n=5 fail=none` | 18,895 | PASS |
| `w=3 s=6 n=5 fail=all` | 13,516 | PASS |
| `later-failure` | 3,723 | PASS |
