# JAMMA protocol verification audit

The audit selected five existing concurrency protocols and two integer targets.
The shared joining-executor protocol was added and checked as part of fixing
interrupted cleanup. Four issue PRs keep each fix with its models, deterministic
regressions and verification report. The models are on independent branches
until those PRs merge; this overview records the combined audit evidence.

| Target | Actors, state, waits and resource ownership | Verdict | Evidence |
|---|---|---|---|
| Chunk rotation handover | Foreground and one executor worker; current/future, chunk counter and two rotation buffers; result wait, then cleanup before BLAS restore | 17 PASS, 1,158 summed states; 13 mutations detected | [PR #495](https://github.com/michael-denyer/jamma/pull/495) |
| LOCO eigen stream | Consumer and 1/2/3 daemon workers; FIFO queue/pending Futures, matrix/eigenvector leases, BLAS scope; result wait, sentinels and join on normal close | 15 PASS, 20,118 states; 15 mutations detected | [PR #493](https://github.com/michael-denyer/jamma/pull/493) |
| Native matrix writer | Consumer and 1/2/3 workers; FIFO pending, 2*workers buffers and one file writer; future wait, worker drain before close/publication | 57 PASS, 53,473 states; 11 mutations detected | [PR #495](https://github.com/michael-denyer/jamma/pull/495) |
| Joining executor | One submit/close caller and 1/2/3 workers; Condition, token dictionary and queued/running callables; explicit notified/spurious wait and mutex recheck | 36 PASS, 712,608 states; 11 mutations detected | [PR #495](https://github.com/michael-denyer/jamma/pull/495) |
| Timed progress | Consumer and daemon worker; result/error boxes and Event; timed wait, stdout failure and normal join, intentional abandonment on caller interrupt | 8 PASS, 319 states; 9 mutations detected | Progress PR, this branch |
| Spawn text pool | Parent and 1/2/3 processes; ordered task results, original process handles and disjoint memmap slices; timed result poll, terminate/join before cleanup | 12 PASS, 59,367 states; 5 mutations detected | [PR #494](https://github.com/michael-denyer/jamma/pull/494) |
| Chunk modulo arithmetic | Python unbounded nonnegative counter and positive buffer count; `chunk_runner_numpy.py:304-305` | Lean PASS, 7 declarations audited; 2 mutations detected | [PR #495](https://github.com/michael-denyer/jamma/pull/495) |
| Matrix block arithmetic | Python unbounded nonnegative rows, positive columns/workers; `_native_matrix_writer.py:48-52` | Lean PASS, 17 declarations audited; 5 mutations detected | [PR #495](https://github.com/michael-denyer/jamma/pull/495) |
| SNP statistics column ranges | Disjoint pthread writes in `jlinalg/src/snp_stats.c:175-185` | Race/sanitizer target, no protocol model | Existing sanitizer coverage retained |

There are 145 passing production TLA+ runs and 847,043 summed distinct states.
A one-buffer chunk pipeline boundary fails ownership as expected, but shipped
`LmmChunkPlan` always provides two buffers when pipelining. That boundary is
retained, not reported as a production bug. Every checked property has a
relevant detected mutation. Counts are bounded-model states, not real-code
execution counts.

## Confirmed issues and reproductions

1. LOCO workers started before the cleanup guard. A later OS thread-start
   failure left earlier workers blocked indefinitely. BLAS entry failure
   could leak all workers. Startup now occurs inside guarded cleanup after
   successful BLAS entry. The original startup suite had three failures;
   the fixed startup/worker/resource suite passed 31 tests.
2. SIGINT during executor shutdown let native formatting keep borrowing its
   matrix after return and let background chunk rotation outlive BLAS restore.
   Actual-signal tests reproduced both traces. The joining helper drains
   callables and protects partial submission/cancellation. Its caller suite
   passed 66 tests, plus the enqueue/start-failure regression.
3. A dead process worker lost its ordered result while Pool replaced the
   process. `SystemExit` and abrupt exit reproductions hung in `imap`.
   Worker-health polling now reports the error and terminates/joins the pool.
   The targeted matrix I/O suite passed 59 tests.
4. Progress claimed 100% on worker errors, consumer cancellation or stdout
   failure with unfinished work. Real progressbar's default `finish()` caused
   an implicit completion redraw that earlier fakes did not record. Completed
   success is now checked explicitly and unsuccessful finalization is dirty.
   Real-library and existing progress tests pass.

Per-run tables, reduced trace mappings, mutation details and exact commands are
in each issue's `docs/formal-*.md` reports. Models cite their source files and
line numbers. The targeted tests and local commit/push hooks passed; GitHub CI
is tracked separately and is not represented by these local PASS counts.

## Rechecking

Run from the branch that owns a model, or after all issue PRs merge. Install the
formal-verify plugin and set its actual installation directory. On this Mac:

```sh
VERIFY_SKILL=/Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify
export JAVA=/opt/homebrew/opt/openjdk@21/bin/java
export TLC_WORKERS=2
bash "$VERIFY_SKILL/scripts/setup.sh" tla
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/LocoWorkers.matrix"
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/SpawnPool.matrix"
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/ChunkPipeline.matrix" slots=2
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/MatrixWriter.matrix"
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/JoiningThreadPool.matrix"
bash "$VERIFY_SKILL/scripts/tlc-matrix.sh" "$PWD/tla/TimedProgress.matrix"
bash "$VERIFY_SKILL/scripts/setup.sh" lean "$PWD/lean/ChunkSlots"
bash "$VERIFY_SKILL/scripts/setup.sh" lean "$PWD/lean/MatrixBlocks"
bash "$VERIFY_SKILL/scripts/lean-check.sh" "$PWD/lean/ChunkSlots"
bash "$VERIFY_SKILL/scripts/lean-check.sh" "$PWD/lean/MatrixBlocks"
python lean/check_mutations.py "$VERIFY_SKILL/scripts/lean-check.sh"
```

The unfiltered chunk matrix includes the expected one-buffer failure. Use a
separate existing JVM temporary directory for each concurrently run TLC matrix,
via `JAVA_TOOL_OPTIONS=-Djava.io.tmpdir=/absolute/unique/directory`.

## Limits

TLC checks calling protocols under abstract library queue/Future/executor
semantics. It does not establish memory ordering below synchronization, floating
point correctness, CPython internals, OS file-descriptor accounting or recovery
from catastrophic allocation failures during cleanup. Tasks must eventually
finish; termination assumes finite interruptions. The joining helper names
strong mutex-acquisition fairness because a diagnostic trace showed pure
starvation by endlessly spurious wakes. Safety does not depend on that fairness.

Lean uses unbounded nonnegative arithmetic under explicit input hypotheses. The
matrix capacity proof assumes 32 bytes suffice per formatted value; it does not
prove the native formatter. Neither Lean project has admitted proofs or extra
axioms. Existing numerical validation, sanitizers and race checks remain needed.
No CI job or numerical algorithm was changed.
