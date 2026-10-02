# Chunk and matrix arithmetic verification

| Target | Source and integer domain | Property | Verdict |
|---|---|---|---|
| Chunk buffer slot | `chunk_runner_numpy.py:304-305`, Python unbounded nonnegative counter and positive buffer count | Modulo slot is in bounds; adjacent counters use different slots when buffers >= 2 | PASS, 7 declarations audited |
| Matrix blocks | `_native_matrix_writer.py:48-52`, Python unbounded rows >= 0, columns > 0 and requested workers > 0 | Positive row block, contiguous clipped slices, strictly decreasing remainder, byte capacity bound, worker and buffer caps | PASS, 17 declarations audited |

## Checker results and assumptions

```text
PASS /Users/mdenyer/VSCode/jamma/lean/ChunkSlots: no sorry, no extra axioms, 7 declarations audited
PASS /Users/mdenyer/VSCode/jamma/lean/MatrixBlocks: no sorry, no extra axioms, 17 declarations audited
```

Both projects pin Lean 4.34.1 and have no external packages. Every declaration
was audited; there are no admitted proofs or additional axioms. Natural numbers
match the nonnegative Python inputs under the stated domain. Python arithmetic
has no fixed-width overflow. These proofs make no statement about the C
formatter's `size_t` arithmetic or floating-point numerical results.

The slot search exhausts 0-20 counters and 1-8 buffers, above the shipped two
slots. The block transition search exhausts row counts and block sizes 0-19
and every in-range start. Column counts through 65,537 cover the shipped
65,536-value block constant and its wider-row boundary. Worker search exhausts
rows/block sizes 0-20 and requests 0-4. Each search has `#eval` counterexample
output and a rejecting `#guard`. Theorems establish their properties for every
size under explicit positivity/bounds hypotheses.

The slice proof clips the last `range(0, rows, block)` endpoint to `rows`, as
NumPy slicing does. It proves that the next slice begins at the previous end
and that each nonempty iteration reduces the remaining rows. The capacity
proof assumes formatting each float requires no more than 32 bytes; native
format correctness is outside Lean. Protocol models separately check leases
on the two chunk buffers and the writer's `2*workers` bytearrays.

## Mutation sensitivity

Every mutation runs in an isolated project copy. All seven mutations cause
both a bounded property failure and a rejected proof or explicit boundary
guard. The real source has none of these mutations, so no arithmetic fix is
proposed.

The runner first requires both untouched projects to pass the checker. Each
mutation names its bounded property guard; detection requires Lean's
evaluated-false diagnostic for that expression at its current source location.
An unrelated checker, import or syntax error fails the run. Checks have a
60-second timeout. The CLI regressions in `tests/test_lean_mutations.py` use
an external checker fake, so they run in ordinary CI without a Lean installation.

| Mutation | Concrete counterexample | Detecting property |
|---|---|---|
| Add buffer count to modulo result | counter=0, buffers=1 produces slot 1 | Slot bound |
| Divide counter by two before modulo | counters 0 and 1, buffers=2 both use slot 0 | Adjacent-slot separation |
| Remove minimum block size | values=65,536, columns=65,537 gives block 0 | Positive block / bounded column search |
| Advance by block minus one | rows=1, block=1, start=0 makes no progress | Exact slice length and decreasing remainder |
| Allocate one fewer row | rows=1, block=1, columns=1 requires 32 bytes, capacity becomes 0 | Slice capacity |
| Replace worker min with max | rows=2, block=1, request=1 yields 2 workers | Worker and buffer caps |
| Use floor division for chunk count | rows=1, block=2, request=1 yields 0 workers | Positive worker count |

Existing runtime checks cover chunk sizing, buffer lifetime and matrix output.
The executor/chunk/writer test group passed 66 tests. Proof models are
transcriptions, so the source references and caller assumptions must be
maintained when those functions change.

## Commands

From the repository root:

```sh
VERIFY_SKILL=/Users/mdenyer/.codex/plugins/cache/agent-formal-verify/agent-formal-verify/0.1.11/skills/formal-verify
bash "$VERIFY_SKILL/scripts/setup.sh" lean "$PWD/lean/ChunkSlots"
bash "$VERIFY_SKILL/scripts/setup.sh" lean "$PWD/lean/MatrixBlocks"
bash "$VERIFY_SKILL/scripts/lean-check.sh" "$PWD/lean/ChunkSlots"
bash "$VERIFY_SKILL/scripts/lean-check.sh" "$PWD/lean/MatrixBlocks"
python lean/check_mutations.py "$VERIFY_SKILL/scripts/lean-check.sh"
```

No CI job was added. Keep existing sanitizer and numerical validation coverage;
these integer proofs do not replace either.
