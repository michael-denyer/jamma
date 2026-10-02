import Std.Tactic
/-!
Source: src/jamma/io/_native_matrix_writer.py:48-52,55-61,66-92.
Python integer arithmetic with rows >= 0, columns > 0, n_workers > 0.
Rows and block sizes use Nat because callers reject negative dimensions.
The native formatter's byte bound is an assumption, not proved here.
-/

def block_rows (values columns : Nat) : Nat := max 1 (values / columns)
def next_start (rows block start : Nat) : Nat := min rows (start + block)
def slice_rows (rows block start : Nat) : Nat := min block (rows - start)
def capacity (rows block columns bytes : Nat) : Nat := min rows block * columns * bytes

def StepOK (rows block start : Nat) : Prop :=
  next_start rows block start ≤ rows ∧
  start ≤ next_start rows block start ∧
  next_start rows block start - start = slice_rows rows block start ∧
  (start < rows → start < next_start rows block start) ∧
  slice_rows rows block start ≤ min rows block
instance (rows block start : Nat) : Decidable (StepOK rows block start) := by
  unfold StepOK; infer_instance

def badSteps : List (Nat × Nat × Nat) :=
  (List.range 20).flatMap fun rows =>
    (List.range 20).flatMap fun block =>
      (List.range (rows + 1)).filterMap fun start =>
        if 0 < block ∧ (¬ StepOK rows block start ∨
          slice_rows rows block start * 32 > capacity rows block 1 32) then some (rows, block, start) else none
#eval badSteps
#guard badSteps.isEmpty
#guard block_rows 65536 65537 = 1
#guard block_rows 65536 1 = 65536
#guard capacity 3 65536 1 32 = 96

def badBlockSizes : List Nat :=
  (List.range 65538).filter fun columns =>
    0 < columns && block_rows 65536 columns == 0
#eval badBlockSizes
#guard badBlockSizes.isEmpty

theorem block_rows_positive (values columns : Nat) : 0 < block_rows values columns := by
  unfold block_rows; omega

theorem block_step (rows block start : Nat) (hb : 0 < block) (hs : start ≤ rows) :
    StepOK rows block start := by
  unfold StepOK next_start slice_rows
  omega

theorem slice_fits_capacity (rows block start columns bytes : Nat) (hs : start ≤ rows) :
    slice_rows rows block start * columns * bytes ≤ capacity rows block columns bytes := by
  have h : slice_rows rows block start ≤ min rows block := by unfold slice_rows; omega
  exact Nat.mul_le_mul_right bytes (Nat.mul_le_mul_right columns h)

/-- Each successful iteration decreases the remaining row count. This is the
termination measure for Python range(0, rows, block), with no skipped rows. -/
theorem remaining_decreases (rows block start : Nat) (hb : 0 < block) (hs : start < rows) :
    rows - next_start rows block start < rows - start := by
  unfold next_start; omega

/-- The same ceiling division and cap used at writer line 52. -/
def workers (rows block requested : Nat) : Nat :=
  min requested ((rows + block - 1) / block)

theorem workers_bounded (rows block requested : Nat) :
    workers rows block requested ≤ requested := Nat.min_le_left _ _

theorem workers_positive (rows block requested : Nat)
    (hr : 0 < rows) (hb : 0 < block) (hw : 0 < requested) :
    0 < workers rows block requested := by
  have h : 0 < (rows + block - 1) / block := Nat.div_pos (by omega) hb
  unfold workers; omega

theorem buffers_bounded (rows block requested : Nat) :
    2 * workers rows block requested ≤ 2 * requested :=
  Nat.mul_le_mul_left 2 (workers_bounded rows block requested)

#guard workers 0 65536 3 = 0
#guard workers 65537 65536 3 = 2


def badWorkerCounts : List (Nat × Nat × Nat) :=
  (List.range 21).flatMap fun rows =>
    (List.range 21).flatMap fun block =>
      (List.range 5).filterMap fun requested =>
        if 0 < rows ∧ 0 < block ∧ 0 < requested ∧
          (workers rows block requested = 0 ∨ requested < workers rows block requested)
        then some (rows, block, requested) else none
#eval badWorkerCounts
#guard badWorkerCounts.isEmpty
