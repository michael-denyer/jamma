import Std.Tactic
/-!
Source: src/jamma/lmm/chunk_runner_numpy.py:304-305 and
src/jamma/lmm/chunk_sizing.py:256-267,288-293.
Python integers are unbounded. Counters are nonnegative and slot counts positive.
The shipped overlapped pipeline uses two slots. This proves arithmetic only;
TLA+ separately checks that the foreground relinquishes a slot before reuse.
-/

def slot (counter buffers : Nat) : Nat := counter % buffers

def Valid (counter buffers : Nat) : Prop :=
  slot counter buffers < buffers ∧
  (2 ≤ buffers → slot counter buffers ≠ slot (counter + 1) buffers)
instance (counter buffers : Nat) : Decidable (Valid counter buffers) := by
  unfold Valid; infer_instance

def badPairs : List (Nat × Nat) :=
  (List.range 9).flatMap fun buffers =>
    (List.range 21).filterMap fun counter =>
      if 0 < buffers ∧ ¬ Valid counter buffers then some (counter, buffers) else none
#eval badPairs
#guard badPairs.isEmpty

theorem slot_in_bounds (counter buffers : Nat) (h : 0 < buffers) :
    slot counter buffers < buffers := Nat.mod_lt _ h

theorem consecutive_slots_distinct (counter buffers : Nat) (h : 2 ≤ buffers) :
    slot counter buffers ≠ slot (counter + 1) buffers := by
  unfold slot
  have hb : 0 < buffers := by omega
  have hr := Nat.mod_lt counter hb
  have hone : 1 % buffers = 1 := Nat.mod_eq_of_lt (by omega)
  rw [Nat.add_mod, hone]
  by_cases hs : counter % buffers + 1 < buffers
  · rw [Nat.mod_eq_of_lt hs]
    omega
  · have heq : counter % buffers + 1 = buffers := by omega
    rw [heq, Nat.mod_self]
    omega

theorem valid_slot (counter buffers : Nat) (h : 0 < buffers) : Valid counter buffers :=
  ⟨slot_in_bounds counter buffers h, consecutive_slots_distinct counter buffers⟩
