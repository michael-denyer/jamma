---- MODULE ChunkPipeline ----
(* Copyright (c) 2026 Michael Denyer
   SPDX-License-Identifier: GPL-3.0-only

   src/jamma/lmm/chunk_pipeline.py:98-186 and
   src/jamma/lmm/chunk_runner_numpy.py:266-372.
   LmmChunkPlan enforces two buffers for pipelines (chunk_sizing.py:256-267,
   288-293). Slots=1 intentionally checks the unsafe boundary beneath that policy.

   Foreground initial prepare, successor submit, compute and Future.result
   are separate steps. Background allocation and unlocked rotation are separate.
   Library Future synchronization and the completion barrier proved in
   JoiningThreadPool.tla are atomic here; this is not a model of
   their condition variables. No hand-written mutex occurs in this protocol.
   No allocations/exit/longjmp run under a calling-code lock. Python errors
   unwind into executor shutdown; join precedes process-wide BLAS restoration.

   'idle' is the consumer boundary after a yielded item, before compute.
   Close choices cover before first next, yielded idle, after end and after
   error; compute/prepare failures also trigger cleanup. A consumer may stop
   requesting indefinitely. NoCloseReachesEnd therefore assumes continued
   requests, as the production driver's for-loop does. Close is a conservative
   wrapper abstraction: the production API offers full drive, not public next.
   Weak fairness applies to every deterministic foreground/background phase,
   not to request/close choices. No mutex-starvation assumption is needed.
   Finiteness of raw iteration, rotation, compute and sink calls is assumed.
   Buffer ownership here is logical lifetime; actual arrays are GC-managed and
   belong to the engine beyond executor close. NoFreeLeak means no live lease.
   ChunkPipeline.mutations lists the bugs the properties detect.
*)
EXTENDS Naturals, Sequences, FiniteSets
CONSTANTS Slots, Items, FailAt, ComputeFailAt, InterruptJoin
ASSUME Slots >= 1 /\ Items >= 0
VARIABLES fg, bg, nextItem, current, future, building, owner, contents,
          allocated, output, discarded, report, closeCalled, blasHeld
leases == <<current, future, building, owner>>
source == <<nextItem, contents, allocated>>
results == <<output, discarded, report>>
data == <<leases, source, results>>
lifecycle == <<closeCalled, blasHeld>>
vars == <<fg, bg, data, lifecycle>>
Slot(i) == ((i - 1) % Slots) + 1
OutputSet == {output[k] : k \in 1..Len(output)}
Held == ({current, future, building} \ {0})
Settled == {"ready", "error", "end"}
Blocked == fg \in {"initial", "compute", "await", "join"}
Init == /\ fg = "new" /\ bg = "off" /\ nextItem = 1
        /\ current = 0 /\ future = 0 /\ building = 0
        /\ owner = [s \in 1..Slots |-> "free"]
        /\ contents = [s \in 1..Slots |-> 0]
        /\ allocated = {} /\ output = <<>> /\ discarded = {}
        /\ report = "none" /\ closeCalled = FALSE /\ blasHeld = TRUE
\* Read the next item into its slot and lease that slot to `role`.
Lease(role) == /\ nextItem' = nextItem + 1
               /\ owner' = [owner EXCEPT ![Slot(nextItem)] = role]
               /\ contents' = [contents EXCEPT ![Slot(nextItem)] = nextItem]
               /\ allocated' = allocated \cup {nextItem}
Start == /\ fg = "new" /\ fg' = "initial"
         /\ UNCHANGED <<bg, data, lifecycle>>
Initial == \E failChoice \in BOOLEAN :
           /\ fg = "initial"
           /\ IF nextItem > Items
                 THEN /\ fg' = "ended" /\ report' = "end"
                      /\ UNCHANGED <<current, owner, source>>
                 ELSE IF failChoice /\ nextItem \in FailAt
                      THEN /\ fg' = "error" /\ report' = "prepare-error"
                           /\ UNCHANGED <<current, owner, source>>
                      ELSE /\ fg' = "submit" /\ current' = nextItem
                           /\ Lease("current")
                           /\ UNCHANGED report
           /\ UNCHANGED <<bg, future, building, output, discarded, lifecycle>>
Submit == /\ fg = "submit" /\ bg = "off"
          /\ bg' = "queued" /\ fg' = "idle"
          /\ UNCHANGED <<data, lifecycle>>
Request == /\ fg = "idle" /\ fg' = "compute"
           /\ UNCHANGED <<bg, data, lifecycle>>
Compute == \E failChoice \in BOOLEAN :
           /\ fg = "compute"
           /\ IF failChoice /\ current \in ComputeFailAt
                 THEN /\ report' = "compute-error" /\ fg' = "error"
                      /\ discarded' = discarded \cup {current}
                      /\ UNCHANGED output
                 ELSE /\ output' = Append(output, contents[Slot(current)])
                      /\ fg' = "await" /\ UNCHANGED <<report, discarded>>
           /\ owner' = [owner EXCEPT ![Slot(current)] = "free"]
           /\ current' = 0
           /\ UNCHANGED <<bg, future, building, source, lifecycle>>
BeginPrepare == /\ bg = "queued"
                /\ IF nextItem > Items
                      THEN /\ bg' = "end" /\ UNCHANGED <<building, owner, source>>
                      ELSE /\ building' = nextItem /\ bg' = "rotate"
                           /\ Lease("background")
                /\ UNCHANGED <<fg, current, future, results, lifecycle>>
FinishPrepare == \E failChoice \in BOOLEAN :
                 /\ bg = "rotate"
                 /\ IF failChoice /\ building \in FailAt
                       THEN /\ bg' = "error" /\ discarded' = discarded \cup {building}
                            /\ owner' = [owner EXCEPT ![Slot(building)] = "free"]
                            /\ UNCHANGED future
                       ELSE /\ bg' = "ready" /\ future' = building
                            /\ owner' = [owner EXCEPT ![Slot(building)] = "future"]
                            /\ UNCHANGED discarded
                 /\ building' = 0
                 /\ UNCHANGED <<fg, current, source, output, report, lifecycle>>
Await == /\ fg = "await" /\ bg \in Settled
         /\ IF bg = "ready"
               THEN /\ current' = future /\ future' = 0 /\ fg' = "submit"
                    /\ owner' = [owner EXCEPT ![Slot(future)] = "current"]
                    /\ UNCHANGED report
               ELSE /\ current' = 0 /\ future' = 0
                    /\ fg' = IF bg = "error" THEN "error" ELSE "ended"
                    /\ report' = IF bg = "error" THEN "prepare-error" ELSE "end"
                    /\ UNCHANGED owner
         /\ bg' = "off"
         /\ UNCHANGED <<building, source, output, discarded, lifecycle>>
Close == /\ fg \in {"new", "idle", "ended", "error"}
         /\ fg' = "join" /\ closeCalled' = TRUE
         /\ UNCHANGED <<bg, data, blasHeld>>
AutoUnwind == /\ fg \in {"ended", "error"} /\ fg' = "join"
              /\ UNCHANGED <<bg, data, lifecycle>>
Join == /\ fg = "join" /\ bg \in Settled \cup {"off"}
        /\ discarded' = discarded \cup ({current, future} \ {0})
        /\ current' = 0 /\ future' = 0 /\ bg' = "off" /\ fg' = "restore"
        /\ owner' = [s \in 1..Slots |-> "free"]
        /\ UNCHANGED <<building, source, output, report, lifecycle>>
(* SIGINT during the executor's __exit__. The completion barrier retries the
   join, so the foreground stays there until the worker's callable completes. *)
InterruptedJoin == /\ InterruptJoin /\ fg = "join" /\ report = "compute-error" /\ bg = "rotate"
                   /\ fg' = "join"
                   /\ UNCHANGED <<bg, data, lifecycle>>
Restore == /\ fg = "restore" /\ fg' = "closed" /\ blasHeld' = FALSE
           /\ UNCHANGED <<bg, data, closeCalled>>
Done == fg = "closed" /\ UNCHANGED vars
Next == Start \/ Initial \/ Submit \/ Request \/ Compute \/ BeginPrepare \/
        FinishPrepare \/ Await \/ Close \/ AutoUnwind \/ Join \/ InterruptedJoin \/ Restore \/ Done
Spec == Init /\ [][Next]_vars /\ WF_vars(Start) /\ WF_vars(Initial) /\
        WF_vars(Submit) /\ WF_vars(Compute) /\ WF_vars(BeginPrepare) /\
        WF_vars(FinishPrepare) /\ WF_vars(Await) /\ WF_vars(AutoUnwind) /\
        WF_vars(Join) /\ WF_vars(Restore)
TypeOK == /\ fg \in {"new", "initial", "submit", "idle", "compute", "await", "error", "ended", "join", "restore", "closed"}
          /\ bg \in {"off", "queued", "rotate", "ready", "error", "end"}
          /\ nextItem \in 1..(Items+1)
          /\ current \in 0..Items /\ future \in 0..Items /\ building \in 0..Items
          /\ owner \in [1..Slots -> {"free", "current", "background", "future"}]
          /\ contents \in [1..Slots -> 0..Items]
          /\ allocated \subseteq 1..Items /\ discarded \subseteq 1..Items
          /\ output \in Seq(1..Items)
          /\ report \in {"none", "end", "prepare-error", "compute-error"}
          /\ closeCalled \in BOOLEAN /\ blasHeld \in BOOLEAN
Ownership == /\ current # 0 => owner[Slot(current)] = "current"
             /\ future # 0 => owner[Slot(future)] = "future"
             /\ building # 0 => owner[Slot(building)] = "background"
             /\ Cardinality(Held) = Cardinality({Slot(i) : i \in Held})
Partition == /\ allocated = OutputSet \cup discarded \cup Held
             /\ OutputSet \cap discarded = {} /\ OutputSet \cap Held = {}
             /\ discarded \cap Held = {}
Order == \A k \in 1..Len(output) : output[k] = k
NoDuplicate == Cardinality(OutputSet) = Len(output)
ErrorOrder == report = "prepare-error" => Len(output) = IF nextItem = 1 THEN 0 ELSE nextItem - 2
NoLeak == fg = "closed" => /\ Held = {} /\ \A s \in 1..Slots : owner[s] = "free"
BlasLifetime == ~blasHeld => /\ bg = "off" /\ building = 0 /\ fg = "closed"
NoOutputAfterTerminal == [][(report # "none" => output' = output)]_vars
BlockingReturns == [](Blocked => <>(~Blocked))
CloseTerminates == [](closeCalled => <>(fg = "closed"))
UnwindTerminates == [](report # "none" => <>(fg = "closed"))
NoCloseReachesEnd == (([](~closeCalled) /\ WF_vars(Request)) => <>(fg = "closed"))
====
