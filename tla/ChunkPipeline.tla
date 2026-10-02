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
*)
EXTENDS Naturals, Sequences, FiniteSets
CONSTANTS Slots, Items, FailAt, ComputeFailAt, InterruptJoin, CompletionBarrier
ASSUME Slots >= 1 /\ Items >= 0
VARIABLES fg, bg, nextItem, current, future, building, owner, contents,
          allocated, output, discarded, report, closeCalled, blasHeld
vars == <<fg, bg, nextItem, current, future, building, owner, contents,
          allocated, output, discarded, report, closeCalled, blasHeld>>
Slot(i) == ((i - 1) % Slots) + 1
OutputSet == {output[k] : k \in 1..Len(output)}
Held == ({current, future, building} \ {0})
Init == /\ fg = "new" /\ bg = "off" /\ nextItem = 1
        /\ current = 0 /\ future = 0 /\ building = 0
        /\ owner = [s \in 1..Slots |-> "free"]
        /\ contents = [s \in 1..Slots |-> 0]
        /\ allocated = {} /\ output = <<>> /\ discarded = {}
        /\ report = "none" /\ closeCalled = FALSE /\ blasHeld = TRUE
Start == /\ fg = "new" /\ fg' = "initial"
         /\ UNCHANGED <<bg, nextItem, current, future, building, owner,
                        contents, allocated, output, discarded, report,
                        closeCalled, blasHeld>>
Initial == \E failChoice \in BOOLEAN :
           /\ fg = "initial"
           /\ IF nextItem > Items
                 THEN /\ fg' = "ended" /\ report' = "end"
                      /\ UNCHANGED <<nextItem, current, owner, contents, allocated>>
                 ELSE IF failChoice /\ nextItem \in FailAt
                      THEN /\ fg' = "error" /\ report' = "prepare-error"
                           /\ UNCHANGED <<nextItem, current, owner, contents, allocated>>
                      ELSE /\ fg' = "submit" /\ current' = nextItem
                           /\ nextItem' = nextItem + 1
                           /\ owner' = [owner EXCEPT ![Slot(nextItem)] = "current"]
                           /\ contents' = [contents EXCEPT ![Slot(nextItem)] = nextItem]
                           /\ allocated' = allocated \cup {nextItem}
                           /\ UNCHANGED report
           /\ UNCHANGED <<bg, future, building, output, discarded,
                          closeCalled, blasHeld>>
Submit == /\ fg = "submit" /\ bg = "off"
          /\ bg' = "queued" /\ fg' = "idle"
          /\ UNCHANGED <<nextItem, current, future, building, owner, contents,
                         allocated, output, discarded, report, closeCalled, blasHeld>>
Request == /\ fg = "idle" /\ fg' = "compute"
           /\ UNCHANGED <<bg, nextItem, current, future, building, owner, contents,
                          allocated, output, discarded, report, closeCalled, blasHeld>>
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
           /\ UNCHANGED <<bg, nextItem, future, building, contents, allocated,
                          closeCalled, blasHeld>>
BeginPrepare == /\ bg = "queued"
                /\ IF nextItem > Items
                      THEN /\ bg' = "end" /\ UNCHANGED <<building, nextItem, owner, contents, allocated>>
                      ELSE /\ building' = nextItem /\ nextItem' = nextItem + 1
                           /\ bg' = "rotate"
                           /\ owner' = [owner EXCEPT ![Slot(nextItem)] = "background"]
                           /\ contents' = [contents EXCEPT ![Slot(nextItem)] = nextItem]
                           /\ allocated' = allocated \cup {nextItem}
                /\ UNCHANGED <<fg, current, future, output, discarded, report, closeCalled, blasHeld>>
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
                 /\ UNCHANGED <<fg, nextItem, current, contents, allocated, output,
                                report, closeCalled, blasHeld>>
Await == /\ fg = "await" /\ bg \in {"ready", "error", "end"}
         /\ IF bg = "ready"
               THEN /\ current' = future /\ future' = 0 /\ fg' = "submit"
                    /\ owner' = [owner EXCEPT ![Slot(future)] = "current"]
                    /\ UNCHANGED report
               ELSE /\ current' = 0 /\ future' = 0
                    /\ fg' = IF bg = "error" THEN "error" ELSE "ended"
                    /\ report' = IF bg = "error" THEN "prepare-error" ELSE "end"
                    /\ UNCHANGED owner
         /\ bg' = "off"
         /\ UNCHANGED <<nextItem, building, contents, allocated, output, discarded,
                        closeCalled, blasHeld>>
Close == /\ fg \in {"new", "idle", "ended", "error"}
         /\ fg' = "join" /\ closeCalled' = TRUE
         /\ UNCHANGED <<bg, nextItem, current, future, building, owner, contents,
                        allocated, output, discarded, report, blasHeld>>
AutoUnwind == /\ fg \in {"ended", "error"} /\ fg' = "join"
              /\ UNCHANGED <<bg, nextItem, current, future, building, owner, contents,
                             allocated, output, discarded, report, closeCalled, blasHeld>>
Join == /\ fg = "join" /\ bg \in {"off", "ready", "error", "end"}
        /\ discarded' = discarded \cup ({current, future} \ {0})
        /\ current' = 0 /\ future' = 0 /\ bg' = "off" /\ fg' = "restore"
        /\ owner' = [s \in 1..Slots |-> "free"]
        /\ UNCHANGED <<nextItem, building, contents, allocated, output, report,
                       closeCalled, blasHeld>>
(* CompletionBarrier=TRUE is the current helper. FALSE reproduces the old
   ThreadPoolExecutor: SIGINT during __exit__ unwinds its outer BLAS context
   while a worker remains active. Barrier retries wait for callable completion. *)
InterruptedJoin == /\ InterruptJoin /\ fg = "join" /\ report = "compute-error" /\ bg = "rotate"
                   /\ fg' = IF CompletionBarrier THEN "join" ELSE "restore"
                   /\ UNCHANGED <<bg, nextItem, current, future, building, owner, contents,
                                  allocated, output, discarded, report, closeCalled, blasHeld>>
Restore == /\ fg = "restore" /\ fg' = "closed" /\ blasHeld' = FALSE
           /\ UNCHANGED <<bg, nextItem, current, future, building, owner, contents,
                          allocated, output, discarded, report, closeCalled>>
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
BlockingReturns == [](fg \in {"initial", "compute", "await", "join"} => <>(fg \notin {"initial", "compute", "await", "join"}))
CloseTerminates == [](closeCalled => <>(fg = "closed"))
UnwindTerminates == [](report # "none" => <>(fg = "closed"))
NoCloseReachesEnd == (([](~closeCalled) /\ WF_vars(Request)) => <>(fg = "closed"))
====
