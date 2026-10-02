---- MODULE LocoWorkers ----
(* Copyright (c) 2026 Michael Denyer
   SPDX-License-Identifier: GPL-3.0-only

   Caller protocol of solve_eigen_pairs, src/jamma/lmm/loco_workers.py:86-158,
   and closing wrapper src/jamma/lmm/loco_eigen.py:405-422. Queue and Future
   critical sections are library-owned atomic actions, not a model of their
   locks or memory ordering. Queue waits have explicit worker pc="wait";
   enqueue wakes them, and awakened workers recheck queue contents. Future
   waits have consumer pc="wait"; settlement wakes to "await", which rereads
   the outcome. Spurious wakes have no fairness. Thus a removed wake can
   violate liveness without constituting a TLC deadlock.

   Each input/output resource has one logical owner. Pending Future references
   are metadata; while solving the worker owns K, after settlement the Future
   owns the result, and after yield the caller owns it. External retained
   outputs and exception tracebacks are outside the pool's ownership claim.
   Worker deletion of Future/K (:126) is part of settlement; its short Python
   reference-release lag is omitted. Inputs and numerical solves terminate,
   unless the consumer intentionally abandons workers on KeyboardInterrupt.
   No user mutex is held during an allocation or exit. Internal SimpleQueue
   and Future allocation failure is not modelled. Setup failures at start
   (:137) and scope entry (:134) are modelled, with cleanup covering both
   phases. LocoWorkers.mutations lists the bugs the properties detect,
   including the setup order whose failures escaped cleanup.

   Fairness covers internal consumer and worker steps only. Drain=TRUE also
   requires a caller to keep pulling; otherwise idle callers may stop forever
   or close from any idle point. Close before first next starts no workers.
   Sentinels are item 0 and are enqueued after all submitted jobs. *)
EXTENDS Integers, Sequences, FiniteSets, TLC
CONSTANTS Workers, Items, FailAt, InputFailAt, SetupFailure, Drain, Interrupt
W == 1..Workers
I == 1..Items
WorkerOwner(w) == "worker" \o ToString(w)
Owners == {"unproduced", "queue", "future", "caller", "free"}
          \cup {WorkerOwner(w): w \in W}
VARIABLES cpc, wpc, held, started, jobs, pending, submitted, current,
          owner, outcome, delivered, error, ended, scope
workers == <<wpc, held>>
items == <<jobs, owner, outcome>>
window == <<pending, submitted, current>>
results == <<delivered, error, ended>>
vars == <<cpc, started, scope, workers, items, window, results>>
\* The consumer's phase after each setup step, and where a setup failure goes.
Setup == [first |-> "scope", afterScope |-> "starting", afterStart |-> "pull", onError |-> "closing"]
Wake(ws) == [w \in W |-> IF ws[w] = "wait" THEN "ready" ELSE ws[w]]
WorkersDone == \A w \in 1..started: wpc[w] = "done"
StartFails == (SetupFailure = 1 /\ started = 1) \/ (SetupFailure = 3 /\ started = 0)
\* The awaited Future holds a result the consumer may yield.
Yields == outcome[current] = "ok"
Init == /\ cpc = "idle" /\ wpc = [w \in W |-> "new"]
        /\ held = [w \in W |-> 0] /\ started = 0
        /\ jobs = <<>> /\ pending = <<>> /\ submitted = 0 /\ current = 0
        /\ owner = [i \in I |-> "unproduced"]
        /\ outcome = [i \in I |-> "none"]
        /\ delivered = <<>> /\ error = -1 /\ ended = FALSE /\ scope = FALSE
\* The consumer moves to `to` and nothing else changes.
Goto(to) == cpc' = to /\ UNCHANGED <<started, scope, workers, items, window, results>>
\* The consumer records a failure before any item and moves to `to`.
Fail(to) == /\ cpc' = to /\ error' = 0
            /\ UNCHANGED <<started, scope, workers, items, window, delivered, ended>>
StartCall == cpc = "idle" /\ Goto(Setup.first)
CloseUnstarted == cpc = "idle" /\ ~Drain /\ Goto("closed")
StartWorker == /\ cpc = "starting" /\ started < Workers /\ ~StartFails
               /\ started' = started + 1
               /\ wpc' = [wpc EXCEPT ![started + 1] = "ready"]
               /\ UNCHANGED <<cpc, scope, held, items, window, results>>
SetupError == /\ (cpc = "starting" /\ StartFails) \/ (cpc = "scope" /\ SetupFailure = 2)
              /\ Fail(Setup.onError)
FinishStart == cpc = "starting" /\ started = Workers /\ Goto(Setup.afterStart)
EnterScope == /\ cpc = "scope" /\ SetupFailure # 2
              /\ cpc' = Setup.afterScope /\ scope' = TRUE
              /\ UNCHANGED <<started, workers, items, window, results>>
InputFailure == cpc = "pull" /\ submitted + 1 \in InputFailAt /\ Fail("closing")
Submit == /\ cpc = "pull" /\ submitted < Items
          /\ submitted' = submitted + 1
          /\ owner' = [owner EXCEPT ![submitted + 1] = "queue"]
          /\ outcome' = [outcome EXCEPT ![submitted + 1] = "queued"]
          /\ jobs' = Append(jobs, submitted + 1)
          /\ pending' = Append(pending, submitted + 1)
          /\ wpc' = Wake(wpc)
          /\ cpc' = IF Len(pending') = Workers THEN "pop" ELSE "pull"
          /\ UNCHANGED <<held, started, current, results, scope>>
EndInput == /\ cpc = "pull" /\ submitted = Items
            /\ cpc' = IF Len(pending) > 0 THEN "pop" ELSE "closing"
            /\ ended' = (Len(pending) = 0)
            /\ UNCHANGED <<started, scope, workers, items, window, delivered, error>>
Pop == /\ cpc = "pop" /\ Len(pending) > 0 /\ cpc' = "await"
       /\ current' = Head(pending) /\ pending' = Tail(pending)
       /\ UNCHANGED <<started, scope, workers, items, submitted, results>>
Await == /\ cpc = "await"
         /\ cpc' = CASE Yields -> "yield"
                        [] outcome[current] = "err" -> "closing"
                        [] OTHER -> "wait"
         /\ delivered' = IF Yields THEN Append(delivered, current) ELSE delivered
         /\ owner' = IF Yields THEN [owner EXCEPT ![current] = "caller"] ELSE owner
         /\ error' = IF outcome[current] = "err" THEN current ELSE error
         /\ UNCHANGED <<started, scope, workers, jobs, outcome, window, ended>>
Resume == /\ cpc = "yield" /\ cpc' = "pull" /\ current' = 0
          /\ UNCHANGED <<started, scope, workers, items, pending, submitted, results>>
Close == cpc = "yield" /\ ~Drain /\ Goto("closing")
Interrupted == Interrupt /\ cpc \in {"pull", "await", "wait", "pop"} /\ Goto("abandoning")
Cleanup == /\ cpc \in {"closing", "abandoning"}
           /\ cpc' = IF cpc = "abandoning" THEN "abandoned" ELSE "joining"
           /\ jobs' = jobs \o [k \in 1..Workers |-> 0]
           /\ wpc' = Wake(wpc)
           /\ outcome' = [i \in I |-> IF outcome[i] = "queued" THEN "cancelled"
                                     ELSE outcome[i]]
           /\ scope' = IF cpc = "abandoning" THEN FALSE ELSE scope
           /\ UNCHANGED <<held, started, owner, window, results>>
Joined == /\ cpc = "joining" /\ WorkersDone
          /\ cpc' = "closed" /\ scope' = FALSE /\ pending' = <<>>
          /\ owner' = [i \in I |-> IF owner[i] \in {"future", "queue"} THEN "free" ELSE owner[i]]
          /\ UNCHANGED <<started, workers, jobs, outcome, submitted, current, results>>
WorkerGet(w) ==
    /\ wpc[w] = "ready"
    /\ IF Len(jobs) = 0
       THEN /\ wpc' = [wpc EXCEPT ![w] = "wait"]
            /\ UNCHANGED <<held, items>>
       ELSE LET j == Head(jobs) IN
            /\ jobs' = Tail(jobs)
            /\ held' = [held EXCEPT ![w] = j]
            /\ wpc' = [wpc EXCEPT ![w] = IF j = 0 THEN "done"
                             ELSE IF outcome[j] = "cancelled" THEN "discard"
                                  ELSE "solve"]
            /\ owner' = IF j = 0 THEN owner
                         ELSE [owner EXCEPT ![j] = WorkerOwner(w)]
            /\ outcome' = IF j = 0 \/ outcome[j] = "cancelled"
                           THEN outcome ELSE [outcome EXCEPT ![j] = "running"]
    /\ UNCHANGED <<cpc, started, scope, window, results>>
WorkerFinish(w) ==
    /\ wpc[w] \in {"solve", "discard"}
    /\ \E result \in {"ok", "err"}:
        /\ result = "ok" \/ held[w] \in FailAt
        /\ outcome' = [outcome EXCEPT ![held[w]] =
                              IF wpc[w] = "discard" THEN "cancelled" ELSE result]
    /\ owner' = [owner EXCEPT ![held[w]] =
                              IF wpc[w] = "discard" THEN "free" ELSE "future"]
    /\ wpc' = [wpc EXCEPT ![w] = "ready"] /\ held' = [held EXCEPT ![w] = 0]
    /\ cpc' = IF cpc = "wait" /\ current = held[w] THEN "await" ELSE cpc
    /\ UNCHANGED <<started, scope, jobs, window, results>>
Spurious ==
    \/ cpc = "wait" /\ Goto("await")
    \/ /\ \E w \in W: /\ wpc[w] = "wait"
                         /\ wpc' = [wpc EXCEPT ![w] = "ready"]
       /\ UNCHANGED <<cpc, started, scope, held, items, window, results>>
Internal == StartWorker \/ SetupError \/ FinishStart \/ EnterScope \/ InputFailure
            \/ Submit \/ EndInput \/ Pop \/ Await \/ Cleanup \/ Joined
Next == StartCall \/ CloseUnstarted \/ Internal \/ Resume \/ Close \/ Interrupted
        \/ (\E w \in W: WorkerGet(w) \/ WorkerFinish(w)) \/ Spurious
        \/ UNCHANGED vars
Spec == Init /\ [][Next]_vars
        /\ WF_vars(Internal)
        /\ \A w \in W: WF_vars(WorkerGet(w)) /\ WF_vars(WorkerFinish(w))
        /\ (Drain => (WF_vars(StartCall) /\ WF_vars(Resume)))
TypeOK == /\ cpc \in {"idle", "starting", "scope", "pull", "pop", "await", "wait",
                       "yield", "closing", "joining", "closed", "abandoning",
                       "abandoned", "escaped"}
          /\ wpc \in [W -> {"new", "ready", "wait", "solve", "discard", "done"}]
          /\ held \in [W -> 0..Items] /\ started \in 0..Workers
          /\ jobs \in Seq(0..Items) /\ pending \in Seq(I)
          /\ owner \in [I -> Owners] /\ submitted \in 0..Items
          /\ current \in 0..Items /\ delivered \in Seq(I)
          /\ outcome \in [I -> {"none", "queued", "running", "ok", "err", "cancelled"}]
          /\ scope \in BOOLEAN /\ ended \in BOOLEAN /\ error \in -1..Items
Ownership == /\ \A i \in I: (owner[i] = "queue") <=> (\E k \in 1..Len(jobs): jobs[k] = i)
             /\ \A w \in W: held[w] > 0 => owner[held[w]] = WorkerOwner(w)
             /\ \A j, k \in 1..Len(jobs): (jobs[j] # 0 /\ jobs[j] = jobs[k]) => j = k
DeliveryOrder == \A k \in 1..Len(delivered): delivered[k] = k
EndOrder == ended => Len(delivered) = submitted
NoDeliveryAfterFailure == \A k \in 1..Len(delivered): outcome[delivered[k]] = "ok"
ErrorOrder == error > 0 => Len(delivered) = error - 1
WindowBound == Cardinality({i \in I: owner[i] \in {"queue", "future"}
                                \/ (\E w \in W: owner[i] = WorkerOwner(w))}) <= Workers
ClosedClean == cpc = "closed" => /\ WorkersDone
                                    /\ \A i \in I: owner[i] \in {"unproduced", "caller", "free"}
                                    /\ ~scope
SetupClean == cpc = "escaped" => started = 0
ScopeCoversSolve == \A w \in W: wpc[w] = "solve" => scope \/ cpc = "abandoned"
CallsReturn == (cpc \in {"starting", "scope", "pull", "pop", "await", "wait"})
               ~> (cpc \in {"yield", "closed", "abandoned"})
CloseReturns == (cpc \in {"closing", "joining"}) ~> (cpc = "closed")
DrainEnds == Drain => <>(cpc \in {"closed", "abandoned"})
====
