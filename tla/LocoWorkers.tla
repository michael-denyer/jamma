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
   (:137) and scope entry (:134) are modelled. SetupCleanup=FALSE preserves
   the original cleanup gap; TRUE models cleanup covering those phases.

   Fairness covers internal consumer and worker steps only. Drain=TRUE also
   requires a caller to keep pulling; otherwise idle callers may stop forever
   or close from any idle point. Close before first next starts no workers.
   Sentinels are item 0 and are enqueued after all submitted jobs. *)
EXTENDS Integers, Sequences, FiniteSets, TLC
CONSTANTS Workers, Items, FailAt, InputFailAt, SetupFailure, SetupCleanup,
          Drain, Interrupt, Mutation
W == 1..Workers
I == 1..Items
WorkerOwner(w) == "worker" \o ToString(w)
Owners == {"unproduced", "queue", "future", "caller", "free"}
          \cup {WorkerOwner(w): w \in W}
VARIABLES cpc, wpc, held, started, jobs, pending, submitted, current,
          owner, outcome, delivered, error, ended, scope
vars == <<cpc, wpc, held, started, jobs, pending, submitted, current,
          owner, outcome, delivered, error, ended, scope>>
Wake(ws) == [w \in W |-> IF ws[w] = "wait" THEN "ready" ELSE ws[w]]
Init == /\ cpc = "idle" /\ wpc = [w \in W |-> "new"]
        /\ held = [w \in W |-> 0] /\ started = 0
        /\ jobs = <<>> /\ pending = <<>> /\ submitted = 0 /\ current = 0
        /\ owner = [i \in I |-> "unproduced"]
        /\ outcome = [i \in I |-> "none"]
        /\ delivered = <<>> /\ error = -1 /\ ended = FALSE /\ scope = FALSE
StartCall == /\ cpc = "idle" /\ cpc' = IF SetupCleanup THEN "scope" ELSE "starting"
             /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                  current, owner, outcome, delivered, error, ended, scope>>
CloseUnstarted == /\ cpc = "idle" /\ ~Drain /\ cpc' = "closed"
                  /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                       current, owner, outcome, delivered, error, ended, scope>>
StartWorker == /\ cpc = "starting" /\ started < Workers
               /\ ~(SetupFailure = 1 /\ started = 1)
               /\ ~(SetupFailure = 3 /\ started = 0)
               /\ started' = started + 1
               /\ wpc' = [wpc EXCEPT ![started + 1] = "ready"]
               /\ UNCHANGED <<cpc, held, jobs, pending, submitted, current,
                                owner, outcome, delivered, error, ended, scope>>
SetupError == /\ ((cpc = "starting" /\ SetupFailure = 1 /\ started = 1)
                  \/ (cpc = "starting" /\ SetupFailure = 3 /\ started = 0)
                  \/ (cpc = "scope" /\ SetupFailure = 2))
              /\ cpc' = IF SetupCleanup THEN "closing" ELSE "escaped"
              /\ error' = 0
              /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                               current, owner, outcome, delivered, ended, scope>>
FinishStart == /\ cpc = "starting" /\ started = Workers /\ cpc' = IF SetupCleanup THEN "pull" ELSE "scope"
               /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                      current, owner, outcome, delivered, error, ended, scope>>
EnterScope == /\ cpc = "scope" /\ SetupFailure # 2
              /\ cpc' = (IF SetupCleanup THEN "starting" ELSE "pull") /\ scope' = TRUE
              /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                           current, owner, outcome, delivered, error, ended>>
InputFailure == /\ cpc = "pull" /\ submitted + 1 \in InputFailAt
                /\ cpc' = "closing" /\ error' = 0
                /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                       current, owner, outcome, delivered, ended, scope>>
Submit == /\ cpc = "pull" /\ submitted < Items
          /\ submitted' = submitted + 1
          /\ owner' = [owner EXCEPT ![submitted + 1] = "queue"]
          /\ outcome' = [outcome EXCEPT ![submitted + 1] = "queued"]
          /\ jobs' = Append(jobs, submitted + 1)
          /\ pending' = Append(pending, submitted + 1)
          /\ wpc' = IF Mutation = "no-queue-wake" THEN wpc ELSE Wake(wpc)
          /\ cpc' = IF Len(pending') = (IF Mutation = "oversubmit" THEN Workers + 1 ELSE Workers) THEN "pop" ELSE "pull"
          /\ UNCHANGED <<held, started, current, delivered, error, ended, scope>>
EndInput == /\ cpc = "pull" /\ submitted = Items
            /\ cpc' = IF Len(pending) > 0 THEN "pop" ELSE "closing"
            /\ ended' = (Len(pending) = 0 \/ Mutation = "early-end")
            /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                           current, owner, outcome, delivered, error, scope>>
Pop == /\ cpc = "pop" /\ Len(pending) > 0 /\ cpc' = "await"
       /\ current' = IF Mutation = "lifo" THEN pending[Len(pending)] ELSE Head(pending)
       /\ pending' = IF Mutation = "lifo" THEN SubSeq(pending, 1, Len(pending)-1)
                     ELSE Tail(pending)
       /\ UNCHANGED <<wpc, held, started, jobs, submitted, owner, outcome,
                      delivered, error, ended, scope>>
Await == /\ cpc = "await"
         /\ cpc' = CASE (outcome[current] = "ok" \/ (Mutation = "ignore-error" /\ outcome[current] = "err")) -> "yield"
                        [] outcome[current] = "err" -> "closing"
                        [] OTHER -> "wait"
         /\ delivered' = IF outcome[current] = "ok" \/ (Mutation = "ignore-error" /\ outcome[current] = "err") THEN Append(delivered, current)
                          ELSE delivered
         /\ owner' = IF outcome[current] = "ok" \/ (Mutation = "ignore-error" /\ outcome[current] = "err")
                     THEN [owner EXCEPT ![current] = "caller"] ELSE owner
         /\ error' = IF outcome[current] = "err" THEN current ELSE error
         /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                         current, outcome, ended, scope>>
Resume == /\ cpc = "yield" /\ cpc' = "pull" /\ current' = 0
          /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted, owner,
                          outcome, delivered, error, ended, scope>>
Close == /\ cpc = "yield" /\ ~Drain /\ cpc' = "closing"
         /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted, current,
                         owner, outcome, delivered, error, ended, scope>>
Interrupted == /\ Interrupt /\ cpc \in {"pull", "await", "wait", "pop"}
               /\ cpc' = "abandoning"
               /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted,
                      current, owner, outcome, delivered, error, ended, scope>>
Cleanup == /\ cpc \in {"closing", "abandoning"}
           /\ cpc' = IF cpc = "abandoning" THEN "abandoned" ELSE "joining"
           /\ jobs' = IF Mutation = "no-sentinels" THEN jobs
                       ELSE jobs \o [k \in 1..Workers |-> 0]
           /\ wpc' = IF Mutation = "no-queue-wake" THEN wpc ELSE Wake(wpc)
           /\ outcome' = [i \in I |-> IF outcome[i] = "queued" THEN "cancelled"
                                     ELSE outcome[i]]
           /\ scope' = IF cpc = "abandoning" \/ Mutation = "early-restore" THEN FALSE ELSE scope
           /\ UNCHANGED <<held, started, pending, submitted, current,
                            owner, delivered, error, ended>>
Joined == /\ cpc = "joining" /\ \A w \in 1..started: wpc[w] = "done"
          /\ cpc' = "closed" /\ scope' = FALSE /\ pending' = <<>>
          /\ owner' = [i \in I |-> IF owner[i] \in {"future", "queue"} /\ Mutation # "retain-future"
                                      THEN "free" ELSE owner[i]]
          /\ UNCHANGED <<wpc, held, started, jobs, submitted, current,
                           outcome, delivered, error, ended>>
WorkerGet(w) ==
    /\ wpc[w] = "ready"
    /\ IF Len(jobs) = 0
       THEN /\ wpc' = [wpc EXCEPT ![w] = "wait"]
            /\ UNCHANGED <<jobs, held, owner, outcome>>
       ELSE /\ jobs' = IF Mutation = "duplicate-job" THEN jobs ELSE Tail(jobs)
            /\ held' = [held EXCEPT ![w] = Head(jobs)]
            /\ wpc' = [wpc EXCEPT ![w] = IF Head(jobs) = 0 THEN "done"
                             ELSE IF outcome[Head(jobs)] = "cancelled" THEN "discard"
                                  ELSE "solve"]
            /\ owner' = IF Head(jobs) = 0 THEN owner
                         ELSE [owner EXCEPT ![Head(jobs)] = IF Mutation = "orphan" THEN "orphan" ELSE WorkerOwner(w)]
            /\ outcome' = IF Head(jobs) = 0 \/ outcome[Head(jobs)] = "cancelled"
                           THEN outcome ELSE [outcome EXCEPT ![Head(jobs)] = "running"]
    /\ UNCHANGED <<cpc, started, pending, submitted, current,
                     delivered, error, ended, scope>>
WorkerFinish(w) ==
    /\ wpc[w] \in {"solve", "discard"}
    /\ \E result \in {"ok", "err"}:
        /\ result = "ok" \/ held[w] \in FailAt
        /\ outcome' = [outcome EXCEPT ![held[w]] =
                              IF wpc[w] = "discard" THEN "cancelled" ELSE result]
    /\ owner' = [owner EXCEPT ![held[w]] =
                              IF wpc[w] = "discard" THEN "free" ELSE "future"]
    /\ wpc' = [wpc EXCEPT ![w] = "ready"] /\ held' = [held EXCEPT ![w] = 0]
    /\ cpc' = IF cpc = "wait" /\ current = held[w] /\ Mutation # "no-future-wake"
               THEN "await" ELSE cpc
    /\ UNCHANGED <<started, jobs, pending, submitted, current,
                     delivered, error, ended, scope>>
Spurious ==
    \/ /\ cpc = "wait" /\ cpc' = "await"
       /\ UNCHANGED <<wpc, held, started, jobs, pending, submitted, current,
                       owner, outcome, delivered, error, ended, scope>>
    \/ /\ \E w \in W: /\ wpc[w] = "wait"
                         /\ wpc' = [wpc EXCEPT ![w] = "ready"]
       /\ UNCHANGED <<cpc, held, started, jobs, pending, submitted, current,
                       owner, outcome, delivered, error, ended, scope>>
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
ClosedClean == cpc = "closed" => /\ \A w \in 1..started: wpc[w] = "done"
                                    /\ \A i \in I: owner[i] \in {"unproduced", "caller", "free"}
                                    /\ ~scope
SetupClean == cpc = "escaped" => started = 0
ScopeCoversSolve == \A w \in W: wpc[w] = "solve" => scope \/ cpc = "abandoned"
CallsReturn == (cpc \in {"starting", "scope", "pull", "pop", "await", "wait"})
               ~> (cpc \in {"yield", "closed", "abandoned"})
CloseReturns == (cpc \in {"closing", "joining"}) ~> (cpc = "closed")
DrainEnds == Drain => <>(cpc \in {"closed", "abandoned"})
====
