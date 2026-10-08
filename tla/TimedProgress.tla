---- MODULE TimedProgress ----
(* Copyright (c) 2026 Michael Denyer
   SPDX-License-Identifier: GPL-3.0-only

   timed_progress, src/jamma/core/progress.py:138-223. Worker fn returns or
   raises at :187-190, publishes a result/error, notifies Event at :191-192,
   then exits. Consumer waits with a positive finite timeout at :200, polls
   at :203-207, finishes at :211-213, joins at :217-219 and reports the
   result/error at :221-223. Progressbar.finish default behavior is an
   implicit update(100), the only full update; dirty=True preserves value.
   Library Event/Thread operations are atomic contracts. There is no caller
   mutex. fn allocations/errors occur outside application locks. Python list
   allocation failure and Event/Thread internals are outside this abstraction.

   Worker pcs distinguish fn completion, box publication, notification and
   actual thread exit. The consumer has an Event wait pc. Timeout or signal
   returns from that wait, then the predicate is reread; no caller CV exists.
   Spurious wakes return to the Event wait and have no fairness. Each worker
   and consumer internal action has weak fairness. fn termination and finite,
   positive poll/join timeouts are assumptions. Consumer interruption choices
   have no fairness and deliberately allow abandonment of the daemon worker.

   Finish is clean, and so draws 100%, only on done and a published result
   with no cancellation; any other finish is dirty. TimedProgress.mutations
   lists the bugs the properties detect, including the clean finish before
   the worker is done and the full redraw on any finish. Resources are the output/error box and the worker; ownership
   transfers from worker to box to caller. Caller-retained exception tracebacks/results are permitted.
   Output modes include empty value, nonempty value, Exception and BaseException;
   values are abstract tags, so no numerical correctness is claimed. *)
EXTENDS Naturals
CONSTANTS Outcome, OutputFailure, AllowInterrupt
VARIABLES pc, wpc, box, done, workerExited, display, falseCompletion,
          cancelled, report, owner
shown == <<display, falseCompletion>>
outcome == <<cancelled, report>>
worker == <<wpc, workerExited>>
shared == <<box, done, owner>>
vars == <<pc, shown, outcome, worker, shared>>
Init == /\ pc = "wait" /\ wpc = "run" /\ box = "none" /\ done = FALSE
        /\ workerExited = FALSE /\ display = 0 /\ falseCompletion = FALSE
        /\ cancelled = FALSE /\ report = "none" /\ owner = "worker"
Succeeded == box \in {"empty", "value"}
Successful == done /\ Succeeded
\* Draw 100% when `drawn` holds; that is a false completion unless `earned`.
Complete(drawn, earned) ==
    /\ display' = IF drawn THEN 100 ELSE display
    /\ falseCompletion' = (falseCompletion \/ (drawn /\ ~earned))
WorkerRun == /\ wpc = "run" /\ wpc' = "publish"
             /\ UNCHANGED <<pc, shown, outcome, workerExited, shared>>
Publish == /\ wpc = "publish" /\ wpc' = "notify"
           /\ box' = Outcome
           /\ owner' = "box"
           /\ UNCHANGED <<pc, shown, outcome, workerExited, done>>
Notify == /\ wpc = "notify" /\ wpc' = "exit"
          /\ done' = TRUE
          /\ pc' = IF pc = "wait" THEN "check" ELSE pc
          /\ UNCHANGED <<shown, outcome, workerExited, box, owner>>
WorkerExit == /\ wpc = "exit" /\ wpc' = "done" /\ workerExited' = TRUE
              /\ UNCHANGED <<pc, shown, outcome, shared>>
Wait == /\ pc = "wait" /\ pc' = IF done THEN "check" ELSE "poll"
        /\ UNCHANGED <<shown, outcome, worker, shared>>
Check == /\ pc = "check" /\ pc' = IF done THEN "finish" ELSE "wait"
         /\ UNCHANGED <<shown, outcome, worker, shared>>
Poll == /\ pc = "poll"
        /\ pc' = IF OutputFailure = "poll" THEN "finish" ELSE "wait"
        /\ display' = IF OutputFailure = "poll" THEN display ELSE 99
        /\ UNCHANGED <<falseCompletion, outcome, worker, shared>>
Finish == /\ pc = "finish" /\ pc' = IF cancelled THEN "cancelled" ELSE "join"
          /\ Complete(Successful /\ ~cancelled, Succeeded /\ ~cancelled)
          /\ UNCHANGED <<outcome, worker, shared>>
Join == /\ pc = "join" /\ workerExited
        /\ pc' = "reported"
        /\ report' = IF Succeeded THEN "result"
                      ELSE IF box \in {"error", "base"} THEN "error" ELSE "missing"
        /\ owner' = "caller"
        /\ UNCHANGED <<shown, cancelled, worker, box, done>>
Cancel == /\ AllowInterrupt /\ cancelled' = TRUE /\ report' = "cancel"
          /\ UNCHANGED <<shown, worker, shared>>
Interrupt == pc \in {"wait", "check", "poll"} /\ pc' = "finish" /\ Cancel
InterruptJoin == pc = "join" /\ pc' = "cancelled" /\ Cancel
Spurious == /\ pc = "wait" /\ pc' = "check"
            /\ UNCHANGED <<shown, outcome, worker, shared>>
Worker == WorkerRun \/ Publish \/ Notify \/ WorkerExit
Consumer == Wait \/ Check \/ Poll \/ Finish \/ Join
Next == Worker \/ Consumer \/ Interrupt \/ InterruptJoin \/ Spurious \/ UNCHANGED vars
Spec == Init /\ [][Next]_vars /\ WF_vars(Worker) /\ WF_vars(Consumer)
TypeOK == /\ pc \in {"wait", "check", "poll", "finish", "join", "reported", "cancelled"}
          /\ wpc \in {"run", "publish", "notify", "exit", "done"}
          /\ box \in {"none", "empty", "value", "error", "base"}
          /\ done \in BOOLEAN /\ workerExited \in BOOLEAN /\ cancelled \in BOOLEAN
          /\ display \in {0, 99, 100} /\ falseCompletion \in BOOLEAN
          /\ report \in {"none", "result", "error", "missing", "cancel"}
          /\ owner \in {"worker", "box", "caller", "free"}
HonestCompletion == ~falseCompletion
JoinedBeforeReport == pc = "reported" => workerExited
OutcomeDelivered == pc = "reported" => (report = "result" <=> Outcome \in {"empty", "value"})
                    /\ report \in {"result", "error"}
Ownership == pc = "reported" => owner = "caller"
Terminates == <>(pc \in {"reported", "cancelled"})
====
