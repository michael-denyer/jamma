---- MODULE TimedProgress ----
(* Copyright (c) 2026 Michael Denyer
   SPDX-License-Identifier: GPL-3.0-only

   timed_progress, src/jamma/core/progress.py:138-226. Worker fn returns or
   raises at :187-190, publishes a result/error, notifies Event at :191-192,
   then exits. Consumer waits with a positive finite timeout at :200, polls
   at :203-207, performs final update at :208-210, finishes at :214-216, joins
   at :220-222 and reports the result/error at :224-226. Progressbar.finish
   default behavior is an implicit update(100); dirty=True preserves value.
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

   Fixed=FALSE reproduces premature explicit full update after polling OSError
   and implicit full redraw on any finish. Fixed=TRUE gates success on done
   and absence of error and performs dirty finish otherwise. Resources are
   the output/error box and the worker; ownership transfers from worker to
   box to caller. Caller-retained exception tracebacks/results are permitted.
   Output modes include empty value, nonempty value, Exception and BaseException;
   values are abstract tags, so no numerical correctness is claimed. *)
EXTENDS Naturals
CONSTANTS Outcome, OutputFailure, AllowInterrupt, Fixed, Mutation
VARIABLES pc, wpc, box, done, workerExited, display, falseCompletion,
          cancelled, report, owner
vars == <<pc, wpc, box, done, workerExited, display, falseCompletion,
          cancelled, report, owner>>
Init == /\ pc = "wait" /\ wpc = "run" /\ box = "none" /\ done = FALSE
        /\ workerExited = FALSE /\ display = 0 /\ falseCompletion = FALSE
        /\ cancelled = FALSE /\ report = "none" /\ owner = "worker"
WorkerRun == /\ wpc = "run" /\ wpc' = "publish"
             /\ UNCHANGED <<pc, box, done, workerExited, display,
                             falseCompletion, cancelled, report, owner>>
Publish == /\ wpc = "publish" /\ wpc' = "notify"
           /\ box' = IF Mutation = "drop-base-catch" /\ Outcome = "base"
                     THEN "none" ELSE Outcome
           /\ owner' = IF Mutation = "drop-base-catch" /\ Outcome = "base"
                       THEN "free" ELSE IF Mutation = "orphan" THEN "orphan" ELSE "box"
           /\ UNCHANGED <<pc, done, workerExited, display, falseCompletion,
                           cancelled, report>>
Notify == /\ wpc = "notify" /\ wpc' = "exit"
          /\ done' = (Mutation # "no-notify")
          /\ pc' = IF pc = "wait" /\ Mutation # "no-notify" THEN "check" ELSE pc
          /\ UNCHANGED <<box, workerExited, display, falseCompletion,
                          cancelled, report, owner>>
WorkerExit == /\ wpc = "exit" /\ wpc' = "done" /\ workerExited' = TRUE
              /\ UNCHANGED <<pc, box, done, display, falseCompletion,
                              cancelled, report, owner>>
Wait == /\ pc = "wait" /\ pc' = IF done THEN "check" ELSE "poll"
        /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion,
                        cancelled, report, owner>>
Check == /\ pc = "check" /\ pc' = IF done THEN "full" ELSE "wait"
         /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion,
                         cancelled, report, owner>>
Poll == /\ pc = "poll"
        /\ pc' = IF OutputFailure = "poll" THEN "full" ELSE "wait"
        /\ display' = IF OutputFailure = "poll" THEN display
                       ELSE IF Mutation = "uncapped" THEN 100 ELSE 99
        /\ falseCompletion' = (falseCompletion
                              \/ (Mutation = "uncapped" /\ box # "empty" /\ box # "value"))
        /\ UNCHANGED <<wpc, box, done, workerExited, cancelled, report, owner>>
Successful == done /\ box \in {"empty", "value"}
Full == /\ pc = "full" /\ pc' = "finish"
        /\ display' = IF (IF Fixed THEN Successful ELSE box \notin {"error", "base"})
                           THEN 100 ELSE display
        /\ falseCompletion' = (falseCompletion
              \/ ((IF Fixed THEN Successful ELSE box \notin {"error", "base"})
                  /\ box \notin {"empty", "value"}))
        /\ UNCHANGED <<wpc, box, done, workerExited, cancelled, report, owner>>
Finish == /\ pc = "finish" /\ pc' = IF cancelled THEN "cancelled" ELSE "join"
          /\ display' = IF ~Fixed \/ (Successful /\ ~cancelled) THEN 100 ELSE display
          /\ falseCompletion' = (falseCompletion
                     \/ ((~Fixed \/ (Successful /\ ~cancelled))
                         /\ (box \notin {"empty", "value"} \/ cancelled)))
          /\ UNCHANGED <<wpc, box, done, workerExited, cancelled, report, owner>>
Join == /\ pc = "join" /\ (workerExited \/ Mutation = "no-join")
        /\ pc' = "reported"
        /\ report' = IF box \in {"empty", "value"} THEN "result"
                      ELSE IF box \in {"error", "base"} THEN "error" ELSE "missing"
        /\ owner' = IF Mutation = "retain-box" THEN owner ELSE "caller"
        /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion, cancelled>>
Interrupt == /\ AllowInterrupt /\ pc \in {"wait", "check", "poll", "full"}
             /\ pc' = "finish" /\ cancelled' = TRUE /\ report' = "cancel"
             /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion, owner>>
InterruptJoin == /\ AllowInterrupt /\ pc = "join"
                 /\ pc' = "cancelled" /\ cancelled' = TRUE /\ report' = "cancel"
                 /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion, owner>>
Spurious == /\ pc = "wait" /\ pc' = "check"
            /\ UNCHANGED <<wpc, box, done, workerExited, display, falseCompletion,
                            cancelled, report, owner>>
Worker == WorkerRun \/ Publish \/ Notify \/ WorkerExit
Consumer == Wait \/ Check \/ Poll \/ Full \/ Finish \/ Join
Next == Worker \/ Consumer \/ Interrupt \/ InterruptJoin \/ Spurious \/ UNCHANGED vars
Spec == Init /\ [][Next]_vars /\ WF_vars(Worker) /\ WF_vars(Consumer)
TypeOK == /\ pc \in {"wait", "check", "poll", "full", "finish", "join", "reported", "cancelled"}
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
