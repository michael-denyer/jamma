---- MODULE JoiningThreadPool ----
(* Copyright (c) 2026 Michael Denyer
   SPDX-License-Identifier: GPL-3.0-only

   src/jamma/core/thread_pool.py:21-92. One submitting/closing thread and
   Workers executor threads. Condition protects the _outstanding dictionary
   of pending/running tokens; token absence means settled. Token membership
   and value change atomically. state=new/settled is ghost submitted history.
   Library enqueue, dequeue,
   Future cancellation and thread start are atomic abstractions. Calls may
   fail after enqueue; a dequeued wrapper must recheck state under the lock
   before borrowing arguments. Shutdown(cancel_futures=True) first removes
   pending tokens, then cancels queued Futures; their callbacks can be skipped
   by an interrupt. Running callables settle in
   finally. Interruptions during drain release the Condition and retry.

   Each mutex critical section and unlocked callable phase is separate.
   A Condition wait is pc=wait, with the lock released. Only notify_all,
   spurious wakeup or KeyboardInterrupt can leave this wait set. Reacquisition
   then rereads outstanding. Spurious wakeups and interrupts have no fairness.
   Lost notify appears as a liveness violation, not a TLC deadlock, because
   explicit stuttering is allowed in blocked states. Weak fairness applies to
   real deterministic phases and library callbacks; strong mutex fairness is
   named below to exclude starvation from an infinitely spuriously waking
   shutdown thread. Finite callable termination and finite interrupts are
   prerequisites for practical termination; the model records at most one
   interrupt, sufficient to check cleanup on that path.

   Logical borrowed arguments are owned by at most one worker per job. Python
   callable exceptions, including MemoryError, still execute finally. No
   longjmp/os._exit exists in the helper. Dictionary allocation in submit and
   library failures are abstracted by SubmitFailures. Interpreter/OS death,
   catastrophic allocation failure in exception cleanup, mutex ordering below
   Condition, and arbitrary concurrent submitters are outside this instance.
   JoiningThreadPool.mutations lists the bugs the properties detect.
*)
EXTENDS Naturals, FiniteSets, Sequences
CONSTANTS Items, Workers, SubmitFailures, CancelQueued
VARIABLES main, lock, nextJob, outstanding, state, queue, submitted, wpc, job,
          interrupted, released, cancelJob
tokens == <<outstanding, state>>
workers == <<wpc, job>>
caller == <<nextJob, submitted, cancelJob>>
flags == <<interrupted, released>>
vars == <<main, lock, tokens, queue, workers, caller, flags>>
W == 1..Workers
J == 1..Items
Live == {j \in J : state[j] \in {"pending", "running"}}
Running == {w \in W : wpc[w] \in {"run", "settle-lock", "settle"}}
Borrowed == {job[w] : w \in Running}
Wake(pc) == IF pc = "wait" THEN "drain-lock" ELSE pc
Init == /\ main = "idle" /\ lock = 0 /\ nextJob = 1 /\ outstanding = {}
        /\ state = [j \in J |-> "new"] /\ queue = <<>> /\ submitted = {}
        /\ wpc = [w \in W |-> "idle"] /\ job = [w \in W |-> 0]
        /\ interrupted = FALSE /\ released = FALSE /\ cancelJob = 0
\* The main thread takes the Condition's mutex and moves to `to`.
MainAcquire(to) == /\ lock = 0 /\ lock' = 1 /\ main' = to
                   /\ UNCHANGED <<tokens, queue, workers, caller, flags>>
\* The main thread moves to `to` without touching shared state.
MainGoes(to) == main' = to /\ UNCHANGED <<lock, tokens, queue, workers, caller, flags>>
\* Worker w takes the Condition's mutex and moves to `to`.
WorkerAcquire(w, to) == /\ lock = 0 /\ lock' = w + 1
                        /\ wpc' = [wpc EXCEPT ![w] = to]
                        /\ UNCHANGED <<main, tokens, queue, job, caller, flags>>
SubmitAcquire == main = "idle" /\ nextJob <= Items /\ MainAcquire("increment")
Increment == /\ main = "increment" /\ lock = 1
             /\ state' = [state EXCEPT ![nextJob] = "pending"] /\ outstanding' = outstanding \cup {nextJob}
             /\ submitted' = submitted \cup {nextJob} /\ main' = "enqueue" /\ lock' = 0
             /\ UNCHANGED <<nextJob, cancelJob, queue, workers, flags>>
Enqueue == /\ main = "enqueue" /\ queue' = Append(queue, nextJob)
           /\ main' = "submit-result"
           /\ UNCHANGED <<lock, tokens, workers, caller, flags>>
SubmitResult == \E fail \in BOOLEAN :
                /\ main = "submit-result"
                /\ IF fail /\ nextJob \in SubmitFailures
                      THEN /\ cancelJob' = nextJob /\ main' = "cancel-lock"
                           /\ UNCHANGED nextJob
                      ELSE /\ nextJob' = nextJob + 1 /\ main' = "idle"
                           /\ UNCHANGED cancelJob
                /\ UNCHANGED <<lock, tokens, queue, workers, submitted, flags>>
CancelAcquire == main = "cancel-lock" /\ MainAcquire("cancel")
CancelPending == /\ main = "cancel" /\ lock = 1
                 /\ IF state[cancelJob] = "pending"
                       THEN /\ state' = [state EXCEPT ![cancelJob] = "settled"] /\ outstanding' = outstanding \ {cancelJob}
                       ELSE /\ UNCHANGED tokens
                 /\ main' = "shutdown" /\ lock' = 0 /\ cancelJob' = 0
                 /\ UNCHANGED <<nextJob, submitted, queue, workers, flags>>
Close == main = "idle" /\ MainGoes("shutdown")
BeginShutdown == main = "shutdown" /\ MainGoes("cancel-queued")
CancelQueuedAcquire == main = "cancel-queued" /\ CancelQueued /\ MainAcquire("cancel-queued-settle")
CancelQueuedSettle == /\ main = "cancel-queued-settle" /\ lock = 1
                      /\ outstanding' = {j \in outstanding : state[j] = "running"}
                      /\ state' = [j \in J |-> IF state[j] = "pending" THEN "settled" ELSE state[j]]
                      /\ main' = "cancel-library" /\ lock' = 0
                      /\ UNCHANGED <<queue, workers, caller, flags>>
CancelLibrary == /\ main = "cancel-library" /\ queue' = <<>> /\ main' = "drain-lock"
                 /\ UNCHANGED <<lock, tokens, workers, caller, flags>>
DrainBegin == main = "cancel-queued" /\ ~CancelQueued /\ MainGoes("drain-lock")
DrainAcquire == main = "drain-lock" /\ MainAcquire("check")
Check == /\ main = "check" /\ lock = 1 /\ lock' = 0
         /\ main' = IF outstanding = {} THEN "join" ELSE "wait"
         /\ UNCHANGED <<tokens, queue, workers, caller, flags>>
SpuriousWake == main = "wait" /\ MainGoes("drain-lock")
Interrupt == /\ main \in {"wait", "check", "join", "cancel-library"} /\ ~interrupted
             /\ main' = "shutdown" /\ interrupted' = TRUE /\ lock' = IF lock = 1 THEN 0 ELSE lock
             /\ UNCHANGED <<tokens, queue, workers, caller, released>>
Join == /\ main = "join" /\ main' = "closed" /\ released' = TRUE
        /\ UNCHANGED <<lock, tokens, queue, workers, caller, interrupted>>
Take(w) == /\ wpc[w] = "idle" /\ Len(queue) > 0
           /\ job' = [job EXCEPT ![w] = Head(queue)] /\ queue' = Tail(queue)
           /\ wpc' = [wpc EXCEPT ![w] = "run-lock"]
           /\ UNCHANGED <<main, lock, tokens, caller, flags>>
RunAcquire(w) == wpc[w] = "run-lock" /\ WorkerAcquire(w, "start")
StartRun(w) == /\ wpc[w] = "start" /\ lock = w + 1 /\ lock' = 0
               /\ IF state[job[w]] = "settled"
                     THEN /\ wpc' = [wpc EXCEPT ![w] = "idle"] /\ job' = [job EXCEPT ![w] = 0]
                          /\ UNCHANGED state
                     ELSE /\ state' = [state EXCEPT ![job[w]] = "running"]
                          /\ wpc' = [wpc EXCEPT ![w] = "run"] /\ UNCHANGED job
               /\ UNCHANGED <<main, outstanding, queue, caller, flags>>
Run(w) == /\ wpc[w] = "run" /\ wpc' = [wpc EXCEPT ![w] = "settle-lock"]
          /\ UNCHANGED <<main, lock, tokens, queue, job, caller, flags>>
SettleAcquire(w) == wpc[w] = "settle-lock" /\ WorkerAcquire(w, "settle")
Settle(w) == /\ wpc[w] = "settle" /\ lock = w + 1 /\ lock' = 0
             /\ state' = [state EXCEPT ![job[w]] = "settled"] /\ outstanding' = outstanding \ {job[w]}
             /\ wpc' = [wpc EXCEPT ![w] = "idle"] /\ job' = [job EXCEPT ![w] = 0]
             /\ main' = Wake(main)
             /\ UNCHANGED <<queue, caller, flags>>
Pause == UNCHANGED vars
Next == SubmitAcquire \/ Increment \/ Enqueue \/ SubmitResult \/ CancelAcquire \/ CancelPending \/
        Close \/ BeginShutdown \/ CancelQueuedAcquire \/ CancelQueuedSettle \/ CancelLibrary \/ DrainBegin \/
        DrainAcquire \/ Check \/ SpuriousWake \/ Interrupt \/ Join \/ Pause \/
        (\E w \in W : Take(w) \/ RunAcquire(w) \/ StartRun(w) \/ Run(w) \/ SettleAcquire(w) \/ Settle(w))
(* Strong mutex acquisition fairness excludes infinite spurious-wakeup
   starvation only; it does not make notification or interruption fair. *)
MutexFairness == WF_vars(DrainAcquire) /\
                 (\A w \in W : SF_vars(RunAcquire(w) \/ SettleAcquire(w)))
MainStep == SubmitAcquire \/ Increment \/ Enqueue \/ SubmitResult \/ CancelAcquire \/ CancelPending \/
            BeginShutdown \/ CancelQueuedAcquire \/ CancelQueuedSettle \/ CancelLibrary \/ DrainBegin \/
            DrainAcquire \/ Check \/ Join
WorkerStep(w) == Take(w) \/ RunAcquire(w) \/ StartRun(w) \/ Run(w) \/ SettleAcquire(w) \/ Settle(w)
Spec == Init /\ [][Next]_vars /\ MutexFairness /\ WF_vars(MainStep) /\
        (\A w \in W : WF_vars(WorkerStep(w)))
TypeOK == /\ main \in {"idle", "increment", "enqueue", "submit-result", "cancel-lock", "cancel", "shutdown", "cancel-queued", "cancel-queued-settle", "cancel-library", "drain-lock", "check", "wait", "join", "closed"}
          /\ lock \in 0..(Workers+1) /\ nextJob \in 1..(Items+1) /\ outstanding \subseteq J
          /\ state \in [J -> {"new", "pending", "running", "settled"}]
          /\ queue \in Seq(J) /\ submitted \subseteq J
          /\ wpc \in [W -> {"idle", "run-lock", "start", "run", "settle-lock", "settle"}]
          /\ job \in [W -> 0..Items] /\ interrupted \in BOOLEAN /\ released \in BOOLEAN
          /\ cancelJob \in 0..Items
TokensExact == outstanding = Live
Ownership == Cardinality(Borrowed) = Cardinality(Running)
BorrowedRunning == \A j \in Borrowed : state[j] = "running"
NoBorrowAfterRelease == released => Borrowed = {}
NoLeak == main = "closed" => outstanding = {}
BlockingReturns == [](main \in {"shutdown", "cancel-queued", "cancel-queued-settle", "cancel-library", "drain-lock", "check", "wait", "join"} => <>(main = "closed"))
NoCloseTerminates == (WF_vars(Close) => <>(main = "closed"))
====
