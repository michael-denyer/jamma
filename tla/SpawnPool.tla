---------------------------- MODULE SpawnPool ----------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC
(***************************************************************************
Source: src/jamma/io/_parallel_text.py:88-152, matrix_reader.py:225-248,
matrix_writer.py:190-207. Model library-owned ordered imap and process handles
as atomic transitions, not library locks. Each worker borrows one disjoint
memmap slice. Parent polling is a real timeout, not a fairness-free spurious
wake. A fatal worker exit loses its result even if the library replaces it.
Without the poll the parent waits for that result for ever, a liveness
violation; SpawnPool.mutations lists it with the other bugs the properties
detect.
Assume finite parsing/formatting, reliable process termination/join, one
parent, no task-pickling/IPC failure or further SIGINT during terminate/join.
No handwritten mutex, allocation under lock, longjmp or condition variable.
Normal close is after draining; caller interruption can abort from any wait.
***************************************************************************)
CONSTANTS Items, Workers, FailAt
W == 1..Workers
I == 1..Items
VARIABLES pc, status, job, worker, delivered, stopped, resources
pool == <<status, job, worker>>
parent == <<pc, delivered, stopped, resources>>
vars == <<parent, pool>>
Init == /\ pc = "wait"
        /\ status = [i \in I |-> "queued"]
        /\ job = [w \in W |-> 0]
        /\ worker = [w \in W |-> "idle"]
        /\ delivered = <<>> /\ stopped = FALSE /\ resources = "parent"
TypeOK == /\ pc \in {"wait","terminate","join","closed"}
          /\ status \in [I -> {"queued","running","ok","error","lost","consumed","freed"}]
          /\ job \in [W -> 0..Items]
          /\ worker \in [W -> {"idle","running","dead","closed"}]
          /\ delivered \in Seq(I) /\ stopped \in BOOLEAN
          /\ resources \in {"parent","freed"}
Own == /\ \A w \in W : (worker[w] = "running") <=> (job[w] \in I)
       /\ \A w,v \in W : (job[w] # 0 /\ job[w] = job[v]) => w = v
       /\ \A i \in I : status[i] = "running" => \E w \in W : job[w] = i
Order == delivered = [k \in 1..Len(delivered) |-> k]
NoLeak == pc = "closed" =>
          (resources = "freed" /\ \A w \in W : worker[w] = "closed")

Acquire(w) == /\ pc = "wait" /\ worker[w] = "idle" /\ ~stopped
              /\ \E i \in I :
                  /\ status[i] = "queued"
                  /\ \A j \in 1..(i-1) : status[j] # "queued"
                  /\ status' = [status EXCEPT ![i] = "running"]
                  /\ job' = [job EXCEPT ![w] = i]
                  /\ worker' = [worker EXCEPT ![w] = "running"]
              /\ UNCHANGED parent
\* A running worker gives up its job with `result` and becomes `state`.
Leave(w, result, state) ==
    /\ worker[w] = "running" /\ ~stopped
    /\ status' = [status EXCEPT ![job[w]] = result]
    /\ job' = [job EXCEPT ![w] = 0]
    /\ worker' = [worker EXCEPT ![w] = state]
    /\ UNCHANGED parent
Finish(w) == Leave(w, "ok", "idle")
Fail(w) == job[w] \in FailAt /\ Leave(w, "error", "idle")
Die(w) == job[w] \in FailAt /\ Leave(w, "lost", "dead")
Receive == /\ pc = "wait" /\ Len(delivered) < Items
           /\ LET i == Len(delivered)+1 IN
                /\ status[i] = "ok"
                /\ delivered' = Append(delivered, i)
                /\ status' = [status EXCEPT ![i] = "consumed"]
           /\ UNCHANGED <<pc, job, worker, stopped, resources>>
\* The parent leaves its wait to terminate the pool.
Abort == /\ pc = "wait" /\ pc' = "terminate"
         /\ UNCHANGED <<pool, delivered, stopped, resources>>
Raise == /\ Len(delivered) < Items /\ status[Len(delivered)+1] = "error"
         /\ Abort
Poll == (\E w \in W : worker[w] = "dead") /\ Abort
Interrupt == Abort
End == Len(delivered) = Items /\ Abort
Terminate == /\ pc = "terminate" /\ stopped' = TRUE /\ pc' = "join"
             /\ UNCHANGED <<pool, delivered, resources>>
Stop(w) == /\ stopped /\ worker[w] # "closed"
           /\ worker' = [worker EXCEPT ![w] = "closed"]
           /\ status' = IF job[w] = 0 THEN status ELSE [status EXCEPT ![job[w]] = "freed"]
           /\ job' = [job EXCEPT ![w] = 0]
           /\ UNCHANGED parent
Join == /\ pc = "join"
        /\ \A w \in W : worker[w] = "closed"
        /\ pc' = "closed" /\ resources' = "freed"
        /\ status' = [i \in I |-> IF status[i] = "consumed" THEN "consumed" ELSE "freed"]
        /\ UNCHANGED <<job, worker, delivered, stopped>>
Done == /\ pc = "closed" /\ UNCHANGED vars
Next == Done \/ Receive \/ Raise \/ Poll \/ Interrupt \/ End \/ Terminate \/ Join \/
        (\E w \in W : Acquire(w) \/ Finish(w) \/ Fail(w) \/ Die(w) \/ Stop(w))
Fair == /\ WF_vars(Receive) /\ WF_vars(Raise) /\ WF_vars(Poll)
        /\ WF_vars(End) /\ WF_vars(Terminate) /\ WF_vars(Join)
        /\ \A w \in W : WF_vars(Acquire(w)) /\ WF_vars(Finish(w)) /\ WF_vars(Stop(w))
Closed == <> (pc = "closed")
Spec == Init /\ [][Next]_vars /\ Fair
=============================================================================
