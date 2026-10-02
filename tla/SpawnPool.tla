---------------------------- MODULE SpawnPool ----------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC
(***************************************************************************
Source: src/jamma/io/_parallel_text.py:88-152, matrix_reader.py:225-248,
matrix_writer.py:190-207. Model library-owned ordered imap and process handles
as atomic transitions, not library locks. Each worker borrows one disjoint
memmap slice. Parent polling is a real timeout, not a fairness-free spurious
wake. A fatal worker exit loses its result even if the library replaces it.
DetectDeath=FALSE preserves that calling-code bug as a liveness violation.
Assume finite parsing/formatting, reliable process termination/join, one
parent, no task-pickling/IPC failure or further SIGINT during terminate/join.
No handwritten mutex, allocation under lock, longjmp or condition variable.
Normal close is after draining; caller interruption can abort from any wait.
***************************************************************************)
CONSTANTS Items, Workers, FailAt, DetectDeath, Mutation
W == 1..Workers
I == 1..Items
VARIABLES pc, status, job, worker, delivered, stopped, resources
vars == <<pc, status, job, worker, delivered, stopped, resources>>
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
              /\ UNCHANGED <<pc, delivered, stopped, resources>>
Finish(w) == /\ worker[w] = "running" /\ ~stopped
             /\ status' = [status EXCEPT ![job[w]] = "ok"]
             /\ job' = IF Mutation = "own" THEN job ELSE [job EXCEPT ![w] = 0]
             /\ worker' = [worker EXCEPT ![w] = IF Mutation = "type" THEN "broken" ELSE "idle"]
             /\ UNCHANGED <<pc, delivered, stopped, resources>>
Fail(w) == /\ worker[w] = "running" /\ job[w] \in FailAt /\ ~stopped
           /\ status' = [status EXCEPT ![job[w]] = "error"]
           /\ job' = [job EXCEPT ![w] = 0]
           /\ worker' = [worker EXCEPT ![w] = "idle"]
           /\ UNCHANGED <<pc, delivered, stopped, resources>>
Die(w) == /\ worker[w] = "running" /\ job[w] \in FailAt /\ ~stopped
          /\ status' = [status EXCEPT ![job[w]] = "lost"]
          /\ job' = [job EXCEPT ![w] = 0]
          /\ worker' = [worker EXCEPT ![w] = "dead"]
          /\ UNCHANGED <<pc, delivered, stopped, resources>>
Receive == /\ pc = "wait" /\ Len(delivered) < Items
           /\ LET i == Len(delivered)+1 IN
                /\ status[i] = "ok"
                /\ delivered' = Append(delivered, IF Mutation = "order" THEN Items ELSE i)
                /\ status' = [status EXCEPT ![i] = "consumed"]
           /\ UNCHANGED <<pc, job, worker, stopped, resources>>
Raise == /\ pc = "wait" /\ Len(delivered) < Items
         /\ status[Len(delivered)+1] = "error" /\ pc' = "terminate"
         /\ UNCHANGED <<status, job, worker, delivered, stopped, resources>>
Poll == /\ pc = "wait" /\ DetectDeath
        /\ \E w \in W : worker[w] = "dead"
        /\ pc' = "terminate"
        /\ UNCHANGED <<status, job, worker, delivered, stopped, resources>>
Interrupt == /\ pc = "wait" /\ pc' = "terminate"
             /\ UNCHANGED <<status, job, worker, delivered, stopped, resources>>
End == /\ pc = "wait" /\ Len(delivered) = Items /\ pc' = "terminate"
       /\ UNCHANGED <<status, job, worker, delivered, stopped, resources>>
Terminate == /\ pc = "terminate" /\ stopped' = TRUE /\ pc' = "join"
             /\ UNCHANGED <<status, job, worker, delivered, resources>>
Stop(w) == /\ stopped /\ worker[w] # "closed"
           /\ worker' = [worker EXCEPT ![w] = "closed"]
           /\ status' = IF job[w] = 0 THEN status ELSE [status EXCEPT ![job[w]] = "freed"]
           /\ job' = [job EXCEPT ![w] = 0]
           /\ UNCHANGED <<pc, delivered, stopped, resources>>
Join == /\ pc = "join"
        /\ (Mutation = "join" \/ \A w \in W : worker[w] = "closed")
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
