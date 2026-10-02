----------------------------- MODULE MatrixWriter -----------------------------
(* src/jamma/io/_native_matrix_writer.py:65-99;
   src/jamma/utils/atomic_publish.py:68-80;
   src/jamma/core/thread_pool.py, completion-barrier contract.
   Calling-code model, not a model of Python's executor implementation.
   Future.result() blocking is pc="wait"; atomic Future completion wakes it.
   No explicit mutex or spurious wakeup is invented for library-owned Futures.
   Buffer references in pending and workers are aliases; workers have exclusive
   mutation rights until completion. Consumer alone owns the temporary file.
   Each submitted formatter terminates under weak fairness. Workers=1 and Slots
   below 2*Workers exercise hypothetical reductions of the threaded protocol;
   production uses its sequential branch for effective workers <=1.
   Interrupts during shutdown set an error but retain the completion barrier.
   The JoiningThreadPoolExecutor contract is assumed here; its Condition
   implementation requires a separate protocol model.
   MatrixWriter.mutations lists the bugs the properties detect. *)
EXTENDS Naturals, Sequences, FiniteSets, TLC
CONSTANTS Items, Slots, Workers, FailAt, InterruptJoin
VARIABLES pc, nextItem, pending, current, state, buffer, worker, written,
          error, file, published, joinInterrupted, reportAt
queue == <<nextItem, pending, current>>
jobs == <<state, buffer, worker>>
output == <<written, file, published>>
failure == <<error, joinInterrupted, reportAt>>
vars == <<pc, queue, jobs, output, failure>>
Jobs == 1..Items
Buffers == 1..Slots
Threads == 1..Workers
Live == {"queued","running","done","failed"}
Held == Live \cup {"cancelled"}
Pending == {pending[i] : i \in 1..Len(pending)}
WorkersIdle == \A w \in Threads : worker[w] = 0
Init == /\ pc = "fill" /\ nextItem = 1 /\ pending = <<>> /\ current = 0
        /\ state = [j \in Jobs |-> "new"]
        /\ buffer = [j \in Jobs |-> 0]
        /\ worker = [w \in Threads |-> 0]
        /\ written = <<>> /\ error = "none" /\ file = "open"
        /\ published = FALSE /\ joinInterrupted = FALSE /\ reportAt = 0
FreeBuffers == Buffers \ {buffer[j] : j \in {k \in Jobs : state[k] \in Live}}
Submit == /\ pc \in {"fill","resubmit"} /\ nextItem <= Items
          /\ FreeBuffers # {}
          /\ LET b == CHOOSE b \in FreeBuffers : TRUE IN
             /\ buffer' = [buffer EXCEPT ![nextItem] = b]
             /\ state' = [state EXCEPT ![nextItem] = "queued"]
          /\ pending' = Append(pending,nextItem) /\ nextItem' = nextItem + 1
          /\ pc' = IF pc = "resubmit" THEN "pick" ELSE "fill"
          /\ current' = 0
          /\ UNCHANGED <<worker, output, failure>>
FinishFill == /\ pc = "fill" /\ (nextItem > Items \/ FreeBuffers = {})
              /\ pc' = "pick" /\ UNCHANGED <<queue, jobs, output, failure>>
Pick == /\ pc = "pick" /\ Len(pending) > 0
        /\ current' = Head(pending) /\ pending' = Tail(pending) /\ pc' = "wait"
        /\ UNCHANGED <<nextItem, jobs, output, failure>>
End == /\ pc = "pick" /\ pending = <<>> /\ pc' = "shutdown"
       /\ UNCHANGED <<queue, jobs, output, failure>>
Start(w) == /\ worker[w] = 0 /\ pc \notin {"join","close","closed"}
            /\ \E j \in Jobs : /\ state[j] = "queued"
                /\ worker' = [worker EXCEPT ![w] = j]
                /\ state' = [state EXCEPT ![j] = "running"]
            /\ UNCHANGED <<pc, queue, buffer, output, failure>>
Finish(w) == /\ worker[w] # 0
             /\ LET j == worker[w] IN state' = [state EXCEPT ![j] = IF j \in FailAt THEN "failed" ELSE "done"]
             /\ worker' = [worker EXCEPT ![w] = 0]
             /\ UNCHANGED <<pc, queue, buffer, output, failure>>
Result == /\ pc = "wait" /\ state[current] \in {"done","failed"}
          /\ LET failed == state[current] = "failed" IN
             /\ pc' = IF failed THEN "shutdown" ELSE "write"
             /\ error' = IF failed THEN "format" ELSE error
             /\ reportAt' = IF failed THEN current ELSE reportAt
          /\ UNCHANGED <<queue, jobs, output, joinInterrupted>>
Write == /\ pc = "write" /\ error = "none" /\ file = "open"
         /\ written' = Append(written,current)
         /\ state' = [state EXCEPT ![current] = "written"]
         /\ pc' = "resubmit"
         /\ UNCHANGED <<queue, buffer, worker, file, published, failure>>
NoMore == /\ pc = "resubmit" /\ nextItem > Items
          /\ current' = 0 /\ pc' = "pick"
          /\ UNCHANGED <<nextItem, pending, jobs, output, failure>>
ExternalError == /\ pc \in {"fill","pick","wait","write","resubmit"}
                 /\ error' = IF pc = "write" THEN "write" ELSE IF pc \in {"fill","resubmit"} THEN "submit" ELSE "interrupt"
                 /\ pc' = "shutdown"
                 /\ UNCHANGED <<queue, jobs, output, joinInterrupted, reportAt>>
Shutdown == /\ pc = "shutdown" /\ pc' = "join"
            /\ state' = [j \in Jobs |-> IF state[j] = "queued" THEN "cancelled" ELSE state[j]]
            /\ UNCHANGED <<queue, buffer, worker, output, failure>>
Join == /\ pc = "join" /\ WorkersIdle
        /\ pc' = "close"
        /\ state' = [j \in Jobs |-> IF state[j] \in Held THEN "freed" ELSE state[j]]
        /\ pending' = <<>> /\ current' = 0
        /\ UNCHANGED <<nextItem, buffer, worker, output, failure>>
AbortJoin == /\ InterruptJoin /\ pc = "join" /\ ~WorkersIdle
             /\ pc' = "join" /\ error' = "interrupt" /\ joinInterrupted' = TRUE
             /\ UNCHANGED <<queue, jobs, output, reportAt>>
Close == /\ pc = "close" /\ pc' = "closed" /\ file' = "closed"
         /\ published' = (error = "none")
         /\ UNCHANGED <<queue, jobs, written, failure>>
Done == pc = "closed" /\ UNCHANGED vars
WorkerStep(w) == Start(w) \/ Finish(w)
Foreground == Submit \/ FinishFill \/ Pick \/ End \/ Result \/ Write \/ NoMore \/ Shutdown \/ Join \/ Close
Next == Foreground \/ ExternalError \/ AbortJoin \/ (\E w \in Threads : WorkerStep(w)) \/ Done
Spec == Init /\ [][Next]_vars /\ WF_vars(Foreground)
        /\ \A w \in Threads : WF_vars(WorkerStep(w))
TypeOK == /\ pc \in {"fill","pick","wait","write","resubmit","shutdown","join","close","closed"}
          /\ nextItem \in 1..(Items+1) /\ pending \in Seq(Jobs)
          /\ current \in Jobs \cup {0}
          /\ state \in [Jobs -> {"new","queued","running","done","failed","written","cancelled","freed"}]
          /\ buffer \in [Jobs -> Buffers \cup {0}]
          /\ worker \in [Threads -> Jobs \cup {0}]
          /\ written \in Seq(Jobs) /\ file \in {"open","closed"}
          /\ reportAt \in Jobs \cup {0}
Ownership == /\ \A i,k \in 1..Len(pending) : i # k => pending[i] # pending[k]
             /\ (current = 0 \/ current \notin Pending)
             /\ \A j \in Jobs : state[j] \in Live => j = current \/ j \in Pending
             /\ \A j,k \in Jobs : (j # k /\ state[j] \in Live /\ state[k] \in Live) => buffer[j] # buffer[k]
             /\ \A w,v \in Threads : (w # v /\ worker[w] # 0) => worker[w] # worker[v]
             /\ \A w \in Threads : worker[w] # 0 => state[worker[w]] = "running"
             /\ \A j \in Jobs : state[j] = "running" => \E w \in Threads : worker[w] = j
DeliveryOrder == \A i \in 1..Len(written) : written[i] = i
ErrorOrder == error = "format" => /\ reportAt = Len(written)+1 /\ reportAt \in FailAt
NoDeliveryAfterError == error # "none" => pc \notin {"write","resubmit"}
PublishComplete == published => error = "none" /\ Len(written) = Items /\ WorkersIdle
JoinedAtReturn == pc = "closed" => WorkersIdle
NoLeakAtReturn == pc = "closed" => /\ pending = <<>> /\ current = 0
                                  /\ \A j \in Jobs : state[j] \notin Held
Terminates == <>(pc = "closed")
ResultReturns == [](pc = "wait" => <>(pc # "wait"))
CloseReturns == [](pc \in {"shutdown","join","close"} => <>(pc = "closed"))
=============================================================================
