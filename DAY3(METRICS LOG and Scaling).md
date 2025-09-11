
## Day 3 – Baseline Workload + Metrics Collection

###  Goal
- Run controlled CPU workloads with varying **thread counts** (scaling).  
- Observe how performance changes when scaling from **light load → heavy load**.  
- Collect **application-level** and **system-level** metrics.  
- Automate logging into files for reproducibility.

---

###  Concept: What is Scaling?
- **Scaling** = Running the same workload with different numbers of threads or processes.  
- In our case, we use `sysbench cpu`:
  - **1 thread** → Light load (single-threaded baseline).
  - **2 threads** → Moderate load (parallelism starts).
  - **4 threads** → Matches a typical core CPU (balanced).
  - **8 threads** → Oversubscription (more threads than CPU cores).


#### Why Scaling?
- To see **how well the CPU handles parallel workloads**.  
- To identify the **saturation point** (when adding more threads no longer improves performance).  
- To study **latency vs throughput trade-offs**.  

####  Effects of Scaling:
1. **Throughput** (events/sec):
   - Increases with more threads **until CPU saturates**.
   - Flattens or even decreases once scheduling overhead dominates.
2. **Latency** (ms per event):
   - Stays low for light loads.
   - Increases as more threads contend for CPU.
3. **Runnable Queue Length**:
   - Roughly matches number of runnable threads when CPU is busy.
4. **Context Switches**:
   - Increase with more threads, because CPU must switch execution between them.
5. **Fairness**:
   - Ideally all threads get similar CPU share.
   - Can break down under oversubscription.

---

### Automating Commands & Logging

Instead of running commands manually, let’s create a **script file** (`run_experiments.sh`) to:
- Run workloads with 1, 2, 4, and 8 threads.
- Save each run’s results into a log file.
- Also capture system-level stats in parallel.

