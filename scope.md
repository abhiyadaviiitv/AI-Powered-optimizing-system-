DAY 1

# AI-Powered Self-Optimizing Operating System
## Project Scope and Objectives

### 1. What does "self-optimizing" mean?
In the context of this project, **self-optimizing** refers to an OS-level system that can:
- **Observe** its current workload and resource usage patterns in real time (via low-overhead telemetry like eBPF).
- **Analyze** these signals using machine learning models (classification, prediction, reinforcement learning).
- **Actuate** decisions to improve system performance automatically by tuning kernel-level knobs (e.g., CPU scheduling, cgroups, I/O priorities).
- **Adapt** continuously as workload conditions change.

The goal is to **replace static heuristics** in resource management with an **adaptive AI-driven controller** that:
- Allocates resources dynamically.
- Improves overall system responsiveness.
- Reduces manual tuning required by administrators.

---

### 2. Focus Area (First Iteration)
To keep the project achievable in 6 weeks, the first iteration will focus on **CPU and scheduling optimization**, since:
- CPU scheduling is critical for most workloads (interactive + batch).
- Kernel already exposes controllable parameters (cgroups v2, `nice`, CPU governor).
- Easier to measure improvement through well-defined benchmarks.

Future extensions (if time permits):
- **Memory management** (swappiness, page cache tuning).
- **I/O scheduling** (block device schedulers, I/O priorities).
- **Energy optimization** (DVFS governors, core pinning).

---

### 3. Metrics to Measure
We will measure and track both **performance** and **system health** metrics.

#### Performance Metrics
- **Throughput**: number of operations completed per second (e.g., requests/sec in benchmarks).
- **Mean latency**: average time per operation.
- **95th / 99th percentile latency**: tail latencies, critical for interactive workloads.
- **CPU utilization**: percentage of CPU cycles used over time.
- **Runnable queue length**: number of processes waiting for CPU (from scheduler telemetry).
- **Fairness index**: degree to which CPU is fairly distributed among competing processes.

#### System Health / Resource Metrics
- **Context switches/sec**: to detect scheduler overhead.
- **Syscalls/sec**: proxy for workload intensity.
- **Cache miss rate** (if measurable from perf/eBPF).
- **I/O wait percentage** (to catch CPU stalls from I/O bottlenecks).

---

### 4. Optimization Goals
The system should aim to:
1. **Reduce latency variance** (stabilize tail latencies).
2. **Maximize throughput** while avoiding starvation of low-priority tasks.
3. **Maintain fairness** across processes (avoid monopolization).
4. **Minimize wasted CPU cycles** (e.g., idle + runnable imbalance).
5. (Optional) **Improve energy efficiency** without sacrificing performance.

---

### 5. Success Criteria
- Demonstrated improvement in **p95 latency** and/or **throughput** compared to baseline kernel defaults.
- Stable operation under diverse workloads (CPU-bound, I/O-heavy, mixed).
- Low overhead (<5%) for telemetry collection and decision-making.
- Safe actuation with rollback capability (no system crashes or starvation events).

---

### 6. Deliverables from Scope Phase
- A clear definition of optimization goals (above).
- A chosen set of workloads/benchmarks for evaluation:
  - **Sysbench** (CPU test).
  - **Phoronix Test Suite** (various real-world workloads).
  - **fio** (for I/O-heavy benchmarks, optional).
- A baseline log of system performance without AI interventions.
