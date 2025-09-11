import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/summary.csv")

print("First rows of data:\n", df.head())
print("\nSummary statistics:\n", df.describe())

# Plot workload-wise
workloads = df["Workload"].unique()

for wl in workloads:
    subdf = df[df["Workload"] == wl].sort_values("Threads")   # <-- ensure sorted order

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Sysbench {wl} Workload - Performance Metrics", fontsize=16)

    # Throughput
    axs[0,0].plot(subdf["Threads"], subdf["EventsPerSec"], marker='o', color='blue')
    axs[0,0].set_title("Throughput vs Threads")
    axs[0,0].set_xlabel("Threads")
    axs[0,0].set_ylabel("Events/sec")
    axs[0,0].grid(True)

    # Latency
    axs[0,1].plot(subdf["Threads"], subdf["AvgLatency_ms"], marker='x', color='red', label="Avg Latency")
    axs[0,1].plot(subdf["Threads"], subdf["95thLatency_ms"], marker='s', color='green', label="95th Latency")
    axs[0,1].set_title("Latency vs Threads")
    axs[0,1].set_xlabel("Threads")
    axs[0,1].set_ylabel("Latency (ms)")
    axs[0,1].legend()
    axs[0,1].grid(True)

    # CPU
    axs[1,0].plot(subdf["Threads"], subdf["CPU_%"], marker='o', color='purple')
    axs[1,0].set_title("CPU Utilization vs Threads")
    axs[1,0].set_xlabel("Threads")
    axs[1,0].set_ylabel("CPU (%)")
    axs[1,0].grid(True)

    # Memory
    axs[1,1].plot(subdf["Threads"], subdf["Mem_MB"], marker='o', color='orange')
    axs[1,1].set_title("Memory Usage vs Threads")
    axs[1,1].set_xlabel("Threads")
    axs[1,1].set_ylabel("Memory (MB)")
    axs[1,1].grid(True)

    # Context Switches
    axs[2,0].plot(subdf["Threads"], subdf["CSwitch/s"], marker='o', color='brown')
    axs[2,0].set_title("Context Switches vs Threads")
    axs[2,0].set_xlabel("Threads")
    axs[2,0].set_ylabel("CSwitch/s")
    axs[2,0].grid(True)

    # I/O Wait
    axs[2,1].plot(subdf["Threads"], subdf["IOWait_%"], marker='o', color='black')
    axs[2,1].set_title("I/O Wait vs Threads")
    axs[2,1].set_xlabel("Threads")
    axs[2,1].set_ylabel("IOWait (%)")
    axs[2,1].grid(True)

    # adjust layout to leave room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
