#!/usr/bin/env python3
"""
Generate Dataset A: system + per-process snapshots.

Produces CSV with columns:
snapshot_id,system_timestamp,system_cpu_percent,system_memory_percent,system_load_1min,
system_process_count,timestamp,pid,name,status,ppid,cpu_percent,memory_percent,
memory_mb,num_threads,nice,process_age,process_type,io_intensity,user_time,system_time,cmdline

Usage:
    python generate_dataset_a.py --out dataset_a.csv --interval 1.0 --duration 60

This will sample every `interval` seconds for `duration` seconds (total snapshots = duration/interval).
"""

import csv
import time
import argparse
import psutil
import os
from datetime import datetime

# Heuristic to map process name/cmdline -> process_type
def infer_process_type(name, cmdline):
    n = (name or "").lower()
    c = (" ".join(cmdline) if cmdline else "").lower()
    if "python" in n or "python" in c or "pip" in n or "node" in n or "java" in n or "gcc" in n:
        return "development"
    if "code" in n or "vscode" in c or "explorer" in n or "chrome" in n or "firefox" in n:
        return "interactive"
    if "system" in n or "svchost" in n or "idle" in n:
        return "system"
    return "other"

def get_load_1min():
    # os.getloadavg not on Windows; psutil.getloadavg available on some platforms
    try:
        if hasattr(os, "getloadavg"):
            return os.getloadavg()[0]
        if hasattr(psutil, "getloadavg"):
            return psutil.getloadavg()[0]
    except Exception:
        pass
    return 0.0

def sample_once():
    """
    Return dicts for processes keyed by pid with initial metrics we need to compute deltas later.
    """
    procs = {}
    for p in psutil.process_iter(attrs=["pid", "name", "status", "ppid", "num_threads", "nice", "create_time"]):
        pid = p.info["pid"]
        try:
            # call cpu_percent once to initialize internal counters (value meaningless until second call)
            p_cpu = p.cpu_percent(interval=None)
            mem = p.memory_percent()
            mem_mb = 0
            try:
                mem_mb = p.memory_info().rss / (1024 * 1024)
            except Exception:
                mem_mb = 0.0
            io = None
            try:
                io = p.io_counters()
            except Exception:
                io = None
            cpu_times = None
            try:
                cpu_times = p.cpu_times()
            except Exception:
                cpu_times = None

            cmdline = []
            try:
                cmdline = p.cmdline()
            except Exception:
                cmdline = []

            procs[pid] = {
                "proc": p,
                "name": p.info.get("name"),
                "status": p.info.get("status"),
                "ppid": p.info.get("ppid"),
                "num_threads": p.info.get("num_threads"),
                "nice": p.info.get("nice"),
                "create_time": p.info.get("create_time"),
                "mem_percent": mem,
                "mem_mb": mem_mb,
                "io_counters": io,
                "cpu_times": cpu_times,
                "cmdline": cmdline,
                # placeholder for later:
                "cpu_percent": 0.0,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue
    return procs

def collect_snapshot(snapshot_id, interval, writer):
    # initial sample
    initial = sample_once()
    # measure system-level baseline
    # call system cpu_percent with interval so it sleeps there (or we can sleep separately)
    # we want to sample system over same interval as process deltas
    system_cpu = psutil.cpu_percent(interval=interval)
    system_mem = psutil.virtual_memory().percent
    system_process_count = len(psutil.pids())
    load_1min = get_load_1min()
    system_timestamp = time.time()

    # take second sample quickly to compute per-process cpu_percent and io deltas
    final = {}
    for pid, info in list(initial.items()):
        p = info["proc"]
        try:
            # cpu_percent called before; now call again to get usage over the interval
            cpu_p = p.cpu_percent(interval=None)
            io2 = None
            try:
                io2 = p.io_counters()
            except Exception:
                io2 = None
            cpu_times2 = None
            try:
                cpu_times2 = p.cpu_times()
            except Exception:
                cpu_times2 = None

            # compute io intensity: sum of read_bytes + write_bytes if available, else 0
            io_intensity = 0.0
            if info["io_counters"] and io2:
                try:
                    prev_bytes = getattr(info["io_counters"], "read_bytes", 0) + getattr(info["io_counters"], "write_bytes", 0)
                    now_bytes = getattr(io2, "read_bytes", 0) + getattr(io2, "write_bytes", 0)
                    io_intensity = float(max(0, now_bytes - prev_bytes)) / max(1.0, interval)  # bytes/sec
                except Exception:
                    io_intensity = 0.0
            else:
                io_intensity = 0.0

            user_t = getattr(cpu_times2, "user", 0.0) if cpu_times2 else 0.0
            system_t = getattr(cpu_times2, "system", 0.0) if cpu_times2 else 0.0

            final[pid] = {
                "cpu_percent": cpu_p,
                "io_intensity": io_intensity,
                "user_time": user_t,
                "system_time": system_t,
                "mem_percent": info["mem_percent"],
                "mem_mb": info["mem_mb"],
                "name": info["name"],
                "status": info["status"],
                "ppid": info["ppid"],
                "num_threads": info["num_threads"],
                "nice": info["nice"],
                "create_time": info["create_time"],
                "cmdline": info["cmdline"],
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue

    # write rows
    timestamp_iso = datetime.fromtimestamp(system_timestamp).isoformat()
    for pid, info in final.items():
        name = info["name"]
        cmdline = info["cmdline"] or []
        create_time = info["create_time"] or system_timestamp
        process_age = float(system_timestamp - create_time)
        process_type = infer_process_type(name, cmdline)
        cpu_percent = info["cpu_percent"]
        memory_percent = info["mem_percent"]
        memory_mb = info["mem_mb"]
        num_threads = info["num_threads"]
        nice = info["nice"]
        io_intensity = info["io_intensity"]
        user_time = info["user_time"]
        system_time = info["system_time"]
        ppid = info["ppid"]
        status = info["status"]
        cmdline_str = " ".join(cmdline) if cmdline else ""

        row = {
            "snapshot_id": snapshot_id,
            "system_timestamp": f"{system_timestamp:.6f}",
            "system_cpu_percent": f"{system_cpu:.3f}",
            "system_memory_percent": f"{system_mem:.3f}",
            "system_load_1min": f"{load_1min:.6f}",
            "system_process_count": system_process_count,
            "timestamp": f"{timestamp_iso}",
            "pid": pid,
            "name": name or "",
            "status": status or "",
            "ppid": ppid or 0,
            "cpu_percent": f"{cpu_percent:.6f}",
            "memory_percent": f"{memory_percent:.12f}",
            "memory_mb": f"{memory_mb:.6f}",
            "num_threads": num_threads or 0,
            "nice": nice if nice is not None else 0,
            "process_age": f"{process_age:.6f}",
            "process_type": process_type,
            "io_intensity": f"{io_intensity:.6f}",
            "user_time": f"{user_time:.6f}",
            "system_time": f"{system_time:.6f}",
            "cmdline": cmdline_str,
        }

        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Generate Dataset A snapshots to CSV.")
    parser.add_argument("--out", "-o", default="dataset_a.csv", help="Output CSV file")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--duration", "-d", type=float, default=60.0, help="Total duration in seconds (0 for infinite)")
    parser.add_argument("--snapshots", "-n", type=int, default=0, help="Number of snapshots (overrides duration if >0)")
    args = parser.parse_args()

    fieldnames = [
        "snapshot_id","system_timestamp","system_cpu_percent","system_memory_percent","system_load_1min",
        "system_process_count","timestamp","pid","name","status","ppid","cpu_percent","memory_percent",
        "memory_mb","num_threads","nice","process_age","process_type","io_intensity","user_time","system_time","cmdline"
    ]

    # open file and write header
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        snapshot_id = 0
        start = time.time()
        if args.snapshots > 0:
            max_snapshots = args.snapshots
        elif args.duration > 0:
            max_snapshots = int(max(1, args.duration / args.interval))
        else:
            max_snapshots = None  # infinite

        while True:
            snapshot_id += 1
            try:
                collect_snapshot(snapshot_id, args.interval, writer)
                f.flush()
            except KeyboardInterrupt:
                print("\nUser requested stop. Exiting.")
                break
            except Exception as e:
                # log and continue
                print(f"[warning] snapshot error: {e}")

            if max_snapshots is not None and snapshot_id >= max_snapshots:
                break

            # if using duration-based termination, check time
            if args.duration > 0 and (time.time() - start) >= args.duration:
                break

    print(f"Done. Output written to {args.out}")

if __name__ == "__main__":
    main()
