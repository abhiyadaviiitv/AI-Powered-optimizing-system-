#!/bin/bash

mkdir -p logs results

WORKLOADS=("cpu" "memory" "fileio")
THREADS=(1 2 4 8 16)

# Remove old logs but keep results
rm -f logs/*

OUTFILE="results/summary.csv"

# Write header only if file does not exist or is empty
if [ ! -s "$OUTFILE" ]; then
  echo "Workload,Threads,EventsPerSec,AvgLatency_ms,95thLatency_ms,CPU_%,Mem_MB,CSwitch/s,IOWait_%" > "$OUTFILE"
fi
for wl in "${WORKLOADS[@]}"; do
  for t in "${THREADS[@]}"; do
    echo ">>> Running $wl with $t threads"

    # Start monitoring tools
    mpstat 1 > logs/mpstat_${wl}_${t}threads.log &
    MPSTAT_PID=$!
    vmstat 1 > logs/vmstat_${wl}_${t}threads.log &
    VMSTAT_PID=$!
    pidstat -w -r -u 1 > logs/pidstat_${wl}_${t}threads.log &
    PIDSTAT_PID=$!

    # --- Run Sysbench workload ---
    if [ "$wl" == "fileio" ]; then
      sysbench fileio --file-num=128 --file-total-size=2G prepare > /dev/null 2>&1
      sysbench fileio --file-num=128 --file-total-size=2G --file-test-mode=rndrw --threads=$t --time=10 run > logs/sysbench_${wl}_${t}threads.log
      sysbench fileio cleanup > /dev/null 2>&1
    else
      sysbench $wl --threads=$t --time=10 run > logs/sysbench_${wl}_${t}threads.log
    fi

    # Stop monitors
    kill $MPSTAT_PID $VMSTAT_PID $PIDSTAT_PID 2>/dev/null
    wait $MPSTAT_PID $VMSTAT_PID $PIDSTAT_PID 2>/dev/null

    # --- Extract Sysbench metrics ---
    logfile="logs/sysbench_${wl}_${t}threads.log"

    case "$wl" in
      cpu)
            eps=$(awk  '/events per second:/ {print $4}'  "$logfile")
            avg_lat=$(awk  '/avg:/ {print $2}'            "$logfile")
            p95_lat=$(awk  '/95th percentile:/ {print $3}' "$logfile")
            ;;
      memory)
            eps=$(awk  '/Total operations:/ {print $3}'   "$logfile")
            avg_lat=$(awk  '/avg:/ {print $2}'            "$logfile")
            p95_lat=$(awk  '/95th percentile:/ {print $3}' "$logfile")
            ;;
      fileio)
            # We'll take total events / total time as throughput
              eps=$(awk '
              /total number of events:/ {ev=$NF}
              /total time:/ {val=$NF; gsub("s","",val); t=val}
              END {if(t>0) printf "%.2f", ev/t; else print 0}
            ' "$logfile")

            avg_lat=$(awk '/avg:/ {print $2}' "$logfile")
            p95_lat=$(awk '/95th percentile:/ {print $3}' "$logfile")
            avg_lat=$(awk  '/avg:/ {print $2}'            "$logfile")
            p95_lat=$(awk  '/95th percentile:/ {print $3}' "$logfile")
            ;;
    esac

    # safe defaults
    [ -z "$eps" ]     && eps=0
    [ -z "$avg_lat" ] && avg_lat=0
    [ -z "$p95_lat" ] && p95_lat=0

    # --- Extract system metrics ---
    mpfile="logs/mpstat_${wl}_${t}threads.log"
    vmfile="logs/vmstat_${wl}_${t}threads.log"

    # CPU utilization (%)
    cpu=$(awk '/all/ {print 100 - $13}' $mpfile | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

    # IOWait (%)
    iowait=$(awk '/all/ {print $6}' $mpfile | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

    # Context Switches (per second)
    cs=$(awk 'NR>2 {print $12}' $vmfile | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

    # Memory usage (MB, from free column of vmstat)
    mem=$(awk 'NR>2 {print $4/1024}' $vmfile | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

    # Append row to CSV
    echo "$wl,$t,$eps,$avg_lat,$p95_lat,$cpu,$mem,$cs,$iowait" >> $OUTFILE

    echo ">>> Finished $wl with $t threads"
    echo
  done
done

echo "✅ All experiments complete. Results saved in $OUTFILE"
