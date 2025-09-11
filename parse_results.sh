#!/bin/bash
# ================================
# Parse Sysbench + Monitoring Logs
# ================================

mkdir -p results
OUTFILE="results/summary.csv"

# CSV Header
echo "Workload,Threads,EventsPerSec,AvgLatency_ms,95thLatency_ms,CPU_%,Mem_MB,CSwitch/s,IOWait_%" > $OUTFILE

for f in logs/sysbench_*threads.log; do
  workload=$(echo $f | sed -E 's/.*sysbench_([a-zA-Z]+)_[0-9]+threads.*/\1/')
  threads=$(echo $f | sed -E 's/.*_([0-9]+)threads.*/\1/')

  eps=$(grep "events per second:" $f | awk '{print $4}')
  [ -z "$eps" ] && eps=0

  avg_lat=$(grep "avg:" $f | awk '{print $2}')
  [ -z "$avg_lat" ] && avg_lat=0

  p95_lat=$(grep "95th percentile:" $f | awk '{print $3}')
  [ -z "$p95_lat" ] && p95_lat=0

  mpfile="logs/mpstat_${workload}_${threads}threads.log"
  vmfile="logs/vmstat_${workload}_${threads}threads.log"

  cpu=$(awk '/all/ {print 100 - $13}' $mpfile | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
  iowait=$(awk '/all/ {print $6}' $mpfile | tail -n +2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

  cs=$(awk 'NR>2 {print $12}' $vmfile | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
  mem=$(awk 'NR>2 {print $4/1024}' $vmfile | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

  echo "$workload,$threads,$eps,$avg_lat,$p95_lat,$cpu,$mem,$cs,$iowait" >> $OUTFILE
done

echo "✅ Parsed results saved in $OUTFILE"
