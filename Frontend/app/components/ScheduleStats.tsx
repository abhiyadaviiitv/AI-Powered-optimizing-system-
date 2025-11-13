'use client';

import { useMemo } from 'react';

export interface ScheduleMetricsView {
  averageBurstTime: number;
  averageTurnaroundTime: number;
  averageWaitingTime: number;
  throughput: number;
  cpuUtilization: number;
  makespan: number;
  totalBurst: number;
  idleTime: number;
}

interface ScheduleStatsProps {
  metrics: ScheduleMetricsView;
}

function formatNumber(value: number, fractionDigits = 2) {
  if (!Number.isFinite(value)) {
    return '0';
  }
  return value.toFixed(fractionDigits);
}

export function ScheduleStats({ metrics }: ScheduleStatsProps) {
  const items = useMemo(
    () => [
      { label: 'Avg Burst Time', value: formatNumber(metrics.averageBurstTime) },
      { label: 'Avg Turnaround', value: formatNumber(metrics.averageTurnaroundTime) },
      { label: 'Avg Waiting', value: formatNumber(metrics.averageWaitingTime) },
      { label: 'Throughput', value: `${formatNumber(metrics.throughput, 3)} jobs/unit` },
      { label: 'CPU Utilization', value: `${formatNumber(metrics.cpuUtilization)}%` }
    ],
    [metrics]
  );

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: 16
      }}
    >
      {items.map((item) => (
        <div key={item.label} className="metric-card">
          <span>{item.label}</span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}
