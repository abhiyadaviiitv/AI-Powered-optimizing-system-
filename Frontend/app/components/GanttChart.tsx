'use client';

import { useMemo } from 'react';

export interface GanttSegment {
  pid: number | null;
  tag: string;
  label: string;
  start: number;
  end: number;
  duration: number;
  color: string;
  queueLevel?: number;
  timeQuantum?: number | null;
}

interface GanttChartProps {
  segments: GanttSegment[];
}

function formatTime(value: number) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

export function GanttChart({ segments }: GanttChartProps) {
  const { totalSpan, ticks } = useMemo(() => {
    if (!segments.length) {
      return { totalSpan: 0, ticks: [] as number[] };
    }
    const span = segments[segments.length - 1].end;
    const tickCount = Math.min(Math.max(Math.ceil(span / 2), 4), 12);
    const step = tickCount > 0 ? span / tickCount : span;
    const generated: number[] = [];
    for (let i = 0; i <= tickCount; i += 1) {
      generated.push(Number((i * step).toFixed(2)));
    }
    if (generated[generated.length - 1] !== span) {
      generated[generated.length - 1] = span;
    }
    return { totalSpan: span, ticks: generated };
  }, [segments]);

  if (!segments.length || totalSpan <= 0) {
    return (
      <div className="gantt-container" style={{ padding: 24, textAlign: 'center', color: 'var(--text-secondary)' }}>
        No schedulable processes for this algorithm yet.
      </div>
    );
  }

  return (
    <div>
      <div className="gantt-container">
        <div className="gantt-track">
          {segments.map((segment) => {
            const isIdle = segment.pid === null;
            const relative = segment.duration / totalSpan;
            return (
              <div
                key={`${segment.tag}-${segment.start}`}
                className={`gantt-segment${isIdle ? ' idle' : ''}`}
                style={{
                  flexGrow: Math.max(segment.duration, 0.0001),
                  minWidth: relative < 0.05 ? '68px' : undefined,
                  background: isIdle
                    ? 'rgba(255,255,255,0.05)'
                    : `linear-gradient(135deg, ${segment.color}, rgba(255,255,255,0.15))`,
                  borderRight: '1px solid rgba(0, 0, 0, 0.12)',
                  boxShadow: isIdle ? 'none' : `0 8px 25px ${segment.color}33`
                }}
              >
                <div className="segment-label">
                  <span>{segment.label}</span>
                  {!isIdle && typeof segment.timeQuantum === 'number' && (
                    <span className="segment-subtitle">tq {segment.timeQuantum}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="gantt-axis">
        {ticks.map((tick, index) => (
          <div className="gantt-axis-tick" key={`${tick}-${index}`}>
            <span>{formatTime(tick)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
