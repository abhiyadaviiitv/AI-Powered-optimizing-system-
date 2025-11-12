'use client';

import { useState } from 'react';

export interface ProcessItem {
  pid: number;
  tag: string;
  name: string;
  algorithm: string;
  cpuPercent: number;
  memoryMb: number;
  status: string;
  predictedBurstTime?: number | null;
  predictedTimeQuantum?: number | null;
  bestAlgorithm?: string;
}

interface ProcessListProps {
  processes: ProcessItem[];
  highlightAlgorithm?: string;
}

const INITIAL_DISPLAY_COUNT = 10;

export function ProcessList({ processes, highlightAlgorithm }: ProcessListProps) {
  const [displayCount, setDisplayCount] = useState(INITIAL_DISPLAY_COUNT);
  const displayedProcesses = processes.slice(0, displayCount);
  const hasMore = processes.length > displayCount;

  const handleLoadMore = () => {
    setDisplayCount((prev) => Math.min(prev + 10, processes.length));
  };

  return (
    <div className="panel" style={{ flex: 1, minWidth: 400 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <p className="badge">Processes</p>
          <h2 style={{ fontSize: 24, marginTop: 12 }}>Live process feed</h2>
        </div>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 44,
            height: 44,
            borderRadius: '50%',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}
        >
          📡
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          maxHeight: 600,
          overflowY: 'auto',
          paddingRight: 6
        }}
      >
        {displayedProcesses.map((process) => {
          const isHighlighted = highlightAlgorithm === process.algorithm;
          return (
            <div
              key={process.pid}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
                padding: '16px 18px',
                borderRadius: 'var(--radius-md)',
                border: `1px solid ${isHighlighted ? 'rgba(255, 97, 210, 0.65)' : 'rgba(255, 255, 255, 0.05)'}`,
                background: isHighlighted ? 'rgba(255, 97, 210, 0.08)' : 'rgba(255, 255, 255, 0.02)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                  <strong style={{ fontSize: 15 }}>{process.name || 'Process'}</strong>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    PID {process.pid} • {process.status}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, fontSize: 13 }}>
                  <span style={{ color: 'var(--text-secondary)' }}>CPU {process.cpuPercent.toFixed(2)}%</span>
                  <span style={{ color: 'var(--text-secondary)' }}>Mem {process.memoryMb.toFixed(2)} MB</span>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12, marginTop: 4 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ color: 'var(--text-secondary)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Predicted Burst
                  </span>
                  <span style={{ color: 'var(--text-primary)' }}>
                    {process.predictedBurstTime != null ? `${process.predictedBurstTime.toFixed(2)} ms` : '—'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ color: 'var(--text-secondary)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Predicted Time Quantum
                  </span>
                  <span style={{ color: 'var(--text-primary)' }}>
                    {process.predictedTimeQuantum != null ? `${process.predictedTimeQuantum.toFixed(2)} ms` : '—'}
                  </span>
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: 8, borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    Current Algorithm
                  </span>
                  <span style={{ fontSize: 12, color: isHighlighted ? 'var(--accent)' : 'var(--text-secondary)' }}>
                    {process.algorithm}
                  </span>
                </div>
                {process.bestAlgorithm && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'flex-end' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                      Best Algorithm
                    </span>
                    <span style={{ fontSize: 12, color: '#00c9a7', fontWeight: 600 }}>
                      {process.bestAlgorithm}
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {hasMore && (
        <button
          onClick={handleLoadMore}
          style={{
            marginTop: 16,
            padding: '12px 24px',
            borderRadius: 'var(--radius-md)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.02)',
            color: 'var(--text-primary)',
            fontFamily: 'inherit',
            fontSize: 14,
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            width: '100%'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'rgba(255, 97, 210, 0.35)';
            e.currentTarget.style.background = 'rgba(255, 97, 210, 0.08)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
          }}
        >
          Load More ({processes.length - displayCount} remaining)
        </button>
      )}
    </div>
  );
}
