'use client';

import { useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';
import dayjs from 'dayjs';
import { AlgorithmSelector, type AlgorithmOption } from './components/AlgorithmSelector';
import { ProcessList, type ProcessItem } from './components/ProcessList';
import { GanttChart, type GanttSegment } from './components/GanttChart';
import { ScheduleStats, type ScheduleMetricsView } from './components/ScheduleStats';
import { ScheduleTable, type ScheduleRow } from './components/ScheduleTable';

interface ApiResponse {
  algorithms: AlgorithmOption[];
  selectedAlgorithm: string;
  algorithmDescription: string;
  processes: ProcessItem[];
  schedule: {
    segments: GanttSegment[];
    table: ScheduleRow[];
    metrics: ScheduleMetricsView;
  } | null;
  allSchedules: Record<string, {
    segments: GanttSegment[];
    table: ScheduleRow[];
    metrics: ScheduleMetricsView;
  } | null>;
  lastUpdated: string | null;
}

const fetcher = (url: string) => fetch(url).then((res) => {
  if (!res.ok) {
    throw new Error('Failed to fetch scheduler data');
  }
  return res.json();
});

function AnimatedTitle({ text }: { text: string }) {
  const [animatedChars, setAnimatedChars] = useState<boolean[]>([]);

  useEffect(() => {
    const animate = () => {
      setAnimatedChars(new Array(text.length).fill(false));
      
      // Animate each character with a delay
      text.split('').forEach((_, i) => {
        setTimeout(() => {
          setAnimatedChars((prev) => {
            const newState = [...prev];
            newState[i] = true;
            return newState;
          });
        }, i * 50);
      });
    };

    // Initial animation
    animate();

    // Repeat animation every 12 seconds
    const interval = setInterval(animate, 12000);

    return () => clearInterval(interval);
  }, [text]);

  return (
    <div style={{ width: '100%', display: 'flex', justifyContent: 'center', paddingBottom: 20 }}>
      <div style={{ overflow: 'hidden', fontSize: 'clamp(24px, 4vw, 36px)', padding: '8px 0', display: 'flex', cursor: 'default' }}>
        {text.split('').map((char, i) => (
          <h1 
            key={i} 
            style={{ 
              fontFamily: 'monospace', 
              opacity: animatedChars[i] ? 1 : 0,
              transform: animatedChars[i] ? 'translateY(0)' : 'translateY(20px)',
              transition: 'opacity 0.3s ease, transform 0.3s ease',
              margin: 0, 
              width: char === ' ' ? 12 : 'auto' 
            }}
          >
            {char}
          </h1>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const swrKey = useMemo(
    () => `/api/processes?algorithm=${encodeURIComponent(selectedAlgorithm ?? '')}`,
    [selectedAlgorithm]
  );

  const { data, error, isLoading } = useSWR<ApiResponse>(swrKey, fetcher, {
    refreshInterval: 5000,
    revalidateOnFocus: true
  });

  useEffect(() => {
    if (!selectedAlgorithm && data?.selectedAlgorithm) {
      setSelectedAlgorithm(data.selectedAlgorithm);
    }
  }, [data?.selectedAlgorithm, selectedAlgorithm]);

  const handleAlgorithmChange = (next: string) => {
    setSelectedAlgorithm(next);
  };

  const formattedUpdated = data?.lastUpdated ? dayjs(data.lastUpdated).format('YYYY-MM-DD HH:mm:ss') : '—';
  const isCompareAll = selectedAlgorithm === 'Compare All';
  
  // Calculate overall best algorithm based on minimum average waiting time
  const overallBestAlgorithm = useMemo(() => {
    if (!data?.allSchedules || !isCompareAll) return null;
    
    let bestAlg: string | null = null;
    let bestWaiting = Infinity;
    
    Object.entries(data.allSchedules).forEach(([algName, schedule]) => {
      if (schedule?.metrics) {
        const avgWaiting = schedule.metrics.averageWaitingTime;
        if (avgWaiting < bestWaiting) {
          bestWaiting = avgWaiting;
          bestAlg = algName;
        }
      }
    });
    
    return bestAlg;
  }, [data?.allSchedules, isCompareAll]);

  return (
    <main
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 32,
        padding: '48px clamp(24px, 6vw, 72px)',
        minHeight: '100vh'
      }}
    >
      <header style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <AnimatedTitle text="SCHEDULING ALGORITHM SIMULATOR" />
        <p style={{ color: 'var(--text-secondary)', maxWidth: 560, fontSize: 16 }}>
          Stream live process telemetry, explore scheduling algorithms, and inspect performance metrics across the first ten schedulable processes for each policy.
        </p>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', display: 'flex', gap: 16 }}>
          <span>Status: {error ? 'Error fetching data' : isLoading ? 'Loading…' : 'Live'}</span>
          <span>Last refreshed: {formattedUpdated}</span>
        </div>
      </header>

      <section
        className="layout-columns"
        style={{
          display: 'flex',
          gap: 24,
          alignItems: 'stretch'
        }}
      >
        <AlgorithmSelector
          algorithms={data?.algorithms ?? []}
          selected={selectedAlgorithm || data?.selectedAlgorithm}
          description={data?.algorithmDescription}
          onSelect={handleAlgorithmChange}
        />
        <ProcessList
          processes={data?.processes ?? []}
          highlightAlgorithm={selectedAlgorithm || data?.selectedAlgorithm}
        />
      </section>

      <section style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <p className="badge">{isCompareAll ? 'Algorithm Comparison' : 'Gantt Chart'}</p>
              <h2 style={{ fontSize: 26, marginTop: 12 }}>
                {isCompareAll ? 'Gantt Charts for All Algorithms' : (selectedAlgorithm || data?.selectedAlgorithm || 'Scheduling Timeline')}
              </h2>
            </div>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Showing the first 10 schedulable processes (system PIDs 0 and 4 removed)
            </span>
          </div>
          
          {isCompareAll && overallBestAlgorithm && (
            <div style={{ 
              padding: '16px 20px', 
              borderRadius: 'var(--radius-md)', 
              background: 'rgba(0, 201, 167, 0.1)',
              border: '1px solid rgba(0, 201, 167, 0.3)',
              marginBottom: 16
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 18 }}>🏆</span>
                <div>
                  <strong style={{ color: '#00c9a7', fontSize: 16 }}>Overall Best Algorithm:</strong>
                  <span style={{ marginLeft: 8, fontSize: 16, fontWeight: 600 }}>{overallBestAlgorithm}</span>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>
                    Based on minimum average waiting time: {data?.allSchedules[overallBestAlgorithm]?.metrics.averageWaitingTime.toFixed(2) || '—'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {isCompareAll ? (
            data?.allSchedules ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
                {Object.entries(data.allSchedules)
                  .filter(([_, schedule]) => schedule !== null)
                  .map(([algName, schedule]) => (
                    <div key={algName} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 style={{ fontSize: 18, fontWeight: 600 }}>
                          {algName}
                          {overallBestAlgorithm === algName && (
                            <span style={{ marginLeft: 12, fontSize: 14, color: '#00c9a7' }}>🏆 Best</span>
                          )}
                        </h3>
                      </div>
                      {schedule && (
                        <div style={{ marginBottom: 8 }}>
                          <ScheduleStats metrics={schedule.metrics} />
                        </div>
                      )}
                      {schedule ? (
                        <GanttChart segments={schedule.segments.slice(0, 30)} />
                      ) : (
                        <div style={{ padding: 24, color: 'var(--text-secondary)' }}>
                          No schedule available for this algorithm.
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            ) : (
              <div style={{ padding: 24, color: 'var(--text-secondary)' }}>
                Waiting for processes with arrival and burst times.
              </div>
            )
          ) : (
            data?.schedule ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ marginBottom: 8 }}>
                  <ScheduleStats metrics={data.schedule.metrics} />
                </div>
                <GanttChart segments={data.schedule.segments.slice(0, 30)} />
                {data.schedule.table.length > 0 && (
                  <ScheduleTable rows={data.schedule.table} />
                )}
              </div>
            ) : (
              <div style={{ padding: 24, color: 'var(--text-secondary)' }}>
                Waiting for processes with arrival and burst times for this algorithm.
              </div>
            )
          )}
        </div>
      </section>

      {data?.allSchedules && isCompareAll && (
        <section style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div>
              <p className="badge">Comparison</p>
              <h2 style={{ fontSize: 26, marginTop: 12 }}>Best Algorithm per Process</h2>
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 18, lineHeight: 1.55 }}>
              This table shows waiting times for each process across all algorithms. The best algorithm (lowest waiting time) is highlighted in green.
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Process</th>
                    <th>PID</th>
                    {Object.keys(data.allSchedules)
                      .filter((alg) => data.allSchedules[alg] !== null)
                      .map((alg) => {
                        // Shorten algorithm names for table headers
                        const shortName = alg
                          .replace('First Come First Serve', 'FCFS')
                          .replace('Shortest Job First', 'SJF')
                          .replace('Shortest Remaining Time First', 'SRTF')
                          .replace('Priority Scheduling', 'Priority')
                          .replace('Multilevel Feedback Queue', 'MLFQ');
                        return (
                          <th key={alg} style={{ textAlign: 'center' }}>
                            {shortName}
                            <div style={{ fontSize: 10, fontWeight: 400, color: 'var(--text-secondary)', marginTop: 4 }}>
                              (waiting)
                            </div>
                          </th>
                        );
                      })}
                    <th>Best Algorithm</th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const allPids = new Set<number>();
                    Object.values(data.allSchedules).forEach((schedule) => {
                      if (schedule?.table) {
                        schedule.table.forEach((row) => allPids.add(row.pid));
                      }
                    });
                    const pidList = Array.from(allPids).sort((a, b) => a - b);
                    const algorithmNames = Object.keys(data.allSchedules).filter(
                      (alg) => data.allSchedules[alg] !== null
                    );

                    return pidList.map((pid) => {
                      const processRow = Object.values(data.allSchedules)
                        .find((schedule) => schedule?.table.find((r) => r.pid === pid))
                        ?.table.find((r) => r.pid === pid);
                      
                      if (!processRow) return null;

                      const waitingTimes: Record<string, number> = {};
                      algorithmNames.forEach((algName) => {
                        const schedule = data.allSchedules[algName];
                        const row = schedule?.table.find((r) => r.pid === pid);
                        if (row) {
                          waitingTimes[algName] = row.waitingTime;
                        }
                      });

                      const bestAlg = Object.entries(waitingTimes).reduce((best, [alg, time]) => {
                        return !best || time < best.time ? { alg, time } : best;
                      }, null as { alg: string; time: number } | null);

                      return (
                        <tr key={pid}>
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <span className="tag-dot" style={{ background: processRow.color }} />
                              <div>
                                <strong style={{ fontSize: 14 }}>{processRow.tag}</strong>
                                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{processRow.name}</div>
                              </div>
                            </div>
                          </td>
                          <td>{pid}</td>
                          {algorithmNames.map((algName) => {
                            const waitingTime = waitingTimes[algName];
                            const isBest = bestAlg?.alg === algName;
                            return (
                              <td key={algName} style={{ textAlign: 'center' }}>
                                <span
                                  style={{
                                    color: isBest ? '#00c9a7' : 'var(--text-primary)',
                                    fontWeight: isBest ? 600 : 400
                                  }}
                                >
                                  {waitingTime != null ? waitingTime.toFixed(2) : '—'}
                                </span>
                              </td>
                            );
                          })}
                          <td>
                            <span style={{ color: '#00c9a7', fontWeight: 600 }}>
                              {bestAlg ? bestAlg.alg : '—'}
                            </span>
                          </td>
                        </tr>
                      );
                    });
                  })()}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {error && (
        <div className="panel" style={{ borderColor: 'rgba(255, 97, 210, 0.4)', color: 'var(--accent)' }}>
          Failed to refresh data: {error.message}
        </div>
      )}
    </main>
  );
}
