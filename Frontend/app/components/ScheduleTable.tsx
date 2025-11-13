'use client';

export interface ScheduleRow {
  pid: number;
  tag: string;
  name: string;
  arrivalTime: number;
  burstTime: number;
  waitingTime: number;
  turnaroundTime: number;
  completionTime: number;
  color: string;
  timeQuantum?: number | null;
}

interface ScheduleTableProps {
  rows: ScheduleRow[];
}

function format(value: number) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

export function ScheduleTable({ rows }: ScheduleTableProps) {
  const showTimeQuantum = rows.some((row) => row.timeQuantum !== undefined && row.timeQuantum !== null);

  return (
    <div className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <div>
          <p className="badge">Summary</p>
          <h2 style={{ fontSize: 22, marginTop: 12 }}>Per-process metrics</h2>
        </div>
      </div>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 18, maxWidth: 640, lineHeight: 1.55 }}>
        For each of the first ten processes we calculate when it arrived, how long it ran (burst), how long it waited,
        and its turnaround/completion times based on the simulated schedule. {showTimeQuantum ? 'For multilevel feedback queues we also display the predicted time quantum used for that task.' : ''} Use this to compare fairness across algorithms.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="table">
          <thead>
            <tr>
              <th>Process</th>
              <th>Arrival</th>
              <th>Burst</th>
              <th>Waiting</th>
              <th>Turnaround</th>
              <th>Completion</th>
              {showTimeQuantum && <th>Time Quantum</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.tag}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span className="tag-dot" style={{ background: row.color }} />
                    <div>
                      <strong style={{ fontSize: 14 }}>{row.tag}</strong>
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{row.name}</div>
                    </div>
                  </div>
                </td>
                <td>{format(row.arrivalTime)}</td>
                <td>{format(row.burstTime)}</td>
                <td>{format(row.waitingTime)}</td>
                <td>{format(row.turnaroundTime)}</td>
                <td>{format(row.completionTime)}</td>
                {showTimeQuantum && <td>{row.timeQuantum != null ? format(row.timeQuantum) : '—'}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
