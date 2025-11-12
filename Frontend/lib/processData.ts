import { promises as fs } from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';

type AlgorithmKey = string;

export interface ProcessRecord {
  snapshotId: number;
  systemTimestamp: number;
  timestamp: string;
  pid: number;
  name: string;
  status: string;
  ppid: number | null;
  cpuPercent: number;
  memoryPercent: number;
  memoryMb: number;
  numThreads: number | null;
  nice: number | null;
  processAge: number | null;
  processType: number | null;
  ioIntensity: number | null;
  userTime: number | null;
  systemTime: number | null;
  burstTime: number | null;
  predictedBurstTime: number | null;
  arrivalTime: number | null;
  predictedTimeQuantum: number | null;
  predictedPriority: number | null;
  recommendedAlgorithm: AlgorithmKey;
}

export interface ProcessOverview {
  pid: number;
  tag: string;
  name: string;
  algorithm: AlgorithmKey;
  cpuPercent: number;
  memoryMb: number;
  status: string;
  predictedBurstTime?: number | null;
  predictedTimeQuantum?: number | null;
  bestAlgorithm?: AlgorithmKey;
}

export interface ProcessTableRow {
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

export interface ScheduleSegment {
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

export interface ScheduleMetrics {
  averageBurstTime: number;
  averageTurnaroundTime: number;
  averageWaitingTime: number;
  throughput: number;
  cpuUtilization: number;
  makespan: number;
  totalBurst: number;
  idleTime: number;
}

export interface SchedulePayload {
  segments: ScheduleSegment[];
  table: ProcessTableRow[];
  metrics: ScheduleMetrics;
}

export interface ProcessesPayload {
  algorithms: AlgorithmSummary[];
  selectedAlgorithm: AlgorithmKey;
  algorithmDescription: string;
  processes: ProcessOverview[];
  schedule: SchedulePayload | null;
  allSchedules: Record<AlgorithmKey, SchedulePayload | null>;
  lastUpdated: string | null;
}

export interface AlgorithmSummary {
  name: AlgorithmKey;
  description: string;
  available: boolean;
}

const ALGORITHM_DESCRIPTIONS: Record<AlgorithmKey, string> = {
  'First Come First Serve (FCFS)':
    'Processes are executed in the order they arrive. Simple but can lead to longer waiting times for later jobs.',
  'Shortest Job First (SJF)':
    'Chooses the process with the smallest execution time next. Minimizes average waiting time but requires knowing burst lengths.',
  'Shortest Remaining Time First (SRTF)':
    'Preemptive version of SJF that always runs the job with the least remaining time.',
  'Priority Scheduling':
    'Executes processes based on priority levels. Lower priority numbers indicate higher importance by default.',
  'Multilevel Feedback Queue (MLFQ)':
    'Uses multiple queues with variable priorities and time quantums, adapting to process behavior over time.'
};

const EXCLUDED_ALGORITHMS = new Set<AlgorithmKey>(['Round Robin (RR)']);

const DEFAULT_DATA_FILE = path.join(process.cwd(), 'scheduler_results (1).csv');

const COLOR_PALETTE = [
  '#ff61d2',
  '#845ec2',
  '#f9f871',
  '#ff9671',
  '#00c9a7',
  '#0081cf',
  '#f24c00',
  '#ff6f91',
  '#2c73d2',
  '#00c9c8'
];

interface InternalProcess extends ProcessRecord {
  remainingTime: number;
  normalizedArrival: number;
  tag: string;
  color: string;
}

interface SimulationResult {
  segments: ScheduleSegment[];
  stats: Map<string, {
    arrival: number;
    burst: number;
    completion: number;
  }>;
  idleTime: number;
  makespan: number;
  totalBurst: number;
}

function toNumber(value: string | undefined): number | null {
  if (!value || value.trim() === '') {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

async function readCsvRecords(): Promise<ProcessRecord[]> {
  const filePath = process.env.SCHEDULER_DATA_FILE ?? DEFAULT_DATA_FILE;
  const content = await fs.readFile(filePath, 'utf8');
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  }) as Record<string, string>[];

  const latestByPid = new Map<number, ProcessRecord>();

  for (const record of records) {
    const pid = toNumber(record.pid);
    if (pid === null || pid === 0 || pid === 4) {
      continue;
    }

    const burstTime = toNumber(record.burst_time) ?? toNumber(record.predicted_burst_time);
    const arrivalTime = toNumber(record.arrival_time);
    const systemTimestamp = toNumber(record.system_timestamp) ?? 0;

    const parsed: ProcessRecord = {
      snapshotId: toNumber(record.snapshot_id) ?? 0,
      systemTimestamp,
      timestamp: record.timestamp,
      pid,
      name: record.name || 'Unknown',
      status: record.status || 'unknown',
      ppid: toNumber(record.ppid),
      cpuPercent: toNumber(record.cpu_percent) ?? 0,
      memoryPercent: toNumber(record.memory_percent) ?? 0,
      memoryMb: toNumber(record.memory_mb) ?? 0,
      numThreads: toNumber(record.num_threads),
      nice: toNumber(record.nice),
      processAge: toNumber(record.process_age),
      processType: toNumber(record.process_type),
      ioIntensity: toNumber(record.io_intensity),
      userTime: toNumber(record.user_time),
      systemTime: toNumber(record.system_time),
      burstTime,
      predictedBurstTime: toNumber(record.predicted_burst_time),
      arrivalTime,
      predictedTimeQuantum: toNumber(record.predicted_time_quantum),
      predictedPriority: toNumber(record.predicted_priority),
      recommendedAlgorithm: record.recommended_algorithm || 'First Come First Serve (FCFS)'
    };

    if (EXCLUDED_ALGORITHMS.has(parsed.recommendedAlgorithm)) {
      continue;
    }

    const existing = latestByPid.get(pid);
    if (!existing || parsed.systemTimestamp >= existing.systemTimestamp) {
      latestByPid.set(pid, parsed);
    }
  }

  return [...latestByPid.values()].sort((a, b) => a.pid - b.pid);
}

function normalizeProcesses(processes: ProcessRecord[]): InternalProcess[] {
  const arrivalValues = processes
    .map((p) => p.arrivalTime)
    .filter((value): value is number => value !== null);

  const minArrival = arrivalValues.length ? Math.min(...arrivalValues) : 0;

  return processes.map((process, index) => {
    const burst = process.burstTime ?? process.predictedBurstTime ?? 0;
    const arrival = process.arrivalTime ?? minArrival;
    const normalizedArrival = arrival - minArrival;

    return {
      ...process,
      remainingTime: burst,
      normalizedArrival,
      tag: `P${index + 1}`,
      color: COLOR_PALETTE[index % COLOR_PALETTE.length]
    };
  });
}

function addIdleSegment(
  segments: ScheduleSegment[],
  currentTime: number,
  nextTime: number
): void {
  if (nextTime <= currentTime) {
    return;
  }
  segments.push({
    pid: null,
    tag: 'IDLE',
    label: 'Idle',
    start: currentTime,
    end: nextTime,
    duration: nextTime - currentTime,
    color: 'rgba(255,255,255,0.08)'
  });
}

function simulateFCFS(processes: InternalProcess[]): SimulationResult {
  const sorted = [...processes].sort((a, b) => a.normalizedArrival - b.normalizedArrival);
  const segments: ScheduleSegment[] = [];
  const stats = new Map<string, { arrival: number; burst: number; completion: number }>();

  let time = 0;
  const totalBurst = sorted.reduce((acc, proc) => acc + (proc.burstTime ?? proc.predictedBurstTime ?? 0), 0);

  for (const process of sorted) {
    const arrival = process.normalizedArrival;
    if (time < arrival) {
      addIdleSegment(segments, time, arrival);
      time = arrival;
    }

    const burst = process.burstTime ?? process.predictedBurstTime ?? 0;
    const start = time;
    const end = start + burst;

    segments.push({
      pid: process.pid,
      tag: process.tag,
      label: `${process.tag}`,
      start,
      end,
      duration: burst,
      color: process.color
    });

    time = end;
    stats.set(process.tag, {
      arrival,
      burst,
      completion: end
    });
  }

  const makespan = segments.length ? segments[segments.length - 1].end : 0;
  const idleTime = segments
    .filter((segment) => segment.pid === null)
    .reduce((acc, seg) => acc + seg.duration, 0);

  return { segments, stats, idleTime, makespan, totalBurst };
}

function simulatePriority(processes: InternalProcess[]): SimulationResult {
  const segments: ScheduleSegment[] = [];
  const stats = new Map<string, { arrival: number; burst: number; completion: number }>();
  const remaining = new Map(processes.map((p) => [p.tag, p.burstTime ?? p.predictedBurstTime ?? 0]));

  let time = 0;
  const sortedByArrival = [...processes].sort((a, b) => a.normalizedArrival - b.normalizedArrival);
  const totalBurst = sortedByArrival.reduce((acc, proc) => acc + (proc.burstTime ?? proc.predictedBurstTime ?? 0), 0);

  const ready: InternalProcess[] = [];
  let idx = 0;

  while (stats.size < processes.length) {
    while (idx < sortedByArrival.length && sortedByArrival[idx].normalizedArrival <= time) {
      ready.push(sortedByArrival[idx]);
      idx += 1;
    }

    if (!ready.length) {
      const nextArrival = sortedByArrival[idx]?.normalizedArrival;
      if (nextArrival === undefined) {
        break;
      }
      addIdleSegment(segments, time, nextArrival);
      time = nextArrival;
      continue;
    }

    ready.sort((a, b) => {
      const priorityA = a.predictedPriority ?? Number.MAX_SAFE_INTEGER;
      const priorityB = b.predictedPriority ?? Number.MAX_SAFE_INTEGER;
      if (priorityA === priorityB) {
        return a.normalizedArrival - b.normalizedArrival;
      }
      return priorityA - priorityB;
    });

    const current = ready.shift()!;
    const burst = remaining.get(current.tag) ?? 0;
    const start = Math.max(time, current.normalizedArrival);
    const end = start + burst;

    if (start > time) {
      addIdleSegment(segments, time, start);
      time = start;
    }

    segments.push({
      pid: current.pid,
      tag: current.tag,
      label: current.tag,
      start,
      end,
      duration: burst,
      color: current.color
    });

    time = end;
    stats.set(current.tag, {
      arrival: current.normalizedArrival,
      burst,
      completion: end
    });
  }

  const makespan = segments.length ? segments[segments.length - 1].end : 0;
  const idleTime = segments
    .filter((segment) => segment.pid === null)
    .reduce((acc, seg) => acc + seg.duration, 0);

  return { segments, stats, idleTime, makespan, totalBurst };
}

function simulateMLFQ(processes: InternalProcess[]): SimulationResult {
  const quantums = [4, 8, 16];
  const queues: InternalProcess[][] = [[], [], []];
  const stats = new Map<string, { arrival: number; burst: number; completion: number }>();
  const remaining = new Map(processes.map((p) => [p.tag, p.burstTime ?? p.predictedBurstTime ?? 0]));
  const segments: ScheduleSegment[] = [];

  const sorted = [...processes].sort((a, b) => a.normalizedArrival - b.normalizedArrival);
  const totalBurst = sorted.reduce((acc, proc) => acc + (proc.burstTime ?? proc.predictedBurstTime ?? 0), 0);

  let time = 0;
  let idx = 0;

  const enqueueNewArrivals = () => {
    while (idx < sorted.length && sorted[idx].normalizedArrival <= time) {
      queues[0].push(sorted[idx]);
      idx += 1;
    }
  };

  const hasReady = () => queues.some((q) => q.length > 0);

  while (stats.size < processes.length) {
    enqueueNewArrivals();

    if (!hasReady()) {
      const nextArrival = sorted[idx]?.normalizedArrival;
      if (nextArrival === undefined) {
        break;
      }
      addIdleSegment(segments, time, nextArrival);
      time = nextArrival;
      enqueueNewArrivals();
      continue;
    }

    let level = queues.findIndex((q) => q.length > 0);
    if (level === -1) {
      level = queues.length - 1;
    }

    const current = queues[level].shift()!;
    const remainingTime = remaining.get(current.tag) ?? 0;
    const quantum = quantums[level] ?? quantums[quantums.length - 1];
    const predictedQuantum = current.predictedTimeQuantum ?? quantum;
    const actualRun = Math.min(quantum, remainingTime);
    const start = Math.max(time, current.normalizedArrival);

    if (start > time) {
      addIdleSegment(segments, time, start);
      time = start;
    }

    const end = time + actualRun;

    segments.push({
      pid: current.pid,
      tag: current.tag,
      label: `${current.tag} (Q${level + 1})`,
      start: time,
      end,
      duration: actualRun,
      color: current.color,
      queueLevel: level + 1,
      timeQuantum: predictedQuantum
    });

    time = end;
    enqueueNewArrivals();

    const updatedRemaining = remainingTime - actualRun;
    remaining.set(current.tag, updatedRemaining);

    if (updatedRemaining > 0.0001) {
      const nextLevel = Math.min(level + 1, queues.length - 1);
      queues[nextLevel].push(current);
    } else {
      const burst = current.burstTime ?? current.predictedBurstTime ?? 0;
      stats.set(current.tag, {
        arrival: current.normalizedArrival,
        burst,
        completion: time
      });
    }
  }

  const makespan = segments.length ? segments[segments.length - 1].end : 0;
  const idleTime = segments
    .filter((segment) => segment.pid === null)
    .reduce((acc, seg) => acc + seg.duration, 0);

  return { segments, stats, idleTime, makespan, totalBurst };
}

/**
 * Computes scheduling metrics for all processes
 * 
 * METRICS CALCULATION EXPLANATION:
 * 
 * 1. Average Waiting Time:
 *    - For each process: Waiting Time = Completion Time - Arrival Time - Burst Time
 *    - Average = Sum of all waiting times / Number of processes
 *    - This measures how long processes wait in the ready queue before execution
 * 
 * 2. Average Turnaround Time:
 *    - For each process: Turnaround Time = Completion Time - Arrival Time
 *    - Average = Sum of all turnaround times / Number of processes
 *    - This measures total time from arrival to completion (waiting + execution)
 * 
 * 3. Average Burst Time:
 *    - Average = Sum of all burst times / Number of processes
 *    - This is the average execution time needed by processes
 * 
 * 4. Throughput:
 *    - Throughput = Number of processes completed / Makespan
 *    - Measures how many processes complete per unit time
 * 
 * 5. CPU Utilization:
 *    - Busy Time = Makespan - Idle Time
 *    - CPU Utilization = (Busy Time / Makespan) * 100%
 *    - Measures percentage of time CPU is busy executing processes
 * 
 * 6. Makespan:
 *    - Makespan = End time of the last segment (total time to complete all processes)
 * 
 * 7. Idle Time:
 *    - Sum of all idle segments (time when CPU has no processes to execute)
 */
function computeMetrics(
  stats: Map<string, { arrival: number; burst: number; completion: number }>,
  segments: ScheduleSegment[],
  totalBurst: number,
  idleTime: number
): ScheduleMetrics {
  const values = [...stats.values()];
  if (!values.length) {
    return {
      averageBurstTime: 0,
      averageTurnaroundTime: 0,
      averageWaitingTime: 0,
      throughput: 0,
      cpuUtilization: 0,
      makespan: 0,
      totalBurst: 0,
      idleTime: 0
    };
  }

  // Makespan = total time from start to finish of all processes
  const makespan = segments.length ? segments[segments.length - 1].end : 0;

  // Turnaround Time = Completion Time - Arrival Time (total time in system)
  const totalTurnaround = values.reduce((acc, { arrival, completion }) => acc + (completion - arrival), 0);
  
  // Waiting Time = Completion Time - Arrival Time - Burst Time (time spent waiting)
  const totalWaiting = values.reduce((acc, { arrival, completion, burst }) => acc + (completion - arrival - burst), 0);
  
  // Average Burst Time = Sum of burst times / Number of processes
  const avgBurst = values.reduce((acc, { burst }) => acc + burst, 0) / values.length;

  // Throughput = Number of processes / Total time (processes per unit time)
  const throughput = makespan > 0 ? values.length / makespan : 0;
  
  // Busy Time = Total time - Idle time (time CPU was executing processes)
  const busyTime = Math.max(makespan - idleTime, 0);
  
  // CPU Utilization = (Busy Time / Total Time) * 100%
  const cpuUtilization = makespan > 0 ? (busyTime / makespan) * 100 : 0;

  return {
    averageBurstTime: avgBurst,
    averageTurnaroundTime: totalTurnaround / values.length,
    averageWaitingTime: totalWaiting / values.length,
    throughput,
    cpuUtilization,
    makespan,
    totalBurst,
    idleTime
  };
}

function buildTable(
  stats: Map<string, { arrival: number; burst: number; completion: number }>,
  processes: InternalProcess[],
  colorByTag: Map<string, string>
): ProcessTableRow[] {
  const tagToProcess = new Map(processes.map((proc) => [proc.tag, proc]));

  return [...stats.entries()]
    .map(([tag, { arrival, burst, completion }]) => {
      const process = tagToProcess.get(tag);
      if (!process) {
        throw new Error(`Missing process for tag ${tag}`);
      }

      return {
        pid: process.pid,
        tag,
        name: process.name,
        arrivalTime: Number(arrival.toFixed(2)),
        burstTime: Number(burst.toFixed(2)),
        waitingTime: Number((completion - arrival - burst).toFixed(2)),
        turnaroundTime: Number((completion - arrival).toFixed(2)),
        completionTime: Number(completion.toFixed(2)),
        color: colorByTag.get(tag) ?? '#999',
        timeQuantum: process.predictedTimeQuantum ?? null
      };
    })
    .sort((a, b) => a.arrivalTime - b.arrivalTime);
}

function formatSegments(segments: ScheduleSegment[]): ScheduleSegment[] {
  return segments.map((segment) => ({
    ...segment,
    start: Number(segment.start.toFixed(2)),
    end: Number(segment.end.toFixed(2)),
    duration: Number(segment.duration.toFixed(2))
  }));
}

function simulateSJF(processes: InternalProcess[]): SimulationResult {
  const sorted = [...processes].sort((a, b) => {
    const burstA = a.burstTime ?? a.predictedBurstTime ?? 0;
    const burstB = b.burstTime ?? b.predictedBurstTime ?? 0;
    if (Math.abs(burstA - burstB) < 0.001) {
      return a.normalizedArrival - b.normalizedArrival;
    }
    return burstA - burstB;
  });
  const segments: ScheduleSegment[] = [];
  const stats = new Map<string, { arrival: number; burst: number; completion: number }>();

  let time = 0;
  const totalBurst = sorted.reduce((acc, proc) => acc + (proc.burstTime ?? proc.predictedBurstTime ?? 0), 0);

  for (const process of sorted) {
    const arrival = process.normalizedArrival;
    if (time < arrival) {
      addIdleSegment(segments, time, arrival);
      time = arrival;
    }

    const burst = process.burstTime ?? process.predictedBurstTime ?? 0;
    const start = time;
    const end = start + burst;

    segments.push({
      pid: process.pid,
      tag: process.tag,
      label: `${process.tag}`,
      start,
      end,
      duration: burst,
      color: process.color
    });

    time = end;
    stats.set(process.tag, {
      arrival,
      burst,
      completion: end
    });
  }

  const makespan = segments.length ? segments[segments.length - 1].end : 0;
  const idleTime = segments
    .filter((segment) => segment.pid === null)
    .reduce((acc, seg) => acc + seg.duration, 0);

  return { segments, stats, idleTime, makespan, totalBurst };
}

function simulateSRTF(processes: InternalProcess[]): SimulationResult {
  const segments: ScheduleSegment[] = [];
  const stats = new Map<string, { arrival: number; burst: number; completion: number }>();
  const remaining = new Map(processes.map((p) => [p.tag, p.burstTime ?? p.predictedBurstTime ?? 0]));

  let time = 0;
  const sortedByArrival = [...processes].sort((a, b) => a.normalizedArrival - b.normalizedArrival);
  const totalBurst = sortedByArrival.reduce((acc, proc) => acc + (proc.burstTime ?? proc.predictedBurstTime ?? 0), 0);

  const ready: InternalProcess[] = [];
  let idx = 0;

  while (stats.size < processes.length) {
    // Add newly arrived processes
    while (idx < sortedByArrival.length && sortedByArrival[idx].normalizedArrival <= time) {
      ready.push(sortedByArrival[idx]);
      idx += 1;
    }

    if (!ready.length) {
      const nextArrival = sortedByArrival[idx]?.normalizedArrival;
      if (nextArrival === undefined) {
        break;
      }
      addIdleSegment(segments, time, nextArrival);
      time = nextArrival;
      continue;
    }

    // Sort by remaining time (SRTF)
    ready.sort((a, b) => {
      const remainingA = remaining.get(a.tag) ?? 0;
      const remainingB = remaining.get(b.tag) ?? 0;
      if (Math.abs(remainingA - remainingB) < 0.001) {
        return a.normalizedArrival - b.normalizedArrival;
      }
      return remainingA - remainingB;
    });

    const current = ready[0];
    const remainingTime = remaining.get(current.tag) ?? 0;
    
    // Find next event: either next arrival or completion
    const nextArrival = sortedByArrival[idx]?.normalizedArrival;
    const timeUntilNext = nextArrival !== undefined ? nextArrival - time : Infinity;
    const runTime = Math.min(remainingTime, timeUntilNext);
    
    const start = time;
    const end = start + runTime;

    // Check if we should merge with previous segment
    const lastSegment = segments[segments.length - 1];
    if (lastSegment && lastSegment.pid === current.pid && Math.abs(lastSegment.end - start) < 0.0001) {
      lastSegment.end = end;
      lastSegment.duration += runTime;
    } else {
      segments.push({
        pid: current.pid,
        tag: current.tag,
        label: current.tag,
        start,
        end,
        duration: runTime,
        color: current.color
      });
    }

    time = end;
    const updatedRemaining = remainingTime - runTime;
    remaining.set(current.tag, updatedRemaining);

    if (updatedRemaining < 0.0001) {
      // Process completed
      const burst = current.burstTime ?? current.predictedBurstTime ?? 0;
      stats.set(current.tag, {
        arrival: current.normalizedArrival,
        burst,
        completion: end
      });
      ready.shift();
    }
    // If not completed, it stays in ready queue and will be resorted on next iteration
  }

  const makespan = segments.length ? segments[segments.length - 1].end : 0;
  const idleTime = segments
    .filter((segment) => segment.pid === null)
    .reduce((acc, seg) => acc + seg.duration, 0);

  return { segments, stats, idleTime, makespan, totalBurst };
}

function simulate(
  algorithm: AlgorithmKey,
  processes: InternalProcess[]
): SimulationResult {
  switch (algorithm) {
    case 'First Come First Serve (FCFS)':
      return simulateFCFS(processes);
    case 'Shortest Job First (SJF)':
      return simulateSJF(processes);
    case 'Shortest Remaining Time First (SRTF)':
      return simulateSRTF(processes);
    case 'Priority Scheduling':
      return simulatePriority(processes);
    case 'Multilevel Feedback Queue (MLFQ)':
      return simulateMLFQ(processes);
    default:
      return simulateFCFS(processes);
  }
}

export async function getProcessesPayload(preferredAlgorithm?: AlgorithmKey): Promise<ProcessesPayload> {
  const records = await readCsvRecords();
  const runningProcesses = records
    .filter((record) => record.status.toLowerCase() === 'running')
    .sort((a, b) => b.cpuPercent - a.cpuPercent);

  const dataAlgorithms = new Set(records.map((record) => record.recommendedAlgorithm));
  const canonicalAlgorithms = Object.keys(ALGORITHM_DESCRIPTIONS);
  const mergedAlgorithms = new Set<string>([...canonicalAlgorithms, ...dataAlgorithms]);

  const algorithms: AlgorithmSummary[] = [
    ...([...mergedAlgorithms]
      .filter((name) => name !== 'Shortest Remaining Time First (SRTF)') // Exclude SRTF from selector
      .sort((a, b) => a.localeCompare(b))
      .map((name) => ({
        name,
        description: ALGORITHM_DESCRIPTIONS[name] ?? 'No description available.',
        available: dataAlgorithms.has(name)
      }))),
    // Add "Compare All" option
    {
      name: 'Compare All',
      description: 'View all algorithms side-by-side and see which performs best overall based on average waiting time.',
      available: true
    }
  ];

  // Get first available algorithm (excluding "Compare All")
  const firstAvailable = [...dataAlgorithms][0] ?? 
    algorithms.find(alg => alg.name !== 'Compare All')?.name ?? 
    'First Come First Serve (FCFS)';

  // Handle "Compare All" - don't treat it as a real algorithm for scheduling
  const selectedAlgorithm =
    preferredAlgorithm === 'Compare All'
      ? 'Compare All'
      : (preferredAlgorithm && mergedAlgorithms.has(preferredAlgorithm))
      ? preferredAlgorithm
      : firstAvailable;

  // Get all processes that have burst time and arrival time for scheduling
  const schedulableProcesses = records
    .filter((record) =>
      (record.burstTime ?? record.predictedBurstTime ?? 0) > 0 &&
      record.arrivalTime !== null
    )
    .sort((a, b) => (a.arrivalTime ?? 0) - (b.arrivalTime ?? 0));

  const limited = schedulableProcesses.slice(0, 10);
  const internalProcesses = normalizeProcesses(limited);

  // Compute schedules for all algorithms (excluding SRTF since it's not in the dataset)
  const allSchedules: Record<AlgorithmKey, SchedulePayload | null> = {};
  const algorithmMetrics: Record<AlgorithmKey, { avgWaiting: number; avgTurnaround: number }> = {};

  // Filter out SRTF from algorithms to compute (only compute algorithms that are in the dataset or available)
  const algorithmsToCompute = canonicalAlgorithms.filter(
    (algName) => algName !== 'Shortest Remaining Time First (SRTF)'
  );

  for (const algName of algorithmsToCompute) {
    if (internalProcesses.length) {
      const simulation = simulate(algName, internalProcesses);
      const metrics = computeMetrics(
        simulation.stats,
        simulation.segments,
        simulation.totalBurst,
        simulation.idleTime
      );

      const colorMap = new Map(internalProcesses.map((proc) => [proc.tag, proc.color]));
      const table = buildTable(simulation.stats, internalProcesses, colorMap);

      allSchedules[algName] = {
        segments: formatSegments(simulation.segments),
        table,
        metrics
      };

      algorithmMetrics[algName] = {
        avgWaiting: metrics.averageWaitingTime,
        avgTurnaround: metrics.averageTurnaroundTime
      };
    } else {
      allSchedules[algName] = null;
    }
  }

  // Determine best algorithm for each process based on waiting time
  const processBestAlgorithm = new Map<number, AlgorithmKey>();
  for (const process of limited) {
    let bestAlg: AlgorithmKey | null = null;
    let bestWaiting = Infinity;

    for (const [algName, schedule] of Object.entries(allSchedules)) {
      if (!schedule) continue;
      const processRow = schedule.table.find((row) => row.pid === process.pid);
      if (processRow && processRow.waitingTime < bestWaiting) {
        bestWaiting = processRow.waitingTime;
        bestAlg = algName as AlgorithmKey;
      }
    }

    if (bestAlg) {
      processBestAlgorithm.set(process.pid, bestAlg);
    }
  }

  // Get schedule for selected algorithm (skip if "Compare All")
  const schedule = selectedAlgorithm === 'Compare All' ? null : (allSchedules[selectedAlgorithm] ?? null);

  const processOverview: ProcessOverview[] = runningProcesses.map((process) => {
    const record = records.find((r) => r.pid === process.pid);
    return {
      pid: process.pid,
      tag: `PID ${process.pid}`,
      name: process.name,
      algorithm: process.recommendedAlgorithm,
      cpuPercent: Number(process.cpuPercent.toFixed(2)),
      memoryMb: Number(process.memoryMb.toFixed(2)),
      status: process.status,
      predictedBurstTime: record?.predictedBurstTime ?? null,
      predictedTimeQuantum: record?.predictedTimeQuantum ?? null,
      bestAlgorithm: processBestAlgorithm.get(process.pid)
    };
  });

  let latestTimestamp: string | null = null;
  for (const record of records) {
    if (!record.timestamp) continue;
    const millis = Date.parse(record.timestamp);
    if (!Number.isFinite(millis)) {
      continue;
    }
    if (!latestTimestamp || millis > Date.parse(latestTimestamp)) {
      latestTimestamp = record.timestamp;
    }
  }

  return {
    algorithms,
    selectedAlgorithm,
    algorithmDescription: selectedAlgorithm === 'Compare All' 
      ? 'View all algorithms side-by-side and see which performs best overall based on average waiting time.'
      : (ALGORITHM_DESCRIPTIONS[selectedAlgorithm] ?? 'Scheduling visualization'),
    processes: processOverview,
    schedule,
    allSchedules,
    lastUpdated: latestTimestamp
  };
}
