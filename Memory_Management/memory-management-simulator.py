"""
Memory Management Simulator (single-file)

What this does
- Simulates Secondary Memory (disk), Primary Memory (swap area), and RAM as OS-visible folders.
- Pages are represented by simple files inside these folders so you can open your OS File Explorer and watch pages move.
- Supports multiple page replacement strategies:
    - LRU (baseline)
    - FIFO (baseline)
    - ML-based predictor (RandomForest) that predicts the probability a page will be used in the near future; choose to evict page with lowest future-use probability.
    - (Optional) Optimal (requires full future trace, for evaluation only)
- Records access history to `history.csv` (used to train the model) and can train a model from that history.
- Simple CLI to initialize, run synthetic traces, replay traces, train model, and run simulations.

How pages are represented on disk (so they are visible in File Explorer)
- A directory named `memsim_workspace` is created in the current working directory.
- Inside it you'll find three folders:
    - `secondary`  (simulates disk)   — contains all pages initially
    - `primary`    (simulates swap)   — optional mid-level storage
    - `ram`        (simulates RAM)    — contains currently loaded pages (files moved here)
- Pages are text files named `page_<id>.pg` containing a tiny JSON-like metadata line + optional payload.

Requirements
- Python 3.8+
- Recommended (for ML): scikit-learn, pandas, joblib
    pip install scikit-learn pandas joblib numpy

Run examples
- Initialize workspace with 100 pages and RAM size 8:
    python memory-management-simulator.py --init --pages 100 --ram 8

- Run a synthetic trace (random locality) of 500 accesses using ML disabled (LRU):
    python memory-management-simulator.py --run --trace synthetic --steps 500 --policy lru

- Train ML model from history.csv (will create model.joblib):
    python memory-management-simulator.py --train

- Run simulation using ML policy (if model exists):
    python memory-management-simulator.py --run --trace synthetic --steps 1000 --policy ml

After running, open the folder `memsim_workspace` in your file explorer to see pages in `ram`, `primary`, and `secondary`.

Short design notes
- The ML model is trained to predict whether a page will be accessed within the next K accesses (binary classification). Features include: last_access_delta (time since last access), access_count, recency rank, page id (one-hot-ish via hashing), and global step index.
- At eviction time, for each page in RAM we compute feature vector and run the model to get probability of future access; evict the page with lowest probability.
- The simulator logs each access, hit/miss, and evictions to `events.log` and `history.csv`.

"""

import os
import sys
import argparse
import random
import csv
import time
import json
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

# Optional ML imports. Fall back gracefully if not available.
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

WORKSPACE = os.path.join(os.getcwd(), "memsim_workspace")
SECONDARY = os.path.join(WORKSPACE, "secondary")
PRIMARY = os.path.join(WORKSPACE, "primary")
RAM = os.path.join(WORKSPACE, "ram")
HISTORY_CSV = os.path.join(WORKSPACE, "history.csv")
EVENT_LOG = os.path.join(WORKSPACE, "events.log")
MODEL_FILE = os.path.join(WORKSPACE, "model.joblib")

# Parameters used for ML label window
FUTURE_WINDOW = 10  # label = will the page be used within next FUTURE_WINDOW accesses

# Utility: make sure folders exist
def ensure_workspace():
    os.makedirs(SECONDARY, exist_ok=True)
    os.makedirs(PRIMARY, exist_ok=True)
    os.makedirs(RAM, exist_ok=True)

@dataclass
class PageMeta:
    pid: int
    last_access_step: Optional[int] = None
    access_count: int = 0
    last_loaded_step: Optional[int] = None

    def to_dict(self):
        return {"pid": self.pid, "last_access_step": self.last_access_step, "access_count": self.access_count}


class MemorySimulator:
    def __init__(self, pages:int=100, ram_size:int=8, primary_enabled:bool=True):
        ensure_workspace()
        self.pages = pages
        self.ram_size = ram_size
        self.primary_enabled = primary_enabled

        # In-memory metadata
        self.meta: Dict[int, PageMeta] = {i: PageMeta(pid=i) for i in range(pages)}
        self.ram_order = deque()  # for FIFO
        self.time_step = 0

        # event logging
        open(EVENT_LOG, 'a').close()
        # initialize history file header if not exists
        if not os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step","process_id","page_id","in_ram","in_primary","in_secondary",
                    "last_access_delta","access_count","referenced_bit","dirty_bit",
                    "resident_duration","free_frames","ram_utilization"
                ])

    def init_workspace(self, overwrite:bool=False):
        # Remove existing files? Only create pages if not present
        ensure_workspace()
        existing = os.listdir(SECONDARY)
        if len(existing) >= self.pages and not overwrite:
            print(f"Secondary already has {len(existing)} pages. Use --overwrite to recreate.")
            return
        # Clean folders
        for d in (SECONDARY, PRIMARY, RAM):
            for f in os.listdir(d):
                os.remove(os.path.join(d,f))
        # Create page files in secondary
        for i in range(self.pages):
            path = os.path.join(SECONDARY, f"page_{i}.pg")
            with open(path, 'w') as fh:
                meta = {"pid": i, "created": time.time()}
                fh.write(json.dumps(meta) + "\n")
                fh.write(f"payload: this is page {i}\n")
        # reset metadata
        self.meta = {i: PageMeta(pid=i) for i in range(self.pages)}
        self.ram_order = deque()
        self.time_step = 0
        print(f"Initialized workspace with {self.pages} pages. Open '{WORKSPACE}' in file explorer to inspect.")


    def _sample_referenced(self, pid):
        # realistic: set 1 when accessed; may remain 1 until periodic reset
        return 1

    def _sample_dirty(self, pid):
        # probabilistic write: if this access is a write (simulate)
        return random.random() < 0.2

    def _resident_duration(self, pid):
        # compute time since loaded into RAM; use meta or file timestamps
        last_loaded = self.meta[pid].last_loaded_step
        if last_loaded is None:
            return 0
        return self.time_step - last_loaded


    def log_event(self, text:str):
        with open(EVENT_LOG, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} STEP {self.time_step}: {text}\n")

    def page_location(self, pid:int) -> str:
        # return folder name where file currently resides. If missing, assume secondary.
        for d,name in [(RAM,'ram'), (PRIMARY,'primary'), (SECONDARY,'secondary')]:
            if os.path.exists(os.path.join(d, f"page_{pid}.pg")):
                return name
        return 'secondary'

    def load_page_to_ram(self, pid:int, policy:str='lru'):
        # ensure page file present in secondary or primary
        src_dir = SECONDARY
        if self.primary_enabled and os.path.exists(os.path.join(PRIMARY, f"page_{pid}.pg")):
            src_dir = PRIMARY
        src = os.path.join(src_dir, f"page_{pid}.pg")
        dst = os.path.join(RAM, f"page_{pid}.pg")
        if os.path.exists(dst):
            return  # already in RAM
        # if RAM full, evict
        if len(os.listdir(RAM)) >= self.ram_size:
            self.evict_one(policy)
        # move file
        try:
            os.replace(src, dst)
        except Exception:
            # if missing source, create a placeholder in RAM
            with open(dst,'w') as fh:
                fh.write(json.dumps({'pid':pid,'placeholder':True})+'\n')
        # update metadata
        self.meta[pid].access_count += 1
        self.meta[pid].last_access_step = self.time_step
        # set loaded time if not set
        if self.meta[pid].last_loaded_step is None:
            self.meta[pid].last_loaded_step = self.time_step
        # maintain FIFO order
        if pid not in self.ram_order:
            self.ram_order.append(pid)
        self.log_event(f"Loaded page {pid} into RAM (from {src_dir}).")

    def evict_one(self, policy:str='lru'):
        # choose victim according to policy
        if policy == 'fifo':
            # evict oldest in ram_order
            if not self.ram_order:
                # fallback: choose random
                victim = random.choice(os.listdir(RAM))
                victim_pid = int(victim.split('_')[1].split('.')[0])
            else:
                victim_pid = self.ram_order.popleft()
        elif policy == 'lru':
            # choose page in RAM with smallest last_access_step (oldest)
            candidates = os.listdir(RAM)
            best_pid = None
            best_last = float('inf')
            for c in candidates:
                pid = int(c.split('_')[1].split('.')[0])
                last = self.meta[pid].last_access_step if self.meta[pid].last_access_step is not None else -1
                # older means smaller last -> evict the smallest
                if last < best_last:
                    best_last = last
                    best_pid = pid
            victim_pid = best_pid
        elif policy == 'ml' and ML_AVAILABLE and os.path.exists(MODEL_FILE):
            # use model predictions
            model = joblib.load(MODEL_FILE)
            candidates = os.listdir(RAM)
            ids = []
            X = []
            for c in candidates:
                pid = int(c.split('_')[1].split('.')[0])
                feat = self._make_feature(pid)
                X.append(feat)
                ids.append(pid)
            if len(X) == 0:
                return
            probs = model.predict_proba(np.array(X))[:,1]  # prob of future use
            # choose lowest prob
            idx = int(np.argmin(probs))
            victim_pid = ids[idx]
        else:
            # fallback: random
            cands = os.listdir(RAM)
            victim_pid = int(random.choice(cands).split('_')[1].split('.')[0])

        # move victim to primary if enabled, else to secondary
        src = os.path.join(RAM, f"page_{victim_pid}.pg")
        dst_dir = PRIMARY if self.primary_enabled else SECONDARY
        dst = os.path.join(dst_dir, f"page_{victim_pid}.pg")
        os.replace(src, dst)
        # reset loaded timestamp for victim
        self.meta[victim_pid].last_loaded_step = None
        self.log_event(f"Evicted page {victim_pid} from RAM -> {os.path.basename(dst_dir)}.")
        return victim_pid

    def access_page(self, pid:int, policy:str='lru') -> bool:
        # simulate an access to page pid; return True if hit
        self.time_step += 1
        in_loc = self.page_location(pid)
        in_ram = (in_loc == 'ram')

        # compute last_delta based on previous last_access_step BEFORE we update it
        prev_last = self.meta[pid].last_access_step
        last_delta = -1 if prev_last is None else (self.time_step - prev_last)

        if in_ram:
            # hit
            self.meta[pid].access_count += 1
            self.meta[pid].last_access_step = self.time_step
            hit = True
            self.log_event(f"Accessed page {pid} -> HIT")
        else:
            # miss: load into RAM
            self.log_event(f"Accessed page {pid} -> MISS (in {in_loc})")
            self.load_page_to_ram(pid, policy=policy)
            hit = False

        # simulate hardware bits and other runtime-only fields
        referenced = self._sample_referenced(pid)   # simulate or use real hardware hook
        dirty = self._sample_dirty(pid)             # simulate writes (deterministic if you like)
        resident_duration = self._resident_duration(pid)  # steps since loaded into RAM
        free_frames = max(0, self.ram_size - len(os.listdir(RAM)))
        ram_util = len(os.listdir(RAM)) / max(1, self.ram_size)

        row = [
            self.time_step, # step
            0,              # process_id  (single-process sim -> 0; extend later)
            pid,
            int(in_ram),
            int(in_loc == 'primary'),
            int(in_loc == 'secondary'),
            last_delta if last_delta>=0 else 999999,
            self.meta[pid].access_count,
            int(referenced),
            int(dirty),
            resident_duration,
            free_frames,
            ram_util
        ]
        with open(HISTORY_CSV, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        return hit

    def _make_feature(self, pid:int) -> List[float]:
        # construct feature vector used for ML (must match offline dataset transforms)
        last = self.meta[pid].last_access_step
        last_delta = self.time_step - last if last is not None else 999999
        access_count = self.meta[pid].access_count
        # hashed pid to a small number to encode identity without huge one-hot
        pid_hash = (pid * 2654435761) % 1024
        return [last_delta, access_count, pid_hash, self.time_step]

    # Dataset creation: from history.csv build supervised examples where label=1 if page used within next FUTURE_WINDOW accesses
    def build_dataset(self) -> Optional[str]:
        if not os.path.exists(HISTORY_CSV):
            print("No history.csv found. Run accesses first to generate history.")
            return None
        df = pd.read_csv(HISTORY_CSV)
        # adjust to new column names: page_id
        df = df.sort_values('step').reset_index(drop=True)
        df['label'] = 0
        n = len(df)
        pages = df['page_id'].values
        for i in range(n):
            pid = pages[i]
            label = 0
            for j in range(i+1, min(n, i+1+FUTURE_WINDOW)):
                if pages[j] == pid:
                    label = 1
                    break
            df.at[i,'label'] = label
        # produce X,y
        X = []
        for i,row in df.iterrows():
            last = row['last_access_delta'] if row['last_access_delta']>=0 else 999999
            ac = row['access_count']
            pid_hash = (int(row['page_id']) * 2654435761) % 1024
            X.append([last, ac, pid_hash, int(row['step'])])
        y = df['label'].values
        dataset_path = os.path.join(WORKSPACE, 'dataset.parquet')
        df.to_parquet(dataset_path)
        print(f"Built dataset with {len(X)} rows. Saved to {dataset_path}")
        return dataset_path

    def train_model(self, test_size:float=0.2, random_state:int=42):
        if not ML_AVAILABLE:
            print("ML packages not available. Install scikit-learn, pandas, joblib, numpy to enable training.")
            return
        if not os.path.exists(HISTORY_CSV):
            print("No history to train on. Generate accesses first.")
            return
        df = pd.read_csv(HISTORY_CSV)
        df = df.sort_values('step').reset_index(drop=True)
        # label assignment as in build_dataset
        df['label'] = 0
        n = len(df)
        pages = df['page_id'].values
        for i in range(n):
            label = 0
            for j in range(i+1, min(n, i+1+FUTURE_WINDOW)):
                if pages[j] == pages[i]:
                    label = 1
                    break
            df.at[i,'label'] = label
        X = []
        for i,row in df.iterrows():
            last = row['last_access_delta'] if row['last_access_delta']>=0 else 999999
            ac = row['access_count']
            pid_hash = (int(row['page_id']) * 2654435761) % 1024
            X.append([last, ac, pid_hash, int(row['step'])])
        y = df['label'].values
        X = np.array(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        joblib.dump(model, MODEL_FILE)
        print(f"Trained RandomForest model and saved to {MODEL_FILE}. Test accuracy: {acc:.3f}")
        self.log_event(f"Trained model. Test acc {acc:.3f}")

    # Synthetic trace generator — produces locality of reference using working sets
    def synthetic_trace(self, steps:int=500, working_set:int=20) -> List[int]:
        trace = []
        ws = list(range(min(self.pages, working_set)))
        for s in range(steps):
            # occasionally shift working set to simulate phase changes
            if random.random() < 0.02:
                start = random.randint(0, max(0,self.pages - working_set))
                ws = list(range(start, start+working_set))
            # pick page with locality
            if random.random() < 0.85:
                pid = random.choice(ws)
            else:
                pid = random.randint(0, self.pages-1)
            trace.append(pid)
        return trace

    def run_trace(self, trace:List[int], policy:str='lru', verbose:bool=False):
        hits = 0
        for pid in trace:
            hit = self.access_page(pid, policy=policy)
            hits += 1 if hit else 0
            if verbose:
                print(f"Step {self.time_step}: Access {pid} -> {'HIT' if hit else 'MISS'}")
        total = len(trace)
        hitrate = hits/total if total>0 else 0
        print(f"Done. Steps={total}, Hits={hits}, Hit-rate={hitrate:.3f}")
        self.log_event(f"Trace finished. Steps={total} Hits={hits} Hit-rate={hitrate:.3f}")
        return hitrate


# CLI wiring

def main():
    parser = argparse.ArgumentParser(description='Memory Management Simulator')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--pages', type=int, default=100)
    parser.add_argument('--ram', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--trace', choices=['synthetic','file'], default='synthetic')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--policy', choices=['lru','fifo','ml'], default='lru')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--working_set', type=int, default=20)
    args = parser.parse_args()

    sim = MemorySimulator(pages=args.pages, ram_size=args.ram)
    if args.init:
        sim.init_workspace(overwrite=args.overwrite)
        print(f"Workspace at: {WORKSPACE}")
        return
    if args.run:
        if args.trace == 'synthetic':
            trace = sim.synthetic_trace(steps=args.steps, working_set=args.working_set)
        else:
            # read file named trace.txt in workspace, one pid per line
            tfile = os.path.join(WORKSPACE, 'trace.txt')
            if not os.path.exists(tfile):
                print(f"No trace file at {tfile}")
                return
            with open(tfile,'r') as f:
                trace = [int(line.strip()) for line in f if line.strip()]
        print(f"Running trace with policy={args.policy} (RAM size={args.ram})")
        sim.run_trace(trace, policy=args.policy, verbose=False)
        print("Open the folder 'memsim_workspace' in your file explorer to see pages in RAM/primary/secondary.")
        return
    if args.train:
        sim.train_model()
        return
    parser.print_help()

if __name__ == '__main__':
    main()
