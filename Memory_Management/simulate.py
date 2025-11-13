import os
import csv
HISTORY_REAL_CSV = 'history_real.csv'

pid = 0

# when initializing workspace, create header if missing
if not os.path.exists(HISTORY_REAL_CSV):
    with open(HISTORY_REAL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "step","process_id","page_id","in_ram","in_primary","in_secondary",
            "last_access_delta","access_count","referenced_bit","dirty_bit",
            "resident_duration","free_frames","ram_utilization"
        ])

# inside access_page(pid, ...) after updating meta...
referenced = self._sample_referenced(pid)   # simulate or use real hardware hook
dirty = self._sample_dirty(pid)             # simulate writes (deterministic if you like)
resident_duration = self._resident_duration(pid)  # steps since loaded into RAM
free_frames = max(0, self.ram_size - len(os.listdir(RAM)))
ram_util = len(os.listdir(RAM)) / max(1, self.ram_size)

row = [
    self.time_step, # step
    0,              # process_id  (single-process sim -> 0; extend later)
    pid,
    int(in_loc == 'ram'),
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
with open(HISTORY_REAL_CSV, 'a', newline='') as f:
    csv.writer(f).writerow(row)
