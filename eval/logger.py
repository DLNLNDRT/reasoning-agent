# eval/logger.py
import os, csv, time
from pathlib import Path
from typing import Optional, Dict, Any

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "log.csv"

COLUMNS = [
    "timestamp", "run_id", "mode",
    "provider", "model", "family",
    "technique", "subject", "question_index",
    "prediction", "reference", "correct",
    "latency_sec",
    # hyperparams (nullable)
    "self_consistency_n", "self_ask_steps", "few_shot_k"
]

class Logger:
    def __init__(self, csv_path: Optional[str] = None):
        self.path = Path(csv_path) if csv_path else RESULTS_CSV
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    def append(self, row: Dict[str, Any]):
        # ensure all columns exist
        out = {k: row.get(k) for k in COLUMNS}
        out["timestamp"] = out.get("timestamp") or int(time.time())
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(out)
