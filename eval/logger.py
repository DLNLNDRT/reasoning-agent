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
    "self_consistency_n", "self_ask_steps", "few_shot_k",
    # NEW: tracing
    "prompt_snippet", "output_snippet",
]

class Logger:
    def __init__(self, csv_path: Optional[str] = None):
        self.path = Path(csv_path) if csv_path else RESULTS_CSV
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
        else:
            self._ensure_columns()

    def _ensure_columns(self):
        """If existing CSV misses new columns, rewrite it in place with added columns."""
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            old_cols = reader.fieldnames or []
            if set(COLUMNS).issubset(set(old_cols)):
                return  # nothing to do
            rows = list(reader)

        # Build new rows with all columns
        new_rows = []
        for r in rows:
            nr = {c: r.get(c, "") for c in COLUMNS}
            new_rows.append(nr)

        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(new_rows)

    def append(self, row: Dict[str, Any]):
        out = {k: row.get(k) for k in COLUMNS}
        out["timestamp"] = out.get("timestamp") or int(time.time())
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(out)
