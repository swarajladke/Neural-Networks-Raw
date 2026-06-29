"""
Raw AGNIS — src/agnis/utils/logging.py

Structured metric logging utilities.
"""

import json
import os
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime


class ExperimentLogger:
    """
    Structured logger for experiment metrics.

    Writes metrics to:
    - A JSONL file (one record per step)
    - A CSV summary file at end of experiment

    Parameters
    ----------
    log_dir : str
        Directory to write log files.
    experiment_name : str
        Prefix for log files.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.jsonl")
        self._records: List[Dict[str, Any]] = []

    def log(self, step: int, data: Dict[str, Any]):
        """Log a record at a given step."""
        record = {"step": step, "timestamp": datetime.now().isoformat(), **data}
        self._records.append(record)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self, summary: Dict[str, Any], filename: Optional[str] = None):
        """Save experiment summary to JSON."""
        if filename is None:
            filename = f"{self.experiment_name}_summary.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Logger] Summary saved to {path}")

    def save_csv(self, filename: Optional[str] = None):
        """Save all step records to CSV."""
        if not self._records:
            return
        if filename is None:
            filename = f"{self.experiment_name}_steps.csv"
        path = os.path.join(self.log_dir, filename)
        keys = list(self._records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for record in self._records:
                writer.writerow({k: record.get(k, "") for k in keys})
        print(f"[Logger] Step log saved to {path}")

    def print_latest(self, n: int = 5):
        """Print the last n log records."""
        for record in self._records[-n:]:
            print(record)
