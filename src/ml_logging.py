import csv
from pathlib import Path
from typing import List, Dict, Any
from .features import FEATURE_ORDER


class DecisionLogger:
    """Append decision data (features + chosen flag + score) to CSV."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._file = None

    def _ensure_writer(self):
        if self._writer is not None:
            return
        exists = self.path.exists()
        self._file = self.path.open("a", newline="", encoding="utf-8")
        fieldnames = ["decision_type","instance","time","veh_idx","req_idx","chosen","score"] + FEATURE_ORDER
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not exists:
            self._writer.writeheader()

    def log(self, decision_type: str, instance: str, time: float, veh_idx: int, req_idx: int,
            chosen: bool, score: float, features: List[float]):
        self._ensure_writer()
        row = {
            "decision_type": decision_type,
            "instance": instance,
            "time": time,
            "veh_idx": veh_idx,
            "req_idx": req_idx,
            "chosen": int(chosen),
            "score": score,
        }
        for i, name in enumerate(FEATURE_ORDER):
            row[name] = features[i] if i < len(features) else None
        self._writer.writerow(row)
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
