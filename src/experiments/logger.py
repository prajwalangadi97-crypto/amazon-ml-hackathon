"""Experiment logging utility for Amazon ML Challenge 2026.

Records hypothesis, configuration, validation split, models, hyperparameters,
macro F0.5, precision, recall, execution timings, and qualitative conclusions.
Uses only the standard library (json, pathlib, datetime, dataclasses).
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentRecord:
    """Complete metadata and evaluation metrics for one experiment."""

    experiment_id: str
    hypothesis: str
    data_split_description: str
    blocking_method: str
    features: List[str]
    model: str
    hyperparameters: Dict[str, Any]
    validation_method: str
    macro_F0_5: float
    precision: float
    recall: float
    training_time_seconds: float
    inference_time_seconds: float
    notes: str
    conclusion: str
    next_experiment: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Instantiate record from dictionary."""
        return cls(**data)


class ExperimentLogger:
    """Manages experiment persistence and retrieval under experiments/."""

    def __init__(self, experiments_dir: Union[str, Path] = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.experiments_dir / "experiments_log.jsonl"

    def log(self, record: ExperimentRecord) -> Path:
        """Save experiment record to individual JSON file and append to JSONL log.

        Args:
            record: Populated ExperimentRecord.

        Returns:
            Path to the saved individual experiment JSON file.
        """
        record_dict = record.to_dict()

        # 1. Save individual experiment file
        exp_file = self.experiments_dir / f"{record.experiment_id}.json"
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(record_dict, f, indent=2)

        # 2. Append to JSONL log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")

        return exp_file

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve an experiment by its ID."""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        if not exp_file.is_file():
            return None
        with open(exp_file, "r", encoding="utf-8") as f:
            return ExperimentRecord.from_dict(json.load(f))

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List summary entries of all logged experiments."""
        if not self.log_file.is_file():
            return []

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def summary_table(self) -> str:
        """Return formatted table summarizing all logged experiments."""
        entries = self.list_experiments()
        if not entries:
            return "No experiments logged yet."

        header = f"{'Experiment ID':<22} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'Model':<15} | {'Hypothesis':<30}"
        sep = "-" * len(header)
        rows = [header, sep]
        for e in entries:
            exp_id = e.get("experiment_id", "")[:22]
            f05 = f"{e.get('macro_F0_5', 0.0):.4f}"
            p = f"{e.get('precision', 0.0):.4f}"
            r = f"{e.get('recall', 0.0):.4f}"
            model = e.get("model", "")[:15]
            hyp = e.get("hypothesis", "")[:30]
            rows.append(
                f"{exp_id:<22} | {f05:<10} | {p:<10} | {r:<10} | {model:<15} | {hyp:<30}"
            )
        return "\n".join(rows)
