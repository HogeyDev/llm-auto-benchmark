"""
Replay checkpoints for benchmark inference calls.

A checkpoint is a JSONL file containing one record per model call. Records are
keyed by model identity, task, dataset revision, call mode, exact messages or
prompt, and generation parameters, so a later run can serve identical calls
without touching the model.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

EVALUATOR_VERSION = "2.0.0"


class CheckpointStore:
    """Read and append replay records for one model/quantization pair."""

    def __init__(
        self,
        path: str,
        model_id: str,
        quantization: str,
        evaluator_version: str = EVALUATOR_VERSION,
    ):
        self.path = Path(path)
        self.model_id = model_id
        self.quantization = quantization
        self.evaluator_version = evaluator_version
        self.entries: Dict[str, dict] = {}
        self.loaded = 0
        self.hits = 0
        self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("model_id") != self.model_id:
                    continue
                if record.get("quantization") != self.quantization:
                    continue
                if record.get("evaluator_version") != self.evaluator_version:
                    continue
                key = record.get("key")
                if key:
                    self.entries[key] = record
                    self.loaded += 1

    @staticmethod
    def make_key(
        model_id: str,
        quantization: str,
        evaluator_version: str,
        task_id: str,
        dataset_revision: str,
        mode: str,
        payload: Dict[str, Any],
    ) -> str:
        material = json.dumps(
            {
                "model_id": model_id,
                "quantization": quantization,
                "evaluator_version": evaluator_version,
                "task_id": task_id,
                "dataset_revision": dataset_revision,
                "mode": mode,
                "payload": payload,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self.entries.get(key)

    def put(self, record: Dict[str, Any]) -> None:
        self.entries[record["key"]] = record
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
