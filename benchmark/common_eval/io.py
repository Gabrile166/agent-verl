"""Common I/O helpers for benchmark evaluation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple


class JsonlPredictionWriter:
    """Single-writer helper to append JSONL records from worker threads safely."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._queue: "Queue[Optional[Dict[str, Any]]]" = Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        with open(self.output_path, "a", encoding="utf-8") as handle:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                self._queue.task_done()

    def write(self, item: Dict[str, Any]) -> None:
        self._queue.put(item)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()


def ensure_output_dir(output_dir: str) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.jsonl")
    metrics_path = os.path.join(output_dir, "metrics.json")
    summary_path = os.path.join(output_dir, "predictions_summary.json")
    return prediction_path, metrics_path, summary_path


def prepare_prediction_file(prediction_path: str, resume: bool) -> None:
    if resume:
        return
    with open(prediction_path, "w", encoding="utf-8"):
        pass


def load_existing_predictions(prediction_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(prediction_path):
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    with open(prediction_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse predictions file {prediction_path} at line {line_no}: "
                    f"{exc.msg}"
                ) from exc
            sample_id = str(item.get("sample_id", "")).strip()
            if sample_id:
                existing[sample_id] = item
    return existing


def dump_json(path: str, payload: Any) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def dump_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_benchmark_file(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Load either a JSON array or JSONL benchmark file."""

    input_path = Path(path)
    raw_text = input_path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON array file {input_path}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})"
            ) from exc
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {input_path}, but got {type(payload).__name__}")
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected object items in JSON array {input_path}, but item #{index} is {type(item).__name__}"
                )
            rows.append(dict(item))
            if max_samples > 0 and len(rows) >= max_samples:
                break
        return rows

    rows = []
    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSONL file {input_path} at line {line_no}: {exc.msg} "
                f"(column {exc.colno})"
            ) from exc
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected a JSON object in JSONL file {input_path} at line {line_no}, "
                f"but got {type(item).__name__}"
            )
        rows.append(dict(item))
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows
