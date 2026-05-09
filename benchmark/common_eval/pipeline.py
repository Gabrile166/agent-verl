"""Shared threaded execution helpers for evaluation runners."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .io import JsonlPredictionWriter


def evaluate_samples(
    samples: Iterable[Any],
    existing_predictions: Dict[str, Dict[str, Any]],
    writer: JsonlPredictionWriter,
    worker_fn: Callable[[Any], Dict[str, Any]],
    num_workers: int,
    print_every: int,
    failure_builder: Callable[[Any, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = list(existing_predictions.values())
    pending_samples = [sample for sample in samples if getattr(sample, "sample_id", "") not in existing_predictions]
    total_count = len(results) + len(pending_samples)
    completed_count = len(results)

    if not pending_samples:
        return results

    max_workers = min(len(pending_samples), max(1, num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(worker_fn, sample): sample for sample in pending_samples}

        while future_to_sample:
            done, _ = wait(list(future_to_sample.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                sample = future_to_sample.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = failure_builder(sample, repr(exc))
                writer.write(result)
                results.append(result)
                completed_count += 1
                if print_every > 0 and (completed_count % print_every == 0 or completed_count == total_count):
                    print(f"[progress] completed {completed_count}/{total_count}")

    return results


def order_results_by_sample_id(results: Sequence[Dict[str, Any]], samples: Sequence[Any]) -> List[Dict[str, Any]]:
    result_by_id = {str(item.get("sample_id", "")): item for item in results if str(item.get("sample_id", ""))}
    return [result_by_id[sample.sample_id] for sample in samples if sample.sample_id in result_by_id]
