"""Run progress delta classification benchmark evaluation with local model inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

for candidate_root in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (candidate_root / "benchmarks").exists() and str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))
        break

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from benchmarks.common_eval import (
        GenericAnswerExtractorClient,
        JsonlPredictionWriter,
        LocalModelRunner,
        PromptLengthEstimator,
        build_local_model_config,
        dump_json,
        dump_jsonl,
        ensure_output_dir,
        evaluate_samples,
        load_benchmark_file,
        load_existing_predictions,
        order_results_by_sample_id,
        prepare_prediction_file,
    )
    from benchmarks.progress_delta_classification.core.render import QUESTION_TEXT, render_model_input
    from benchmarks.progress_delta_classification.core.schema import ProgressDeltaSample
    from benchmarks.progress_delta_classification.eval.metrics import summarize_results
else:
    from ...common_eval import (
        GenericAnswerExtractorClient,
        JsonlPredictionWriter,
        LocalModelRunner,
        PromptLengthEstimator,
        build_local_model_config,
        dump_json,
        dump_jsonl,
        ensure_output_dir,
        evaluate_samples,
        load_benchmark_file,
        load_existing_predictions,
        order_results_by_sample_id,
        prepare_prediction_file,
    )
    from ..core.render import QUESTION_TEXT, render_model_input
    from ..core.schema import ProgressDeltaSample
    from .metrics import summarize_results

BENCHMARK_NAME = "progress_delta_classification"
ALLOWED_LABELS = ["increase", "decrease", "same"]
FORBIDDEN_PROMPT_SUBSTRINGS = [
    "progress_before",
    "progress_after",
    "progress_delta",
    "scorer_details",
    "source label",
    "subgoal_idx",
    "label_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run progress delta classification benchmark evaluation")
    parser.add_argument("--input", required=True, help="Benchmark JSON or JSONL file path")
    parser.add_argument("--output-dir", "--output_folder", dest="output_dir", required=True)
    parser.add_argument("--model-path", "--model_name", dest="model_path", required=False, default="")
    parser.add_argument("--infer-backend", "--infer_backend", dest="infer_backend", default="vllm", choices=["vllm"])
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--num-workers", "--eval_num_threads", dest="num_workers", type=int, default=8)
    parser.add_argument("--max-retries", "--max_retries", dest="max_retries", type=int, default=2)
    parser.add_argument("--retry-sleep", "--retry_sleep", dest="retry_sleep", type=float, default=1.0)
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--print-every", "--print_every", dest="print_every", type=int, default=20)
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", dest="gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", "--max_model_len", dest="max_model_len", type=int, default=0)
    parser.add_argument("--max-num-seqs", "--max_num_seqs", dest="max_num_seqs", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", "--trust_remote_code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--chat-template", "--chat_template", dest="chat_template", default="")
    parser.add_argument(
        "--answer-extractor-urls",
        "--eval_base_urls",
        "--eval_base_url",
        dest="answer_extractor_urls",
        default="http://127.0.0.1:8080/v1,http://127.0.0.1:8081/v1",
    )
    parser.add_argument("--answer-extractor-model", "--eval_model_name", dest="answer_extractor_model", default="")
    parser.add_argument(
        "--answer-extractor-api-key",
        "--eval_api_key",
        "--eval-api-key",
        dest="answer_extractor_api_key",
        default=os.environ.get("ANSWER_EXTRACTOR_API_KEY", ""),
    )
    parser.add_argument("--answer-extractor-max-tokens", type=int, default=16)
    parser.add_argument("--answer-extractor-timeout", type=float, default=60.0)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dump-prompts", action="store_true")
    return parser.parse_args()


def _parse_answer_extractor_urls(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def load_samples(input_path: str, max_samples: int = 0) -> List[ProgressDeltaSample]:
    rows = load_benchmark_file(input_path, max_samples=max_samples)
    samples: List[ProgressDeltaSample] = []
    for index, row in enumerate(rows, start=1):
        try:
            samples.append(ProgressDeltaSample.from_dict(row))
        except Exception as exc:
            raise ValueError(f"Failed to parse sample #{index} from {input_path}: {exc}") from exc
    return samples


def _assert_prompt_is_clean(sample: ProgressDeltaSample, prompt: str) -> None:
    lowered_prompt = prompt.lower()
    for forbidden in FORBIDDEN_PROMPT_SUBSTRINGS:
        if forbidden in lowered_prompt:
            raise ValueError(f"Prompt leakage detected for sample {sample.sample_id}: contains {forbidden!r}")
    if "highest milestone" in lowered_prompt or "closest milestone" in lowered_prompt:
        raise ValueError(f"Prompt leakage detected for sample {sample.sample_id}: wrong milestone wording")
    required_fragments = [QUESTION_TEXT.lower(), "increase", "decrease", "same"]
    for fragment in required_fragments:
        if fragment not in lowered_prompt:
            raise ValueError(f"Prompt validation failed for sample {sample.sample_id}: missing fragment {fragment!r}")


def _build_prompt_records(samples: List[ProgressDeltaSample]) -> List[Dict[str, Any]]:
    prompt_rows: List[Dict[str, Any]] = []
    for sample in samples:
        prompt = render_model_input(sample)
        _assert_prompt_is_clean(sample, prompt)
        prompt_rows.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": BENCHMARK_NAME,
                "allowed_labels": ALLOWED_LABELS,
                "prompt": prompt,
                "ground_truth_label": sample.label,
            }
        )
    return prompt_rows


def _build_failure_result(sample: ProgressDeltaSample, error_text: str) -> Dict[str, Any]:
    generation_mode = str(sample.details.get("generation_mode") or "")
    return {
        "sample_id": sample.sample_id,
        "benchmark": BENCHMARK_NAME,
        "env": sample.env,
        "predicted_label": "INVALID",
        "ground_truth_label": sample.label,
        "correct": False,
        "raw_response": "",
        "extractor_output": "",
        "prompt": "",
        "prompt_tokens": None,
        "latency_sec": None,
        "error": error_text,
        "metadata": {
            "progress_before": sample.progress_before,
            "progress_after": sample.progress_after,
            "progress_delta": sample.progress_delta,
            "generation_mode": generation_mode or None,
        },
    }


def evaluate_one_sample(
    sample: ProgressDeltaSample,
    model_runner: LocalModelRunner,
    answer_extractor: GenericAnswerExtractorClient,
    estimator: PromptLengthEstimator,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    prompt = render_model_input(sample)
    _assert_prompt_is_clean(sample, prompt)
    prompt_tokens = estimator.count_tokens(prompt)

    started_at = time.time()
    raw_response = ""
    last_error = "Unknown evaluation error"
    max_attempts = max(1, max_retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            raw_response = model_runner.generate(prompt)
            last_error = ""
            break
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_attempts:
                time.sleep(max(0.0, retry_sleep))
    else:
        raise RuntimeError(last_error)

    extraction = answer_extractor.extract_answer_with_metadata(
        raw_response=raw_response,
        allowed_labels=ALLOWED_LABELS,
        task_name=BENCHMARK_NAME,
        question=QUESTION_TEXT,
    )
    predicted_label = extraction["label"]
    generation_mode = str(sample.details.get("generation_mode") or "")
    latency_sec = time.time() - started_at

    return {
        "sample_id": sample.sample_id,
        "benchmark": BENCHMARK_NAME,
        "env": sample.env,
        "predicted_label": predicted_label,
        "ground_truth_label": sample.label,
        "correct": predicted_label == sample.label,
        "raw_response": raw_response,
        "extractor_output": extraction.get("extractor_output", ""),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "latency_sec": latency_sec,
        "error": extraction.get("error"),
        "metadata": {
            "progress_before": sample.progress_before,
            "progress_after": sample.progress_after,
            "progress_delta": sample.progress_delta,
            "generation_mode": generation_mode or None,
        },
    }


class ProgressDeltaEvaluationPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prediction_path, self.metrics_path, self.summary_path = ensure_output_dir(args.output_dir)

    def run(self) -> None:
        samples = load_samples(self.args.input, max_samples=self.args.max_samples)
        prompt_rows = _build_prompt_records(samples)

        if self.args.dry_run:
            self._run_dry_run(samples, prompt_rows)
            return

        if not self.args.model_path:
            raise ValueError("--model-path is required unless --dry-run is enabled")

        prepare_prediction_file(self.prediction_path, self.args.resume)
        existing_predictions = self._load_existing_predictions(samples)
        estimator = PromptLengthEstimator(self.args.tokenizer)
        model_runner = LocalModelRunner(build_local_model_config(self.args))
        answer_extractor = GenericAnswerExtractorClient(
            base_urls=_parse_answer_extractor_urls(self.args.answer_extractor_urls),
            model_name=self.args.answer_extractor_model,
            api_key=self.args.answer_extractor_api_key,
            temperature=0.0,
            max_tokens=self.args.answer_extractor_max_tokens,
            timeout=self.args.answer_extractor_timeout,
        )

        writer = JsonlPredictionWriter(self.prediction_path)
        try:
            results = evaluate_samples(
                samples=samples,
                existing_predictions=existing_predictions,
                writer=writer,
                worker_fn=lambda sample: evaluate_one_sample(
                    sample,
                    model_runner=model_runner,
                    answer_extractor=answer_extractor,
                    estimator=estimator,
                    max_retries=self.args.max_retries,
                    retry_sleep=self.args.retry_sleep,
                ),
                num_workers=self.args.num_workers,
                print_every=self.args.print_every,
                failure_builder=_build_failure_result,
            )
        finally:
            writer.close()

        ordered_results = order_results_by_sample_id(results, samples)
        metrics = summarize_results(ordered_results, samples)
        metrics_payload = {
            "benchmark": BENCHMARK_NAME,
            "input": self.args.input,
            **metrics,
        }
        dump_json(self.metrics_path, metrics_payload)
        summary_payload = {
            "benchmark": BENCHMARK_NAME,
            "input": self.args.input,
            "output_dir": self.args.output_dir,
            "model_path": self.args.model_path,
            "timestamp": int(time.time()),
            "num_samples": len(samples),
            "metrics_path": self.metrics_path,
            "predictions_path": self.prediction_path,
            "config": vars(self.args),
        }
        dump_json(self.summary_path, summary_payload)
        print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True))

    def _load_existing_predictions(self, samples: List[ProgressDeltaSample]) -> Dict[str, Dict[str, Any]]:
        if not self.args.resume:
            return {}
        sample_ids = {sample.sample_id for sample in samples}
        existing = load_existing_predictions(self.prediction_path)
        return {sample_id: row for sample_id, row in existing.items() if sample_id in sample_ids}

    def _run_dry_run(self, samples: List[ProgressDeltaSample], prompt_rows: List[Dict[str, Any]]) -> None:
        prompts_path = os.path.join(self.args.output_dir, "prompts.jsonl")
        dump_jsonl(prompts_path, prompt_rows)
        summary = {
            "benchmark": BENCHMARK_NAME,
            "input": self.args.input,
            "output_dir": self.args.output_dir,
            "num_samples": len(samples),
            "prompts_path": prompts_path,
            "dry_run": True,
            "config": vars(self.args),
        }
        dump_json(os.path.join(self.args.output_dir, "dry_run_summary.json"), summary)
        if self.args.dump_prompts:
            print(json.dumps(prompt_rows[: min(3, len(prompt_rows))], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    ProgressDeltaEvaluationPipeline(args).run()


if __name__ == "__main__":
    main()
