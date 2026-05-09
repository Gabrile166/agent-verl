"""Run pairwise benchmark evaluation with local model inference and answer extraction services."""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]

for candidate_root in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (candidate_root / "benchmarks").exists() and str(candidate_root) not in sys.path:
        sys.path.insert(0, str(candidate_root))
        break

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from benchmarks.common_eval.io import load_benchmark_file
    from benchmarks.pairwise_phi_ranking.core.schema import BenchmarkSample
    from benchmarks.pairwise_phi_ranking.eval.metrics import (
        compute_accuracy_by_pair_type,
        compute_invalid_response_rate,
        compute_pairwise_choice_acc,
        compute_subset_accuracies,
    )
    from benchmarks.pairwise_phi_ranking.eval.prompt_builder import PairwiseComparisonPromptBuilder
else:
    from ...common_eval.io import load_benchmark_file
    from ..core.schema import BenchmarkSample
    from .metrics import (
        compute_accuracy_by_pair_type,
        compute_invalid_response_rate,
        compute_pairwise_choice_acc,
        compute_subset_accuracies,
    )
    from .prompt_builder import PairwiseComparisonPromptBuilder


ANSWER_EXTRACTOR_SYSTEM_PROMPT = (
    "You extract the final decision from another model's reasoning for a two-choice A/B benchmark. "
    "Return only one token: A, B, or INVALID. Do not output anything else."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise phi ranking benchmark evaluation")
    parser.add_argument("--input", required=True, help="Benchmark JSONL file path")
    parser.add_argument(
        "--output-dir",
        "--output_folder",
        dest="output_dir",
        required=True,
        help="Directory to write predictions and metrics",
    )
    parser.add_argument(
        "--model-path",
        "--model_name",
        dest="model_path",
        required=True,
        help="Local model path used for benchmark inference",
    )
    parser.add_argument("--model-type", default="", help="Compatibility option; native vLLM does not use it.")
    parser.add_argument(
        "--model_type",
        dest="model_type",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--infer-backend",
        "--infer_backend",
        dest="infer_backend",
        default="vllm",
        choices=["vllm"],
        help="Local inference backend for loading the benchmark model",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt passed to the local benchmark model",
    )
    parser.add_argument(
        "--max-tokens",
        "--max_tokens",
        dest="max_tokens",
        type=int,
        default=8192,
        help="Max completion tokens per benchmark sample",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for benchmark model")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0, help="Top-p for vLLM sampling")
    parser.add_argument(
        "--num-workers",
        "--eval_num_threads",
        dest="num_workers",
        type=int,
        default=8,
        help="Concurrent evaluation worker threads",
    )
    parser.add_argument(
        "--max-retries",
        "--max_retries",
        dest="max_retries",
        type=int,
        default=2,
        help="Retries per sample after local generation failures",
    )
    parser.add_argument(
        "--retry-sleep",
        "--retry_sleep",
        dest="retry_sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between retries",
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=int,
        default=0,
        help="Only evaluate first N samples when > 0",
    )
    parser.add_argument("--resume", action="store_true", help="Skip samples already present in predictions.jsonl")
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer name/path used to estimate prompt lengths via transformers",
    )
    parser.add_argument(
        "--print-every",
        "--print_every",
        dest="print_every",
        type=int,
        default=20,
        help="Print progress every N completed samples",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tensor_parallel_size",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size when infer-backend=vllm",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        "--gpu_memory_utilization",
        dest="gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization when infer-backend=vllm",
    )
    parser.add_argument(
        "--max-model-len",
        "--max_model_len",
        dest="max_model_len",
        type=int,
        default=0,
        help="Max model context length passed to the inference backend. Use 0 to keep backend default.",
    )
    parser.add_argument(
        "--max-num-seqs",
        "--max_num_seqs",
        dest="max_num_seqs",
        type=int,
        default=64,
        help="Maximum in-flight sequences when infer-backend=vllm",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype passed to native vLLM, e.g. auto, bfloat16, float16.",
    )
    parser.add_argument(
        "--trust-remote-code",
        "--trust_remote_code",
        dest="trust_remote_code",
        action="store_true",
        default=True,
        help="Allow custom model/tokenizer code when loading with native vLLM.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code for vLLM/tokenizer loading.",
    )
    parser.add_argument(
        "--chat-template",
        "--chat_template",
        dest="chat_template",
        default="",
        help="Optional raw chat template string passed to tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--answer-extractor-urls",
        "--eval_base_urls",
        "--eval_base_url",
        dest="answer_extractor_urls",
        default="http://127.0.0.1:8080/v1,http://127.0.0.1:8081/v1",
        help="Comma-separated OpenAI-style eval endpoints used to extract final A/B answers",
    )
    parser.add_argument(
        "--answer-extractor-model",
        "--eval_model_name",
        dest="answer_extractor_model",
        default="",
        help="Eval model name for answer extraction. If empty, auto-detect via /models.",
    )
    parser.add_argument(
        "--answer-extractor-api-key",
        "--eval_api_key",
        dest="answer_extractor_api_key",
        default=os.environ.get("ANSWER_EXTRACTOR_API_KEY", "EMPTY"),
        help="API key passed to the eval answer extractor services",
    )
    parser.add_argument(
        "--eval_output_path",
        default="",
        help="Optional metrics JSON path. Defaults to <output-dir>/metrics.json.",
    )
    parser.add_argument(
        "--eval_result_detail",
        default="simple",
        choices=["simple", "detailed"],
        help="Compatibility with template scripts. detailed keeps prompt/raw_response in summary.",
    )
    parser.add_argument(
        "--answer-extractor-max-tokens",
        type=int,
        default=16,
        help="Max completion tokens for answer extractor services",
    )
    parser.add_argument(
        "--answer-extractor-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each answer extractor request",
    )
    parser.add_argument("--suffix", default="", help="Optional experiment suffix printed for template-style runs.")
    parser.add_argument("--gpu_groups", default="", help="Compatibility with template scripts; not used here.")
    parser.add_argument("--enable_data_parallel", action="store_true", help="Compatibility flag; not used here.")
    parser.add_argument("--eval_mode", default="sync", help="Compatibility with template scripts; pairwise eval always runs.")
    parser.add_argument("--model", default="", help="Deprecated. Local inference now uses --model-path.")
    parser.add_argument(
        "--base-url",
        default="",
        help="Deprecated. Local inference now uses --model-path instead of a remote base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="Deprecated. Kept only for backward compatibility with older launch scripts.",
    )
    return parser.parse_args()


class JsonlPredictionWriter:
    """Single-writer helper to append prediction records safely from worker threads."""

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


class PromptLengthEstimator:
    """Estimate prompt token lengths when a transformers tokenizer is available."""

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer_name_or_path = tokenizer_name_or_path.strip()
        self.tokenizer = None
        if self.tokenizer_name_or_path:
            if AutoTokenizer is None:
                raise ImportError("transformers is required when --tokenizer is provided")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=True)

    def count_tokens(self, text: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        encoded = self.tokenizer(text, add_special_tokens=False)
        return len(encoded["input_ids"])


class LocalModelRunner:
    """Shared local benchmark model runner backed by native vLLM."""

    def __init__(self, config: Dict[str, Any]):
        if LLM is None or SamplingParams is None:
            raise ImportError("vllm is required for local benchmark inference")
        if AutoTokenizer is None:
            raise ImportError("transformers is required to build chat prompts for native vLLM inference")

        self.model_path = str(config.get("model_path", "")).strip()
        self.infer_backend = str(config.get("infer_backend", "vllm")).strip().lower()
        self.system_prompt = str(config.get("system_prompt", ""))
        self.max_tokens = int(config.get("max_tokens", 256))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 1.0))
        self.tensor_parallel_size = int(config.get("tensor_parallel_size", 1))
        self.gpu_memory_utilization = float(config.get("gpu_memory_utilization", 0.9))
        self.max_model_len = config.get("max_model_len")
        self.max_num_seqs = int(config.get("max_num_seqs", 64))
        self.dtype = str(config.get("dtype", "auto"))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.chat_template = str(config.get("chat_template", ""))
        self._lock = threading.Lock()
        self.tokenizer = self._load_tokenizer()
        self.engine = self._build_engine()
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def _load_tokenizer(self) -> Any:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        return AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)

    def _build_engine(self) -> Any:
        if not self.model_path:
            raise ValueError("model_path must not be empty")

        if self.infer_backend == "vllm":
            kwargs = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_num_seqs": self.max_num_seqs,
                "dtype": self.dtype,
                "trust_remote_code": self.trust_remote_code,
            }
            if self.max_model_len is not None:
                kwargs["max_model_len"] = self.max_model_len
            return LLM(**kwargs)

        raise ValueError(f"Unsupported infer_backend: {self.infer_backend}")

    def generate(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        rendered_prompt = self._apply_chat_template(messages)
        with self._lock:
            responses = self.engine.generate([rendered_prompt], self.sampling_params, use_tqdm=False)
        if not responses:
            return ""
        outputs = getattr(responses[0], "outputs", None) or []
        if not outputs:
            return ""
        return str(getattr(outputs[0], "text", "") or "")

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=self.chat_template or None,
            )
            return str(rendered)
        except Exception:
            parts: List[str] = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)


class AnswerExtractorClient:
    """Round-robin client for local answer-extractor model services."""

    def __init__(self, config: Dict[str, Any]):
        self.urls = _parse_service_urls(str(config.get("urls", "")))
        self.model_override = str(config.get("model", "")).strip()
        self.api_key = str(config.get("api_key", "EMPTY"))
        self.timeout = float(config.get("timeout", 60.0))
        self.max_tokens = int(config.get("max_tokens", 16))
        self.temperature = float(config.get("temperature", 0.0))
        self._cursor = 0
        self._cursor_lock = threading.Lock()
        self._model_cache: Dict[str, str] = {}
        self._model_cache_lock = threading.Lock()
        self._session_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            setattr(self._session_local, "session", session)
        return session

    def _ordered_urls(self) -> List[str]:
        with self._cursor_lock:
            start_idx = self._cursor % len(self.urls)
            self._cursor += 1
        return self.urls[start_idx:] + self.urls[:start_idx]

    def _resolve_model(self, base_url: str) -> str:
        if self.model_override:
            return self.model_override

        with self._model_cache_lock:
            cached = self._model_cache.get(base_url)
            if cached:
                return cached

        session = self._get_session()
        response = session.get(f"{base_url}/models", timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            raise RuntimeError(f"No model was returned by answer extractor service: {base_url}")

        model_name = str(data[0].get("id", "")).strip()
        if not model_name:
            raise RuntimeError(f"Invalid /models response from answer extractor service: {base_url}")

        with self._model_cache_lock:
            self._model_cache[base_url] = model_name
        return model_name

    def extract_answer(self, sample: BenchmarkSample, raw_response: str) -> str:
        if not raw_response.strip():
            return "INVALID"

        last_error: Optional[Exception] = None
        messages = _build_answer_extractor_messages(sample, raw_response)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for base_url in self._ordered_urls():
            try:
                model_name = self._resolve_model(base_url)
                session = self._get_session()
                response = session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices returned by answer extractor service: {base_url}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                return str(content or "")
            except Exception as exc:  # pragma: no cover - network/service failure path
                last_error = exc
                continue

        if last_error is None:
            raise RuntimeError("No answer extractor URL is available")
        raise RuntimeError(repr(last_error))


def _parse_service_urls(text: str) -> List[str]:
    urls: List[str] = []
    for part in text.split(","):
        url = part.strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        urls.append(url.rstrip("/"))
    if not urls:
        raise ValueError("At least one answer extractor URL must be provided")
    return urls


def _normalize_extracted_label(text: Any) -> str:
    if text is None:
        return "INVALID"

    raw_text = str(text).strip()
    if not raw_text:
        return "INVALID"

    upper = raw_text.upper()
    if upper in {"A", "B", "INVALID"}:
        return upper

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("answer", "label", "choice", "final_answer", "final_label"):
            if key in payload:
                return _normalize_extracted_label(payload[key])

    answer_match = re.search(r"answer\s*[:=]\s*(A|B|INVALID)\b", raw_text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    standalone = re.findall(r"\b(A|B|INVALID)\b", raw_text, flags=re.IGNORECASE)
    if len(standalone) == 1:
        return standalone[0].upper()

    return "INVALID"


def _strip_think_content(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _build_answer_extractor_messages(sample: BenchmarkSample, raw_response: str) -> List[Dict[str, str]]:
    cleaned_response = _strip_think_content(raw_response)
    response_for_extraction = cleaned_response if cleaned_response else raw_response
    user_prompt = (
        "Read the following model response for a pairwise benchmark and identify the final choice.\n\n"
        f"Task description:\n{sample.task_description}\n\n"
        "Model response:\n"
        f"{response_for_extraction}\n\n"
        "Return only one token:\n"
        "- A if the response prefers trajectory A\n"
        "- B if the response prefers trajectory B\n"
        "- INVALID if the response does not determine a final choice\n"
    )
    return [
        {"role": "system", "content": ANSWER_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def load_samples(input_path: str, max_samples: int = 0) -> List[BenchmarkSample]:
    rows = load_benchmark_file(input_path, max_samples=max_samples)
    return [BenchmarkSample.from_dict(row) for row in rows]


def load_existing_predictions(prediction_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(prediction_path):
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    with open(prediction_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", "")).strip()
            if sample_id:
                existing[sample_id] = item
    return existing


def build_local_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "infer_backend": args.infer_backend,
        "system_prompt": args.system_prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len if args.max_model_len > 0 else None,
        "max_num_seqs": args.max_num_seqs,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "chat_template": args.chat_template,
    }


def build_answer_extractor_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "urls": args.answer_extractor_urls,
        "model": args.answer_extractor_model,
        "api_key": args.answer_extractor_api_key,
        "timeout": args.answer_extractor_timeout,
        "max_tokens": args.answer_extractor_max_tokens,
        "temperature": 0.0,
    }


def _build_failure_result(sample: BenchmarkSample, error_text: str) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "predicted_label": "INVALID",
        "ground_truth_label": sample.label,
        "correct": False,
        "pair_type": sample.pair_type,
        "difficulty": sample.difficulty,
        "track": sample.track,
        "task_type": sample.task_type,
        "is_subset_pair": sample.is_subset_pair,
        "uses_expert_branch": sample.uses_expert_branch,
        "num_steps_a": len(sample.trajectory_a.steps),
        "num_steps_b": len(sample.trajectory_b.steps),
        "prompt_tokens": None,
        "latency_sec": None,
        "attempts": None,
        "raw_response": "",
        "extractor_output": "",
        "prompt": "",
        "error": error_text,
    }


def evaluate_one_sample(
    sample: BenchmarkSample,
    prompt_builder: PairwiseComparisonPromptBuilder,
    model_runner: LocalModelRunner,
    answer_extractor: AnswerExtractorClient,
    estimator: PromptLengthEstimator,
    max_retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    prompt = prompt_builder.build_query(sample)
    prompt_tokens = estimator.count_tokens(prompt)

    started_at = time.time()
    max_attempts = max(1, max_retries + 1)
    last_error = "Unknown evaluation error"
    raw_response = ""
    attempts_used: Optional[int] = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw_response = model_runner.generate(prompt)
            attempts_used = attempt
            last_error = ""
            break
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_attempts:
                time.sleep(max(0.0, retry_sleep))
    else:
        raise RuntimeError(last_error)

    extractor_output = ""
    extractor_error = ""
    try:
        extractor_output = answer_extractor.extract_answer(sample, raw_response)
    except Exception as exc:  # pragma: no cover - network/service failure path
        extractor_error = repr(exc)

    predicted_label = _normalize_extracted_label(extractor_output)
    latency_sec = time.time() - started_at
    error_text = extractor_error or None

    return {
        "sample_id": sample.sample_id,
        "predicted_label": predicted_label,
        "ground_truth_label": sample.label,
        "correct": predicted_label == sample.label,
        "pair_type": sample.pair_type,
        "difficulty": sample.difficulty,
        "track": sample.track,
        "task_type": sample.task_type,
        "is_subset_pair": sample.is_subset_pair,
        "uses_expert_branch": sample.uses_expert_branch,
        "num_steps_a": len(sample.trajectory_a.steps),
        "num_steps_b": len(sample.trajectory_b.steps),
        "prompt_tokens": prompt_tokens,
        "latency_sec": latency_sec,
        "attempts": attempts_used,
        "raw_response": raw_response,
        "extractor_output": extractor_output,
        "prompt": prompt,
        "error": error_text,
    }


def evaluate_samples(
    samples: Iterable[BenchmarkSample],
    existing_predictions: Dict[str, Dict[str, Any]],
    writer: JsonlPredictionWriter,
    model_runner: LocalModelRunner,
    answer_extractor: AnswerExtractorClient,
    estimator: PromptLengthEstimator,
    num_workers: int,
    print_every: int,
    max_retries: int,
    retry_sleep: float,
) -> List[Dict[str, Any]]:
    prompt_builder = PairwiseComparisonPromptBuilder()
    results: List[Dict[str, Any]] = list(existing_predictions.values())

    pending_samples = [sample for sample in samples if sample.sample_id not in existing_predictions]
    total_count = len(results) + len(pending_samples)
    completed_count = len(results)

    if not pending_samples:
        return results

    max_workers = min(len(pending_samples), max(1, num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(
                evaluate_one_sample,
                sample,
                prompt_builder,
                model_runner,
                answer_extractor,
                estimator,
                max_retries,
                retry_sleep,
            ): sample
            for sample in pending_samples
        }

        while future_to_sample:
            done, _ = wait(list(future_to_sample.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                sample = future_to_sample.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _build_failure_result(sample, repr(exc))
                writer.write(result)
                results.append(result)
                completed_count += 1
                if print_every > 0 and (completed_count % print_every == 0 or completed_count == total_count):
                    print(f"[progress] completed {completed_count}/{total_count}")

    return results


def summarize_results(results: List[Dict[str, Any]], samples: List[BenchmarkSample]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "num_samples": len(samples),
        "num_results": len(results),
        "pairwise_choice_acc": compute_pairwise_choice_acc(results),
        "invalid_response_rate": compute_invalid_response_rate(results),
    }
    metrics.update(compute_subset_accuracies(results, samples))
    metrics.update(compute_accuracy_by_pair_type(results, samples))

    prompt_tokens = [item["prompt_tokens"] for item in results if isinstance(item.get("prompt_tokens"), int)]
    latencies = [item["latency_sec"] for item in results if isinstance(item.get("latency_sec"), (int, float))]
    attempts = [item["attempts"] for item in results if isinstance(item.get("attempts"), int)]

    metrics["avg_prompt_tokens"] = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
    metrics["max_prompt_tokens"] = max(prompt_tokens) if prompt_tokens else None
    metrics["avg_latency_sec"] = sum(latencies) / len(latencies) if latencies else None
    metrics["avg_attempts"] = sum(attempts) / len(attempts) if attempts else None
    metrics["num_failures"] = sum(1 for item in results if item.get("error"))
    return metrics


def dump_metrics(metrics: Dict[str, Any], output_dir: str, eval_output_path: str = "") -> str:
    output_path = eval_output_path.strip() or os.path.join(output_dir, "metrics.json")
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def dump_summary(results: List[Dict[str, Any]], output_dir: str, result_detail: str = "simple") -> str:
    summary_path = os.path.join(output_dir, "predictions_summary.json")
    lightweight_results: List[Dict[str, Any]] = []
    for item in results:
        if result_detail == "detailed":
            lightweight_results.append(dict(item))
        else:
            lightweight_results.append({key: value for key, value in item.items() if key not in {"prompt", "raw_response"}})
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(lightweight_results, handle, ensure_ascii=False, indent=2)
    return summary_path


def ensure_output_dir(output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.jsonl")
    metrics_path = os.path.join(output_dir, "metrics.json")
    return prediction_path, metrics_path


def prepare_prediction_file(prediction_path: str, resume: bool) -> None:
    if resume:
        return
    with open(prediction_path, "w", encoding="utf-8"):
        pass


class PairwiseEvaluationPipeline:
    """Template-style pipeline for pairwise phi ranking inference and evaluation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prediction_path, _ = ensure_output_dir(args.output_dir)
        self.model_runner: Optional[LocalModelRunner] = None
        self.answer_extractor: Optional[AnswerExtractorClient] = None
        self.estimator: Optional[PromptLengthEstimator] = None

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("🚀 Starting Pairwise Phi Ranking Evaluation Pipeline")
        print(f"   Mode: Local {self.args.infer_backend} inference + eval model answer extraction")
        print(f"   Evaluation: {self.args.eval_mode}")
        print("=" * 60)

        prepare_prediction_file(self.prediction_path, self.args.resume)
        samples = self._load_samples()
        existing_predictions = self._load_existing_predictions(samples)

        self.estimator = self._setup_prompt_length_estimator()
        self.model_runner = self._load_model()
        self.answer_extractor = self._setup_answer_extractor()

        results = self._process_samples(samples, existing_predictions)
        ordered_results = self._order_results(results, samples)
        metrics_path, summary_path = self._save_results(ordered_results, samples)

        print("\n" + "=" * 60)
        print("✅ Pairwise evaluation completed successfully!")
        print(f"   Predictions: {self.prediction_path}")
        print(f"   Metrics: {metrics_path}")
        print(f"   Summary: {summary_path}")
        print("=" * 60)

    def _load_samples(self) -> List[BenchmarkSample]:
        print("\n📂 Loading benchmark samples...")
        samples = load_samples(self.args.input, max_samples=self.args.max_samples)
        print(f"   Loaded samples: {len(samples)}")
        return samples

    def _load_existing_predictions(self, samples: List[BenchmarkSample]) -> Dict[str, Dict[str, Any]]:
        if not self.args.resume:
            return {}

        print("\n🔁 Resume mode enabled; loading existing predictions...")
        sample_by_id = {sample.sample_id: sample for sample in samples}
        existing_predictions = load_existing_predictions(self.prediction_path)
        filtered_predictions = {
            sample_id: item for sample_id, item in existing_predictions.items() if sample_id in sample_by_id
        }
        print(f"   Reused predictions: {len(filtered_predictions)}")
        return filtered_predictions

    def _setup_prompt_length_estimator(self) -> PromptLengthEstimator:
        if self.args.tokenizer:
            print("\n🔢 Loading tokenizer for prompt length estimation...")
        return PromptLengthEstimator(self.args.tokenizer)

    def _load_model(self) -> LocalModelRunner:
        print("\n🤖 Loading local benchmark model...")
        local_model_config = build_local_model_config(self.args)
        max_model_len = local_model_config["max_model_len"]
        if max_model_len is not None:
            print(f"   Max model len: {max_model_len}")
        else:
            print("   Max model len: backend default")
        return LocalModelRunner(local_model_config)

    def _setup_answer_extractor(self) -> AnswerExtractorClient:
        print("\n🔍 Setting up eval model answer extractor...")
        endpoints = _parse_service_urls(self.args.answer_extractor_urls)
        print(f"   Eval endpoints: {len(endpoints)}")
        print(f"   Eval model: {self.args.answer_extractor_model or 'auto-detect'}")
        answer_extractor_config = build_answer_extractor_config(self.args)
        return AnswerExtractorClient(answer_extractor_config)

    def _process_samples(
        self,
        samples: List[BenchmarkSample],
        existing_predictions: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self.model_runner is None or self.answer_extractor is None or self.estimator is None:
            raise RuntimeError("Pipeline components are not initialized")

        print("\n⚙️  Starting inference and answer extraction...")
        writer = JsonlPredictionWriter(self.prediction_path)
        try:
            return evaluate_samples(
                samples=samples,
                existing_predictions=existing_predictions,
                writer=writer,
                model_runner=self.model_runner,
                answer_extractor=self.answer_extractor,
                estimator=self.estimator,
                num_workers=self.args.num_workers,
                print_every=self.args.print_every,
                max_retries=self.args.max_retries,
                retry_sleep=self.args.retry_sleep,
            )
        finally:
            writer.close()

    def _order_results(
        self,
        results: List[Dict[str, Any]],
        samples: List[BenchmarkSample],
    ) -> List[Dict[str, Any]]:
        results_by_id = {str(item.get("sample_id", "")): item for item in results if str(item.get("sample_id", ""))}
        return [results_by_id[sample.sample_id] for sample in samples if sample.sample_id in results_by_id]

    def _save_results(
        self,
        ordered_results: List[Dict[str, Any]],
        samples: List[BenchmarkSample],
    ) -> Tuple[str, str]:
        print("\n📊 Computing metrics...")
        metrics = summarize_results(ordered_results, samples)
        metrics_path = dump_metrics(metrics, self.args.output_dir, self.args.eval_output_path)
        summary_path = dump_summary(ordered_results, self.args.output_dir, self.args.eval_result_detail)

        print("\n📊 Evaluation Summary")
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
        return metrics_path, summary_path


def print_configuration(args: argparse.Namespace) -> None:
    print("\n📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print("   Loader: native vLLM")
    if args.model_type:
        print(f"   Model Type: {args.model_type} (ignored by native vLLM)")
    print(f"   Backend: {args.infer_backend}")
    print(f"   Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"   Max Model Len: {args.max_model_len or 'backend default'}")
    print(f"   DType: {args.dtype}")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output_dir}")
    print(f"   Evaluation Mode: {args.eval_mode}")
    print(f"   Eval Model: {args.answer_extractor_model or 'auto-detect'}")
    print(f"   Eval Endpoints: {args.answer_extractor_urls}")
    print(f"   Eval Threads: {args.num_workers}")
    print(f"   Eval Result Detail: {args.eval_result_detail}")
    if args.suffix:
        print(f"   Suffix: {args.suffix}")
    if args.gpu_groups:
        print(f"   GPU Groups: {args.gpu_groups}")
    if args.enable_data_parallel:
        print("   Data Parallel: flag received; pairwise runner uses one shared local engine")


def main() -> None:
    pipeline: Optional[PairwiseEvaluationPipeline] = None
    try:
        args = parse_args()
        print_configuration(args)
        pipeline = PairwiseEvaluationPipeline(args)
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\n❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

