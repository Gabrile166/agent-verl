"""Shared local model loading helpers for benchmark evaluation."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore[assignment]


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
        if self.infer_backend != "vllm":
            raise ValueError(f"Unsupported infer_backend: {self.infer_backend}")

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


def build_local_model_config(args: Any) -> Dict[str, Any]:
    return {
        "model_path": getattr(args, "model_path", ""),
        "infer_backend": getattr(args, "infer_backend", "vllm"),
        "system_prompt": getattr(args, "system_prompt", ""),
        "max_tokens": getattr(args, "max_tokens", 256),
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", 1.0),
        "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
        "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.9),
        "max_model_len": getattr(args, "max_model_len", 0) or None,
        "max_num_seqs": getattr(args, "max_num_seqs", 64),
        "dtype": getattr(args, "dtype", "auto"),
        "trust_remote_code": getattr(args, "trust_remote_code", True),
        "chat_template": getattr(args, "chat_template", ""),
    }
