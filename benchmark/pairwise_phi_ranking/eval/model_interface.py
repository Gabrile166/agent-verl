"""Model interface abstractions for pairwise benchmark evaluation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


class ModelInterface(ABC):
    """Unified model interface used by benchmark evaluation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelInterface":
        """Factory method that builds a concrete model interface from config."""
        model_type = str(config.get("type", "")).strip().lower()

        if model_type in {"openai_compatible", "openai-compatible", "openai"}:
            base_url = str(config.get("base_url", "")).strip()
            model = str(config.get("model", "")).strip()
            api_key = str(config.get("api_key", "EMPTY"))
            timeout_raw = config.get("timeout")
            max_tokens_raw = config.get("max_tokens")
            system_prompt = str(config.get("system_prompt", "")).strip()
            timeout = float(timeout_raw) if timeout_raw not in {None, ""} else None
            max_tokens = int(max_tokens_raw) if max_tokens_raw not in {None, ""} else None
            return OpenAICompatibleModel(
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

        if model_type in {"manual", "manual_input", "manual-input"}:
            return ManualInputModel()

        raise ValueError(f"Unsupported model type: {model_type}")


class OpenAICompatibleModel(ModelInterface):
    """OpenAI-compatible chat completion wrapper with deterministic decoding."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: str = "",
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAICompatibleModel")
        if not model:
            raise ValueError("model must not be empty")

        self.client = OpenAI(base_url=base_url or None, api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if content is None:
            return ""
        return str(content)


class ManualInputModel(ModelInterface):
    """Manual stdin-backed model for debugging local benchmark logic."""

    def generate(self, prompt: str) -> str:
        print("\n========== Pairwise Benchmark Prompt ==========")
        print(prompt)
        print("==============================================\n")

        while True:
            value = input("Manual answer (A/B): ").strip().upper()
            if value in {"A", "B"}:
                return f"Answer: {value}"
            print("Invalid input. Please enter A or B.")
