from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint cannot serve a request."""


@dataclass(frozen=True)
class APISettings:
    base_url: str
    api_key: str = field(default="EMPTY", repr=False)
    model: str = ""
    timeout_seconds: float = 120
    retries: int = 3
    retry_base_seconds: float = 2
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 512
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("OpenAI base URL is required")
        if not self.model:
            raise ValueError("OpenAI model name is required")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: dict[str, int]
    latency_seconds: float


class OpenAICompatibleClient:
    """Small dependency-free client for the OpenAI chat-completions protocol."""

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
        payload.update(self.settings.extra_body)
        started = time.monotonic()
        response = self._request("POST", "chat/completions", payload)
        latency = time.monotonic() - started

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError("Response does not contain choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise APIError("Response contains an empty choices[0].message.content")

        raw_usage = response.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
        }
        usage["total_tokens"] = int(
            raw_usage.get(
                "total_tokens",
                usage["prompt_tokens"] + usage["completion_tokens"],
            )
            or 0
        )
        return ChatResult(content=content, usage=usage, latency_seconds=latency)

    def list_models(self) -> list[str]:
        response = self._request("GET", "models")
        data = response.get("data", [])
        return [
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        ]

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        for attempt in range(self.settings.retries + 1):
            request = Request(url, data=data, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise APIError("API response must be a JSON object")
                return decoded
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429} or exc.code >= 500
                error = APIError(f"HTTP {exc.code} from {url}: {body[:500]}")
                if not retryable or attempt >= self.settings.retries:
                    raise error from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = APIError(f"Request to {url} failed: {exc}")
                if attempt >= self.settings.retries:
                    raise error from exc

            delay = min(
                self.settings.retry_base_seconds * (2**attempt),
                30,
            )
            time.sleep(delay + random.uniform(0, delay * 0.1))

        raise APIError(f"Request to {url} failed")
