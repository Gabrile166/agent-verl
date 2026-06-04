"""OpenAI-compatible API rollout wrapper for eval-only agent runs."""

from __future__ import annotations

import itertools
import time
from typing import Any, Iterable

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


class APIActorWrapper:
    """Small duck-typed replacement for the rollout worker group.

    TrajectoryCollector only requires ``world_size`` and ``generate_sequences``.
    This wrapper calls an OpenAI-compatible chat completion endpoint and then
    reconstructs the same tensor fields returned by the vLLM rollout path.
    """

    def __init__(
        self,
        tokenizer,
        api_base_urls: list[str],
        model_name: str,
        api_key: str = "EMPTY",
        max_prompt_length: int = 4096,
        max_response_length: int = 2048,
        temperature: float = 0.4,
        top_p: float = 1.0,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool | None = None,
        max_tokens: int = 2048,
        timeout: int = 120,
        retries: int = 3,
    ) -> None:
        if not api_base_urls:
            raise ValueError("api_base_urls must contain at least one endpoint")
        self.tokenizer = tokenizer
        self.api_base_urls = [url.rstrip("/") for url in api_base_urls]
        self.model_name = model_name
        self.api_key = api_key
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.presence_penalty = None if presence_penalty is None else float(presence_penalty)
        self.top_k = None if top_k is None else int(top_k)
        self.min_p = None if min_p is None else float(min_p)
        self.repetition_penalty = None if repetition_penalty is None else float(repetition_penalty)
        self.enable_thinking = enable_thinking
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self._url_cycle = itertools.cycle(self.api_base_urls)

    @property
    def world_size(self) -> int:
        return 1

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        pad_token_id = int(prompts.meta_info.get("pad_token_id", self.tokenizer.pad_token_id))
        eos_token = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        eos_token_id = _first_token_id(eos_token, fallback=pad_token_id)

        response_ids: list[list[int]] = []
        for item in range(batch_size):
            messages = self._build_messages(prompts, item)
            text = self._call_with_retries(messages)
            ids = self._encode_response(text, eos_token_id)
            response_ids.append(ids)

        responses = pad_2d_list_to_length(
            response_ids,
            pad_token_id=pad_token_id,
            max_length=self.max_response_length,
        ).to(device=idx.device, dtype=idx.dtype)

        seq = torch.cat([idx, responses], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=responses,
            eos_token=eos_token,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([attention_mask, response_attention_mask.to(idx.device)], dim=-1)
        full_position_ids = self._append_response_position_ids(position_ids, responses.size(1))

        return DataProto.from_dict(
            tensors={
                "prompts": idx,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            non_tensors=dict(prompts.non_tensor_batch),
            meta_info=dict(prompts.meta_info),
        )

    def _build_messages(self, prompts: DataProto, item: int) -> list[dict[str, str]]:
        raw_prompt = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompt is not None:
            messages = _to_plain(raw_prompt[item])
            if isinstance(messages, list) and messages:
                return [_normalize_message(message) for message in messages]

        raw_prompt_ids = prompts.non_tensor_batch.get("raw_prompt_ids")
        if raw_prompt_ids is not None:
            ids = _to_plain(raw_prompt_ids[item])
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            if ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                return [{"role": "user", "content": text}]

        prompt_ids = prompts.batch["input_ids"][item]
        text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        return [{"role": "user", "content": text}]

    def _call_with_retries(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(max(self.retries, 1)):
            try:
                return self._call_chat_completion(messages)
            except Exception as exc:  # noqa: BLE001 - surface endpoint failures after retries.
                last_error = exc
                if attempt + 1 < max(self.retries, 1):
                    time.sleep(min(2**attempt, 8))
        print(f"[APIActorWrapper] API call failed after {self.retries} attempts: {last_error}")
        return ""

    def _call_chat_completion(self, messages: list[dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for API evaluation.") from exc

        base_url = next(self._url_cycle)
        client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=self.timeout)
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.presence_penalty

        extra_body: dict[str, Any] = {}
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty is not None:
            extra_body["repetition_penalty"] = self.repetition_penalty
        if self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if content:
            return content
        reasoning = getattr(response.choices[0].message, "reasoning", None)
        return reasoning or ""

    def _encode_response(self, text: str, eos_token_id: int) -> list[int]:
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        ids = [int(token_id) for token_id in ids]
        if self.max_response_length <= 1:
            return [eos_token_id]
        ids = ids[: self.max_response_length - 1]
        ids.append(eos_token_id)
        return ids or [eos_token_id]

    @staticmethod
    def _append_response_position_ids(position_ids: torch.Tensor, response_length: int) -> torch.Tensor:
        batch_size = position_ids.size(0)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size,
                position_ids.size(1),
                -1,
            )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        return torch.cat([position_ids, response_position_ids], dim=-1)


def _first_token_id(token: Any, fallback: int) -> int:
    if token is None:
        return int(fallback)
    if isinstance(token, torch.Tensor):
        token = token.tolist()
    if isinstance(token, np.ndarray):
        token = token.tolist()
    if isinstance(token, Iterable) and not isinstance(token, (str, bytes)):
        token_list = list(token)
        return int(token_list[0]) if token_list else int(fallback)
    return int(token)


def _to_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_message(message: Any) -> dict[str, str]:
    if isinstance(message, np.ndarray):
        message = message.item() if message.shape == () else message.tolist()
    if not isinstance(message, dict):
        return {"role": "user", "content": str(message)}
    return {
        "role": str(message.get("role", "user")),
        "content": str(message.get("content", "")),
    }
