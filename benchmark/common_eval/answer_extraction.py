"""Generic answer extraction helpers for benchmark evaluation."""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


_GENERIC_SYSTEM_PROMPT = (
    "You extract the final decision from another model's reasoning for a benchmark task. "
    "Return only one label from the allowed labels, or INVALID if the answer cannot be determined. "
    "Do not output anything else."
)


class GenericAnswerExtractorClient:
    """Answer extractor that supports dynamic label sets and regex fallback."""

    def __init__(
        self,
        base_urls: Sequence[str],
        model_name: str,
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 16,
        timeout: float = 60.0,
    ):
        self.urls = self._normalize_urls(base_urls)
        self.model_override = str(model_name or "").strip()
        self.api_key = str(api_key or "")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self._cursor = 0
        self._cursor_lock = threading.Lock()
        self._model_cache: Dict[str, str] = {}
        self._model_cache_lock = threading.Lock()
        self._session_local = threading.local()

    def extract_answer(self, raw_response: str, allowed_labels: List[str], task_name: str, question: str = "") -> str:
        return self.extract_answer_with_metadata(raw_response, allowed_labels, task_name, question=question)["label"]

    def extract_answer_with_metadata(
        self,
        raw_response: str,
        allowed_labels: List[str],
        task_name: str,
        question: str = "",
    ) -> Dict[str, Any]:
        labels = [str(label).strip() for label in allowed_labels if str(label).strip()]
        if not raw_response or not raw_response.strip():
            return {
                "label": "INVALID",
                "extractor_output": "",
                "used_fallback": True,
                "error": None,
            }
        if not labels:
            raise ValueError("allowed_labels must not be empty")

        extractor_output = ""
        extractor_error = None
        if self.urls:
            try:
                extractor_output = self._extract_via_service(raw_response, labels, task_name, question=question)
                normalized = self._normalize_candidate(extractor_output, labels)
                if normalized != "INVALID":
                    return {
                        "label": normalized,
                        "extractor_output": extractor_output,
                        "used_fallback": False,
                        "error": None,
                    }
            except Exception as exc:  # pragma: no cover - network/service failure path
                extractor_error = repr(exc)

        fallback = self._fallback_extract(raw_response, labels, task_name)
        return {
            "label": fallback,
            "extractor_output": extractor_output or f"regex_fallback:{fallback}",
            "used_fallback": True,
            "error": extractor_error,
        }

    def _extract_via_service(self, raw_response: str, allowed_labels: List[str], task_name: str, question: str) -> str:
        last_error: Optional[Exception] = None
        messages = self._build_messages(raw_response, allowed_labels, task_name, question=question)
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
                return str(message.get("content") or "")
            except Exception as exc:  # pragma: no cover - network/service failure path
                last_error = exc
                continue

        if last_error is None:
            raise RuntimeError("No answer extractor URL is available")
        raise RuntimeError(repr(last_error))

    def _ordered_urls(self) -> List[str]:
        with self._cursor_lock:
            start_idx = self._cursor % len(self.urls)
            self._cursor += 1
        return self.urls[start_idx:] + self.urls[:start_idx]

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            setattr(self._session_local, "session", session)
        return session

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

    def _build_messages(
        self,
        raw_response: str,
        allowed_labels: List[str],
        task_name: str,
        question: str,
    ) -> List[Dict[str, str]]:
        labels_text = ", ".join(allowed_labels)
        cleaned_response = self._strip_think_content(raw_response) or raw_response
        user_prompt = (
            f"Task: {task_name}\n"
            f"Allowed labels: {labels_text}\n"
            "Choose exactly one label from the allowed labels above. "
            "If the model response does not determine a final answer, return INVALID.\n"
        )
        if question.strip():
            user_prompt += f"Question: {question.strip()}\n"
        user_prompt += f"Model response:\n{cleaned_response}\n\nReturn only one token or phrase from the allowed labels, or INVALID."
        return [
            {"role": "system", "content": _GENERIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _normalize_urls(base_urls: Sequence[str]) -> List[str]:
        urls: List[str] = []
        for part in base_urls:
            url = str(part or "").strip()
            if not url:
                continue
            if not url.startswith(("http://", "https://")):
                url = f"http://{url}"
            urls.append(url.rstrip("/"))
        return urls

    @staticmethod
    def _strip_think_content(text: str) -> str:
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        return cleaned.strip()

    def _fallback_extract(self, raw_response: str, allowed_labels: List[str], task_name: str) -> str:
        cleaned = self._strip_think_content(raw_response) or str(raw_response or "")
        if not cleaned.strip():
            return "INVALID"

        explicit_patterns = [
            r"answer\s*[:=]\s*([^\n\r]+)",
            r"final answer(?:\s+is|\s*[:=])\s*([^\n\r.]+)",
            r"i choose\s+([^\n\r.]+)",
            r"i pick\s+([^\n\r.]+)",
            r"milestone\s+([A-Za-z0-9_\-]+)",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if not match:
                continue
            normalized = self._normalize_candidate(match.group(1), allowed_labels)
            if normalized != "INVALID":
                return normalized

        if self._is_milestone_task(allowed_labels, task_name):
            milestone_mentions = self._collect_label_mentions(cleaned, allowed_labels)
            if milestone_mentions:
                return milestone_mentions[-1]
            return "INVALID"

        mentions = self._collect_label_mentions(cleaned, allowed_labels)
        unique_mentions = []
        for label in mentions:
            if label not in unique_mentions:
                unique_mentions.append(label)
        if len(unique_mentions) == 1:
            return unique_mentions[0]
        return "INVALID"

    def _collect_label_mentions(self, text: str, allowed_labels: List[str]) -> List[str]:
        mentions: List[Tuple[int, str]] = []
        for alias, canonical in self._alias_pairs(allowed_labels):
            pattern = self._alias_pattern(alias)
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                mentions.append((match.start(), canonical))
        mentions.sort(key=lambda item: item[0])
        return [label for _, label in mentions]

    def _normalize_candidate(self, value: Any, allowed_labels: List[str]) -> str:
        raw_text = str(value or "").strip()
        if not raw_text:
            return "INVALID"
        upper = raw_text.upper()
        if upper == "INVALID":
            return "INVALID"

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "label", "choice", "final_answer", "final_label"):
                if key in payload:
                    return self._normalize_candidate(payload[key], allowed_labels)

        direct = self._match_exact_label(raw_text, allowed_labels)
        if direct is not None:
            return direct

        mentions = self._collect_label_mentions(raw_text, allowed_labels)
        unique_mentions = []
        for label in mentions:
            if label not in unique_mentions:
                unique_mentions.append(label)
        if len(unique_mentions) == 1:
            return unique_mentions[0]
        return "INVALID"

    @staticmethod
    def _is_milestone_task(allowed_labels: List[str], task_name: str) -> bool:
        if "milestone" in str(task_name or "").lower():
            return True
        return all(re.fullmatch(r"[A-Za-z0-9_\-]+", label) and len(label) <= 4 for label in allowed_labels)

    def _match_exact_label(self, text: str, allowed_labels: List[str]) -> Optional[str]:
        normalized = str(text or "").strip()
        for label in allowed_labels:
            if normalized.lower() == label.lower():
                return label
        if self._is_progress_labels(allowed_labels):
            alias_map = {alias.lower(): canonical for alias, canonical in self._alias_pairs(allowed_labels)}
            return alias_map.get(normalized.lower())
        return None

    @staticmethod
    def _is_progress_labels(allowed_labels: Sequence[str]) -> bool:
        return {label.lower() for label in allowed_labels} == {"increase", "decrease", "same"}

    def _alias_pairs(self, allowed_labels: Sequence[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for label in allowed_labels:
            canonical = str(label).strip()
            if not canonical:
                continue
            pairs.append((canonical, canonical))
            if self._is_progress_labels(allowed_labels):
                lower = canonical.lower()
                if lower == "increase":
                    pairs.extend([
                        ("increased", canonical),
                        ("increasing", canonical),
                        ("progress increased", canonical),
                        ("moved closer", canonical),
                    ])
                elif lower == "decrease":
                    pairs.extend([
                        ("decreased", canonical),
                        ("decreasing", canonical),
                        ("progress decreased", canonical),
                        ("moved farther", canonical),
                    ])
                elif lower == "same":
                    pairs.extend([
                        ("unchanged", canonical),
                        ("no change", canonical),
                        ("stay the same", canonical),
                        ("stayed the same", canonical),
                        ("remained the same", canonical),
                    ])
        deduped: List[Tuple[str, str]] = []
        seen = set()
        for alias, canonical in pairs:
            key = (alias.lower(), canonical)
            if key in seen:
                continue
            deduped.append((alias, canonical))
            seen.add(key)
        return deduped

    @staticmethod
    def _alias_pattern(alias: str) -> str:
        escaped = re.escape(alias)
        if re.fullmatch(r"[A-Za-z0-9_\-]+", alias):
            return rf"\b{escaped}\b"
        return escaped
