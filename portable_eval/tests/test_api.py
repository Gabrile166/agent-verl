import json
import unittest
from unittest.mock import patch

from agent_eval.api import APISettings, OpenAICompatibleClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_chat_completion_request(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "choices": [{"message": {"content": "<action>look</action>"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            )

        settings = APISettings(
            base_url="http://model-host:8000/v1/",
            api_key="secret-token",
            model="served-model",
            timeout_seconds=9,
            retries=0,
        )
        client = OpenAICompatibleClient(settings)
        with patch("agent_eval.api.urlopen", side_effect=fake_urlopen):
            result = client.chat([{"role": "user", "content": "act"}])

        self.assertEqual(captured["url"], "http://model-host:8000/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(captured["body"]["model"], "served-model")
        self.assertEqual(captured["body"]["messages"][0]["content"], "act")
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(result.content, "<action>look</action>")
        self.assertEqual(result.usage["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
