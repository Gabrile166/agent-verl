import tempfile
import unittest
from pathlib import Path

from agent_eval.core import (
    EpisodeRecorder,
    build_agent_prompt,
    extract_action,
    summarize_episodes,
)


class ActionParsingTests(unittest.TestCase):
    def test_extracts_one_tagged_action(self):
        self.assertEqual(
            extract_action("<think>check the room</think><action>open fridge 1</action>"),
            "open fridge 1",
        )

    def test_rejects_missing_or_ambiguous_tags(self):
        self.assertIsNone(extract_action("open fridge 1"))
        self.assertIsNone(
            extract_action("<action>look</action><action>inventory</action>")
        )
        self.assertIsNone(extract_action("<action>   </action>"))


class PromptTests(unittest.TestCase):
    def test_history_is_bounded(self):
        history = [
            {"observation": f"obs-{index}", "action": f"act-{index}"}
            for index in range(12)
        ]
        prompt = build_agent_prompt(
            environment="ALFWorld",
            task="put the apple in the fridge",
            observation="current",
            available_actions=["look", "open fridge 1"],
            history=history,
            history_steps=10,
            step_number=13,
        )
        self.assertNotIn("observation='obs-0';", prompt)
        self.assertNotIn("observation='obs-1';", prompt)
        self.assertIn("observation='obs-2';", prompt)
        self.assertIn("observation='obs-11';", prompt)
        self.assertIn("<action>", prompt)


class ResultTests(unittest.TestCase):
    def test_scienceworld_success_is_strict(self):
        summary = summarize_episodes(
            [
                {
                    "environment": "sciworld_l1",
                    "success": False,
                    "score": 50,
                    "steps": 10,
                    "invalid_actions": 0,
                },
                {
                    "environment": "sciworld_l1",
                    "success": True,
                    "score": 100,
                    "steps": 20,
                    "invalid_actions": 1,
                },
            ]
        )
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_score"], 75.0)

    def test_recorder_resumes_by_episode_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(Path(tmp), resume=False)
            recorder.append(
                {
                    "episode_id": "alfworld:0",
                    "environment": "alfworld_ood",
                    "success": True,
                    "score": 1,
                    "steps": 4,
                    "invalid_actions": 0,
                }
            )

            resumed = EpisodeRecorder(Path(tmp), resume=True)
            self.assertTrue(resumed.contains("alfworld:0"))
            self.assertEqual(len(resumed.episodes), 1)


if __name__ == "__main__":
    unittest.main()
