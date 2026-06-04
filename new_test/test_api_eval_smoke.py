"""Smoke tests for the API eval scaffold.

These tests avoid real environments and real API endpoints. They verify the
fragile DataProto shapes that TrajectoryCollector expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "new_test") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "new_test"))

from api_actor import APIActorWrapper  # noqa: E402
from eval_runner import build_dummy_gen_batch  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [10 + i for i, _ in enumerate(text.split())]

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(i) for i in ids if not skip_special_tokens or i not in {0, 2})


class FakeAPIActor(APIActorWrapper):
    def _call_chat_completion(self, messages):
        return "take apple"


def test_generate_sequences_shapes_and_masks():
    tokenizer = FakeTokenizer()
    actor = FakeAPIActor(
        tokenizer=tokenizer,
        api_base_urls=["http://unused/v1"],
        model_name="fake",
        max_prompt_length=8,
        max_response_length=5,
    )
    prompts = build_dummy_gen_batch(
        3,
        tokenizer,
        OmegaConf.create(
            {
                "data": {"max_prompt_length": 8},
                "env": {"env_name": "alfworld/AlfredTWEnv"},
            }
        ),
    )
    prompts.batch["input_ids"][:, -2:] = torch.tensor([7, 8])
    prompts.batch["attention_mask"][:, -2:] = 1
    prompts.batch["position_ids"][:, -2:] = torch.tensor([0, 1])

    output = actor.generate_sequences(prompts)

    assert output.batch["prompts"].shape == (3, 8)
    assert output.batch["responses"].shape == (3, 5)
    assert output.batch["input_ids"].shape == (3, 13)
    assert output.batch["attention_mask"].shape == (3, 13)
    assert output.batch["position_ids"].shape == (3, 13)
    assert output.batch["responses"][:, 2].tolist() == [tokenizer.eos_token_id] * 3
    assert output.batch["attention_mask"][:, 8:].tolist() == [[1, 1, 1, 0, 0]] * 3
    assert output.batch["position_ids"][:, 8:].tolist() == [[2, 3, 4, 5, 6]] * 3


def test_dummy_gen_batch_required_fields():
    tokenizer = FakeTokenizer()
    config = OmegaConf.create(
        {
            "data": {"max_prompt_length": 4},
            "env": {"env_name": "sciworld/ScienceWorldEnv"},
        }
    )
    batch = build_dummy_gen_batch(2, tokenizer, config)

    assert batch.batch["input_ids"].shape == (2, 4)
    assert set(batch.non_tensor_batch) == {"raw_prompt", "data_source", "raw_prompt_ids"}
    assert isinstance(batch.non_tensor_batch["raw_prompt"], np.ndarray)
    assert batch.non_tensor_batch["data_source"].tolist() == ["sciworld/ScienceWorldEnv"] * 2
    assert batch.meta_info["eos_token_id"] == tokenizer.eos_token_id
