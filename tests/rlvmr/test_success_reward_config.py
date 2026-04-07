import sys
import types

from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs
from agent_system.environments.reward_utils import (
    DEFAULT_SUCCESS_REWARD,
    compute_binary_success_reward,
)


class _FakeManager:
    def __init__(self, envs, projection_f, *args):
        self.envs = envs
        self.projection_f = projection_f
        self.args = args


def _build_base_config(env_name: str, success_reward: float = DEFAULT_SUCCESS_REWARD):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 2, "val_batch_size": 1},
            "env": {
                "env_name": env_name,
                "seed": 7,
                "success_reward": success_reward,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
                "rollout": {"n": 1},
                "alfworld": {"eval_dataset": "eval_in_distribution"},
                "sciworld": {
                    "generalization_level": 0,
                    "simplifications_preset": "easy",
                    "env_step_limit": 50,
                    "jar_path": None,
                },
            },
            "algorithm": {"expert": {"enable": False}},
        }
    )


def test_compute_binary_success_reward_uses_default_and_override():
    assert compute_binary_success_reward(True) == DEFAULT_SUCCESS_REWARD
    assert compute_binary_success_reward(True, success_reward=1.5) == 1.5
    assert compute_binary_success_reward(False, success_reward=99) == 0.0


def test_make_envs_passes_success_reward_to_alfworld(monkeypatch):
    calls = []

    fake_alfworld = types.ModuleType("agent_system.environments.env_package.alfworld")

    def fake_build(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"kind": "alfworld-env", "kwargs": kwargs}

    fake_alfworld.build_alfworld_envs = fake_build
    fake_alfworld.alfworld_projection = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "agent_system.environments.env_package.alfworld", fake_alfworld)
    monkeypatch.setattr("agent_system.environments.env_manager.AlfWorldEnvironmentManager", _FakeManager)

    config = _build_base_config("alfworld/AlfredTWEnv", success_reward=1.0)
    envs, val_envs = make_envs(config)

    assert len(calls) == 2
    assert calls[0]["kwargs"]["success_reward"] == 1.0
    assert calls[1]["kwargs"]["success_reward"] == 1.0
    assert envs.envs["kwargs"]["success_reward"] == 1.0
    assert val_envs.envs["kwargs"]["success_reward"] == 1.0


def test_make_envs_passes_success_reward_to_sciworld(monkeypatch):
    calls = []

    fake_sciworld = types.ModuleType("agent_system.environments.env_package.sciworld")

    def fake_build(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"kind": "sciworld-env", "kwargs": kwargs}

    fake_sciworld.build_sciworld_envs = fake_build
    fake_sciworld.sciworld_projection = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "agent_system.environments.env_package.sciworld", fake_sciworld)
    monkeypatch.setattr("agent_system.environments.env_manager.SciWorldEnvironmentManager", _FakeManager)

    fake_variation_data = '{"train": [[0, 0]], "test": [[1, 1]]}'

    def fake_open(*args, **kwargs):
        from io import StringIO

        return StringIO(fake_variation_data)

    monkeypatch.setattr("builtins.open", fake_open)

    config = _build_base_config("sciworld", success_reward=2.5)
    envs, val_envs = make_envs(config)

    assert len(calls) == 2
    assert calls[0]["kwargs"]["success_reward"] == 2.5
    assert calls[1]["kwargs"]["success_reward"] == 2.5
    assert envs.envs["kwargs"]["success_reward"] == 2.5
    assert val_envs.envs["kwargs"]["success_reward"] == 2.5
