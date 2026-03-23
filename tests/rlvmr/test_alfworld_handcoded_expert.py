import importlib
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALFWORLD_ROOT = PROJECT_ROOT / "agent_system" / "environments" / "env_package" / "alfworld"


def load_handcoded_expert_module():
    if str(ALFWORLD_ROOT) not in sys.path:
        sys.path.insert(0, str(ALFWORLD_ROOT))

    if "textworld" not in sys.modules:
        textworld_stub = types.ModuleType("textworld")

        class Agent:
            pass

        textworld_stub.Agent = Agent
        sys.modules["textworld"] = textworld_stub

    importlib.invalidate_caches()
    return importlib.import_module("alfworld.agents.expert.handcoded_expert")


def test_get_objects_and_classes_splits_and_connected_objects():
    handcoded_expert = load_handcoded_expert_module()
    policy = handcoded_expert.BasePolicy(task_params={})

    obs = "On the sidetable 1, you see a alarmclock 1 and a desklamp 1. Your task is to: turn on the lamp."

    assert policy.get_objects_and_classes(obs) == {
        "alarmclock 1": "alarmclock",
        "desklamp 1": "desklamp",
    }


def test_use_action_falls_back_to_admissible_command_object_ids():
    handcoded_expert = load_handcoded_expert_module()
    policy = handcoded_expert.BasePolicy(task_params={})
    policy.subgoals = [{'action': 'use', 'param': 'desklamp'}]

    game_state = {
        'feedback': "On the sidetable 1, you see a alarmclock 1.",
        'admissible_commands': ["look", "use desklamp 1"],
        'facts': [],
    }

    assert policy.act(game_state, "look") == "use desklamp 1"


def test_use_action_reobserves_instead_of_crashing_on_empty_candidates():
    handcoded_expert = load_handcoded_expert_module()
    policy = handcoded_expert.BasePolicy(task_params={})
    policy.subgoals = [{'action': 'use', 'param': 'desklamp'}]

    game_state = {
        'feedback': "On the sidetable 1, you see a alarmclock 1.",
        'admissible_commands': ["look"],
        'facts': [],
    }

    assert policy.act(game_state, "go to sidetable 1") == "examine sidetable 1"
