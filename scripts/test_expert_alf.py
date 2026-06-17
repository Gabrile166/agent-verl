#!/usr/bin/env python3
"""
CPU-only smoke test for AlfWorld Expert trajectories.

This script does NOT run training. It simply:
1) Builds AlfWorld envs with Expert Workers enabled
2) Resets envs
3) Steps a few times with admissible actions
4) Prints expert trajectory lengths to verify they are non-empty
"""

import argparse
import os
import sys
import time

# Ensure repo root is on sys.path so `agent_system` is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3, help="Number of env steps to run")
    parser.add_argument("--env-num", type=int, default=1, help="Number of env groups")
    parser.add_argument("--group-n", type=int, default=1, help="Policy workers per group")
    parser.add_argument("--eval-dataset", type=str, default="eval_in_distribution")
    return parser.parse_args()


def main():
    args = parse_args()

    # Local import to avoid heavy deps if script isn't used
    from agent_system.environments.env_package.alfworld import build_alfworld_envs

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "agent_system",
        "environments",
        "env_package",
        "alfworld",
        "configs",
        "config_tw.yaml",
    )

    env_kwargs = {"eval_dataset": args.eval_dataset}

    print("[CPU TEST] Building AlfWorld envs with Expert Worker enabled...")
    # Ray requires at least one resource option when using ray.remote(**resources_per_worker)
    resources_per_worker = {"num_cpus": 1}

    envs = build_alfworld_envs(
        alf_config_path=alf_config_path,
        seed=0,
        env_num=args.env_num,
        group_n=args.group_n,
        resources_per_worker=resources_per_worker,
        is_train=True,
        env_kwargs=env_kwargs,
        expert_in_group=True,
    )

    print("[CPU TEST] Resetting envs...")
    text_obs, image_obs, infos = envs.reset()
    print(f"[CPU TEST] Reset done. policy_obs={len(text_obs)}")

    # Step with admissible actions for policy workers only
    for step_idx in range(args.steps):
        admissible = envs.get_admissible_commands
        if not admissible:
            print("[CPU TEST] No admissible commands found.")
            break

        actions = []
        for cmds in admissible:
            if cmds and len(cmds) > 0:
                actions.append(cmds[0])
            else:
                actions.append("look")

        text_obs, image_obs, rewards, dones, infos = envs.step(actions)
        print(f"[CPU TEST] Step {step_idx + 1}/{args.steps} done.")
        time.sleep(0.1)

    print("[CPU TEST] Fetching expert trajectories...")
    expert_trajs = envs.get_expert_trajectories()
    if not expert_trajs:
        print("[CPU TEST] expert_trajs is empty. Expert worker likely not enabled or not producing trajectories.")
    else:
        for group_idx, traj in expert_trajs.items():
            print(f"[CPU TEST] group={group_idx} expert_steps={len(traj)}")

    envs.close()
    print("[CPU TEST] Done.")


if __name__ == "__main__":
    sys.exit(main())
