import torch.multiprocessing as mp
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import sys
import os
import time
import random
from typing import Union
from itertools import product

def compute_reward(info, multi_modal=False):
    reward = 10.0 * float(info['won'])
    return reward

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None, is_expert=False):
    """
    SciWorld worker process.
    
    Expert trajectory format (obs_before, action):
    - obs: observation BEFORE taking the action (what expert saw)
    - action: action taken by expert
    - obs_after: observation AFTER taking the action (optional, for completeness)
    
    Disambiguation handling:
    - Filters out pure digit responses ("0", "1", etc.) used for disambiguation
    - These are internal SciWorld mechanics, not meaningful actions
    """
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    taskNames = env.get_task_names()
    random.seed(seed)
    
    task_id, task_variation = random.choice(variations_idx)
    
    # Expert state
    gold_actions = []
    expert_trajectory = []  # List of {obs, action, obs_after, step_index}
    current_obs = ""        # Track observation BEFORE action
    task_num = 0
    taskName = ""
    
    # Disambiguation patterns to filter
    DISAMBIGUATION_RESPONSES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    
    prev_score = 0
    step_index = 0  # Track step index for expert trajectory
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if is_expert:
                    if len(gold_actions) > 0:
                        action = gold_actions.pop(0)
                    else:
                        action = "look around"  # Fallback
                else:
                    action = data
                
                observation, reward, done, info = env.step(action)
                
                # Update expert trajectory with obs_before format
                if is_expert:
                    # Filter out disambiguation responses (pure digit actions)
                    if action not in DISAMBIGUATION_RESPONSES:
                        expert_trajectory.append({
                            "step_index": step_index,
                            "obs": current_obs,       # Observation BEFORE action
                            "action": action,
                            "obs_after": observation  # Observation AFTER action
                        })
                        step_index += 1
                    # If it's a disambiguation response, just update current_obs without recording
                    
                # Update current_obs for next step
                current_obs = observation
                
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['score'] = info.get('score', 0.0)
                info['task_score'] = info['score']
                isCompleted = done
                prev_score = info['score']
                info["won"] = isCompleted and info["score"] > 0
                reward = compute_reward(info)
                
                info['is_expert'] = is_expert
                if is_expert:
                    info['expert_trajectory'] = expert_trajectory
                    info['task_name'] = taskName
                    info['task_variation'] = task_variation
                
                remote.send((observation, reward, isCompleted, info))
                
            elif cmd == 'reset':
                if data is None:
                    task_id, task_variation = random.choice(variations_idx)
                    task_num = task_id
                    taskName = taskNames[task_num]
                else:
                    variation_idx = data
                
                simplification_str = simplifications_preset if simplifications_preset else ""
                
                # Load with gold path generation if expert
                env.load(taskName, task_variation, simplification_str, generateGoldPath=is_expert)
                
                if is_expert:
                    try:
                        gold_actions = env.get_gold_action_sequence()
                        expert_trajectory = []
                        step_index = 0
                    except Exception as e:
                        print(f"[SciWorld Expert] Error getting gold actions: {e}")
                        gold_actions = []
                        expert_trajectory = []
                        step_index = 0

                observation, info = env.reset()
                
                # Store initial observation for expert trajectory
                current_obs = observation
                
                task_description = env.get_task_description()
                info['task_description'] = task_description
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['won'] = False
                info['task_num'] = task_num
                prev_score = 0
                
                info['is_expert'] = is_expert
                if is_expert:
                    info['expert_trajectory'] = expert_trajectory
                    info['task_name'] = taskName
                    info['task_variation'] = task_variation
                    info['gold_path_length'] = len(gold_actions)

                remote.send((observation, info))
            
            elif cmd == 'get_expert_trajectory':
                if is_expert:
                    # Return trajectory with metadata
                    trajectory_with_metadata = {
                        "task_info": {
                            "task_name": taskName,
                            "task_num": task_num,
                            "variation": task_variation
                        },
                        "steps": expert_trajectory,
                        "total_steps": len(expert_trajectory)
                    }
                    remote.send(trajectory_with_metadata)
                else:
                    remote.send({"task_info": {}, "steps": [], "total_steps": 0})

            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")
    finally:
        env.close()


class SciWorldMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        task_nums: list = [1], 
        split: str = "train", 
        simplifications_preset: str = "", 
        env_step_limit: int = 100,
        jar_path: str = None,
        variations_idx: list = None,
        expert_in_group: bool = False
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.expert_in_group = expert_in_group
        
        # Consistent with AlfWorld: if expert_in_group, we have (group_n + 1) workers per group
        if expert_in_group:
            self.workers_per_group = group_n + 1
        else:
            self.workers_per_group = group_n
            
        self.env_num = env_num
        self.num_processes = env_num * self.workers_per_group
        
        self.split = split
        self.task_nums = task_nums
        self.variations_idx = variations_idx
        self.simplifications_preset = simplifications_preset
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path
        random.seed(seed)
        self._rng = np.random.RandomState(seed)
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        
        self.policy_indices = []
        self.expert_indices = []
        
        ctx = mp.get_context('spawn')
        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            
            # Determine group and position
            group_idx = i // self.workers_per_group
            position_in_group = i % self.workers_per_group
            
            # Last worker in group is expert (if enabled)
            is_expert = expert_in_group and (position_in_group == group_n)
            
            if is_expert:
                self.expert_indices.append(i)
            else:
                self.policy_indices.append(i)

            # Important: Expert and Policy in same group must share base seed to pick same task
            seed_i = seed + group_idx
            
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx, is_expert),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()
            
        self.prev_available_actions = [[] for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]

        if expert_in_group:
            print(f"[SciWorldEnvs] Expert Workers enabled: {len(self.expert_indices)} experts, {len(self.policy_indices)} policy workers.")

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        variations = [None for _ in range(self.num_processes)]
        for remote, variation in zip(self._parent_remotes, variations):
            remote.send(('reset', variation))
        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, info_list

    def get_expert_trajectories(self):
        """
        Collect expert trajectories from Expert Workers.
        
        Returns: Dict[int, Dict] - Group Index -> Trajectory with metadata
            {
                "task_info": {"task_name": ..., "task_num": ..., "variation": ...},
                "steps": [...],  # List of {step_index, obs, action, obs_after}
                "total_steps": int
            }
        """
        if not self.expert_in_group:
            return {}
        
        expert_trajs = {}
        # Request trajectories
        for expert_idx in self.expert_indices:
            self._parent_remotes[expert_idx].send(('get_expert_trajectory', None))
            
        # Collect trajectories  
        for expert_idx in self.expert_indices:
            traj_with_metadata = self._parent_remotes[expert_idx].recv()
            # Map back to group index
            group_idx = expert_idx // self.workers_per_group
            expert_trajs[group_idx] = traj_with_metadata
            
        return expert_trajs

    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        return self.prev_possible_actions

    def close(self):
        if getattr(self, '_closed', False):
            return
        for remote in self._parent_remotes:
            remote.send(('close', None))
        for worker in self._workers:
            worker.join()
        self._closed = True

    def __del__(self):
        self.close()

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    task_nums: Union[int, list] = 1, 
    split: str = "train", 
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str = None,
    variations_idx: list = None,
    expert_in_group: bool = False
):
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        task_nums=task_nums,
        split=split,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        variations_idx=variations_idx,
        expert_in_group=expert_in_group
    ) 
