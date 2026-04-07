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
from agent_system.environments.reward_utils import (
    DEFAULT_SUCCESS_REWARD,
    compute_binary_success_reward,
)

def compute_reward(info, multi_modal=False, success_reward: float = DEFAULT_SUCCESS_REWARD):
    return compute_binary_success_reward(info['won'], success_reward)

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path,
            split=None, variations_idx=None, is_expert=False, shared_port=None,
            jvm_lock=None, success_reward: float = DEFAULT_SUCCESS_REWARD):
    """
    SciWorld worker process.
    
    Expert trajectory format (obs_before, action):
    - obs: observation BEFORE taking the action (what expert saw)
    - action: action taken by expert
    - obs_after: observation AFTER taking the action (optional, for completeness)
    
    Disambiguation handling:
    - Filters out pure digit responses ("0", "1", etc.) used for disambiguation
    - These are internal SciWorld mechanics, not meaningful actions
    
    Args:
        shared_port: If provided, connect to existing JVM on this port (shared mode).
                     If None, launch a new JVM process (standalone mode).
    """
    from scienceworld import ScienceWorldEnv
    if shared_port is not None:
        # Shared JVM mode: connect to existing JVM, create independent PythonInterface
        import tempfile
        import types
        from py4j.java_gateway import JavaGateway, GatewayParameters
        env = ScienceWorldEnv.__new__(ScienceWorldEnv)
        env._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(auto_field=True, port=shared_port))
        env.server = env._gateway.jvm.scienceworld.runtime.pythonapi.PythonInterface()
        env.lastStepScore = 0
        env.taskName = None
        env.envStepLimit = env_step_limit
        env.goldPathGenerated = False
        env._obj_tree_tempdir = tempfile.TemporaryDirectory()
        env.runHistories = {}
        # Override close/del: disconnect only, don't kill the shared JVM
        def _shared_close(self):
            try:
                self._gateway.close()
            except Exception:
                pass
        env.close = types.MethodType(_shared_close, env)
        env.__del__ = types.MethodType(lambda self: None, env)
    else:
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
                reward = compute_reward(info, success_reward=success_reward)
                
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
                # Acquire JVM lock to prevent concurrent access to Scala singletons
                if jvm_lock:
                    jvm_lock.acquire()
                try:
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
                finally:
                    if jvm_lock:
                        jvm_lock.release()
                
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
        expert_in_group: bool = False,
        shared_jvm: bool = True,
        success_reward: float = DEFAULT_SUCCESS_REWARD
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
        self.success_reward = float(success_reward)
        random.seed(seed)
        self._rng = np.random.RandomState(seed)
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        
        self.policy_indices = []
        self.expert_indices = []
        
        # --- Shared JVM mode: share JVMs across groups to limit total JVM count ---
        self.shared_jvm = shared_jvm
        self.group_jvm_procs = []  # JVM Popen objects for cleanup
        self.group_ports = {}     # group_idx -> port
        
        ctx = mp.get_context('spawn')
        
        if shared_jvm:
            from py4j.java_gateway import launch_gateway
            from scienceworld.constants import BASEPATH, JAR_PATH
            import math
            _jar = jar_path or JAR_PATH
            # Cap JVM count: at most 1 JVM per workers_per_group groups
            # e.g. 128 groups with group_n=1 → 128/1=128 workers → ceil(128/8)=16 JVMs
            # e.g. 16 groups with group_n=8 → 16*9=144 workers → ceil(16/1)=16 JVMs
            max_workers_per_jvm = max(self.workers_per_group, 8)
            num_jvms = max(1, math.ceil(env_num / max(1, max_workers_per_jvm // self.workers_per_group)))
            num_jvms = min(num_jvms, env_num)  # never more JVMs than groups
            
            jvm_ports = []
            jvm_locks = []
            for j in range(num_jvms):
                port, proc = launch_gateway(
                    classpath=_jar, die_on_exit=True, cwd=BASEPATH,
                    javaopts=['-Xmx4G'], return_proc=True)
                jvm_ports.append(port)
                jvm_locks.append(ctx.Lock())
                self.group_jvm_procs.append(proc)
            
            # Map each group to a JVM (round-robin)
            self.group_locks = {}
            for g in range(env_num):
                jvm_idx = g % num_jvms
                self.group_ports[g] = jvm_ports[jvm_idx]
                self.group_locks[g] = jvm_locks[jvm_idx]
            print(f"[SciWorldEnvs] Shared JVM mode: {env_num} groups sharing {num_jvms} JVMs")
        
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
            
            # Determine shared port and lock for this worker's group
            shared_port = self.group_ports.get(group_idx) if shared_jvm else None
            jvm_lock = self.group_locks.get(group_idx) if shared_jvm else None
            
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx,
                      is_expert, shared_port, jvm_lock, self.success_reward),
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
        # If expert_in_group, pipeline sends POLICY-ONLY actions; expand for all workers
        if self.expert_in_group:
            if len(actions) != len(self.policy_indices):
                raise ValueError(
                    f'Expected {len(self.policy_indices)} policy actions, got {len(actions)}',
                )
            full_actions = [None] * self.num_processes
            for i, policy_idx in enumerate(self.policy_indices):
                full_actions[policy_idx] = actions[i]
            for expert_idx in self.expert_indices:
                full_actions[expert_idx] = "look around"  # Dummy action for expert
        else:
            if len(actions) != self.num_processes:
                raise ValueError(
                    f'Expected {self.num_processes} actions, got {len(actions)}',
                )
            full_actions = actions

        for remote, action in zip(self._parent_remotes, full_actions):
            remote.send(('step', action))

        all_obs, all_rewards, all_dones, all_infos = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            all_obs.append(obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_infos.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]

        # Filter: return only policy worker results
        if self.expert_in_group:
            obs_list = [all_obs[i] for i in self.policy_indices]
            reward_list = [all_rewards[i] for i in self.policy_indices]
            done_list = [all_dones[i] for i in self.policy_indices]
            info_list = [all_infos[i] for i in self.policy_indices]
        else:
            obs_list = all_obs
            reward_list = all_rewards
            done_list = all_dones
            info_list = all_infos

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        variations = [None for _ in range(self.num_processes)]
        for remote, variation in zip(self._parent_remotes, variations):
            remote.send(('reset', variation))

        all_obs, all_infos = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            all_obs.append(obs)
            all_infos.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]

        # Filter: return only policy worker results
        if self.expert_in_group:
            obs_list = [all_obs[i] for i in self.policy_indices]
            info_list = [all_infos[i] for i in self.policy_indices]
        else:
            obs_list = all_obs
            info_list = all_infos

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
        if self.expert_in_group:
            return [self.prev_available_actions[i] for i in self.policy_indices]
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        if self.expert_in_group:
            return [self.prev_available_actions[i] for i in self.policy_indices]
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        if self.expert_in_group:
            return [self.prev_possible_actions[i] for i in self.policy_indices]
        return self.prev_possible_actions

    def close(self):
        if getattr(self, '_closed', False):
            return
        for remote in self._parent_remotes:
            remote.send(('close', None))
        for worker in self._workers:
            worker.join()
        # Shared JVM mode: kill group JVM processes
        for proc in getattr(self, 'group_jvm_procs', []):
            if proc.poll() is None:
                try:
                    proc.stdin.write("\n".encode("utf-8"))
                    proc.stdin.flush()
                except Exception:
                    proc.terminate()
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
    expert_in_group: bool = False,
    shared_jvm: bool = True,
    success_reward: float = DEFAULT_SUCCESS_REWARD
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
        expert_in_group=expert_in_group,
        shared_jvm=shared_jvm,
        success_reward=success_reward
    ) 
