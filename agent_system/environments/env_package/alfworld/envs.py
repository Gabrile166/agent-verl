# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, is_expert=False):
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)
        self.is_expert = is_expert
        
        # Expert Worker 专用状态
        self.current_expert_plan = []
        self.expert_trajectory = []
    
    def step(self, action):
        """Execute a step in the environment"""
        if self.is_expert:
            # Expert 模式：使用 expert plan 中的下一个动作
            # 优先使用 infos 中更新的 plan，如果没有则使用缓存的
            if self.current_expert_plan:
                action = self.current_expert_plan[0]
            else:
                action = "look"  # fallback
        
        if isinstance(action, list):
            action = action[0]
        action = str(action)
        
        obs, scores, dones, infos = self.env.step([action])
        infos['observation_text'] = obs
        infos['is_expert'] = self.is_expert
        
        if self.is_expert:
            # 记录 Expert 执行的动作
            infos['expert_executed_action'] = action
            
            # 更新 expert plan (必需解包 batch dimension)
            # 注意：base_env.step 返回的 infos 中包含 'extra.expert_plan'
            batch_plan = infos.get('extra.expert_plan', [])
            self.current_expert_plan = batch_plan[0] if batch_plan else []
            
            # 累积轨迹
            current_obs = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
            self.expert_trajectory.append({
                "observation": current_obs,
                "action": action
            })
            infos['expert_trajectory'] = self.expert_trajectory
        
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        infos['is_expert'] = self.is_expert
        
        if self.is_expert:
            self.expert_trajectory = []
            batch_plan = infos.get('extra.expert_plan', [])
            self.current_expert_plan = batch_plan[0] if batch_plan else []
            infos['expert_trajectory'] = self.expert_trajectory
        
        return obs, infos
    
    def get_expert_trajectory(self):
        """Return the accumulated expert trajectory"""
        if self.is_expert:
            return self.expert_trajectory.copy()
        return []
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, 
                 is_train=True, env_kwargs={}, expert_in_group=False):
        """
        Initialize Alfworld environments with optional Expert Worker support.
        """
        super().__init__()
        
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        
        # N+1 Expert Worker 架构
        self.policy_per_group = group_n
        self.group_n = group_n
        self.expert_in_group = expert_in_group
        self.env_num = env_num
        
        if expert_in_group:
            self.workers_per_group = group_n + 1
            self.num_processes = env_num * self.workers_per_group
        else:
            self.workers_per_group = group_n
            self.num_processes = env_num * group_n

        self.expert_indices = []
        self.policy_indices = []

        # Create Ray remote actors
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        
        for i in range(self.num_processes):
            group_idx = i // self.workers_per_group
            position_in_group = i % self.workers_per_group
            
            # Expert 是每组的最后一个
            is_expert = expert_in_group and (position_in_group == self.policy_per_group)
            
            if is_expert:
                self.expert_indices.append(i)
            else:
                self.policy_indices.append(i)
            
            # 同组所有 Worker（包括 Expert）使用相同的 seed
            worker = env_worker.remote(config, seed + group_idx, base_env, is_expert)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]
        
        if expert_in_group:
            print(f"[AlfworldEnvs] Expert Workers enabled:")
            print(f"[AlfworldEnvs]   Total processes: {self.num_processes}")
            print(f"[AlfworldEnvs]   Policy Workers: {len(self.policy_indices)}")

    def step(self, actions):
        """
        Step function.
        If expert_in_group is enabled, expects `actions` to correspond to POLICY workers only.
        It will automatically pad dummy actions for EXPERT workers.
        """
        real_actions = actions
        
        # 如果启用了 Expert 且输入动作数量等于 Policy 数量，则进行填充
        if self.expert_in_group and len(actions) == len(self.policy_indices):
            
            # 构建 full_actions
            full_actions = [None] * self.num_processes
            
            # 填充 Policy 动作
            for i, policy_idx in enumerate(self.policy_indices):
                full_actions[policy_idx] = actions[i]
                
            # 填充 Expert 动作 (Dummy, Worker 内部会忽略)
            for expert_idx in self.expert_indices:
                full_actions[expert_idx] = "look"
                
            real_actions = full_actions

        assert len(real_actions) == self.num_processes, \
            f"The num of actions ({len(real_actions)}) must be equal to the num of processes ({self.num_processes})"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(real_actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        
        # 结果过滤：只返回 Policy Workers 的结果
        # 但我们需要先处理所有结果，保留 prev_admissible_commands 等状态
        
        all_text_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []
        
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in list(info.keys()):
                if isinstance(info[k], (list, tuple)) and len(info[k]) > 0 and k != 'expert_trajectory':
                    info[k] = info[k][0]

            all_text_obs.append(obs[0])
            all_dones.append(dones[0])
            all_infos.append(info)
            self.prev_admissible_commands[i] = info['admissible_commands']
            all_rewards.append(compute_reward(info, self.multi_modal))

        # 过滤并返回 Policy 结果
        if self.expert_in_group:
            text_obs_list = [all_text_obs[i] for i in self.policy_indices]
            rewards_list = [all_rewards[i] for i in self.policy_indices]
            dones_list = [all_dones[i] for i in self.policy_indices]
            info_list = [all_infos[i] for i in self.policy_indices]
        else:
            text_obs_list = all_text_obs
            rewards_list = all_rewards
            dones_list = all_dones
            info_list = all_infos

        if self.multi_modal:
            # TODO: handle multi-modal expert masking if needed
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Reset function. 
        Resets all workers, but filters output to return only POLICY worker observations.
        """
        text_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        all_text_obs = []
        all_infos = []
        
        for i, (obs, info) in enumerate(results):
            for k in list(info.keys()):
                if isinstance(info[k], (list, tuple)) and len(info[k]) > 0 and k != 'expert_trajectory':
                    info[k] = info[k][0]
            
            all_text_obs.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            all_infos.append(info)

        # 过滤并返回 Policy 结果
        if self.expert_in_group:
            text_obs_list = [all_text_obs[i] for i in self.policy_indices]
            info_list = [all_infos[i] for i in self.policy_indices]
        else:
            text_obs_list = all_text_obs
            info_list = all_infos

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        results = ray.get(futures)
        
        if self.expert_in_group:
            return [results[i] for i in self.policy_indices]
        return results
    
    def get_expert_trajectories(self):
        """
        Collect expert trajectories from Expert Workers.
        
        Returns:
            Dict[int, List[Dict]]: Mapping from group_idx to expert trajectory.
            Each trajectory is a list of {"observation": str, "action": str}.
        """
        if not self.expert_in_group:
            return {}
        
        expert_trajs = {}
        
        # 向每个 Expert Worker 请求轨迹
        futures = []
        for expert_idx in self.expert_indices:
            future = self.workers[expert_idx].get_expert_trajectory.remote()
            futures.append((expert_idx, future))
        
        # 收集结果
        for expert_idx, future in futures:
            traj = ray.get(future)
            # 计算组索引
            group_idx = expert_idx // self.workers_per_group
            expert_trajs[group_idx] = traj
        
        return expert_trajs

    @property
    def get_admissible_commands(self):
        if self.expert_in_group:
            return [self.prev_admissible_commands[i] for i in self.policy_indices]
        return self.prev_admissible_commands

    def close(self):
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, 
                        is_train=True, env_kwargs={}, expert_in_group=False):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, 
                        is_train, env_kwargs, expert_in_group)