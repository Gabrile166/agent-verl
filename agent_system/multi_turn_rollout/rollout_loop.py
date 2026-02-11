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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None, trajectory_saver=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
            trajectory_saver: Optional TrajectorySaver instance for saving rollout data
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.trajectory_saver = trajectory_saver
        self._batch_count = 0

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            # 保存完整输出（用于 format_policy_trajectory 提取 action）
            batch.non_tensor_batch['full_output'] = text_actions

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        # ==================== Expert Trajectory Collection ====================
        expert_trajectories = []
        if hasattr(envs, 'get_expert_trajectories'):
            expert_trajectories = envs.get_expert_trajectories()
        
        # 获取任务描述
        tasks = []
        if hasattr(envs, 'tasks'):
            tasks = envs.tasks
        
        # 过滤 Expert Worker 数据（如果启用 N+1 架构）
        if hasattr(envs, 'is_expert_in_group_enabled') and envs.is_expert_in_group_enabled():
            policy_indices = envs.get_policy_indices()
            expert_indices = envs.get_expert_indices()
            
            # 关键判断：检查环境是否已经内部过滤了数据
            # agent-verl 的 AlfworldEnvs.step/reset 会返回 Policy-only 数据（已过滤）
            # Agent-PRM 的 AlfworldEnvs.step/reset 返回所有数据（未过滤）
            # 
            # 判断依据：如果 batch_size == policy 数量，说明已过滤；
            #          如果 batch_size == total 数量 (policy + expert)，说明未过滤
            #
            # policy_indices 的 LENGTH 等于 policy 数量
            # 但 policy_indices 的 VALUES 是原始索引 (0,1,2,...,7, 9,10,...)，最大值 > batch_size
            #
            # 因此：检查 max(policy_indices) < len(total_batch_list) 来判断是否需要过滤
            
            needs_filtering = len(total_batch_list) > len(policy_indices) and max(policy_indices) < len(total_batch_list)
            
            if not needs_filtering:
                print(f"[Rollout] Data already filtered by environment (batch_size={len(total_batch_list)}, "
                      f"policy_count={len(policy_indices)}). Skipping re-filtering.")
            else:
                print(f"[Rollout] Filtering Expert Workers: keeping {len(policy_indices)} policy envs, "
                      f"removing {len(expert_indices)} expert envs")
                
                # 只保留 Policy Workers 的轨迹
                total_batch_list = [total_batch_list[i] for i in policy_indices]
                episode_rewards = np.array([episode_rewards[i] for i in policy_indices])
                episode_lengths = np.array([episode_lengths[i] for i in policy_indices])
                traj_uid = np.array([traj_uid[i] for i in policy_indices])
                tool_callings = np.array([tool_callings[i] for i in policy_indices])
                
                # 过滤 success 字典
                if success:
                    for key in success:
                        if isinstance(success[key], np.ndarray):
                            success[key] = np.array([success[key][i] for i in policy_indices])
                
                # 过滤 tasks
                if tasks:
                    tasks = [tasks[i] for i in policy_indices if i < len(tasks)]
            
            # ==================== Expert Trajectory Remapping ====================
            # 无论是否 needs_filtering，都需要重新映射 expert_trajectories
            # 因为 expert_trajectories 是 {group_idx: trajectory} 格式
            # 需要转换为 List[List[Dict]] 与 policy 轨迹一一对应
            if expert_trajectories and isinstance(expert_trajectories, dict):
                workers_per_group = getattr(envs.envs, 'workers_per_group', 
                                           self.config.env.rollout.n + 1 if hasattr(self.config.env, 'rollout') else 2)
                remapped_expert_trajs = []
                for policy_idx in policy_indices:
                    group_idx = policy_idx // workers_per_group
                    traj = expert_trajectories.get(group_idx, [])
                    remapped_expert_trajs.append(traj)
                expert_trajectories = remapped_expert_trajs
                print(f"[Rollout] Remapped {len(remapped_expert_trajs)} expert trajectories to policy indices")
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, expert_trajectories, tasks
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_expert_trajectories = []
        total_tasks = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, expert_trajectories, tasks = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            total_expert_trajectories.extend(expert_trajectories[:len(batch_list)])
            total_tasks.extend(tasks[:len(batch_list)])

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_expert_trajectories, total_tasks

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, expert_trajectories, tasks = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, expert_trajectories, tasks = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        # Store expert trajectories in meta_info (for Hybrid Reward calculation)
        gen_batch_output.meta_info["expert_trajectories"] = expert_trajectories
        gen_batch_output.meta_info["tasks"] = tasks
        
        # ==================== Format Policy Trajectories ====================
        # 直接在这里格式化 Policy 轨迹，避免后续从扁平化 batch 重建时数据损坏
        # 这保证了 observation 和 action 的正确对应关系
        try:
            from rlvmr.discriminator_reward import format_policy_trajectory
            
            formatted_policy_trajectories = []
            for traj_idx, trajectory in enumerate(total_batch_list):
                task = tasks[traj_idx] if traj_idx < len(tasks) else ''
                formatted_traj = format_policy_trajectory(trajectory, task)
                formatted_policy_trajectories.append(formatted_traj)
            
            gen_batch_output.meta_info["formatted_policy_trajectories"] = formatted_policy_trajectories
            print(f"[Rollout] Formatted {len(formatted_policy_trajectories)} policy trajectories directly from trajectory_list")
        except Exception as e:
            print(f"[Rollout] Failed to format policy trajectories: {e}")
            gen_batch_output.meta_info["formatted_policy_trajectories"] = []
        
        # ==================== Build PipelineData (Two-Level Model) ====================
        # Consolidates query-level (task, expert) and trajectory-level (reward, success, policy)
        # data into a typed, FK-linked structure for Milestone GAE.
        try:
            from rlvmr.pipeline_data import PipelineData, QueryRecord, TrajectoryRecord
            
            n_trajs = len(total_traj_uid)
            formatted_policy_trajs = gen_batch_output.meta_info.get("formatted_policy_trajectories", [])
            
            # Step A: Source alignment assertions — crash with diagnostics, not silent misalignment
            assert n_trajs == len(tasks), (
                f"traj_uid({n_trajs}) vs tasks({len(tasks)}) mismatch"
            )
            assert n_trajs == len(expert_trajectories), (
                f"traj_uid({n_trajs}) vs expert_trajs({len(expert_trajectories)}) mismatch"
            )
            assert n_trajs == len(total_episode_rewards), (
                f"traj_uid({n_trajs}) vs rewards({len(total_episode_rewards)}) mismatch"
            )
            assert n_trajs == len(formatted_policy_trajs), (
                f"traj_uid({n_trajs}) vs policy_trajs({len(formatted_policy_trajs)}) mismatch"
            )
            
            # Step B: Extract traj_uid → uid mapping from per-step data
            # uid is per-step in non_tensor_batch; same traj_uid always maps to same uid.
            uids_per_step = gen_batch_output.non_tensor_batch.get('uid', [])
            traj_uids_per_step = gen_batch_output.non_tensor_batch.get('traj_uid', [])
            traj_uid_to_uid = {}
            for i in range(len(traj_uids_per_step)):
                t = str(traj_uids_per_step[i])
                if t not in traj_uid_to_uid:
                    traj_uid_to_uid[t] = str(uids_per_step[i])
            
            # Step C: Extract per-trajectory success from success_evaluator output
            # success_evaluator() returns Dict[str, np.ndarray] with per-trajectory values
            success_array = total_success.get("success_rate", np.zeros(n_trajs))
            
            # Step D: Build two-level PipelineData
            queries = {}
            trajectories = {}
            
            for traj_idx, t_uid in enumerate(total_traj_uid):
                t_uid_str = str(t_uid)
                uid_str = traj_uid_to_uid[t_uid_str]  # strict: KeyError if missing
                
                # Query-level: deduplicate (first-seen wins; all copies within a
                # rollout group are identical since they share the same env/task)
                if uid_str not in queries:
                    queries[uid_str] = QueryRecord(
                        uid=uid_str,
                        task=tasks[traj_idx],
                        expert_trajectory=expert_trajectories[traj_idx],
                    )
                
                # Trajectory-level: always unique
                trajectories[t_uid_str] = TrajectoryRecord(
                    traj_uid=t_uid_str,
                    uid=uid_str,
                    policy_trajectory=formatted_policy_trajs[traj_idx],
                    episode_reward=float(total_episode_rewards[traj_idx]),
                    success=bool(success_array[traj_idx]),
                )
            
            pipeline_data = PipelineData(queries=queries, trajectories=trajectories)
            gen_batch_output.meta_info["pipeline_data"] = pipeline_data
            print(f"[PipelineData] Built: {len(queries)} queries, {len(trajectories)} trajectories")
        except Exception as e:
            print(f"[PipelineData] Failed to build: {e}")
            import traceback
            traceback.print_exc()
        
        # Save trajectories to disk if saver is enabled
        if self.trajectory_saver is not None and is_train:
            try:
                rollout_n = self.config.env.rollout.n if hasattr(self.config.env.rollout, 'n') else 1
                self.trajectory_saver.save_batch(
                    batch_id=self._batch_count,
                    trajectory_list=total_batch_list,
                    episode_rewards=total_episode_rewards,
                    expert_trajectories=expert_trajectories,
                    tasks=tasks,
                    traj_uids=total_traj_uid,
                    rollout_n=rollout_n,
                    global_step=self._batch_count,
                )
                self._batch_count += 1
            except Exception as e:
                print(f"[TrajectorySaver] Failed to save batch: {e}")
        
        return gen_batch_output
