
import os
import sys
import json
import asyncio
import argparse
import ray
from omegaconf import OmegaConf, DictConfig

# Add project root to path
# Assuming we are in tests/integration, so root is ../../
# Adjust if needed based on execution location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mock classes to simulate internal dependencies if imports fail or are too heavy
# But we aim for integration, so we try to import real ones.

from agent_system.environments import env_manager
from rlvmr.discriminator_reward import DiscriminatorRewardCalculator
# from rlvmr.discriminator_reward import format_policy_trajectory # Might be instance method

class MockConfig:
    def __init__(self):
        self.env = OmegaConf.create({
            'env_name': 'alfworld',
            'alfworld': {
                'config_path': os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/base_config.yaml'),
                'eval_dataset': 'eval_in_distribution'
            },
            'seed': 42,
            'rollout': {'n': 1},
        })
        self.data = OmegaConf.create({
            'train_batch_size': 2, 
            'val_batch_size': 2
        })
        self.algorithm = OmegaConf.create({
            'expert': {
                'enable': True,
            },
            'discriminator': {
                'enable': True,
                'base_urls': ["https://api.openai.com/v1"],
                'model': 'gpt-4o',
                'api_key': os.environ.get("OPENAI_API_KEY", "mock-key"),
                'prompt_template': 'milestone'
            }
        })
        self.resources = OmegaConf.create({
            'num_cpus': 1
        })
        
        # Add dictionary access for compatibility
        self.__dict__.update(self.algorithm)

class MockActor:
    """Simulates an LLM Policy"""
    def __init__(self):
        self.responses = [
            "look",
            "go to countertop 1",
            "take apple 1 from countertop 1"
        ]
        self.step_cnt = 0

    def generate(self, obs_list):
        actions = []
        for _ in obs_list:
            action = self.responses[self.step_cnt % len(self.responses)]
            actions.append(action)
        self.step_cnt += 1
        return actions

async def run_pipeline_test():
    print(">>> 1. Initializing Config & Environment Manager...")
    # Wrap in DictConfig to ensure behavior matches OmegaConf
    config = OmegaConf.create({
        'env': {
            'env_name': 'alfworld',
            'alfworld': {
                'config_path': os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/base_config.yaml'),
                'eval_dataset': 'eval_in_distribution'
            },
            'seed': 42,
            'rollout': {'n': 1},
        },
        'data': {
            'train_batch_size': 2,
            'val_batch_size': 2
        },
        'algorithm': {
            'expert': {
                'enable': True,
            },
            'discriminator': {
                'enable': True,
                'base_urls': ["https://api.openai.com/v1"],
                'model': 'gpt-4o',
                'api_key': os.environ.get("OPENAI_API_KEY", "mock-key"),
                'prompt_template': 'milestone'
            }
        },
        'resources': {
            'num_cpus': 1
        }
    })
    
    # Init Env Manager logic locally since env_manager wrapper might need deeper integration
    # We'll use AlfworldEnvs directly or via the builder in env_manager
    from agent_system.environments.env_manager import build_alfworld_envs
    
    print(">>> 2. Building Environments (AlfworldEnvs)...")
    envs = build_alfworld_envs(
        alf_config_path=config.env.alfworld.config_path,
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=config.env.rollout.n,
        resources_per_worker={'num_cpus': 1},
        is_train=True,
        expert_in_group=config.algorithm.expert.enable
    )
    
    print(f"    Env initialized. Expert Enabled: {envs.expert_in_group}")
    print(f"    Num Processes: {envs.num_processes}")
    
    print(">>> 3. Starting Interaction Loop...")
    actor = MockActor()
    
    # Reset
    obs_list, _, info_list = envs.reset()
    print(f"    Reset Done. Obs count: {len(obs_list)}")
    
    # Rollout loop
    history = [[] for _ in range(len(obs_list))]
    
    for step in range(3):
        print(f"    --- Step {step+1} ---")
        actions = actor.generate(obs_list)
        print(f"    Actions: {actions}")
        
        # Step
        next_obs, _, rewards, dones, infos = envs.step(actions)
        
        for i in range(len(obs_list)):
            history[i].append({
                'observation': obs_list[i],
                'action': actions[i],
                'reward': rewards[i],
                'done': dones[i]
            })
            
        obs_list = next_obs
        if step == 0:
             # Check for expert info in the first return
             if 'expert_trajectory' in infos[0]: 
                 print("    [Info] expert_trajectory found in info (expected behavior for Worker).")
             
    print(">>> 4. Collecting Expert Trajectories...")
    expert_trajs = envs.get_expert_trajectories()
    print(f"    collected expert trajectories for {len(expert_trajs)} groups.")
    
    if len(expert_trajs) > 0:
        print("    [SUCCESS] Expert trajectories are NOT empty.")
        # Check content
        first_key = list(expert_trajs.keys())[0]
        traj = expert_trajs[first_key]
        print(f"    Sample Expert Traj (Group {first_key}): {json.dumps(traj[:1])}...")
    else:
        print("    [FAILURE] Expert trajectories ARE empty!")

    print(">>> 5. Formatting Data for Discriminator...")
    formatted_policy_trajs = []
    
    # Create a dummy Calculator just to access the formatting method if it's static/utility
    # Or implement simple formatting here matching the logic
    def simple_format(traj):
        formatted = []
        for step in traj:
            formatted.append({
                'observation': step['observation'],
                'action': step['action']
            })
        return json.dumps(formatted)

    for i in range(len(history)):
        formatted_policy_trajs.append(simple_format(history[i]))

    print(f"    Formatted Policy Traj 0: {formatted_policy_trajs[0][:100]}...")

    print(">>> 6. Initializing Discriminator & Computing Rewards...")
    if config.algorithm.discriminator.api_key == "mock-key":
         print("    [Make-Believe] Skipping actual API call (no key). Creating dummy discriminator.")
    
    try:
        discriminator = DiscriminatorRewardCalculator(config)
        
        # Mocking the async method for testing flow
        async def mock_compute(policy_trajs, expert_trajs):
            import numpy as np
            print("        [Mock] compute_rewards called.")
            print(f"        [Mock] Received {len(policy_trajs)} policy trajs and expert trajs keys: {list(expert_trajs.keys()) if expert_trajs else 'None'}")
            batch_size = len(policy_trajs)
            # Return dummy scores
            return np.ones(batch_size), np.ones((batch_size, 3)) # Assuming 3 steps

        discriminator.compute_rewards = mock_compute
        
        episode_rewards, step_rewards = await discriminator.compute_rewards(
            formatted_policy_trajs, 
            expert_trajs
        )
        print("    [SUCCESS] Discriminator pipeline completed.")
        print(f"    Episode Rewards: {episode_rewards}")
    except Exception as e:
        print(f"    [FAILURE] Discriminator computation failed: {e}")
        import traceback
        traceback.print_exc()

    print(">>> Test Complete.")
    
    # Cleanup
    envs.close()

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    asyncio.run(run_pipeline_test())
