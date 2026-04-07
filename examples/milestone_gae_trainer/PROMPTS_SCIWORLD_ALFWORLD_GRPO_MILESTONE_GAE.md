# GRPO 和 Milestone-GAE 中 SciWorld/ALFWorld 的提示词合集

此文件汇集了以下模型使用的所有提示词模板：
- SciWorld 和 ALFWorld 中的策略模型（GRPO 和 Milestone-GAE 都在使用）
- Milestone-GAE 裁判大模型 (Judge LLM)
- Milestone-GAE 里程碑生成器大模型 (Milestone-Generator LLM)

## 1. 模型与提示词映射关系

### GRPO
- SciWorld 策略模型（来自 examples/grpo_trainer/run_sciworld.sh）：Qwen2.5-7B-Instruct
  - 使用 SciWorld 策略提示词：SCIWORLD_TEMPLATE_NO_HIS / SCIWORLD_TEMPLATE
  - 如果 meta_think=true，则切换为 SCIWORLD_TEMPLATE_NO_HIS_MC / SCIWORLD_TEMPLATE_MC
- ALFWorld 策略模型（来自 examples/grpo_trainer/run_alfworld.sh）：Qwen2.5-7B-Instruct
  - 使用 ALFWorld 策略提示词：ALFWORLD_TEMPLATE_NO_HIS / ALFWORLD_TEMPLATE

### Milestone-GAE
- SciWorld 策略模型（来自 examples/milestone_gae_trainer/run_sciworld.sh）：Qwen2.5-3B-Instruct
  - 使用与上述相同的 SciWorld 策略提示词
- SciWorld 裁判/生成器模型（来自 examples/milestone_gae_trainer/run_sciworld.sh）：Qwen3-VL-32B-Instruct-FP8
  - 裁判提示词：MilestoneJudge._build_prompt
  - 生成器提示词：MilestoneGenerator.PROMPT_TEMPLATE

- ALFWorld 策略模型（来自 examples/milestone_gae_trainer/run_alfworld.sh）：Qwen2.5-3B-Instruct
  - 使用与上述相同的 ALFWorld 策略提示词
- ALFWorld 裁判/生成器模型（来自 examples/milestone_gae_trainer/run_alfworld.sh）：Qwen3-VL-32B-Instruct-FP8
  - 裁判提示词：MilestoneJudge._build_prompt
  - 生成器提示词：MilestoneGenerator.PROMPT_TEMPLATE

## 2. SciWorld 策略提示词

来源文件: agent_system/environments/prompts/sciworld.py

### SCIWORLD_TEMPLATE_NO_HIS

~~~text
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
~~~

### SCIWORLD_TEMPLATE

~~~text
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an appropriate action for the current step and present it within <action> </action> tags.
~~~

### SCIWORLD_TEMPLATE_NO_HIS_MC

~~~text
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an appropriate action for the current step within <action> </action> tags.
~~~

### SCIWORLD_TEMPLATE_MC

~~~text
You are an expert agent operating in the ScienceWorld environment, which is a text-based virtual environment centered around accomplishing tasks from the elementary science curriculum.
Your current task is: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Here are the actions you may take:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

Current available actions:
{available_actions}

Your previous overall plan is: {planning}.  Please strictly adhere to your plan.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an appropriate action for the current step within <action> </action> tags.
~~~

## 3. ALFWorld 策略提示词

来源文件: agent_system/environments/prompts/alfworld.py

### ALFWORLD_TEMPLATE_NO_HIS

~~~text
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
~~~

### ALFWORLD_TEMPLATE

~~~text
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
~~~

## 4. Milestone-GAE 裁判提示词 (Judge LLM)

来源文件: rlvmr/milestone/judge.py

~~~text
You are a task progress evaluator.

## Task Description
{task_description}

## Milestone Checklist
M0 (Φ=0.0): Not started — Criteria: No milestone has been achieved
{milestone_list}

## Agent Execution Trajectory

Note: Each step shows the environment state (what the agent observes before acting) followed by the agent's action.
{steps_str}

## Instructions

Evaluate the highest milestone achieved at each step.

Output format (strict JSON):
{
  "judgments": [
    {"step": 1, "highest_milestone": "M0", "phi": 0.0},
    {"step": 2, "highest_milestone": "M1", "phi": 0.15},
    ...
  ],
  "final_success": true/false,
  "reasoning": "Brief explanation of your judgment"
}

Notes:
1. M0 means no milestone has been achieved yet, phi=0.0
2. Milestones are generally monotonically increasing (may occasionally regress due to wrong actions)
3. The highest milestone (phi=1.0) should only be reached when the task is confirmed successful
4. You must output valid JSON
~~~

其中：
- milestone_list 从 milestones 生成，格式为：
  - {id} (Φ={phi}): {name} — Criteria: {criteria}
- steps_str 的生成格式为：
  - Step i:
    - Environment State: {observation}
    - Agent Action: {action}

## 5. Milestone-GAE 生成器提示词 (Generator LLM)

来源文件: rlvmr/milestone/generator.py (PROMPT_TEMPLATE)

~~~text
You are a task decomposition expert.

## Task Description
{task_description}

## Expert Successful Trajectory
{expert_trajectory}

## Instructions
Analyze this successful expert trajectory and decompose the task into key milestones marking progress from the initial state to task completion.

Requirements:
1. Generate between 4 and 10 milestones depending on task complexity — simpler tasks need fewer, complex multi-stage tasks need more
2. Each milestone should describe **what was accomplished** and **what state was reached** (e.g., "Agent has arrived at the living room and picked up the metal pot", "The circuit is fully connected and the light bulb is on")
3. Criteria must be **state-based, not step-based**: do NOT reference step numbers (e.g., "Step 12", "after step 30"). Different agents may reach the same state at different steps
4. Milestones should be **verifiable from environment observations** — describe conditions that can be checked against observation text at any step
5. Phi values should increase from 0.0 to 1.0. Distribute them based on task difficulty of each stage, not necessarily evenly. The last milestone must have phi=1.0
6. Order milestones by logical task progression, not by step index

Output format (strict JSON):
{
  "milestones": [
    {"id": "M1", "name": "Milestone name", "phi": <float>, "criteria": "Criteria: Agent has [accomplished X] and [reached state Y]"},
    {"id": "M2", "name": "Milestone name", "phi": <float>, "criteria": "Criteria: Agent has [accomplished X] and [reached state Y]"},
    ...
    {"id": "M<N>", "name": "Milestone name", "phi": 1.0, "criteria": "Criteria: Agent has [completed final objective] and [environment shows final state]"}
  ],
  "reasoning": "Brief explanation of milestone decomposition"
}
~~~

## 6. 源代码文件列表

- examples/grpo_trainer/run_sciworld.sh
- examples/grpo_trainer/run_alfworld.sh
- examples/milestone_gae_trainer/run_sciworld.sh
- examples/milestone_gae_trainer/run_alfworld.sh
- agent_system/environments/prompts/sciworld.py
- agent_system/environments/prompts/alfworld.py
- agent_system/environments/env_manager.py
- rlvmr/milestone/judge.py
- rlvmr/milestone/generator.py
