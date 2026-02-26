import os
import sys
import json
from pprint import pprint

# 添加项目根目录到 Python 路径，确保能正确 import agent-verl 中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def test_sciworld_expert_trajectory():
    """
    测试 SciWorld 获取 Expert 专家样本的脚本。
    我们将启动一组包含 1 个 Policy Worker 和 1 个 Expert Worker 的环境。
    并在一个完整的 Episode 之后，获取并打印专家轨迹。
    """
    print("==================================================")
    print("🚀 分布式启动 SciWorld 环境 (包含 Expert Worker) ...")
    print("==================================================")
    
    # 模拟通用配置参数
    seed = 42
    env_num = 1      # 我们只开 1 个并行环境
    group_n = 1      # 每组 1 个 Policy Worker。因为 expert_in_group=True, 默认会加 1 个 Expert Worker
    task_nums = 1    # 假设任务类别编号
    
    # 加载 variations_idx，SciWorld 需要从这个文件里采样具体的任务变体
    variation_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'
    ))
    
    with open(variation_path, 'r') as f:
        variations_idx = json.load(f)['train']
        
    print(f"✅ 加载任务变体映射成功，共 {len(variations_idx)} 个变体\n")

    try:
        from agent_system.environments.env_package.sciworld.envs import build_sciworld_envs
        import scienceworld
    except ImportError as e:
        print("❌ 遇到 ImportError:", e)
        print("💡 提示：请确保你在支持的环境服务器上执行此脚本，并且已通过 `pip install scienceworld` 安装了该包。")
        return

    try:
        # 1. 构建带有 Expert 的多进程环境
        print("⏳ 正在启动 Java 进程构建环境 (这需要几秒钟)...")
        envs = build_sciworld_envs(
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            task_nums=task_nums,
            split="train",
            simplifications_preset="easy",
            env_step_limit=15,  # 缩小最大步数以便快速测试
            jar_path=None,      # 如果你的环境变量或默认路径找不到 jar，这里需要填绝对路径
            variations_idx=variations_idx,
            expert_in_group=True # <==== 关键：开启 Expert
        )
        print("✅ 环境构建完成！")
        
        # 2. Reset 开始一个新的 Episode
        print("\n⏳ 正在 Reset 环境 (此时Expert底层会调用 generateGoldPath)...")
        obs_list, info_list = envs.reset()
        
        # 查看分配给这个 Episode 的具体任务描述
        for i, info in enumerate(info_list):
            worker_type = "Expert Worker" if info.get('is_expert', False) else "Policy Worker"
            task_desc = info.get('task_description', 'N/A')
            print(f"  [{worker_type} {i}] 新任务指派: {task_desc}")
            if info.get('is_expert', False):
                print(f"  [{worker_type} {i}] 生成的专家路径长度 (Gold Path Length): {info.get('gold_path_length', 'N/A')} 步")

        print("\n⏳ 步进环境以完成 Episode (为了收集完整的轨迹)...")
        
        done = False
        step_count = 0
        
        while not done and step_count < 15:
            step_count += 1
            
            # Policy 必须提供一个动作。对于 Expert, action 是被忽略的，因为底层自动取 gold action
            actions = ["look around"] * envs.num_processes
            
            obs_list, reward_list, done_list, info_list = envs.step(actions)
            
            # 打印此时 Policy 和 Expert 的当前观测和执行情况
            for i, info in enumerate(info_list):
                if info.get('is_expert', False):
                    # 获取当前积累的历史轨迹部分，显示最后一次执行的动作
                    partial_traj = info.get('expert_trajectory', [])
                    executed = partial_traj[-1].get('action', '未知') if partial_traj else '未知'
                    print(f"  [Step {step_count}] Expert 执行了标准答案动作: '{executed}', 当前积累了 {len(partial_traj)} 步。")
                
            # 检查是否有一个进程完成
            done = any(done_list)
            
        print(f"\n✅ 交互完成！总步数: {step_count}")
        
        # 4. === 核心测试 === 提取最终完整的专家轨迹
        print("\n==================================================")
        print("🎯 获取并解析完整的 Expert Trajectories")
        print("==================================================")
        
        # 这个调用会返回结构化的 Dict，Key是组的编号 (Group Index)
        expert_trajs = envs.get_expert_trajectories()
        
        # 解析打印返回的数据结构
        for group_idx, traj_data in expert_trajs.items():
            print(f"\n[Group {group_idx}] 专家样本结果概览:")
            
            task_info = traj_data.get('task_info', {})
            print(f"  📌 任务名称: {task_info.get('task_name')} (ID: {task_info.get('task_num')} 变体: {task_info.get('variation')})")
            
            steps = traj_data.get('steps', [])
            total_steps = traj_data.get('total_steps', 0)
            print(f"  📌 完整过程共 {total_steps} 步")
            
            print("\n  🔍 专家前 3 步详细记录:")
            for i, step in enumerate(steps[:3]):
                print(f"    - 第 {i+1} 步操作:")
                print(f"      ● [动作] Action: '{step.get('action')}'")
                
                # 为了防止控制台淹没，截断 observation 文本输出
                obs_before = step.get('obs', '').replace('\n', ' ')[:80]
                obs_after = step.get('obs_after', '').replace('\n', ' ')[:80]
                
                print(f"      ● [状态] 动作前观测: '{obs_before}...'")
                print(f"      ● [反应] 动作后观测: '{obs_after}...'")

            if len(steps) > 3:
                print(f"    - ... (省略后续 {len(steps) - 3} 步)")

    except Exception as e:
        print("\n❌ 测试期间发生错误:", e)
        import traceback
        traceback.print_exc()
    finally:
        # 清除多进程资源
        if 'envs' in locals() and hasattr(envs, 'close'):
            envs.close()
            print("\n🧹 环境资源已释放")

if __name__ == "__main__":
    test_sciworld_expert_trajectory()
