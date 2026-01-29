"""
Discriminator API Connection Test

测试 Discriminator API 连接和调用的各种场景，帮助诊断 400 错误。
运行后将日志发送给开发者分析。

Usage:
    # 在服务器上运行
    python tests/rlvmr/test_discriminator_api.py \
        --url "http://127.0.0.1:8080/v1" \
        --model "Qwen3-VL-32B-Instruct-FP8"
"""

import argparse
import sys
import os
import json
import asyncio
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_1_basic_connection(base_url: str):
    """测试 1: 基本 HTTP 连接"""
    print_section("Test 1: Basic HTTP Connection")
    
    import urllib.request
    import urllib.error
    
    # 测试 /v1/models 端点
    models_url = f"{base_url}/models"
    print(f"Testing URL: {models_url}")
    
    try:
        req = urllib.request.Request(models_url)
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            print(f"[SUCCESS] Status: {response.status}")
            print(f"Available models: {json.dumps(data, indent=2)}")
            return data
    except urllib.error.HTTPError as e:
        print(f"[FAILED] HTTP Error: {e.code} - {e.reason}")
        try:
            error_body = e.read().decode()
            print(f"Error body: {error_body}")
        except:
            pass
        return None
    except urllib.error.URLError as e:
        print(f"[FAILED] URL Error: {e.reason}")
        return None
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        return None


def test_2_openai_client(base_url: str, model_name: str, api_key: str = "EMPTY"):
    """测试 2: OpenAI 客户端初始化"""
    print_section("Test 2: OpenAI Client Initialization")
    
    try:
        from openai import OpenAI
        print(f"OpenAI library version: {__import__('openai').__version__}")
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        print(f"[SUCCESS] Client created with base_url={base_url}")
        print(f"[INFO] Model name to use: {model_name}")
        
        return client
    except ImportError:
        print("[FAILED] OpenAI library not installed")
        return None
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        return None


def test_3_list_models(client, base_url: str):
    """测试 3: 列出可用模型"""
    print_section("Test 3: List Available Models via OpenAI Client")
    
    if client is None:
        print("[SKIP] Client not available")
        return []
    
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"[SUCCESS] Found {len(model_ids)} models:")
        for mid in model_ids:
            print(f"  - {mid}")
        return model_ids
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_4_simple_chat(client, model_name: str):
    """测试 4: 简单 Chat 请求"""
    print_section("Test 4: Simple Chat Completion")
    
    if client is None:
        print("[SKIP] Client not available")
        return False
    
    print(f"Model: {model_name}")
    print("Message: 'Hello, how are you?'")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=50,
            temperature=0.7,
        )
        print(f"[SUCCESS] Response received")
        print(f"Response: {response.choices[0].message.content[:200]}...")
        return True
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_discriminator_prompt(client, model_name: str):
    """测试 5: Discriminator 风格的 Prompt"""
    print_section("Test 5: Discriminator-style Prompt")
    
    if client is None:
        print("[SKIP] Client not available")
        return False
    
    # 模拟 Discriminator 的 prompt 格式
    prompt = """### Role Definition
You are an AI agent evaluator.

### Task
Evaluate the following trajectory and determine milestone achievements.

### Policy Trajectory
Step 1:
  Observation: You are in a kitchen.
  Action: go to countertop 1

Step 2:
  Observation: On the countertop 1, you see a mug 1.
  Action: take mug 1 from countertop 1

### Output Format
Return a JSON object with:
{
  "episode_score": 0.0 to 1.0,
  "step_scores": [0.0 to 1.0 for each step]
}
"""
    
    print(f"Model: {model_name}")
    print(f"Prompt length: {len(prompt)} chars")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
        )
        print(f"[SUCCESS] Response received")
        print(f"Response:\n{response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_config_loading():
    """测试 6: 配置加载"""
    print_section("Test 6: Config Loading Simulation")
    
    # 模拟 Hydra/OmegaConf 配置
    from omegaconf import OmegaConf
    
    # 模拟命令行参数
    test_cases = [
        # Case 1: Shell 变量 True
        {"enable": "True", "base_urls": "[http://127.0.0.1:8080/v1]"},
        # Case 2: 小写 true
        {"enable": "true", "base_urls": "[http://127.0.0.1:8080/v1]"},
        # Case 3: 布尔值
        {"enable": True, "base_urls": ["http://127.0.0.1:8080/v1"]},
        # Case 4: 正确的 YAML 格式
        {"enable": True, "base_urls": '["http://127.0.0.1:8080/v1"]'},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}: {case}")
        try:
            cfg = OmegaConf.create({"algorithm": {"discriminator": case}})
            enable = cfg.algorithm.discriminator.enable
            urls = cfg.algorithm.discriminator.base_urls
            
            # 检查类型
            print(f"  enable type: {type(enable)}, value: {enable}")
            print(f"  base_urls type: {type(urls)}, value: {urls}")
            
            # 检查是否为有效布尔值
            if isinstance(enable, bool):
                print(f"  [OK] enable is proper boolean")
            elif enable in ["True", "true", "1"]:
                print(f"  [WARN] enable is string, may not work correctly")
            else:
                print(f"  [ERROR] enable is invalid: {enable}")
            
            # 检查是否为有效列表
            if isinstance(urls, list):
                print(f"  [OK] base_urls is proper list with {len(urls)} items")
            else:
                print(f"  [WARN] base_urls is not a list: {type(urls)}")
                
        except Exception as e:
            print(f"  [ERROR] {e}")


def test_7_discriminator_class(base_url: str, model_name: str, api_key: str = "EMPTY"):
    """测试 7: DiscriminatorRewardCalculator 类"""
    print_section("Test 7: DiscriminatorRewardCalculator Class")
    
    try:
        from rlvmr.discriminator_reward import (
            DiscriminatorRewardCalculator,
            DiscriminatorConfig,
        )
        
        config = DiscriminatorConfig(
            base_urls=[base_url],
            api_key=api_key,
            model_name=model_name,
            max_concurrency_per_url=1,
            request_timeout=30,
        )
        
        print(f"Config created:")
        print(f"  base_urls: {config.base_urls}")
        print(f"  model_name: {config.model_name}")
        print(f"  api_key: {config.api_key}")
        
        calculator = DiscriminatorRewardCalculator(config)
        print(f"[SUCCESS] DiscriminatorRewardCalculator created")
        
        return calculator
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_8_compute_rewards(calculator):
    """测试 8: 计算奖励"""
    print_section("Test 8: Compute Rewards")
    
    if calculator is None:
        print("[SKIP] Calculator not available")
        return False
    
    # 模拟 policy trajectories
    policy_trajectories = [
        [
            {"observation": "You are in a kitchen.", "action": "go to countertop 1"},
            {"observation": "On the countertop, you see a mug.", "action": "take mug 1"},
        ],
        [
            {"observation": "You are in a bedroom.", "action": "go to drawer 1"},
            {"observation": "On the drawer, you see a book.", "action": "take book 1"},
        ],
    ]
    
    print(f"Testing with {len(policy_trajectories)} trajectories")
    
    try:
        episode_rewards, step_rewards = calculator.compute_rewards_sync(
            policy_trajectories=policy_trajectories,
            expert_trajectories=None,
        )
        print(f"[SUCCESS] Rewards computed")
        print(f"  episode_rewards: {episode_rewards}")
        print(f"  step_rewards: {step_rewards}")
        return True
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_raw_http_request(base_url: str, model_name: str, api_key: str = "EMPTY"):
    """测试 9: 原始 HTTP 请求"""
    print_section("Test 9: Raw HTTP Request")
    
    import urllib.request
    import urllib.error
    
    url = f"{base_url}/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10,
    }
    
    print(f"URL: {url}")
    print(f"Model: {model_name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            print(f"[SUCCESS] Status: {response.status}")
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
            return True
    except urllib.error.HTTPError as e:
        print(f"[FAILED] HTTP Error: {e.code} - {e.reason}")
        try:
            error_body = e.read().decode()
            print(f"Error body: {error_body}")
        except:
            pass
        return False
    except Exception as e:
        print(f"[FAILED] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(base_url: str, model_name: str, api_key: str = "EMPTY"):
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  Discriminator API Connection Test Suite")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model Name: {model_name}")
    print(f"  API Key: {api_key[:10]}..." if len(api_key) > 10 else f"  API Key: {api_key}")
    
    results = {}
    
    # Test 1: Basic connection
    results['basic_connection'] = test_1_basic_connection(base_url)
    
    # Test 2: OpenAI client
    client = test_2_openai_client(base_url, model_name, api_key)
    results['openai_client'] = client is not None
    
    # Test 3: List models
    model_ids = test_3_list_models(client, base_url)
    results['list_models'] = len(model_ids) > 0
    
    # 检查模型名是否匹配
    if model_ids:
        if model_name in model_ids:
            print(f"\n[OK] Model '{model_name}' is available")
        else:
            print(f"\n[WARNING] Model '{model_name}' NOT in available models!")
            print(f"Available: {model_ids}")
            print(f"\nSuggestion: Try using one of the available models")
    
    # Test 4: Simple chat
    results['simple_chat'] = test_4_simple_chat(client, model_name)
    
    # Test 5: Discriminator prompt
    results['discriminator_prompt'] = test_5_discriminator_prompt(client, model_name)
    
    # Test 6: Config loading
    test_6_config_loading()
    
    # Test 7: Discriminator class
    calculator = test_7_discriminator_class(base_url, model_name, api_key)
    results['discriminator_class'] = calculator is not None
    
    # Test 8: Compute rewards
    results['compute_rewards'] = test_8_compute_rewards(calculator)
    
    # Test 9: Raw HTTP
    results['raw_http'] = test_9_raw_http_request(base_url, model_name, api_key)
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Discriminator API connection")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1",
                        help="Base URL for the API (e.g., http://127.0.0.1:8080/v1)")
    parser.add_argument("--model", default="discriminator",
                        help="Model name to use")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key for authentication")
    
    args = parser.parse_args()
    
    success = run_all_tests(args.url, args.model, args.api_key)
    sys.exit(0 if success else 1)
