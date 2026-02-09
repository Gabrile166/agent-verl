"""
Milestone Templates for Different Environments

Provides utility functions to load and instantiate milestone templates.
"""

import json
import os
from typing import Dict, List, Any

# 模板目录路径
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')


def load_milestone_template(env_name: str) -> Dict[str, Any]:
    """
    加载指定环境的里程碑模板
    
    Args:
        env_name: 环境名称，如 'alfworld', 'webshop', 'math'
    
    Returns:
        里程碑模板字典
    """
    template_path = os.path.join(TEMPLATE_DIR, f'{env_name}.json')
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Milestone template not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_milestones_for_task(template: Dict[str, Any], task_description: str) -> List[Dict[str, Any]]:
    """
    根据任务描述实例化里程碑列表
    
    Args:
        template: 里程碑模板
        task_description: 具体任务描述
    
    Returns:
        实例化后的里程碑列表
    """
    milestones = template.get('milestones', [])
    # TODO: 从 task_description 中提取变量并填充到 milestones 中
    return milestones


def get_available_templates() -> List[str]:
    """获取所有可用的模板名称"""
    if not os.path.exists(TEMPLATE_DIR):
        return []
    
    templates = []
    for f in os.listdir(TEMPLATE_DIR):
        if f.endswith('.json'):
            templates.append(f[:-5])  # 去掉 .json 后缀
    return templates
