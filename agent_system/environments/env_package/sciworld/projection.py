from typing import List
import re

def sciworld_projection(actions: List[str], available_actions=None, meta_think=False):
    """
    Process SciWorld actions from model output.
    
    Args:
        actions: List of raw model outputs
        available_actions: List of available actions per environment
        meta_think: Whether meta-cognition mode is enabled
        
    Returns:
        Tuple of (processed_actions, valids, plannings, action_available)
        - processed_actions: Extracted action strings
        - valids: Binary validity flags (1=valid, 0=invalid)
        - plannings: Extracted planning content (or None if not present)
        - action_available: Whether the action is in available_actions
    """
    valids = [0] * len(actions)
    action_available = [False] * len(actions)
    processed_actions = []
    plannings = []
    
    # Skill tags for meta_think mode
    skill_tags = [
        r"<planning>.*?</planning>",
        r"<reflection>.*?</reflection>",
        r"<explore>.*?</explore>",
        r"<monitor>.*?</monitor>"
    ]

    for i in range(len(actions)):
        original_str = actions[i]
        valid = 1
        act_str = ""
        planning_content = None
        
        # Extract <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = original_str.find(start_tag)
        end_idx = original_str.find(end_tag)
        
        try:
            if start_idx == -1 or end_idx == -1:
                processed_actions.append(original_str[-20:])
                valids[i] = 0
                plannings.append(None)
                continue
                
            extracted_action = original_str[start_idx + len(start_tag):end_idx].strip()
            act_str = extracted_action
            processed_actions.append(extracted_action)
            valids[i] = 1
            
            # Check if action is in available actions
            if available_actions is not None:
                env_available_actions = available_actions[i]
                if extracted_action in env_available_actions:
                    action_available[i] = True
                    
        except:
            processed_actions.append(original_str[-20:])
            valids[i] = 0
            plannings.append(None)
            continue
        
        # Extract <planning>...</planning> content
        planning_match = re.search(r"<planning>([\s\S]*?)</planning>", original_str, re.IGNORECASE)
        if planning_match:
            planning_inner = planning_match.group(1).strip()
            planning_content = planning_inner if planning_inner else None
        plannings.append(planning_content)
        
        # Meta-think validation: require exactly one skill tag
        if meta_think:
            skill_count = 0
            found_skill = False
            min_action_pos = original_str.lower().find("<action>")
            
            for tag in skill_tags:
                tag_matches = list(re.finditer(tag, original_str, re.IGNORECASE | re.DOTALL))
                skill_count += len(tag_matches)
                for tag_match in tag_matches:
                    # Check if skill tag content is non-empty
                    inner = re.sub(r"<.*?>", "", tag_match.group(0)).strip()
                    if inner:
                        found_skill = True
                        # Skill tag must appear before <action>
                        if original_str.find(tag_match.group(0)) > min_action_pos:
                            valid = 0
            
            if skill_count != 1:
                valid = 0
            if not found_skill:
                valid = 0
        else:
            # Non meta-think mode: require <think>...</think>
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valid = 0
        
        # Check for Chinese characters (invalid)
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valid = 0
        
        valids[i] = valid

    return processed_actions, valids, plannings, action_available
