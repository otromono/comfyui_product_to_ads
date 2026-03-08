"""
Configuration Loader for Product to Ads Node
Loads master prompts, brand identities, and brief prompts from JSON files
"""

import os
import json
from typing import Optional, List, Tuple, Dict, Any

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "Product_to_Ads")
BRANDS_DIR = os.path.join(os.path.dirname(__file__), "configs", "Brands")
BRIEF_GENERATOR_DIR = os.path.join(os.path.dirname(__file__), "configs", "Brief_Generator")


def get_config_dir() -> str:
    """Get the configuration directory path"""
    return CONFIG_DIR


def get_prompt_profiles() -> List[str]:
    """
    Get list of available prompt profiles by scanning JSON files in config directory
    
    Returns:
        List of JSON filenames (without extension) found in configs/Product_to_Ads/
    """
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        return ["Default"]
    
    try:
        profiles = []
        for filename in sorted(os.listdir(CONFIG_DIR)):
            if filename.endswith('.json') and not filename.startswith('.'):
                profile_name = filename[:-5]
                profiles.append(profile_name)
        
        if not profiles:
            return ["Default"]
        
        return profiles
        
    except Exception as e:
        print(f"[Product to Ads] Error scanning profiles: {e}")
        return ["Default"]


def load_master_prompt(profile_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Load the master prompt from a JSON file
    
    The entire content of the JSON file is used as the system instruction.
    
    Args:
        profile_name: The name of the profile (filename without .json extension)
        
    Returns:
        Tuple of (master_prompt_text, error_message)
    """
    prompt_file = os.path.join(CONFIG_DIR, f"{profile_name}.json")
    
    if not os.path.exists(prompt_file):
        txt_file = os.path.join(CONFIG_DIR, f"{profile_name}.txt")
        if os.path.exists(txt_file):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, None
            except Exception as e:
                return _get_default_master_prompt(), f"Error loading txt file: {e}"
        
        return _get_default_master_prompt(), f"Profile file not found: {prompt_file}"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, None
        
    except Exception as e:
        error_msg = f"Error loading master prompt: {e}"
        return _get_default_master_prompt(), error_msg


def _get_default_master_prompt() -> str:
    """Return a default master prompt if no file is available"""
    return """
You are an expert advertising creative system operating as a coordinated team of expert roles:
- Product Intelligence Analyst
- Review & Sentiment Analyst
- Advertising Strategist
- Creative Art Director
- Advertising Photographer (Virtual)
- Visual & Graphic Director
- Talent & Campaign Integrator

MISSION

Given product data and reference images, generate a complete Campaign Blueprint JSON for image generation.

OUTPUT FORMAT

You MUST respond with ONLY a valid JSON object. Generate ONLY the JSON Blueprint. No additional text, explanations, or markdown formatting outside the JSON.
"""


def get_brand_profiles() -> List[str]:
    """
    Get list of available brand profiles by scanning JSON files in configs/Brands/
    
    Returns:
        List of brand JSON filenames (without extension), excluding templates
    """
    if not os.path.exists(BRANDS_DIR):
        os.makedirs(BRANDS_DIR, exist_ok=True)
        return ["No Brand"]
    
    try:
        brands = ["No Brand"]
        for filename in sorted(os.listdir(BRANDS_DIR)):
            if filename.endswith('.json') and not filename.startswith(('_', '.')):
                brand_name = filename[:-5]
                brands.append(brand_name)
        
        return brands
        
    except Exception as e:
        print(f"[Product to Ads] Error scanning brands: {e}")
        return ["No Brand"]


def load_brand_identity(brand_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load brand identity from a JSON file
    
    Args:
        brand_name: The name of the brand (filename without .json extension)
        
    Returns:
        Tuple of (brand_identity_dict, error_message)
    """
    if brand_name == "No Brand":
        return None, None
    
    brand_file = os.path.join(BRANDS_DIR, f"{brand_name}.json")
    
    if not os.path.exists(brand_file):
        return None, f"Brand file not found: {brand_file}"
    
    try:
        with open(brand_file, 'r', encoding='utf-8') as f:
            brand_data = json.load(f)
        return brand_data, None
        
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in brand file: {e}"
    except Exception as e:
        return None, f"Error loading brand identity: {e}"


def load_brief_prompt() -> Tuple[Optional[str], Optional[str]]:
    """
    Load the brief generation prompt from configs/Brief_Generator/
    
    Returns:
        Tuple of (brief_prompt_text, error_message)
    """
    brief_file = os.path.join(BRIEF_GENERATOR_DIR, "brief_prompt.json")
    
    if not os.path.exists(brief_file):
        return _get_default_brief_prompt(), "Brief prompt file not found, using default"
    
    try:
        with open(brief_file, 'r', encoding='utf-8') as f:
            brief_data = json.load(f)
        
        system_instruction = brief_data.get("system_instruction", "")
        output_format = brief_data.get("output_format", {})
        generation_rules = brief_data.get("generation_rules", [])
        
        prompt_text = system_instruction
        if output_format:
            prompt_text += f"\n\nOUTPUT FORMAT:\n{json.dumps(output_format, indent=2)}"
        if generation_rules:
            prompt_text += f"\n\nGENERATION RULES:\n" + "\n".join(f"- {rule}" for rule in generation_rules)
        
        return prompt_text, None
        
    except json.JSONDecodeError as e:
        return _get_default_brief_prompt(), f"Invalid JSON in brief prompt: {e}"
    except Exception as e:
        return _get_default_brief_prompt(), f"Error loading brief prompt: {e}"


def _get_default_brief_prompt() -> str:
    """Get a default brief generation prompt"""
    return """You are a Senior Art Director and Campaign Strategist.
Generate a comprehensive Campaign Brief following the 10-point framework.

Analyze the product data and brand identity, then produce a detailed brief covering:
1. Strategic Objective
2. Central Message
3. Visual Tone of Voice
4. Product Role
5. Visual Language and Brand Coherence
6. Photographer and Equipment
7. Extended Art Direction
8. Environment and Context
9. Texture, Material, and Product Render
10. Final Image Signature

Output ONLY a valid JSON object with the complete campaign brief.
"""
