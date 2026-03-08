"""
Brand Identity Analyzer Node for ComfyUI
Analyzes brand identity using Gemini Flash with Google Search grounding
"""

import os
import json
import time
from typing import Tuple, Optional, Dict, Any

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

GEMINI_FLASH_MODEL = "gemini-3-flash-preview"

def get_config_path() -> str:
    """Get the path to the configs directory"""
    return os.path.join(os.path.dirname(__file__), "configs")

def load_brand_analyzer_prompt() -> Tuple[str, Optional[str]]:
    """Load the brand analyzer system prompt. Returns (prompt, error)"""
    prompt_path = os.path.join(get_config_path(), "Brand_Analyzer", "brand_analyzer_prompt.txt")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read(), None
    except Exception as e:
        return "", f"Failed to load brand analyzer prompt: {str(e)}"

def load_brand_identity_template() -> Tuple[dict, Optional[str]]:
    """Load the brand identity JSON template. Returns (template, error)"""
    template_path = os.path.join(get_config_path(), "Brand_Analyzer", "brand_identity_template.json")
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return {}, f"Failed to load brand identity template: {str(e)}"

def extract_json_from_response(text: str) -> Tuple[dict, Optional[str]]:
    """Extract JSON from response text, handling code fences and surrounding prose."""
    text = text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_str), None
        except json.JSONDecodeError as e:
            return {}, f"Invalid JSON in response: {str(e)}"
    
    return {}, "No valid JSON found in response"

def save_brand_identity(brand_name: str, brand_data: dict) -> Tuple[bool, str]:
    """Save brand identity JSON to configs/Brands/{brand_name}.json"""
    brands_path = os.path.join(get_config_path(), "Brands")
    os.makedirs(brands_path, exist_ok=True)
    
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in brand_name)
    safe_name = safe_name.strip().replace(' ', '_')
    
    file_path = os.path.join(brands_path, f"{safe_name}.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(brand_data, f, indent=2, ensure_ascii=False)
        return True, file_path
    except Exception as e:
        return False, str(e)


class BrandIdentityAnalyzerNode:
    """
    ComfyUI node that analyzes brand identity using Gemini Flash with Google Search grounding.
    Generates a comprehensive brand identity JSON file.
    """
    
    CATEGORY = "Product to Ads/Brand"
    FUNCTION = "analyze_brand"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("brand_json", "file_path", "status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brand_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter brand name (e.g., Emporio Armani)"
                }),
                "gemini_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True
                }),
                "api_key_status": ("STRING", {
                    "multiline": False,
                    "default": "Not Verified",
                    "display": "text"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    def __init__(self):
        self.client = None
        self._api_status = "Not Configured"
    
    def _init_client(self, api_key: str) -> Tuple[bool, str]:
        """Initialize Gemini client with API key"""
        if not GENAI_AVAILABLE:
            self._api_status = "Error: google-genai not installed"
            return False, self._api_status
        
        if not api_key:
            self._api_status = "No API Key"
            return False, "No API key provided"
        
        try:
            self.client = genai.Client(api_key=api_key)
            self._api_status = "Configured"
            return True, "Client initialized"
        except Exception as e:
            self._api_status = f"Error: {str(e)}"
            return False, self._api_status
    
    def verify_api_key(self, api_key: str) -> Tuple[bool, str]:
        """Verify the API key is valid"""
        success, msg = self._init_client(api_key)
        if not success:
            return False, msg
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents="Say 'API OK' if you can read this.",
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="minimal")
                )
            )
            if response:
                response_text = ""
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                
                if response_text:
                    self._api_status = "API Key Valid"
                    return True, "API Key Valid"
            
            self._api_status = "API Error"
            return False, "No response from API"
        except Exception as e:
            self._api_status = f"API Error: {str(e)}"
            return False, str(e)
    
    def analyze_brand_identity(self, brand_name: str) -> Tuple[bool, dict, str]:
        """
        Analyze brand identity using Gemini Flash with Google Search grounding.
        Returns (success, brand_data_dict, message)
        """
        if not self.client:
            return False, {}, "Client not initialized"
        
        system_prompt, prompt_error = load_brand_analyzer_prompt()
        if prompt_error:
            return False, {}, prompt_error
        
        template, template_error = load_brand_identity_template()
        if template_error:
            return False, {}, template_error
        
        if not template:
            return False, {}, "Brand identity template is empty"
        
        template_str = json.dumps(template, indent=2)
        
        user_prompt = f"""Analyze the brand: {brand_name}

Use Google Search to research this brand thoroughly:
1. Find official brand information (website, corporate pages)
2. Search for advertising campaigns on Google Images
3. Identify visual patterns, photography style, and brand behavior

Fill in the following JSON template with your analysis. Return ONLY valid JSON, no other text:

{template_str}"""
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=8192,
                    temperature=0.3,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            
            if not response:
                return False, {}, "No response from Gemini"
            
            response_text = ""
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            if not response_text:
                return False, {}, "Empty response from Gemini"
            
            brand_data, json_error = extract_json_from_response(response_text)
            if json_error:
                return False, {}, json_error
            
            return True, brand_data, "Analysis complete"
                
        except Exception as e:
            return False, {}, f"API error: {str(e)}"
    
    def analyze_brand(
        self,
        brand_name: str,
        gemini_api_key: str,
        api_key_status: str,
        unique_id: str = None
    ) -> Tuple[str, str, str]:
        """
        Main execution function for the node.
        Returns (brand_json, file_path, status)
        """
        if not brand_name.strip():
            return "", "", "Error: Brand name is required"
        
        api_key = gemini_api_key.strip()
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "", "", "Error: API key is required"
        
        valid, verify_msg = self.verify_api_key(api_key)
        
        if not valid:
            return "", "", f"Error: API key invalid - {verify_msg}"
        
        success, brand_data, msg = self.analyze_brand_identity(brand_name)
        
        if not success:
            return "", "", f"Error: {msg}"
        
        save_success, file_path = save_brand_identity(brand_name, brand_data)
        
        if not save_success:
            brand_json = json.dumps(brand_data, indent=2)
            return brand_json, "", f"Analysis complete but save failed: {file_path}"
        
        brand_json = json.dumps(brand_data, indent=2)
        return brand_json, file_path, f"Brand identity saved to {file_path}"


NODE_CLASS_MAPPINGS = {
    "BrandIdentityAnalyzer": BrandIdentityAnalyzerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrandIdentityAnalyzer": "Brand Identity Analyzer",
}
