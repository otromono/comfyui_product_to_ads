"""
Keyword Loader - Loads and filters keyword banks for prompt enrichment
"""
import os
import json
from typing import Dict, List, Any, Optional, Tuple


class KeywordLoader:
    """Loads and filters keywords from the Keyword Bank JSON"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = config_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(current_dir, "configs", "Keyword_Banks")
        
        self._keywords_cache: Optional[Dict[str, Any]] = None
    
    def _load_keywords_file(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Load the keywords JSON file"""
        keywords_file = os.path.join(self.config_path, "keywords.json")
        
        if not os.path.exists(keywords_file):
            return None, f"Keywords file not found: {keywords_file}"
        
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._keywords_cache = data
            return data, None
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in keywords file: {e}"
        except Exception as e:
            return None, f"Error loading keywords: {e}"
    
    def get_keywords(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Get all keywords (cached)"""
        if self._keywords_cache is not None:
            return self._keywords_cache, None
        return self._load_keywords_file()
    
    def get_filtered_keywords(
        self,
        min_weight: int = 3,
        max_tokens_per_category: int = 20,
        categories: Optional[List[str]] = None
    ) -> Tuple[Dict[str, List[str]], Optional[str]]:
        """
        Get keywords filtered by weight and optionally by category
        
        Args:
            min_weight: Minimum weight to include (1-5, default 3)
            max_tokens_per_category: Maximum tokens per category (default 20)
            categories: Optional list of category names to include (None = all)
            
        Returns:
            Tuple of (filtered_keywords_dict, error_message)
        """
        keywords_data, error = self.get_keywords()
        if error:
            return {}, error
        
        if not keywords_data:
            return {}, "No keywords loaded"
        
        filtered = {}
        
        for category_name, category_data in keywords_data.items():
            if categories and category_name not in categories:
                continue
            
            if not isinstance(category_data, dict) or "tokens" not in category_data:
                continue
            
            tokens = category_data.get("tokens", [])
            
            filtered_tokens = [
                t["token"] for t in tokens
                if isinstance(t, dict) 
                and t.get("weight_1to5", 0) >= min_weight
            ]
            
            filtered_tokens = filtered_tokens[:max_tokens_per_category]
            
            if filtered_tokens:
                domain = category_data.get("domain", category_name.lower())
                filtered[category_name] = {
                    "domain": domain,
                    "tokens": filtered_tokens
                }
        
        return filtered, None
    
    def format_for_prompt(
        self,
        min_weight: int = 3,
        max_tokens_per_category: int = 15,
        categories: Optional[List[str]] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Format keywords as a string suitable for injection into prompts
        
        Args:
            min_weight: Minimum weight to include
            max_tokens_per_category: Maximum tokens per category
            categories: Optional list of categories to include
            
        Returns:
            Tuple of (formatted_string, error_message)
        """
        filtered, error = self.get_filtered_keywords(
            min_weight=min_weight,
            max_tokens_per_category=max_tokens_per_category,
            categories=categories
        )
        
        if error:
            return "", error
        
        if not filtered:
            return "", "No keywords matched the filter criteria"
        
        lines = ["KEYWORD BANK - Use these professional terms to enrich the blueprint:"]
        lines.append("")
        
        for category_name, data in filtered.items():
            domain = data.get("domain", "general")
            tokens = data.get("tokens", [])
            
            lines.append(f"## {category_name} ({domain})")
            lines.append(", ".join(tokens))
            lines.append("")
        
        return "\n".join(lines), None
    
    def get_categories(self) -> Tuple[List[str], Optional[str]]:
        """Get list of all available categories"""
        keywords_data, error = self.get_keywords()
        if error:
            return [], error
        
        if not keywords_data:
            return [], "No keywords loaded"
        
        return list(keywords_data.keys()), None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the keyword bank"""
        keywords_data, error = self.get_keywords()
        if error:
            return {"error": error}
        
        if not keywords_data:
            return {"error": "No keywords loaded"}
        
        stats = {
            "total_categories": len(keywords_data),
            "categories": {}
        }
        
        total_tokens = 0
        for category_name, category_data in keywords_data.items():
            if isinstance(category_data, dict) and "tokens" in category_data:
                tokens = category_data.get("tokens", [])
                count = len(tokens)
                total_tokens += count
                
                weight_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                for t in tokens:
                    if isinstance(t, dict):
                        w = t.get("weight_1to5", 0)
                        if w in weight_counts:
                            weight_counts[w] += 1
                
                stats["categories"][category_name] = {
                    "domain": category_data.get("domain", "unknown"),
                    "total_tokens": count,
                    "by_weight": weight_counts
                }
        
        stats["total_tokens"] = total_tokens
        return stats


_keyword_loader_instance: Optional[KeywordLoader] = None


def get_keyword_loader() -> KeywordLoader:
    """Get singleton instance of KeywordLoader"""
    global _keyword_loader_instance
    if _keyword_loader_instance is None:
        _keyword_loader_instance = KeywordLoader()
    return _keyword_loader_instance


def load_keywords_for_prompt(min_weight: int = 3) -> Tuple[str, Optional[str]]:
    """
    Convenience function to load keywords formatted for prompt injection
    
    Args:
        min_weight: Minimum weight to include (default 3)
        
    Returns:
        Tuple of (formatted_keywords_string, error_message)
    """
    loader = get_keyword_loader()
    return loader.format_for_prompt(min_weight=min_weight)
