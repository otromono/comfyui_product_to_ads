"""
Run Logger for Product to Ads Node
Provides structured logging and execution summary
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List


class RunLogger:
    """Logger for tracking node execution and generating summaries"""
    
    def __init__(self):
        self.start_time = time.time()
        self.logs: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        self.product_scrape_result: Optional[Dict[str, Any]] = None
        self.product_data: Optional[Dict[str, Any]] = None
        self.prompt_info: Optional[Dict[str, Any]] = None
        self.image_info: Optional[Dict[str, Any]] = None
        self.blueprint_result: Optional[Dict[str, Any]] = None
        self.nano_banana_result: Optional[Dict[str, Any]] = None
        self.cache_status: str = "Not Used"
        
        self._finalized = False
        self._summary = ""
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(time.time() - self.start_time, 3),
            "level": level,
            "message": message
        }
        self.logs.append(entry)
        print(f"[ProductToAds] [{level}] {message}")
    
    def add_error(self, message: str):
        """Add an error"""
        self.errors.append(message)
        self.log(message, level="ERROR")
    
    def add_warning(self, message: str):
        """Add a warning"""
        self.warnings.append(message)
        self.log(message, level="WARN")
    
    def set_product_scrape_result(
        self,
        model_id: str,
        latency: float,
        success: bool,
        product_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Set product scraping result (Phase 1)"""
        self.product_scrape_result = {
            "model": model_id,
            "latency": round(latency, 3),
            "success": success,
            "error": error
        }
        
        if success and product_data:
            self.product_data = {
                "title": str(product_data.get("title", "N/A"))[:100],
                "brand": product_data.get("brand", "N/A"),
                "price": product_data.get("price", "N/A"),
                "rating": product_data.get("rating", "N/A"),
                "review_count": product_data.get("review_count", 0),
                "category": product_data.get("category", "N/A"),
                "image_count": len(product_data.get("image_urls", [])),
                "asin": product_data.get("asin", "N/A"),
                "sentiment": str(product_data.get("sentiment", "N/A"))[:200]
            }
            self.log(f"Product scraped in {latency:.2f}s: {self.product_data['title'][:50]}...")
            self.log(f"Found {self.product_data['image_count']} product images")
        else:
            self.add_error(f"Product scraping failed: {error}")
    
    def set_prompt_info(self, profile: str, prompt_pack: str = "Product_to_Ads"):
        """Set prompt profile information"""
        self.prompt_info = {
            "prompt_pack": prompt_pack,
            "profile": profile
        }
        self.log(f"Using profile: {profile}")
    
    def set_image_info(self, images: Dict[str, Any]):
        """Set image information"""
        self.image_info = images
        provided = [k for k, v in images.items() if v.get("provided", False)]
        self.log(f"Images provided: {', '.join(provided) if provided else 'None'}")
    
    def set_cache_status(self, status: str):
        """Set cache status"""
        self.cache_status = status
        self.log(f"Cache status: {status}")
    
    def set_blueprint_result(
        self,
        model_id: str,
        latency: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Set Blueprint generation result (Phase 2)"""
        self.blueprint_result = {
            "model": model_id,
            "latency": round(latency, 3),
            "success": success,
            "error": error
        }
        if success:
            self.log(f"Campaign Blueprint generated in {latency:.2f}s")
        else:
            self.add_error(f"Blueprint generation failed: {error}")
    
    def set_nano_banana_result(
        self,
        model_id: str,
        latency: float,
        success: bool,
        aspect_ratio: str,
        resolution: str,
        top_p: float,
        error: Optional[str] = None
    ):
        """Set Nano Banana Pro result"""
        self.nano_banana_result = {
            "model": model_id,
            "latency": round(latency, 3),
            "success": success,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "top_p": top_p,
            "error": error
        }
        if success:
            self.log(f"Nano Banana Pro completed in {latency:.2f}s ({aspect_ratio}, {resolution})")
        else:
            self.add_error(f"Nano Banana Pro failed: {error}")
    
    def set_nano_banana_multi_result(self, results: List[Dict[str, Any]]):
        """Set multiple Nano Banana Pro results (for auto mode)"""
        self.nano_banana_result = {
            "multi_format": True,
            "results": results
        }
        success_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)
        self.log(f"Nano Banana Pro multi-format: {success_count}/{total_count} successful")
    
    def save_nano_banana_prompt(self, prompt: str):
        """Save the generated nano banana prompt"""
        self.log(f"Generated prompt length: {len(prompt)} chars")
    
    def finalize(self):
        """Finalize the log and generate summary"""
        if self._finalized:
            return
        
        self._finalized = True
        total_time = time.time() - self.start_time
        
        lines = []
        lines.append("=" * 50)
        lines.append("PRODUCT TO ADS - EXECUTION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Total Time: {total_time:.2f}s")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append("")
        
        if self.product_scrape_result:
            lines.append("--- Phase 1: Product Scraping ---")
            lines.append(f"Model: {self.product_scrape_result['model']}")
            lines.append(f"Latency: {self.product_scrape_result['latency']}s")
            lines.append(f"Status: {'Success' if self.product_scrape_result['success'] else 'Failed'}")
            if self.product_scrape_result.get('error'):
                lines.append(f"Error: {self.product_scrape_result['error']}")
            lines.append("")
        
        if self.product_data:
            lines.append("--- Scraped Product Data ---")
            lines.append(f"Title: {self.product_data['title']}")
            lines.append(f"Brand: {self.product_data['brand']}")
            lines.append(f"Price: {self.product_data['price']}")
            lines.append(f"Rating: {self.product_data['rating']} ({self.product_data['review_count']} reviews)")
            lines.append(f"Category: {self.product_data['category']}")
            lines.append(f"Images Found: {self.product_data['image_count']}")
            lines.append(f"ASIN: {self.product_data['asin']}")
            lines.append(f"Sentiment: {self.product_data['sentiment']}")
            lines.append("")
        
        if self.prompt_info:
            lines.append("--- Prompt Profile ---")
            lines.append(f"Pack: {self.prompt_info['prompt_pack']}")
            lines.append(f"Profile: {self.prompt_info['profile']}")
            lines.append(f"Cache: {self.cache_status}")
            lines.append("")
        
        if self.blueprint_result:
            lines.append("--- Phase 2: Blueprint Generation ---")
            lines.append(f"Model: {self.blueprint_result['model']}")
            lines.append(f"Latency: {self.blueprint_result['latency']}s")
            lines.append(f"Status: {'Success' if self.blueprint_result['success'] else 'Failed'}")
            if self.blueprint_result.get('error'):
                lines.append(f"Error: {self.blueprint_result['error']}")
            lines.append("")
        
        if self.nano_banana_result:
            lines.append("--- Phase 3: Image Generation ---")
            if self.nano_banana_result.get("multi_format"):
                for r in self.nano_banana_result["results"]:
                    status = "OK" if r.get("success") else "FAILED"
                    lines.append(f"  {r.get('aspect_ratio', 'N/A')}: {status} ({r.get('latency', 0):.2f}s)")
            else:
                lines.append(f"Model: {self.nano_banana_result['model']}")
                lines.append(f"Latency: {self.nano_banana_result['latency']}s")
                lines.append(f"Aspect Ratio: {self.nano_banana_result['aspect_ratio']}")
                lines.append(f"Resolution: {self.nano_banana_result['resolution']}")
                lines.append(f"Status: {'Success' if self.nano_banana_result['success'] else 'Failed'}")
                if self.nano_banana_result.get('error'):
                    lines.append(f"Error: {self.nano_banana_result['error']}")
            lines.append("")
        
        if self.errors:
            lines.append("--- Errors ---")
            for error in self.errors:
                lines.append(f"  ! {error}")
            lines.append("")
        
        if self.warnings:
            lines.append("--- Warnings ---")
            for warning in self.warnings:
                lines.append(f"  ? {warning}")
            lines.append("")
        
        lines.append("=" * 50)
        
        self._summary = "\n".join(lines)
        self.log("Execution finalized")
    
    def get_summary(self) -> str:
        """Get the execution summary"""
        if not self._finalized:
            self.finalize()
        return self._summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Export log data as dictionary"""
        return {
            "total_time": time.time() - self.start_time,
            "product_scrape_result": self.product_scrape_result,
            "product_data": self.product_data,
            "prompt_info": self.prompt_info,
            "image_info": self.image_info,
            "blueprint_result": self.blueprint_result,
            "nano_banana_result": self.nano_banana_result,
            "cache_status": self.cache_status,
            "errors": self.errors,
            "warnings": self.warnings,
            "logs": self.logs
        }
