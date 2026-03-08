"""
Run Artifact Manager - Saves all artifacts from each execution for verification
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List


class RunArtifactManager:
    """Manages saving all artifacts from a single run to a dedicated folder"""
    
    def __init__(self, base_path: str = ".cache/runs"):
        self.base_path = base_path
        self.run_folder: Optional[str] = None
        self.run_id: Optional[str] = None
        
    def _sanitize_name(self, name: str) -> str:
        """Convert a string to a safe folder name"""
        name = re.sub(r'[^\w\s-]', '', name.lower())
        name = re.sub(r'[-\s]+', '_', name)
        return name[:50]
    
    def start_run(self, product_url: str, brand: str = "unknown") -> str:
        """
        Initialize a new run folder
        
        Args:
            product_url: The product URL being processed
            brand: The brand name (if known)
            
        Returns:
            Path to the run folder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_brand = self._sanitize_name(brand) if brand else "unknown"
        
        self.run_id = f"{timestamp}_{safe_brand}"
        self.run_folder = os.path.join(self.base_path, self.run_id)
        
        os.makedirs(self.run_folder, exist_ok=True)
        os.makedirs(os.path.join(self.run_folder, "images"), exist_ok=True)
        
        metadata = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "product_url": product_url,
            "brand": brand
        }
        self._save_json("metadata.json", metadata)
        
        return self.run_folder
    
    def _save_json(self, filename: str, data: Any) -> str:
        """Save data as JSON file"""
        if not self.run_folder:
            return ""
        
        filepath = os.path.join(self.run_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath
    
    def _save_text(self, filename: str, text: str) -> str:
        """Save text to file"""
        if not self.run_folder:
            return ""
        
        filepath = os.path.join(self.run_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        return filepath
    
    def _save_image(self, filename: str, image_bytes: bytes) -> str:
        """Save image bytes to file"""
        if not self.run_folder:
            return ""
        
        filepath = os.path.join(self.run_folder, "images", filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        return filepath
    
    def save_scraped_urls(self, html_urls: List[str], filtered_urls: List[str], gemini_urls: List[str]) -> str:
        """Save all URLs found during scraping"""
        data = {
            "html_scraped_urls": html_urls,
            "filtered_urls_too_small": filtered_urls,
            "gemini_fallback_urls": gemini_urls
        }
        return self._save_json("scraped_urls.json", data)
    
    def save_product_data(self, product_data: Dict[str, Any]) -> str:
        """Save Phase 1 product data from scraping"""
        return self._save_json("product_data.json", product_data)
    
    def save_brief(self, brief: Dict[str, Any]) -> str:
        """Save Phase 2 campaign brief"""
        return self._save_json("brief.json", brief)
    
    def save_blueprint(self, blueprint: Dict[str, Any]) -> str:
        """Save Phase 3 campaign blueprint"""
        return self._save_json("blueprint.json", blueprint)
    
    def save_nano_banana_prompt(self, prompt: str) -> str:
        """Save the final nano_banana_prompt as text file"""
        return self._save_text("nano_banana_prompt.txt", prompt)
    
    def save_input_image(self, name: str, image_bytes: bytes) -> str:
        """Save an input image (talent, product, logo)"""
        extension = self._detect_image_extension(image_bytes)
        filename = f"input_{name}{extension}"
        return self._save_image(filename, image_bytes)
    
    def save_downloaded_image(self, index: int, image_bytes: bytes, source_url: str = "") -> str:
        """Save a downloaded product image"""
        extension = self._detect_image_extension(image_bytes)
        filename = f"downloaded_product_{index}{extension}"
        filepath = self._save_image(filename, image_bytes)
        
        if source_url:
            urls_file = os.path.join(self.run_folder, "images", "source_urls.txt")
            with open(urls_file, 'a', encoding='utf-8') as f:
                f.write(f"{filename}: {source_url}\n")
        
        return filepath
    
    def save_output_image(self, image_bytes: bytes, format_name: str = "output") -> str:
        """Save the generated output image"""
        extension = self._detect_image_extension(image_bytes)
        filename = f"output_{format_name}{extension}"
        return self._save_image(filename, image_bytes)
    
    def _detect_image_extension(self, image_bytes: bytes) -> str:
        """Detect image format from bytes"""
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return ".png"
        elif image_bytes[:2] == b'\xff\xd8':
            return ".jpg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return ".gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return ".webp"
        else:
            return ".png"
    
    def save_execution_log(self, log_text: str) -> str:
        """Save the complete execution log"""
        return self._save_text("execution_log.txt", log_text)
    
    def get_run_folder(self) -> Optional[str]:
        """Get the current run folder path"""
        return self.run_folder
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all saved artifacts"""
        if not self.run_folder:
            return {}
        
        artifacts = {
            "run_folder": self.run_folder,
            "files": []
        }
        
        for root, dirs, files in os.walk(self.run_folder):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, self.run_folder)
                size = os.path.getsize(filepath)
                artifacts["files"].append({
                    "path": rel_path,
                    "size_bytes": size
                })
        
        return artifacts
