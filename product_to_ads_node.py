"""
Product to Ads Nodes for ComfyUI
Main node implementations for generating advertising images from product pages
"""

import json
from typing import Tuple, Optional, Dict, Any, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .gemini_client import GeminiClient, ASPECT_RATIOS, RESOLUTIONS, GEMINI_FLASH_MODEL, NANO_BANANA_MODEL
from .config_loader import get_prompt_profiles, load_master_prompt, get_brand_profiles, load_brand_identity, load_brief_prompt
from .image_utils import tensor_to_bytes, bytes_to_tensor, create_empty_tensor
from .cache_manager import CacheManager
from .logger import RunLogger
from .image_scraper import scrape_product_images
from .run_artifact_manager import RunArtifactManager
from .brief_manager import BriefManager
from .keyword_loader import load_keywords_for_prompt


_gemini_client: Optional[GeminiClient] = None
_cache_manager: Optional[CacheManager] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


def get_cache_manager() -> CacheManager:
    """Get or create the cache manager singleton"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


class ProductToAdsManualNode:
    """
    Product to Ads - Manual Mode
    
    Generates a single advertising image using:
    - Product URL for data scraping via Gemini
    - Brand Identity for campaign brief generation
    - User-provided images (talent, product 1-4, logo)
    - Gemini Flash for brief and blueprint generation
    - Nano Banana Pro for image generation
    
    4-Phase Pipeline:
    1. Gemini scrapes product page → extracts data + image URLs
    2. Gemini generates Campaign Brief using Brand Identity + Product Data
    3. Gemini generates Blueprint JSON using Brief + Master Prompt
    4. Nano Banana Pro generates final image
    """
    
    CATEGORY = "Product Ads"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "log", "product_data_json", "brief_json", "blueprint_json", "nanobana_request_json")
    
    @classmethod
    def INPUT_TYPES(cls):
        profiles = get_prompt_profiles()
        brands = get_brand_profiles()
        
        return {
            "required": {
                "product_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "https://www.amazon.com/dp/... or any product page"
                }),
                "brand_profile": (brands, {
                    "default": brands[0] if brands else "No Brand"
                }),
                "prompt_profile": (profiles, {
                    "default": profiles[0] if profiles else "Master_prompt_01_Awareness"
                }),
                "gemini_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "password": True
                }),
                "api_key_status": ("STRING", {
                    "multiline": False,
                    "default": "Not Verified",
                    "display": "text"
                }),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "rerun_nonce": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1
                }),
                "min_image_size": (["500", "1000", "1500"], {"default": "1000"}),
            },
            "optional": {
                "talent_image": ("IMAGE",),
                "product_image_1": ("IMAGE",),
                "product_image_2": ("IMAGE",),
                "product_image_3": ("IMAGE",),
                "product_image_4": ("IMAGE",),
                "brand_logo": ("IMAGE",),
                "pose_ref_img": ("IMAGE",),
                "photo_style_ref": ("IMAGE",),
                "location_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, rerun_nonce, **kwargs):
        """Force re-execution when rerun_nonce changes"""
        return rerun_nonce
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        return True
    
    def execute(
        self,
        product_url: str,
        brand_profile: str,
        prompt_profile: str,
        gemini_api_key: str,
        api_key_status: str,
        aspect_ratio: str,
        resolution: str,
        top_p: float,
        rerun_nonce: int = 0,
        min_image_size: str = "1000",
        talent_image=None,
        product_image_1=None,
        product_image_2=None,
        product_image_3=None,
        product_image_4=None,
        brand_logo=None,
        pose_ref_img=None,
        photo_style_ref=None,
        location_ref=None,
        unique_id: str = ""
    ) -> Tuple:
        """Execute the manual node with 4-phase pipeline"""
        
        logger = RunLogger()
        logger.log(f"Starting ProductToAds Manual execution (nonce: {rerun_nonce})")
        logger.log(f"Brand: {brand_profile}, Profile: {prompt_profile}")
        
        force_regenerate = True
        brief_json = "{}"
        
        gemini_client = get_gemini_client()
        if gemini_api_key:
            gemini_client.set_api_key(gemini_api_key)
        
        if not gemini_client.api_key:
            error_msg = "No Gemini API key configured"
            logger.add_error(error_msg)
            logger.finalize()
            return (create_empty_tensor(), logger.get_summary(), "{}", "{}", "{}", "{}")
        
        is_valid, verify_msg = gemini_client.verify_api_key()
        if not is_valid:
            logger.add_error(f"Gemini API key verification failed: {verify_msg}")
            logger.finalize()
            return (create_empty_tensor(), logger.get_summary(), "{}", "{}", "{}", "{}")
        
        logger.log(f"Gemini API key verified: {gemini_client.status}")
        logger.set_prompt_info(prompt_profile, "Product_to_Ads")
        
        artifact_manager = RunArtifactManager()
        
        logger.log("=== PHASE 1: Product Scraping ===")
        logger.log(f"Scraping product page: {product_url[:80]}...")
        
        product_data, scrape_status, scrape_latency, scrape_request = gemini_client.scrape_product_page(product_url)
        
        if product_data is None:
            logger.set_product_scrape_result(
                model_id=GEMINI_FLASH_MODEL,
                latency=scrape_latency,
                success=False,
                error=scrape_status
            )
            logger.finalize()
            return (create_empty_tensor(), logger.get_summary(), "{}", "{}", "{}", "{}")
        
        logger.set_product_scrape_result(
            model_id=GEMINI_FLASH_MODEL,
            latency=scrape_latency,
            success=True,
            product_data=product_data
        )
        
        product_data_json = json.dumps(product_data, indent=2, ensure_ascii=False)
        
        brand_name = product_data.get("brand", "unknown")
        run_folder = artifact_manager.start_run(product_url, brand_name)
        logger.log(f"Artifacts folder: {run_folder}")
        
        artifact_manager.save_product_data(product_data)
        
        images = {}
        
        if talent_image is not None:
            talent_bytes = tensor_to_bytes(talent_image)
            if talent_bytes:
                images["talent"] = talent_bytes
                artifact_manager.save_input_image("talent", talent_bytes)
                logger.log(f"Talent image: {len(talent_bytes)} bytes")
        
        if product_image_1 is not None:
            prod1_bytes = tensor_to_bytes(product_image_1)
            if prod1_bytes:
                images["product_1"] = prod1_bytes
                artifact_manager.save_input_image("product_1", prod1_bytes)
                logger.log(f"Product image 1: {len(prod1_bytes)} bytes")
        
        if product_image_2 is not None:
            prod2_bytes = tensor_to_bytes(product_image_2)
            if prod2_bytes:
                images["product_2"] = prod2_bytes
                artifact_manager.save_input_image("product_2", prod2_bytes)
                logger.log(f"Product image 2: {len(prod2_bytes)} bytes")
        
        if product_image_3 is not None:
            prod3_bytes = tensor_to_bytes(product_image_3)
            if prod3_bytes:
                images["product_3"] = prod3_bytes
                artifact_manager.save_input_image("product_3", prod3_bytes)
                logger.log(f"Product image 3: {len(prod3_bytes)} bytes")
        
        if product_image_4 is not None:
            prod4_bytes = tensor_to_bytes(product_image_4)
            if prod4_bytes:
                images["product_4"] = prod4_bytes
                artifact_manager.save_input_image("product_4", prod4_bytes)
                logger.log(f"Product image 4: {len(prod4_bytes)} bytes")
        
        if brand_logo is not None:
            logo_bytes = tensor_to_bytes(brand_logo)
            if logo_bytes:
                images["logo"] = logo_bytes
                artifact_manager.save_input_image("logo", logo_bytes)
                logger.log(f"Brand logo: {len(logo_bytes)} bytes")
        
        reference_images = {}
        
        if pose_ref_img is not None:
            pose_bytes = tensor_to_bytes(pose_ref_img)
            if pose_bytes:
                reference_images["pose_ref"] = pose_bytes
                artifact_manager.save_input_image("pose_ref", pose_bytes)
                logger.log(f"Pose reference: {len(pose_bytes)} bytes")
        
        if photo_style_ref is not None:
            style_bytes = tensor_to_bytes(photo_style_ref)
            if style_bytes:
                reference_images["photo_style_ref"] = style_bytes
                artifact_manager.save_input_image("photo_style_ref", style_bytes)
                logger.log(f"Photo style reference: {len(style_bytes)} bytes")
        
        if location_ref is not None:
            location_bytes = tensor_to_bytes(location_ref)
            if location_bytes:
                reference_images["location_ref"] = location_bytes
                artifact_manager.save_input_image("location_ref", location_bytes)
                logger.log(f"Location reference: {len(location_bytes)} bytes")
        
        if reference_images:
            logger.log(f"Creative references provided: {list(reference_images.keys())}")
        
        user_images = {k: v for k, v in images.items()}
        
        min_dim = int(min_image_size)
        scraped_urls = []
        missing_slots = []
        for i in range(1, 5):
            if not images.get(f"product_{i}"):
                missing_slots.append(f"product_{i}")
        
        if missing_slots:
            logger.log(f"Scraping product images from HTML (min {min_dim}px)...")
            html_urls, filtered_urls, scrape_error, product_code = scrape_product_images(
                product_url,
                min_dimension=min_dim,
                max_images=4
            )
            
            if product_code:
                logger.log(f"Detected product code: {product_code}")
            
            if scrape_error:
                logger.log(f"HTML scraping issue: {scrape_error}")
            
            if filtered_urls:
                logger.log(f"Filtered {len(filtered_urls)} images < {min_dim}px")
            
            scraped_urls = html_urls[:len(missing_slots)] if html_urls else []
            
            if not scraped_urls:
                gemini_urls = product_data.get("image_urls", [])
                valid_urls, _ = gemini_client.validate_image_urls(gemini_urls, referer_url=product_url)
                scraped_urls = valid_urls[:len(missing_slots)] if valid_urls else []
            
            artifact_manager.save_scraped_urls(html_urls, filtered_urls, product_data.get("image_urls", []))
        
        cache_manager = get_cache_manager()
        cached_prompt, cache_status = cache_manager.get_cached_prompt(
            unique_id, product_url, prompt_profile, user_images,
            scraped_urls=scraped_urls,
            force_regenerate=force_regenerate
        )
        logger.set_cache_status(cache_status)
        
        if missing_slots and scraped_urls:
            logger.log(f"Downloading {len(scraped_urls)} images for missing slots: {missing_slots}")
            for url in scraped_urls[:4]:
                logger.log(f"  - {url[:80]}...")
            
            for i, url in enumerate(scraped_urls):
                if i >= len(missing_slots):
                    break
                slot = missing_slots[i]
                slot_num = int(slot.split("_")[-1])
                cached_img = cache_manager.get_cached_image(url)
                if cached_img:
                    images[slot] = cached_img
                    artifact_manager.save_downloaded_image(slot_num, cached_img, url)
                    logger.log(f"{slot} from cache: {len(cached_img)} bytes")
                else:
                    downloaded, _ = gemini_client.download_images_from_urls(
                        [url], max_images=1, referer_url=product_url
                    )
                    if downloaded:
                        img_bytes = downloaded[0]
                        images[slot] = img_bytes
                        cache_manager.save_image_to_cache(url, img_bytes)
                        artifact_manager.save_downloaded_image(slot_num, img_bytes, url)
                        logger.log(f"Downloaded {slot}: {len(img_bytes)} bytes")
                    else:
                        logger.log(f"Failed to download {slot}: {url[:60]}...")
        
        if not images.get("product_1"):
            logger.add_warning("No product images found via HTML or Gemini")
        
        image_info = {name: {"provided": True, "size_bytes": len(img)} for name, img in images.items()}
        logger.set_image_info(image_info)
        
        brief_manager = BriefManager()
        brief_result = None
        brand_identity = None
        
        if brand_profile and brand_profile != "No Brand":
            logger.log("=== PHASE 2: Campaign Brief Generation ===")
            
            brand_identity, brand_error = load_brand_identity(brand_profile)
            if brand_error:
                logger.add_warning(f"Brand identity load issue: {brand_error}")
            else:
                logger.log(f"Loaded brand identity: {brand_profile}")
            
            brief_prompt, brief_error = load_brief_prompt()
            if brief_error:
                logger.add_warning(f"Brief prompt load issue: {brief_error}")
            
            if brand_identity:
                logger.log("Calling Gemini Flash for campaign brief...")
                
                product_images = {k: v for k, v in images.items() if k.startswith("product_")}
                
                brief_result, brief_response, brief_latency, brief_request = gemini_client.generate_brief(
                    brief_prompt=brief_prompt or "",
                    brand_identity=brand_identity,
                    product_data=product_data,
                    product_images=product_images if product_images else None,
                    reference_images=reference_images if reference_images else None,
                    max_retries=1
                )
                
                if brief_result:
                    brief_json = json.dumps(brief_result, indent=2, ensure_ascii=False)
                    logger.log(f"Brief generated in {brief_latency:.2f}s")
                    if reference_images:
                        logger.log(f"Brief includes analysis of references: {list(reference_images.keys())}")
                    
                    brief_file, brief_log = brief_manager.save_brief(
                        brand_name=brand_profile,
                        brief_data=brief_result,
                        product_data=product_data,
                        product_url=product_url
                    )
                    if brief_file:
                        logger.log(brief_log)
                    
                    artifact_manager.save_brief(brief_result)
                else:
                    logger.add_warning(f"Brief generation failed: {brief_response}")
        else:
            logger.log("No brand selected, skipping brief generation (Phase 2)")
        
        blueprint_result = None
        if cached_prompt and not force_regenerate:
            blueprint_result = cached_prompt
            blueprint_json = json.dumps(blueprint_result, indent=2, ensure_ascii=False)
            logger.log("Using cached blueprint")
        else:
            logger.log("=== PHASE 3: Blueprint Generation ===")
            
            master_prompt, load_error = load_master_prompt(prompt_profile)
            if load_error:
                logger.add_warning(f"Master prompt load issue: {load_error}")
            
            if master_prompt:
                master_prompt = master_prompt.replace("{{PRODUCT_JSON}}", product_data_json)
                
                funnel_stage = prompt_profile.split("_")[-1] if "_" in prompt_profile else "Awareness"
                master_prompt = master_prompt.replace("{{FUNNEL_STAGE}}", funnel_stage)
                
                if brief_result:
                    master_prompt = master_prompt.replace("{{CAMPAIGN_BRIEF}}", brief_json)
                else:
                    master_prompt = master_prompt.replace("{{CAMPAIGN_BRIEF}}", "{}")
                
                keywords_text, kw_error = load_keywords_for_prompt(min_weight=3)
                if kw_error:
                    logger.add_warning(f"Keyword loading issue: {kw_error}")
                    master_prompt = master_prompt.replace("{{KEYWORD_BANK}}", "")
                else:
                    master_prompt = master_prompt.replace("{{KEYWORD_BANK}}", keywords_text)
                    logger.log("Keyword bank loaded and injected into prompt")
                
                master_prompt = master_prompt.replace("{{FORMAT}}", aspect_ratio)
            
            logger.log("Calling Gemini Flash for campaign blueprint...")
            
            blueprint_result, blueprint_response, blueprint_latency, blueprint_request = gemini_client.generate_campaign_blueprint(
                system_instruction=master_prompt or "",
                product_data=product_data,
                images=images,
                reference_images=reference_images if reference_images else None,
                max_retries=1
            )
            
            if blueprint_result is None:
                logger.set_blueprint_result(
                    model_id=GEMINI_FLASH_MODEL,
                    latency=blueprint_latency,
                    success=False,
                    error=blueprint_response
                )
                logger.finalize()
                blueprint_json = json.dumps({"error": blueprint_response}, indent=2)
                return (create_empty_tensor(), logger.get_summary(), product_data_json, brief_json, blueprint_json, "{}")
            
            logger.set_blueprint_result(
                model_id=GEMINI_FLASH_MODEL,
                latency=blueprint_latency,
                success=True
            )
            
            blueprint_json = json.dumps(blueprint_result, indent=2, ensure_ascii=False)
            
            artifact_manager.save_blueprint(blueprint_result)
            
            cache_manager.save_prompt(
                unique_id, product_url, prompt_profile, user_images,
                blueprint_result, scraped_urls=scraped_urls, product_data=product_data
            )
            logger.log("Blueprint cached for future re-runs")
        
        logger.log("=== PHASE 4: Image Generation ===")
        logger.log("Calling Nano Banana Pro for image generation...")
        
        binding_images = [img for img in images.values() if img is not None]
        
        image_bytes, gen_status, gen_latency, nanobana_request = gemini_client.generate_image(
            prompt="",
            reference_images=binding_images if binding_images else None,
            soft_reference_images=reference_images if reference_images else None,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            top_p=top_p,
            blueprint_json=blueprint_result
        )
        
        nanobana_request_json = json.dumps(nanobana_request, indent=2, ensure_ascii=False)
        
        if image_bytes is None:
            logger.set_nano_banana_result(
                model_id=NANO_BANANA_MODEL,
                latency=gen_latency,
                success=False,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                top_p=top_p,
                error=gen_status
            )
            logger.finalize()
            return (create_empty_tensor(), logger.get_summary(), product_data_json, brief_json, blueprint_json, nanobana_request_json)
        
        logger.set_nano_banana_result(
            model_id=NANO_BANANA_MODEL,
            latency=gen_latency,
            success=True,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            top_p=top_p
        )
        
        try:
            output_tensor = bytes_to_tensor(image_bytes)
            if output_tensor is None:
                raise ValueError("Failed to convert image bytes to tensor")
            
            artifact_manager.save_output_image(image_bytes, aspect_ratio.replace(":", "x"))
            
        except Exception as e:
            logger.add_error(f"Image conversion failed: {str(e)}")
            logger.finalize()
            return (create_empty_tensor(), logger.get_summary(), product_data_json, brief_json, blueprint_json, nanobana_request_json)
        
        artifact_manager.save_execution_log(logger.get_summary())
        run_folder = artifact_manager.get_run_folder()
        logger.log(f"All artifacts saved to: {run_folder}")
        logger.finalize()
        
        return (output_tensor, logger.get_summary(), product_data_json, brief_json, blueprint_json, nanobana_request_json)


class ProductToAdsAutoNode:
    """
    Product to Ads - Auto Mode
    
    Generates 4 advertising images in different formats using:
    - Product URL for data scraping AND product images via Gemini
    - Brand Identity for campaign brief generation
    - User-provided images (talent, logo only)
    - Gemini Flash for brief and blueprint generation
    - Nano Banana Pro for image generation in 4 aspect ratios
    
    4-Phase Pipeline:
    1. Gemini scrapes product page → extracts data + image URLs
    2. Gemini generates Campaign Brief using Brand Identity + Product Data
    3. Gemini generates Blueprint JSON using Brief + Master Prompt
    4. Nano Banana Pro generates final images (4 formats)
    
    Outputs: 4 images (1:1, 4:5, 5:4, 9:16) + brief_json
    """
    
    CATEGORY = "Product Ads"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_1x1", "image_4x5", "image_5x4", "image_9x16", "log", "product_data_json", "brief_json", "blueprint_json", "nanobana_request_json")
    
    AUTO_FORMATS = ["1:1", "4:5", "5:4", "9:16"]
    
    @classmethod
    def INPUT_TYPES(cls):
        profiles = get_prompt_profiles()
        brands = get_brand_profiles()
        
        return {
            "required": {
                "product_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "https://www.amazon.com/dp/... or any product page"
                }),
                "brand_profile": (brands, {
                    "default": brands[0] if brands else "No Brand"
                }),
                "prompt_profile": (profiles, {
                    "default": profiles[0] if profiles else "Master_prompt_01_Awareness"
                }),
                "gemini_api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "password": True
                }),
                "api_key_status": ("STRING", {
                    "multiline": False,
                    "default": "Not Verified",
                    "display": "text"
                }),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "min_image_size": (["500", "1000", "1500"], {
                    "default": "1000"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "rerun_nonce": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1
                }),
            },
            "optional": {
                "talent_image": ("IMAGE",),
                "brand_logo": ("IMAGE",),
                "pose_ref_img": ("IMAGE",),
                "photo_style_ref": ("IMAGE",),
                "location_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, rerun_nonce, **kwargs):
        """Force re-execution when rerun_nonce changes"""
        return rerun_nonce
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        return True
    
    def execute(
        self,
        product_url: str,
        brand_profile: str,
        prompt_profile: str,
        gemini_api_key: str,
        api_key_status: str,
        resolution: str,
        min_image_size: str,
        top_p: float,
        rerun_nonce: int = 0,
        talent_image=None,
        brand_logo=None,
        pose_ref_img=None,
        photo_style_ref=None,
        location_ref=None,
        unique_id: str = ""
    ) -> Tuple:
        """Execute the auto node with 4-phase pipeline and 4 output formats"""
        
        logger = RunLogger()
        logger.log(f"Starting ProductToAds Auto execution (nonce: {rerun_nonce})")
        logger.log(f"Brand: {brand_profile}, Profile: {prompt_profile}")
        logger.log(f"Will generate 4 formats: {', '.join(self.AUTO_FORMATS)}")
        
        force_regenerate = True
        brief_json = "{}"
        
        gemini_client = get_gemini_client()
        if gemini_api_key:
            gemini_client.set_api_key(gemini_api_key)
        
        if not gemini_client.api_key:
            error_msg = "No Gemini API key configured"
            logger.add_error(error_msg)
            logger.finalize()
            empty = create_empty_tensor()
            return (empty, empty, empty, empty, logger.get_summary(), "{}", "{}", "{}", "{}")
        
        is_valid, verify_msg = gemini_client.verify_api_key()
        if not is_valid:
            logger.add_error(f"Gemini API key verification failed: {verify_msg}")
            logger.finalize()
            empty = create_empty_tensor()
            return (empty, empty, empty, empty, logger.get_summary(), "{}", "{}", "{}", "{}")
        
        logger.log(f"Gemini API key verified: {gemini_client.status}")
        logger.set_prompt_info(prompt_profile, "Product_to_Ads")
        
        artifact_manager = RunArtifactManager()
        
        logger.log("=== PHASE 1: Product Scraping ===")
        logger.log(f"Scraping product page: {product_url[:80]}...")
        
        product_data, scrape_status, scrape_latency, scrape_request = gemini_client.scrape_product_page(product_url)
        
        if product_data is None:
            logger.set_product_scrape_result(
                model_id=GEMINI_FLASH_MODEL,
                latency=scrape_latency,
                success=False,
                error=scrape_status
            )
            logger.finalize()
            empty = create_empty_tensor()
            return (empty, empty, empty, empty, logger.get_summary(), "{}", "{}", "{}", "{}")
        
        logger.set_product_scrape_result(
            model_id=GEMINI_FLASH_MODEL,
            latency=scrape_latency,
            success=True,
            product_data=product_data
        )
        
        product_data_json = json.dumps(product_data, indent=2, ensure_ascii=False)
        
        brand_name = product_data.get("brand", "unknown")
        run_folder = artifact_manager.start_run(product_url, brand_name)
        logger.log(f"Artifacts folder: {run_folder}")
        
        artifact_manager.save_product_data(product_data)
        
        images = {}
        
        user_images = {}
        
        if talent_image is not None:
            talent_bytes = tensor_to_bytes(talent_image)
            if talent_bytes:
                images["talent"] = talent_bytes
                user_images["talent"] = talent_bytes
                artifact_manager.save_input_image("talent", talent_bytes)
                logger.log(f"Talent image: {len(talent_bytes)} bytes")
        
        if brand_logo is not None:
            logo_bytes = tensor_to_bytes(brand_logo)
            if logo_bytes:
                images["logo"] = logo_bytes
                user_images["logo"] = logo_bytes
                artifact_manager.save_input_image("logo", logo_bytes)
                logger.log(f"Brand logo: {len(logo_bytes)} bytes")
        
        reference_images = {}
        
        if pose_ref_img is not None:
            pose_bytes = tensor_to_bytes(pose_ref_img)
            if pose_bytes:
                reference_images["pose_ref"] = pose_bytes
                artifact_manager.save_input_image("pose_ref", pose_bytes)
                logger.log(f"Pose reference: {len(pose_bytes)} bytes")
        
        if photo_style_ref is not None:
            style_bytes = tensor_to_bytes(photo_style_ref)
            if style_bytes:
                reference_images["photo_style_ref"] = style_bytes
                artifact_manager.save_input_image("photo_style_ref", style_bytes)
                logger.log(f"Photo style reference: {len(style_bytes)} bytes")
        
        if location_ref is not None:
            location_bytes = tensor_to_bytes(location_ref)
            if location_bytes:
                reference_images["location_ref"] = location_bytes
                artifact_manager.save_input_image("location_ref", location_bytes)
                logger.log(f"Location reference: {len(location_bytes)} bytes")
        
        if reference_images:
            logger.log(f"Creative references provided: {list(reference_images.keys())}")
        
        min_dim = int(min_image_size)
        logger.log(f"Scraping product images from HTML (min {min_dim}px)...")
        html_urls, filtered_urls, scrape_error, product_code = scrape_product_images(
            product_url,
            min_dimension=min_dim,
            max_images=4
        )
        
        if product_code:
            logger.log(f"Detected product code: {product_code}")
        
        if scrape_error:
            logger.log(f"HTML scraping issue: {scrape_error}")
        
        if filtered_urls:
            logger.log(f"Filtered {len(filtered_urls)} images < {min_dim}px")
        
        scraped_urls = html_urls[:4] if html_urls else []
        
        if not scraped_urls and product_data.get("image_urls"):
            gemini_urls = product_data.get("image_urls", [])
            valid_urls, _ = gemini_client.validate_image_urls(gemini_urls, referer_url=product_url)
            scraped_urls = valid_urls[:4] if valid_urls else []
        
        artifact_manager.save_scraped_urls(html_urls, filtered_urls, product_data.get("image_urls", []))
        
        cache_manager = get_cache_manager()
        cached_prompt, cache_status = cache_manager.get_cached_prompt(
            unique_id, product_url, prompt_profile, user_images,
            scraped_urls=scraped_urls,
            force_regenerate=force_regenerate
        )
        logger.set_cache_status(cache_status)
        
        if html_urls:
            logger.log(f"Found {len(html_urls)} product images via HTML scraping")
            for url in html_urls[:4]:
                logger.log(f"  - {url[:80]}...")
            
            for i, url in enumerate(html_urls[:4]):
                cached_img = cache_manager.get_cached_image(url)
                if cached_img:
                    images[f"product_{i+1}"] = cached_img
                    artifact_manager.save_downloaded_image(i+1, cached_img, url)
                    logger.log(f"Product image {i+1} from cache: {len(cached_img)} bytes")
                else:
                    downloaded, _ = gemini_client.download_images_from_urls(
                        [url], max_images=1, referer_url=product_url
                    )
                    if downloaded:
                        img_bytes = downloaded[0]
                        images[f"product_{i+1}"] = img_bytes
                        cache_manager.save_image_to_cache(url, img_bytes)
                        artifact_manager.save_downloaded_image(i+1, img_bytes, url)
                        logger.log(f"Downloaded product image {i+1}: {len(img_bytes)} bytes")
                    else:
                        logger.log(f"Failed to download image {i+1}: {url[:60]}...")
        
        if not images.get("product_1") and product_data.get("image_urls"):
            logger.log("HTML scraping failed, trying Gemini URLs as fallback...")
            gemini_urls = product_data.get("image_urls", [])
            valid_urls, invalid_urls = gemini_client.validate_image_urls(
                gemini_urls,
                referer_url=product_url
            )
            if invalid_urls:
                logger.log(f"Filtered {len(invalid_urls)} hallucinated URLs from Gemini")
            if valid_urls:
                for i, url in enumerate(valid_urls[:4]):
                    cached_img = cache_manager.get_cached_image(url)
                    if cached_img:
                        images[f"product_{i+1}"] = cached_img
                        artifact_manager.save_downloaded_image(i+1, cached_img, url)
                        logger.log(f"Fallback image {i+1} from cache: {len(cached_img)} bytes")
                    else:
                        downloaded, _ = gemini_client.download_images_from_urls(
                            [url], max_images=1, referer_url=product_url
                        )
                        if downloaded:
                            img_bytes = downloaded[0]
                            images[f"product_{i+1}"] = img_bytes
                            cache_manager.save_image_to_cache(url, img_bytes)
                            artifact_manager.save_downloaded_image(i+1, img_bytes, url)
                            logger.log(f"Downloaded (fallback) image {i+1}: {len(img_bytes)} bytes")
        
        if "product_1" not in images:
            logger.add_warning("AUTO MODE: No product images available. Image quality may be affected.")
        
        image_info = {name: {"provided": True, "size_bytes": len(img)} for name, img in images.items()}
        logger.set_image_info(image_info)
        logger.set_cache_status(cache_status)
        
        brief_manager = BriefManager()
        brief_result = None
        brand_identity = None
        
        if brand_profile and brand_profile != "No Brand":
            logger.log("=== PHASE 2: Campaign Brief Generation ===")
            
            brand_identity, brand_error = load_brand_identity(brand_profile)
            if brand_error:
                logger.add_warning(f"Brand identity load issue: {brand_error}")
            else:
                logger.log(f"Loaded brand identity: {brand_profile}")
            
            brief_prompt, brief_error = load_brief_prompt()
            if brief_error:
                logger.add_warning(f"Brief prompt load issue: {brief_error}")
            
            if brand_identity:
                logger.log("Calling Gemini Flash for campaign brief...")
                
                product_images = {k: v for k, v in images.items() if k.startswith("product_")}
                
                brief_result, brief_response, brief_latency, brief_request = gemini_client.generate_brief(
                    brief_prompt=brief_prompt or "",
                    brand_identity=brand_identity,
                    product_data=product_data,
                    product_images=product_images if product_images else None,
                    reference_images=reference_images if reference_images else None,
                    max_retries=1
                )
                
                if brief_result:
                    brief_json = json.dumps(brief_result, indent=2, ensure_ascii=False)
                    logger.log(f"Brief generated in {brief_latency:.2f}s")
                    if reference_images:
                        logger.log(f"Brief includes analysis of references: {list(reference_images.keys())}")
                    
                    brief_file, brief_log = brief_manager.save_brief(
                        brand_name=brand_profile,
                        brief_data=brief_result,
                        product_data=product_data,
                        product_url=product_url
                    )
                    if brief_file:
                        logger.log(brief_log)
                    
                    artifact_manager.save_brief(brief_result)
                else:
                    logger.add_warning(f"Brief generation failed: {brief_response}")
        else:
            logger.log("No brand selected, skipping brief generation (Phase 2)")
        
        blueprint_result = None
        if cached_prompt and not force_regenerate:
            blueprint_result = cached_prompt
            blueprint_json = json.dumps(blueprint_result, indent=2, ensure_ascii=False)
            logger.log("Using cached blueprint")
        else:
            logger.log("=== PHASE 3: Blueprint Generation ===")
            
            master_prompt, load_error = load_master_prompt(prompt_profile)
            if load_error:
                logger.add_warning(f"Master prompt load issue: {load_error}")
            
            if master_prompt:
                master_prompt = master_prompt.replace("{{PRODUCT_JSON}}", product_data_json)
                
                funnel_stage = prompt_profile.split("_")[-1] if "_" in prompt_profile else "Awareness"
                master_prompt = master_prompt.replace("{{FUNNEL_STAGE}}", funnel_stage)
                
                if brief_result:
                    master_prompt = master_prompt.replace("{{CAMPAIGN_BRIEF}}", brief_json)
                else:
                    master_prompt = master_prompt.replace("{{CAMPAIGN_BRIEF}}", "{}")
                
                keywords_text, kw_error = load_keywords_for_prompt(min_weight=3)
                if kw_error:
                    logger.add_warning(f"Keyword loading issue: {kw_error}")
                    master_prompt = master_prompt.replace("{{KEYWORD_BANK}}", "")
                else:
                    master_prompt = master_prompt.replace("{{KEYWORD_BANK}}", keywords_text)
                    logger.log("Keyword bank loaded and injected into prompt")
                
                master_prompt = master_prompt.replace("{{FORMAT}}", "multi-format")
            
            logger.log("Calling Gemini Flash for campaign blueprint...")
            
            blueprint_result, blueprint_response, blueprint_latency, blueprint_request = gemini_client.generate_campaign_blueprint(
                system_instruction=master_prompt or "",
                product_data=product_data,
                images=images,
                reference_images=reference_images if reference_images else None,
                max_retries=1
            )
            
            if blueprint_result is None:
                logger.set_blueprint_result(
                    model_id=GEMINI_FLASH_MODEL,
                    latency=blueprint_latency,
                    success=False,
                    error=blueprint_response
                )
                logger.finalize()
                empty = create_empty_tensor()
                blueprint_json = json.dumps({"error": blueprint_response}, indent=2)
                return (empty, empty, empty, empty, logger.get_summary(), product_data_json, brief_json, blueprint_json, "{}")
            
            logger.set_blueprint_result(
                model_id=GEMINI_FLASH_MODEL,
                latency=blueprint_latency,
                success=True
            )
            
            blueprint_json = json.dumps(blueprint_result, indent=2, ensure_ascii=False)
            
            artifact_manager.save_blueprint(blueprint_result)
            
            cache_manager.save_prompt(
                unique_id, product_url, prompt_profile, user_images,
                blueprint_result, scraped_urls=scraped_urls, product_data=product_data
            )
            logger.log("Blueprint cached for future re-runs")
        
        logger.log("=== PHASE 4: Multi-Format Image Generation ===")
        logger.log("Calling Nano Banana Pro for 4 format generation...")
        
        binding_images = [img for img in images.values() if img is not None]
        
        generation_results = gemini_client.generate_multiple_formats(
            prompt="",
            reference_images=binding_images if binding_images else None,
            soft_reference_images=reference_images if reference_images else None,
            formats=self.AUTO_FORMATS,
            resolution=resolution,
            top_p=top_p,
            blueprint_json=blueprint_result
        )
        
        output_tensors = []
        all_requests = []
        result_logs = []
        
        for aspect_ratio, image_bytes, gen_status, gen_latency, request in generation_results:
            all_requests.append({
                "aspect_ratio": aspect_ratio,
                "status": gen_status,
                "latency": gen_latency,
                "request": request
            })
            
            result_logs.append({
                "aspect_ratio": aspect_ratio,
                "success": image_bytes is not None,
                "status": gen_status,
                "latency": round(gen_latency, 3)
            })
            
            if image_bytes:
                try:
                    tensor = bytes_to_tensor(image_bytes)
                    if tensor is not None:
                        output_tensors.append(tensor)
                        artifact_manager.save_output_image(image_bytes, aspect_ratio.replace(":", "x"))
                    else:
                        output_tensors.append(create_empty_tensor())
                except Exception as e:
                    logger.add_warning(f"Image conversion failed for {aspect_ratio}: {e}")
                    output_tensors.append(create_empty_tensor())
            else:
                logger.add_warning(f"Generation failed for {aspect_ratio}: {gen_status}")
                output_tensors.append(create_empty_tensor())
        
        logger.set_nano_banana_multi_result(result_logs)
        
        nanobana_request_json = json.dumps(all_requests, indent=2, ensure_ascii=False)
        
        while len(output_tensors) < 4:
            output_tensors.append(create_empty_tensor())
        
        artifact_manager.save_execution_log(logger.get_summary())
        run_folder = artifact_manager.get_run_folder()
        logger.log(f"All artifacts saved to: {run_folder}")
        logger.finalize()
        
        return (
            output_tensors[0],
            output_tensors[1],
            output_tensors[2],
            output_tensors[3],
            logger.get_summary(),
            product_data_json,
            brief_json,
            blueprint_json,
            nanobana_request_json
        )
