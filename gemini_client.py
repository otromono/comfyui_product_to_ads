"""
Gemini API Client for Product to Ads Node
Handles both Gemini Flash (text orchestration) and Nano Banana Pro (image generation)
"""

import os
import base64
import json
import time
from typing import Optional, Dict, Any, List, Tuple

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

try:
    from .playwright_scraper import scrape_product_images as playwright_scrape_images
    PLAYWRIGHT_SCRAPER_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_SCRAPER_AVAILABLE = False
    playwright_scrape_images = None

try:
    from .google_image_scraper import scrape_product_via_google
    GOOGLE_SCRAPER_AVAILABLE = True
except ImportError:
    GOOGLE_SCRAPER_AVAILABLE = False
    scrape_product_via_google = None


GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
NANO_BANANA_MODEL = "gemini-3-pro-image-preview"


def extract_product_code_from_url(url: str) -> Optional[str]:
    """Extract product code/SKU from URL patterns common in e-commerce sites"""
    import re
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    path = parsed.path
    
    patterns = [
        r'/([A-Z0-9]{10,20})\.html',
        r'/([A-Z0-9]{8,15}[A-Z][A-Z0-9]{3,8})(?:\.|/|$)',
        r'/dp/([A-Z0-9]{10})',
        r'/product[s]?/([A-Z0-9_-]{6,20})',
        r'/itm/([0-9]{10,15})',
        r'/item/([0-9]{10,15})',
        r'/listing/([0-9]{8,15})',
        r'-p-([A-Z0-9]{10,20})(?:\?|$)',
        r'/p/([A-Z0-9-]{8,20})',
        r'/sku[=:]([A-Z0-9-]{6,20})',
        r'[?&]id=([A-Z0-9-]{6,20})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path + '?' + (parsed.query or ''), re.IGNORECASE)
        if match:
            code = match.group(1)
            if len(code) >= 6 and not code.isdigit():
                return code
            elif len(code) >= 10:
                return code
    
    return None

ASPECT_RATIOS = ["1:1", "4:5", "5:4", "9:16", "16:9", "2:3", "3:2", "3:4", "4:3", "21:9"]
RESOLUTIONS = ["1K", "2K", "4K"]

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}

try:
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


class GeminiClient:
    """Client for interacting with Gemini Flash and Nano Banana Pro APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._status = "Not Configured"
        
        if not GENAI_AVAILABLE:
            self._status = "Error: google-genai not installed"
            return
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize the Gemini client with API key"""
        if not GENAI_AVAILABLE:
            self._status = "Error: google-genai not installed"
            return
            
        try:
            self.client = genai.Client(api_key=self.api_key)
            self._status = "Configured"
        except Exception as e:
            self._status = f"Error: {str(e)}"
            self.client = None
    
    def set_api_key(self, api_key: str):
        """Set or update the API key"""
        self.api_key = api_key
        self._init_client()
    
    @property
    def status(self) -> str:
        """Get current client status"""
        return self._status
    
    def verify_api_key(self) -> Tuple[bool, str]:
        """Verify the API key is valid by making a test request"""
        if not GENAI_AVAILABLE:
            self._status = "Error: google-genai not installed"
            return False, "google-genai library not installed"
            
        if not self.api_key:
            self._status = "No API Key"
            return False, "No API key provided"
        
        if not self.client:
            self._init_client()
            if not self.client:
                return False, self._status
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Test"
            )
            if response and response.text:
                self._status = "Valid"
                return True, "API key verified successfully"
            else:
                self._status = "Invalid Response"
                return False, "API key verification failed: empty response"
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "invalid" in error_msg.lower():
                self._status = "Invalid"
            elif "quota" in error_msg.lower():
                self._status = "Quota Exceeded"
            else:
                self._status = "Error"
            return False, f"API key verification failed: {error_msg}"
    
    def _create_image_part(self, image_bytes: bytes, mime_type: str = "image/png"):
        """Create an image part for the API request"""
        if not GENAI_AVAILABLE:
            return None
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON object from text that may contain extra content before/after.
        Finds the outermost matching braces.
        """
        # Find the first '{' and last '}'
        start = text.find('{')
        if start == -1:
            return text
        
        # Count braces to find matching closing brace
        depth = 0
        end = -1
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end == -1:
            return text
        
        return text[start:end+1]
    
    def scrape_product_page(
        self,
        product_url: str
    ) -> Tuple[Optional[Dict[str, Any]], str, float, Dict[str, Any]]:
        """
        Scrape product page using hybrid approach:
        - Playwright for reliable image extraction from real DOM
        - Gemini Flash for text analysis (title, description, features, etc.)
        
        Args:
            product_url: URL of the product page (Amazon, eBay, etc.)
            
        Returns:
            Tuple of (product_data, status, latency, request_payload)
        """
        if not self.client:
            return None, "Client not initialized", 0.0, {}
        
        start_time = time.time()
        scraped_images = []
        scraper_status = "Not attempted"
        image_source = "none"
        product_name_for_search = ""
        
        if PLAYWRIGHT_SCRAPER_AVAILABLE and playwright_scrape_images:
            try:
                scraped_images, scraper_status = playwright_scrape_images(product_url, max_images=10)
                if scraped_images:
                    image_source = "playwright"
            except Exception as e:
                scraper_status = f"Playwright error: {str(e)}"
        
        system_instruction = """You are a product data extraction expert. Analyze the provided product page URL and extract all relevant information.

You must respond with ONLY a valid JSON object in this exact format:
{
    "title": "Full product title",
    "brand": "Brand name",
    "sku": "Product code, SKU, or model number (look for alphanumeric codes like 8BN321AWQOF1VED, B08N5WRWNW, etc.)",
    "price": "Price with currency symbol",
    "category": "Product category",
    "description": "Product description",
    "bullet_points": ["Feature 1", "Feature 2", "..."],
    "rating": "Rating value (e.g., 4.5)",
    "review_count": 123,
    "reviews_summary": "Brief summary of customer sentiment from reviews",
    "sentiment": "Overall sentiment analysis (Positive/Negative/Mixed with explanation)",
    "image_urls": ["url1", "url2", "..."],
    "specifications": {"key": "value"},
    "asin": "ASIN if Amazon product, otherwise null"
}

IMPORTANT: Always try to extract the product SKU/code. Look for:
- Product codes in the URL (e.g., /8BN321AWQOF1VED.html)
- Model numbers, reference codes, or SKUs on the page
- Amazon ASINs (10-character alphanumeric codes)

IMPORTANT: Focus on extracting TEXT content accurately. The image_urls field is optional - 
if you cannot find clear, unambiguous image URLs, return an empty array [].
DO NOT invent or guess image URLs.

If any field is not available, use null or empty array.
Do not include any other text, explanation, or markdown formatting."""

        user_content = f"Analyze this product page and extract all product data: {product_url}"
        
        request_payload = {
            "model": GEMINI_FLASH_MODEL,
            "product_url": product_url,
            "scraped_images_found": len(scraped_images),
            "scraper_status": scraper_status,
            "image_source": image_source,
            "config": {
                "temperature": 0.3,
                "max_output_tokens": 4096
            }
        }
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=4096
                )
            )
            
            latency = time.time() - start_time
            
            if not response or not response.text:
                return None, "Empty response from Gemini", latency, request_payload
            
            response_text = response.text.strip()
            
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)
            
            try:
                product_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON response: {e}", latency, request_payload
            
            if scraped_images:
                product_data["image_urls"] = scraped_images
                product_data["image_source"] = image_source
            else:
                product_title = product_data.get("title", "")
                product_brand = product_data.get("brand", "")
                product_sku = product_data.get("sku") or product_data.get("asin") or ""
                
                if not product_sku:
                    product_sku = extract_product_code_from_url(product_url) or ""
                
                if product_sku:
                    product_name_for_search = f"{product_brand} {product_sku}".strip()
                else:
                    product_name_for_search = f"{product_brand} {product_title}".strip()
                
                if product_name_for_search and GOOGLE_SCRAPER_AVAILABLE and scrape_product_via_google:
                    try:
                        google_images, google_status = scrape_product_via_google(
                            product_url, product_name_for_search, max_images=10
                        )
                        if google_images:
                            scraped_images = google_images
                            image_source = "google_images"
                            scraper_status = google_status
                    except Exception as e:
                        scraper_status = f"Google Images error: {str(e)}"
                
                if scraped_images:
                    product_data["image_urls"] = scraped_images
                    product_data["image_source"] = image_source
                else:
                    gemini_urls = product_data.get("image_urls", [])
                    if gemini_urls:
                        product_data["image_source"] = "gemini_fallback"
                    else:
                        product_data["image_source"] = "none"
            
            status_msg = f"Success (images: {product_data.get('image_source', 'unknown')})"
            return product_data, status_msg, latency, request_payload
            
        except Exception as e:
            latency = time.time() - start_time
            return None, f"Scraping error: {str(e)}", latency, request_payload
    
    def download_images_from_urls(
        self,
        image_urls: List[str],
        max_images: int = 5,
        referer_url: Optional[str] = None,
        min_size: Tuple[int, int] = (200, 200),
        max_retries: int = 2
    ) -> Tuple[List[bytes], List[str]]:
        """
        Download images from URLs with proper headers to bypass anti-bot protections.
        Uses robust retry logic and image validation.
        
        Args:
            image_urls: List of image URLs
            max_images: Maximum number of images to download
            referer_url: Original page URL to use as Referer header
            min_size: Minimum (width, height) to consider valid
            max_retries: Number of retry attempts per URL
            
        Returns:
            Tuple of (list of image bytes, list of error messages)
        """
        import requests
        from io import BytesIO
        
        try:
            from PIL import Image
            PIL_AVAILABLE = True
        except ImportError:
            PIL_AVAILABLE = False
        
        images = []
        errors = []
        url_index = 0
        
        while len(images) < max_images and url_index < len(image_urls):
            url = image_urls[url_index]
            url_index += 1
            
            success = False
            for attempt in range(max_retries + 1):
                try:
                    parsed = urlparse(url)
                    domain = f"{parsed.scheme}://{parsed.netloc}"
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': referer_url or 'https://www.google.com/',
                        'Origin': domain
                    }
                    
                    response = requests.get(url, headers=headers, timeout=15, stream=True)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        errors.append(f"Hotlink protected: {url[:60]}...")
                        break
                    
                    data = response.content
                    
                    if len(data) < 1000:
                        errors.append(f"Image too small (bytes): {url[:60]}...")
                        break
                    
                    if PIL_AVAILABLE:
                        try:
                            img_buffer = BytesIO(data)
                            img = Image.open(img_buffer)
                            img.verify()
                            img_buffer.seek(0)
                            img = Image.open(img_buffer)
                            
                            width, height = img.size
                            if width < min_size[0] or height < min_size[1]:
                                errors.append(f"Image too small ({width}x{height}): {url[:50]}...")
                                break
                            
                            if img.format and img.format.upper() == 'WEBP':
                                img = img.convert('RGB')
                                output = BytesIO()
                                img.save(output, 'JPEG', quality=95)
                                data = output.getvalue()
                        except Exception as e:
                            errors.append(f"Invalid image: {str(e)[:30]}")
                            break
                    
                    images.append(data)
                    success = True
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    errors.append(f"Timeout: {url[:60]}...")
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else 0
                    if status_code == 403:
                        errors.append(f"Forbidden (hotlink): {url[:50]}...")
                    elif status_code == 404:
                        errors.append(f"Not found: {url[:60]}...")
                    else:
                        errors.append(f"HTTP {status_code}: {url[:50]}...")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    errors.append(f"Error: {str(e)[:40]}")
        
        return images, errors
    
    def validate_image_urls(
        self,
        image_urls: List[str],
        referer_url: Optional[str] = None,
        timeout: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Validate image URLs by making HEAD requests to check they exist
        
        Args:
            image_urls: List of image URLs to validate
            referer_url: Original page URL to use as Referer header
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (valid_urls, invalid_urls)
        """
        if not URLLIB_AVAILABLE:
            return image_urls, []
        
        valid_urls = []
        invalid_urls = []
        
        for url in image_urls:
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    invalid_urls.append(url)
                    continue
                
                domain = f"{parsed.scheme}://{parsed.netloc}"
                
                headers = DEFAULT_HEADERS.copy()
                headers['Referer'] = referer_url or domain
                headers['Origin'] = domain
                headers['Host'] = parsed.netloc
                
                req = urllib.request.Request(url, headers=headers, method='HEAD')
                
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = resp.getcode()
                    content_type = resp.headers.get('Content-Type', '')
                    
                    if status == 200 and content_type.startswith('image/'):
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)
                        
            except urllib.error.HTTPError as e:
                if e.code == 405:
                    try:
                        req_get = urllib.request.Request(url, headers=headers, method='GET')
                        with urllib.request.urlopen(req_get, timeout=timeout) as resp:
                            content_type = resp.headers.get('Content-Type', '')
                            if content_type.startswith('image/'):
                                valid_urls.append(url)
                            else:
                                invalid_urls.append(url)
                    except:
                        invalid_urls.append(url)
                else:
                    invalid_urls.append(url)
            except Exception:
                invalid_urls.append(url)
        
        return valid_urls, invalid_urls
    
    def generate_campaign_blueprint(
        self,
        system_instruction: str,
        product_data: Dict[str, Any],
        images: Dict[str, bytes],
        reference_images: Optional[Dict[str, bytes]] = None,
        max_retries: int = 1
    ) -> Tuple[Optional[Dict[str, Any]], str, float, Dict[str, Any]]:
        """
        Call Gemini Flash to generate the campaign blueprint
        
        Args:
            system_instruction: The master prompt system instruction
            product_data: Dictionary containing product data from scraping
            images: Dictionary of image name -> bytes
            reference_images: Optional dict of reference images (pose_ref, photo_style_ref, location_ref)
            max_retries: Number of retry attempts
            
        Returns:
            Tuple of (parsed_blueprint, response_text, latency, request_payload)
        """
        if not self.client:
            return None, "Client not initialized", 0.0, {}
        
        asset_manifest = []
        for name, img_bytes in images.items():
            if img_bytes:
                asset_manifest.append({"name": name, "provided": True, "size_bytes": len(img_bytes), "type": "binding"})
            else:
                asset_manifest.append({"name": name, "provided": False})
        
        if reference_images:
            for name, img_bytes in reference_images.items():
                if img_bytes:
                    asset_manifest.append({"name": name, "provided": True, "size_bytes": len(img_bytes), "type": "reference"})
        
        product_data_with_assets = {
            **product_data,
            "assets": asset_manifest
        }
        
        product_json = json.dumps(product_data_with_assets, indent=2, ensure_ascii=False)
        
        user_content = f"""<product_data>
{product_json}
</product_data>

Based on the product data above, execute the full campaign blueprint generation following your system instructions.

Generate ONLY the JSON Blueprint. No additional text, explanations, or markdown formatting outside the JSON.
"""
        
        contents = []
        images_info = []
        
        for name, img_bytes in images.items():
            if img_bytes:
                contents.append(self._create_image_part(img_bytes))
                images_info.append({"name": name, "size_bytes": len(img_bytes), "mime_type": "image/png", "type": "binding"})
        
        if reference_images:
            for name, img_bytes in reference_images.items():
                if img_bytes:
                    contents.append(self._create_image_part(img_bytes))
                    images_info.append({"name": name, "size_bytes": len(img_bytes), "mime_type": "image/png", "type": "reference"})
        
        contents.append(user_content)
        
        request_payload = {
            "model": GEMINI_FLASH_MODEL,
            "system_instruction": system_instruction[:500] + "..." if len(system_instruction) > 500 else system_instruction,
            "product_data": product_data_with_assets,
            "user_content": user_content,
            "images": images_info,
            "config": {
                "temperature": 0.7,
                "max_output_tokens": 8192
            }
        }
        
        start_time = time.time()
        last_error = ""
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        max_output_tokens=4096
                    )
                )
                
                latency = time.time() - start_time
                
                if not response or not response.text:
                    last_error = "Empty response from Gemini Flash"
                    continue
                
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    response_text = "\n".join(lines)
                
                # Extract JSON from response (handles text before/after JSON)
                json_text = self._extract_json_from_text(response_text)
                
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON response: {e}"
                    continue
                
                return parsed, json_text, latency, request_payload
                
            except Exception as e:
                last_error = str(e)
                latency = time.time() - start_time
        
        return None, last_error, time.time() - start_time, request_payload
    
    def generate_brief(
        self,
        brief_prompt: str,
        brand_identity: Dict[str, Any],
        product_data: Dict[str, Any],
        product_images: Optional[Dict[str, bytes]] = None,
        reference_images: Optional[Dict[str, bytes]] = None,
        max_retries: int = 1
    ) -> Tuple[Optional[Dict[str, Any]], str, float, Dict[str, Any]]:
        """
        Generate a campaign brief using brand identity and product data (Phase 2)
        
        Args:
            brief_prompt: The brief generation system instruction
            brand_identity: Brand identity JSON configuration
            product_data: Product data from Phase 1 scraping
            product_images: Optional dict of product image name -> bytes
            reference_images: Optional dict of reference images (pose_ref, photo_style_ref, location_ref)
            max_retries: Number of retry attempts
            
        Returns:
            Tuple of (parsed_brief, response_text, latency, request_payload)
        """
        if not self.client:
            return None, "Client not initialized", 0.0, {}
        
        brand_json = json.dumps(brand_identity, indent=2, ensure_ascii=False)
        product_json = json.dumps(product_data, indent=2, ensure_ascii=False)
        
        system_instruction = brief_prompt.replace("{{BRAND_IDENTITY}}", brand_json)
        system_instruction = system_instruction.replace("{{PRODUCT_JSON}}", product_json)
        
        ref_section = ""
        if reference_images:
            ref_types = []
            if "pose_ref" in reference_images:
                ref_types.append("POSE_REF: Analyze this pose reference - the pose must be REPLICATED EXACTLY in the final image.")
            if "photo_style_ref" in reference_images:
                ref_types.append("PHOTO_STYLE_REF: Analyze this photography style reference - derive camera, lighting, color grading, grain, and mood.")
            if "location_ref" in reference_images:
                ref_types.append("LOCATION_REF: Analyze this location reference - create a similar environment with creative enhancement.")
            
            if ref_types:
                ref_section = "\n\n<creative_references>\n" + "\n".join(ref_types) + "\n</creative_references>"
        
        user_content = f"""Generate a comprehensive Campaign Brief following the framework.

<product_data>
{product_json}
</product_data>

<brand_identity>
{brand_json}
</brand_identity>{ref_section}

Analyze the product, brand identity, and any provided creative reference images.
Generate the complete Campaign Brief JSON.
Output ONLY the JSON object, no additional text or explanation.
"""
        
        contents = []
        images_info = []
        
        if product_images:
            for name, img_bytes in product_images.items():
                if img_bytes:
                    contents.append(self._create_image_part(img_bytes))
                    images_info.append({"name": name, "size_bytes": len(img_bytes), "type": "product"})
        
        if reference_images:
            for name, img_bytes in reference_images.items():
                if img_bytes:
                    contents.append(self._create_image_part(img_bytes))
                    images_info.append({"name": name, "size_bytes": len(img_bytes), "type": "reference"})
        
        contents.append(user_content)
        
        request_payload = {
            "model": GEMINI_FLASH_MODEL,
            "phase": "brief_generation",
            "brand_name": brand_identity.get("brand_info", {}).get("name", "Unknown"),
            "product_title": product_data.get("title", "Unknown"),
            "images_count": len(images_info),
            "config": {
                "temperature": 0.7,
                "max_output_tokens": 8192
            }
        }
        
        start_time = time.time()
        last_error = ""
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        max_output_tokens=8192
                    )
                )
                
                latency = time.time() - start_time
                
                if not response or not response.text:
                    last_error = "Empty response from Gemini Flash"
                    continue
                
                response_text = response.text.strip()
                
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    response_text = "\n".join(lines)
                
                json_text = self._extract_json_from_text(response_text)
                
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON in brief response: {e}"
                    continue
                
                return parsed, json_text, latency, request_payload
                
            except Exception as e:
                last_error = str(e)
                latency = time.time() - start_time
        
        return None, last_error, time.time() - start_time, request_payload
    
    def generate_image(
        self,
        prompt: str,
        reference_images: Optional[List[bytes]] = None,
        soft_reference_images: Optional[Dict[str, bytes]] = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        top_p: float = 0.95,
        blueprint_json: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[bytes], str, float, Dict[str, Any]]:
        """
        Generate an image using Nano Banana Pro (gemini-3-pro-image-preview)
        
        Args:
            prompt: The generation prompt (used if blueprint_json not provided)
            reference_images: List of BINDING reference image bytes (product/talent - strictly enforced)
            soft_reference_images: Dict of SOFT reference images (pose_ref, photo_style_ref, location_ref - inspirational only)
            aspect_ratio: Output aspect ratio
            resolution: Output resolution
            top_p: Top-p sampling parameter
            blueprint_json: Complete blueprint dict to send as structured prompt
            
        Returns:
            Tuple of (image_bytes, status, latency, request_payload)
        """
        if not self.client:
            return None, "Client not initialized", 0.0, {}
        
        if aspect_ratio not in ASPECT_RATIOS:
            aspect_ratio = "1:1"
        
        if resolution not in RESOLUTIONS:
            resolution = "1K"
        
        if blueprint_json:
            actual_prompt = json.dumps(blueprint_json, indent=2, ensure_ascii=False)
            prompt_for_log = blueprint_json
        else:
            actual_prompt = prompt
            prompt_for_log = prompt[:500] + "..." if len(prompt) > 500 else prompt
        
        ref_images_info = []
        if reference_images:
            for i, img_bytes in enumerate(reference_images):
                if img_bytes:
                    ref_images_info.append({"index": i, "size_bytes": len(img_bytes), "mime_type": "image/png", "type": "binding"})
        
        soft_ref_images_info = []
        if soft_reference_images:
            for name, img_bytes in soft_reference_images.items():
                if img_bytes:
                    soft_ref_images_info.append({"name": name, "size_bytes": len(img_bytes), "mime_type": "image/png", "type": "soft_reference"})
        
        request_payload = {
            "model": NANO_BANANA_MODEL,
            "prompt": prompt_for_log,
            "binding_images_count": len(ref_images_info),
            "binding_images": ref_images_info,
            "soft_reference_images_count": len(soft_ref_images_info),
            "soft_reference_images": soft_ref_images_info,
            "config": {
                "response_modalities": ["IMAGE", "TEXT"],
                "top_p": top_p,
                "image_config": {
                    "aspect_ratio": aspect_ratio,
                    "image_size": resolution
                }
            }
        }
        
        start_time = time.time()
        
        try:
            contents = []
            
            if reference_images:
                for img_bytes in reference_images:
                    if img_bytes:
                        contents.append(self._create_image_part(img_bytes))
            
            if soft_reference_images:
                for ref_name, img_bytes in soft_reference_images.items():
                    if img_bytes:
                        contents.append(self._create_image_part(img_bytes))
            
            contents.append(actual_prompt)
            
            response = self.client.models.generate_content(
                model=NANO_BANANA_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    top_p=top_p,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    )
                )
            )
            
            latency = time.time() - start_time
            
            if not response or not response.candidates:
                return None, "No response from Nano Banana Pro", latency, request_payload
            
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return None, "No content parts in response", latency, request_payload
            
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.mime_type and part.inline_data.mime_type.startswith("image/"):
                    image_data = part.inline_data.data
                    return image_data, "Success", latency, request_payload
            
            return None, "No image in response", latency, request_payload
            
        except Exception as e:
            error_msg = str(e)
            latency = time.time() - start_time
            
            if "safety" in error_msg.lower():
                return None, f"Safety filter triggered: {error_msg}", latency, request_payload
            elif "quota" in error_msg.lower():
                return None, f"Quota exceeded: {error_msg}", latency, request_payload
            else:
                return None, f"Generation error: {error_msg}", latency, request_payload
    
    def generate_multiple_formats(
        self,
        prompt: str,
        reference_images: Optional[List[bytes]] = None,
        soft_reference_images: Optional[Dict[str, bytes]] = None,
        formats: List[str] = None,
        resolution: str = "1K",
        top_p: float = 0.95,
        blueprint_json: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Optional[bytes], str, float, Dict[str, Any]]]:
        """
        Generate images in multiple aspect ratios using the same prompt
        
        Args:
            prompt: The generation prompt
            reference_images: List of BINDING reference image bytes (product/talent)
            soft_reference_images: Dict of SOFT reference images (pose_ref, photo_style_ref, location_ref)
            formats: List of aspect ratios to generate
            resolution: Output resolution
            top_p: Top-p sampling parameter
            blueprint_json: Complete blueprint dict to send as structured prompt
            
        Returns:
            List of tuples: (aspect_ratio, image_bytes, status, latency, request_payload)
        """
        if formats is None:
            formats = ["1:1", "4:5", "5:4", "9:16"]
        
        results = []
        for aspect_ratio in formats:
            image_bytes, status, latency, request_payload = self.generate_image(
                prompt=prompt,
                reference_images=reference_images,
                soft_reference_images=soft_reference_images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                top_p=top_p,
                blueprint_json=blueprint_json
            )
            results.append((aspect_ratio, image_bytes, status, latency, request_payload))
        
        return results
