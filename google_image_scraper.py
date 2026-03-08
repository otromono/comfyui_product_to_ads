"""
Google Images Scraper for Product Pages.
Uses site: operator to find images from a specific URL via Google Images.
Includes image download functionality with retry and error handling.
"""

import os
import re
import shutil
import time
import hashlib
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlparse
from PIL import Image

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def find_system_chromium() -> Optional[str]:
    """Find system-installed Chromium executable"""
    for name in ['chromium', 'chromium-browser', 'google-chrome', 'chrome']:
        path = shutil.which(name)
        if path:
            return path
    
    import glob
    common_paths = [
        '/nix/store/*/bin/chromium',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
    ]
    for pattern in common_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None


SYSTEM_CHROMIUM = find_system_chromium()


def extract_domain(url: str) -> str:
    """Extract domain from URL for site: search"""
    parsed = urlparse(url)
    return parsed.netloc


def build_google_images_url(product_url: str, search_terms: str = "", use_full_url: bool = False) -> str:
    """Build Google Images search URL with site: operator
    
    Args:
        product_url: The product page URL
        search_terms: Optional additional search terms (product name works best!)
        use_full_url: If True, use full URL in site: operator. If False, use only domain.
    """
    if search_terms:
        query = search_terms
    elif use_full_url:
        query = f"site:{product_url}"
    else:
        domain = extract_domain(product_url)
        query = f"site:{domain}"
    
    encoded_query = quote_plus(query)
    return f"https://www.google.com/search?q={encoded_query}&tbm=isch&udm=2"


def is_valid_product_image(url: str, min_size: int = 200, allow_google_thumbnails: bool = False) -> bool:
    """Check if URL looks like a valid product image"""
    if not url or not url.startswith('http'):
        return False
    
    url_lower = url.lower()
    
    if allow_google_thumbnails and 'encrypted-tbn' in url_lower:
        return True
    
    exclude_patterns = [
        'logo', 'icon', 'sprite', 'avatar', 'profile',
        'banner', 'ad', 'pixel', 'tracking', 'badge',
        'button', 'arrow', 'social', 'share', 'rating',
        'star', 'cart', 'search', 'menu', 'nav',
        'googlelogo', 'gstatic.com/images'
    ]
    
    for pattern in exclude_patterns:
        if pattern in url_lower:
            return False
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    has_valid_ext = any(ext in url_lower for ext in valid_extensions)
    
    known_ecommerce = ['media-amazon', 'ebayimg', 'fendi.com', 'etsy', 'shopify']
    is_ecommerce = any(domain in url_lower for domain in known_ecommerce)
    
    return has_valid_ext or is_ecommerce


def extract_urls_from_page_content(page, target_domain: str = "") -> List[str]:
    """Extract image URLs from page content and JavaScript data"""
    found_urls = []
    
    try:
        content = page.content()
        
        patterns = [
            r'"ou":"(https?://[^"]+)"',
            r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',
            r'data-src="(https?://[^"]+)"',
            r'"(https?://[^"]*\.(?:jpg|jpeg|png|webp)[^"]*)"',
            r'"(https?://[^"]*(?:media-amazon|ebayimg|etsy|cdn|shopify|images)[^"]*)"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if 'google' in match.lower() or 'gstatic' in match.lower():
                    continue
                if 'encrypted-tbn' in match.lower():
                    continue
                if len(match) < 50:
                    continue
                if is_valid_product_image(match, allow_google_thumbnails=False):
                    found_urls.append(match)
    except Exception:
        pass
    
    unique = list(set(found_urls))
    if target_domain:
        domain_urls = [u for u in unique if target_domain.lower() in u.lower()]
        other_urls = [u for u in unique if target_domain.lower() not in u.lower()]
        return domain_urls + other_urls
    
    return unique


def extract_original_url_from_google(page, max_attempts: int = 5, target_domain: str = "") -> List[str]:
    """Extract original image URLs from Google Images using multiple methods"""
    original_urls = []
    
    urls_from_content = extract_urls_from_page_content(page, target_domain)
    original_urls.extend(urls_from_content)
    
    if len(original_urls) >= max_attempts:
        return original_urls[:max_attempts]
    
    try:
        thumbnails = page.query_selector_all('img[alt]:not([src*="google"]):not([src*="gstatic"])')
        valid_thumbnails = []
        for t in thumbnails:
            src = t.get_attribute('src') or ''
            alt = t.get_attribute('alt') or ''
            if src.startswith('data:image') and alt and len(alt) > 5:
                valid_thumbnails.append(t)
        
        for i, thumb in enumerate(valid_thumbnails[:max_attempts]):
            try:
                thumb.click(force=True, timeout=3000)
                time.sleep(2.5)
                
                new_urls = extract_urls_from_page_content(page, target_domain)
                original_urls.extend(new_urls)
                
                page.keyboard.press('Escape')
                time.sleep(0.3)
                
            except Exception:
                continue
    except Exception:
        pass
    
    return list(set(original_urls))


def scrape_google_images(
    product_url: str,
    search_terms: str = "",
    max_images: int = 10,
    timeout: int = 30000
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Scrape images from Google Images using site: operator.
    
    Args:
        product_url: The product page URL (used for site: filter)
        search_terms: Optional additional search terms (e.g., product name)
        max_images: Maximum number of images to return
        timeout: Page load timeout in milliseconds
    
    Returns:
        Tuple of (list of image URLs, metadata dict)
    """
    metadata = {
        "success": False,
        "source": "google_images",
        "product_url": product_url,
        "search_terms": search_terms,
        "images_found": 0,
        "error": None
    }
    
    if not PLAYWRIGHT_AVAILABLE:
        metadata["error"] = "Playwright not available"
        return [], metadata
    
    if not SYSTEM_CHROMIUM:
        metadata["error"] = "Chromium not found"
        return [], metadata
    
    google_url = build_google_images_url(product_url, search_terms)
    metadata["google_url"] = google_url
    
    found_urls = set()
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=SYSTEM_CHROMIUM,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-blink-features=AutomationControlled'
                ]
            )
            
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            page = context.new_page()
            
            try:
                page.goto(google_url, wait_until='networkidle', timeout=timeout)
            except PlaywrightTimeout:
                page.goto(google_url, wait_until='domcontentloaded', timeout=timeout)
            
            time.sleep(2)
            
            for _ in range(3):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                time.sleep(0.5)
            
            thumbnail_urls = []
            image_elements = page.query_selector_all('img[src*="encrypted-tbn"]')
            
            for img in image_elements:
                try:
                    src = img.get_attribute('src')
                    if src and 'encrypted-tbn' in src:
                        thumbnail_urls.append(src)
                except Exception:
                    continue
            
            target_domain = extract_domain(product_url)
            original_urls = extract_original_url_from_google(page, max_attempts=max_images, target_domain=target_domain)
            
            for url in original_urls:
                found_urls.add(url)
            
            if len(found_urls) < max_images:
                for url in thumbnail_urls:
                    if len(found_urls) >= max_images:
                        break
                    found_urls.add(url)
            
            browser.close()
            
            image_list = list(found_urls)[:max_images]
            
            metadata["success"] = len(image_list) > 0
            metadata["images_found"] = len(image_list)
            
            status_msg = f"Found {len(image_list)} images via Google Images"
            if search_terms:
                status_msg += f" for '{search_terms}'"
            
            return image_list, metadata
            
    except Exception as e:
        metadata["error"] = str(e)
        return [], metadata


def scrape_product_via_google(
    product_url: str,
    product_name: str = "",
    max_images: int = 5
) -> Tuple[List[str], str]:
    """
    Main function to scrape product images via Google Images.
    
    Args:
        product_url: The e-commerce product page URL
        product_name: Optional product name for better search results
        max_images: Maximum images to return
    
    Returns:
        Tuple of (list of image URLs, status message)
    """
    images, metadata = scrape_google_images(
        product_url=product_url,
        search_terms=product_name,
        max_images=max_images
    )
    
    if metadata["success"]:
        status = f"Found {len(images)} images via Google Images"
    else:
        error = metadata.get("error", "Unknown error")
        status = f"Google Images scraping failed: {error}"
    
    return images, status


def download_image(
    url: str,
    save_path: str,
    timeout: int = 15,
    min_size: Tuple[int, int] = (200, 200),
    max_retries: int = 2
) -> Tuple[bool, str]:
    """
    Download a single image from URL and save to disk.
    
    Args:
        url: Image URL to download
        save_path: Full path to save the image (without extension)
        timeout: Request timeout in seconds
        min_size: Minimum (width, height) to consider valid
        max_retries: Number of retry attempts
    
    Returns:
        Tuple of (success, status message or file path)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                return False, "URL returned HTML instead of image (hotlink protection)"
            
            img_data = BytesIO(response.content)
            
            try:
                img = Image.open(img_data)
                img.verify()
                img_data.seek(0)
                img = Image.open(img_data)
            except Exception as e:
                return False, f"Invalid image data: {str(e)}"
            
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                return False, f"Image too small: {width}x{height}"
            
            img_format = img.format or 'JPEG'
            ext_map = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif'}
            ext = ext_map.get(img_format.upper(), '.jpg')
            
            final_path = save_path + ext
            
            if img_format.upper() == 'WEBP':
                img = img.convert('RGB')
                final_path = save_path + '.jpg'
                img.save(final_path, 'JPEG', quality=95)
            else:
                img.save(final_path)
            
            return True, final_path
            
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, "Timeout downloading image"
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code == 403:
                return False, "Access forbidden (hotlink protection)"
            elif status_code == 404:
                return False, "Image not found (404)"
            return False, f"HTTP error: {status_code}"
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, f"Download error: {str(e)}"
    
    return False, "Max retries exceeded"


def download_images_from_urls(
    image_urls: List[str],
    save_dir: str,
    prefix: str = "product",
    min_images: int = 1,
    max_images: int = 5,
    min_size: Tuple[int, int] = (300, 300)
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Download multiple images from URLs, with fallback to next URL on failure.
    
    Args:
        image_urls: List of image URLs to try
        save_dir: Directory to save images
        prefix: Filename prefix (e.g., "product" -> product_001.jpg)
        min_images: Minimum required successful downloads
        max_images: Maximum images to download
        min_size: Minimum image dimensions (width, height)
    
    Returns:
        Tuple of (list of saved file paths, metadata dict)
    """
    metadata = {
        "attempted": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "saved_files": []
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = []
    url_index = 0
    image_count = 0
    
    while image_count < max_images and url_index < len(image_urls):
        url = image_urls[url_index]
        url_index += 1
        metadata["attempted"] += 1
        
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{prefix}_{image_count + 1:03d}_{url_hash}"
        save_path = os.path.join(save_dir, filename)
        
        success, result = download_image(url, save_path, min_size=min_size)
        
        if success:
            saved_paths.append(result)
            metadata["saved_files"].append(result)
            metadata["successful"] += 1
            image_count += 1
        else:
            metadata["failed"] += 1
            metadata["errors"].append({"url": url[:100], "error": result})
    
    if len(saved_paths) < min_images and url_index < len(image_urls):
        metadata["errors"].append({
            "warning": f"Only {len(saved_paths)} images downloaded, minimum was {min_images}"
        })
    
    return saved_paths, metadata


def scrape_and_download_product_images(
    product_url: str,
    product_name: str = "",
    save_dir: str = "",
    max_images: int = 5,
    min_size: Tuple[int, int] = (300, 300)
) -> Tuple[List[str], str, Dict[str, Any]]:
    """
    Complete pipeline: search Google Images and download the results.
    
    Args:
        product_url: E-commerce product page URL
        product_name: Product name/brand for search
        save_dir: Directory to save downloaded images
        max_images: Maximum images to download
        min_size: Minimum image dimensions
    
    Returns:
        Tuple of (list of saved file paths, status message, metadata)
    """
    if not save_dir:
        save_dir = os.path.join(os.getcwd(), ".cache", "product_images")
    
    image_urls, search_status = scrape_product_via_google(
        product_url=product_url,
        product_name=product_name,
        max_images=max_images * 3
    )
    
    if not image_urls:
        return [], search_status, {"search_status": search_status}
    
    prefix = re.sub(r'[^\w]+', '_', product_name[:30]) if product_name else "product"
    
    saved_paths, download_meta = download_images_from_urls(
        image_urls=image_urls,
        save_dir=save_dir,
        prefix=prefix,
        min_images=1,
        max_images=max_images,
        min_size=min_size
    )
    
    download_meta["search_status"] = search_status
    download_meta["urls_found"] = len(image_urls)
    
    if saved_paths:
        status = f"Downloaded {len(saved_paths)} images (from {len(image_urls)} found)"
    else:
        status = f"Found {len(image_urls)} URLs but download failed"
    
    return saved_paths, status, download_meta


if __name__ == "__main__":
    test_url = "https://www.amazon.it/dp/B0G662S925"
    print(f"Testing Google Images scraper for: {test_url}")
    
    images, status = scrape_product_via_google(test_url, "BMW M3", max_images=5)
    print(f"Status: {status}")
    print(f"Images found: {len(images)}")
    for i, img in enumerate(images[:3]):
        print(f"  {i+1}. {img[:80]}..." if len(img) > 80 else f"  {i+1}. {img}")
