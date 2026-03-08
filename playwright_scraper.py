"""
Playwright-based image scraper for e-commerce product pages.
Extracts real image URLs from the DOM, handling JavaScript-loaded content.
"""

import os
import re
import shutil
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

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
    
    common_paths = [
        '/nix/store/*/bin/chromium',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
        '/usr/bin/google-chrome'
    ]
    for pattern in common_paths:
        import glob
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None


SYSTEM_CHROMIUM = find_system_chromium()


ECOMMERCE_SELECTORS = {
    "amazon": {
        "product_images": [
            "#imgTagWrapperId img",
            "#landingImage",
            "#main-image-container img",
            ".imgTagWrapper img",
            "#altImages img",
            ".a-dynamic-image",
            "[data-old-hires]",
            ".image-wrapper img"
        ],
        "gallery_click": "#altImages .a-button-thumbnail",
        "scroll_for_images": True
    },
    "ebay": {
        "product_images": [
            ".ux-image-carousel img",
            ".ux-image-magnify img",
            "#mainImgHldr img",
            ".vi-image-gallery img",
            "[data-zoom-src]",
            ".img-item img"
        ],
        "gallery_click": ".ux-image-carousel-item",
        "scroll_for_images": True
    },
    "etsy": {
        "product_images": [
            ".listing-page-image-carousel img",
            ".image-carousel-container img",
            "[data-listing-image-id] img",
            ".carousel-image img"
        ],
        "gallery_click": ".carousel-pagination button",
        "scroll_for_images": True
    },
    "shopify": {
        "product_images": [
            ".product__media img",
            ".product-single__photo img",
            ".product-featured-img",
            "[data-product-featured-image]",
            ".product__main-photos img",
            ".product-gallery img"
        ],
        "gallery_click": ".product__thumbs button",
        "scroll_for_images": True
    },
    "generic": {
        "product_images": [
            "[class*='product'] img[src*='product']",
            "[class*='gallery'] img",
            "[class*='carousel'] img",
            "[id*='product'] img",
            "main img[src]:not([src*='logo']):not([src*='icon'])",
            ".product-image img",
            ".product-gallery img",
            "[data-zoom-image]",
            "[data-large-image]",
            "[data-src]"
        ],
        "gallery_click": None,
        "scroll_for_images": True
    }
}

MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200
EXCLUDED_PATTERNS = [
    r'logo', r'icon', r'sprite', r'banner', r'nav', r'menu',
    r'button', r'arrow', r'close', r'search', r'cart', r'login',
    r'facebook', r'twitter', r'instagram', r'pinterest', r'youtube',
    r'payment', r'visa', r'mastercard', r'paypal', r'badge',
    r'star', r'rating', r'review', r'avatar', r'profile',
    r'placeholder', r'loading', r'spinner', r'1x1', r'pixel',
    r'tracking', r'analytics', r'ad[_-]?', r'advertisement'
]


def detect_platform(url: str) -> str:
    """Detect e-commerce platform from URL"""
    domain = urlparse(url).netloc.lower()
    
    if 'amazon' in domain:
        return 'amazon'
    elif 'ebay' in domain:
        return 'ebay'
    elif 'etsy' in domain:
        return 'etsy'
    elif any(s in domain for s in ['shopify', 'myshopify']):
        return 'shopify'
    
    return 'generic'


def is_valid_product_image(url: str, width: int = 0, height: int = 0) -> bool:
    """Check if URL is likely a valid product image"""
    if not url or len(url) < 10:
        return False
    
    url_lower = url.lower()
    
    for pattern in EXCLUDED_PATTERNS:
        if re.search(pattern, url_lower):
            return False
    
    if width > 0 and width < MIN_IMAGE_WIDTH:
        return False
    if height > 0 and height < MIN_IMAGE_HEIGHT:
        return False
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif')
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    
    has_valid_ext = any(path_lower.endswith(ext) or ext in path_lower for ext in valid_extensions)
    is_data_url = url.startswith('data:image/')
    has_image_in_path = 'image' in path_lower or 'img' in path_lower or 'photo' in path_lower
    
    return has_valid_ext or is_data_url or has_image_in_path


def get_high_res_url(img_element, base_url: str) -> Optional[str]:
    """Extract highest resolution URL from image element attributes"""
    priority_attrs = [
        'data-old-hires',
        'data-zoom-src',
        'data-zoom-image',
        'data-large-image',
        'data-large-src',
        'data-high-res-src',
        'data-src-large',
        'data-image-large',
        'data-full-src',
        'data-original',
        'data-src',
        'srcset',
        'src'
    ]
    
    try:
        for attr in priority_attrs:
            value = img_element.get_attribute(attr)
            if value:
                if attr == 'srcset':
                    urls = []
                    for part in value.split(','):
                        part = part.strip()
                        if part:
                            url_part = part.split()[0]
                            urls.append(url_part)
                    if urls:
                        value = urls[-1]
                
                if value.startswith('//'):
                    value = 'https:' + value
                elif value.startswith('/'):
                    value = urljoin(base_url, value)
                elif not value.startswith('http'):
                    value = urljoin(base_url, value)
                
                if is_valid_product_image(value):
                    return value
    except Exception:
        pass
    
    return None


def scrape_images_with_playwright(
    url: str,
    timeout_ms: int = 30000,
    max_images: int = 10
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Scrape product images from URL using Playwright
    
    Args:
        url: Product page URL
        timeout_ms: Page load timeout in milliseconds
        max_images: Maximum number of images to return
        
    Returns:
        Tuple of (list of image URLs, metadata dict)
    """
    if not PLAYWRIGHT_AVAILABLE:
        return [], {"error": "Playwright not available", "success": False}
    
    platform = detect_platform(url)
    selectors = ECOMMERCE_SELECTORS.get(platform, ECOMMERCE_SELECTORS['generic'])
    
    metadata = {
        "platform": platform,
        "url": url,
        "success": False,
        "images_found": 0,
        "method": "playwright"
    }
    
    found_urls = set()
    
    try:
        with sync_playwright() as p:
            launch_args = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            }
            
            if SYSTEM_CHROMIUM:
                launch_args['executable_path'] = SYSTEM_CHROMIUM
            
            browser = p.chromium.launch(**launch_args)
            
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            page = context.new_page()
            
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=timeout_ms)
                page.wait_for_timeout(2000)
            except PlaywrightTimeout:
                metadata["error"] = "Page load timeout"
                browser.close()
                return [], metadata
            
            if selectors.get('scroll_for_images'):
                for _ in range(3):
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    page.wait_for_timeout(500)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)
            
            gallery_selector = selectors.get('gallery_click')
            if gallery_selector:
                try:
                    thumbnails = page.locator(gallery_selector).all()
                    for thumb in thumbnails[:5]:
                        try:
                            thumb.click(timeout=1000)
                            page.wait_for_timeout(300)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            for selector in selectors['product_images']:
                try:
                    images = page.locator(selector).all()
                    for img in images:
                        try:
                            img_url = get_high_res_url(img, url)
                            if img_url and img_url not in found_urls:
                                box = img.bounding_box()
                                width = box['width'] if box else 0
                                height = box['height'] if box else 0
                                
                                if is_valid_product_image(img_url, width, height):
                                    found_urls.add(img_url)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            if platform == 'amazon':
                try:
                    scripts = page.evaluate("""
                        () => {
                            const urls = [];
                            if (typeof ImageBlockATF !== 'undefined' && ImageBlockATF.data) {
                                const data = ImageBlockATF.data;
                                for (const key in data) {
                                    if (data[key] && data[key].hiRes) {
                                        urls.push(data[key].hiRes);
                                    }
                                }
                            }
                            return urls;
                        }
                    """)
                    for script_url in scripts:
                        if is_valid_product_image(script_url):
                            found_urls.add(script_url)
                except Exception:
                    pass
            
            browser.close()
            
            image_list = list(found_urls)[:max_images]
            
            metadata["success"] = len(image_list) > 0
            metadata["images_found"] = len(image_list)
            
            return image_list, metadata
            
    except Exception as e:
        metadata["error"] = str(e)
        return [], metadata


def scrape_product_images(url: str, max_images: int = 10) -> Tuple[List[str], str]:
    """
    Main entry point for scraping product images
    
    Args:
        url: Product page URL
        max_images: Maximum images to return
        
    Returns:
        Tuple of (list of image URLs, status message)
    """
    images, metadata = scrape_images_with_playwright(url, max_images=max_images)
    
    if metadata.get("success"):
        status = f"Found {len(images)} images from {metadata['platform']}"
    else:
        error = metadata.get("error", "Unknown error")
        status = f"Scraping failed: {error}"
    
    return images, status
