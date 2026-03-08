"""
HTML-based Product Image Scraper
Extracts product images from e-commerce pages using HTML parsing
More reliable than LLM-based extraction for image URLs
"""

import re
from typing import List, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}

EXCLUDE_URL_PATTERNS = [
    r'/icon[s]?/',
    r'/logo[s]?/',
    r'/banner[s]?/',
    r'/thumb[s]?/',
    r'/thumbnail[s]?/',
    r'/sprite[s]?/',
    r'/badge[s]?/',
    r'/button[s]?/',
    r'/nav/',
    r'/menu/',
    r'/footer/',
    r'/header/',
    r'/social/',
    r'/share/',
    r'/rating/',
    r'/star[s]?/',
    r'/cart/',
    r'/wishlist/',
    r'/payment/',
    r'/shipping/',
    r'/related/',
    r'/recommend/',
    r'/similar/',
    r'/avatar/',
    r'/profile/',
    r'/placeholder/',
    r'\.gif$',
    r'\.svg$',
    r'data:image',
    r'base64',
    r'1x1',
    r'pixel',
    r'spacer',
    r'blank',
    r'loading',
    r'lazy',
]

PRODUCT_GALLERY_SELECTORS = [
    '[itemprop="image"]',
    '[property="og:image"]',
    '.product-gallery img',
    '.product-image img',
    '.product-images img',
    '.gallery-image img',
    '.main-image img',
    '#product-image img',
    '#main-image img',
    '[data-zoom-image]',
    '[data-large-image]',
    '[data-full-image]',
    '[data-src-large]',
    '.pdp-image img',
    '.product-detail img',
    '.product-photo img',
    '.swiper-slide img',
    '.slick-slide img',
    '.carousel-item img',
    'picture source',
    'picture img',
]


def fetch_page_html(url: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch HTML content from a URL
    
    Args:
        url: Page URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (html_content, error_message)
    """
    if not URLLIB_AVAILABLE:
        return None, "urllib not available"
    
    try:
        parsed = urlparse(url)
        headers = DEFAULT_HEADERS.copy()
        headers['Host'] = parsed.netloc
        headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get('Content-Type', '')
            
            encoding = 'utf-8'
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip()
            
            raw_data = resp.read()
            
            if resp.headers.get('Content-Encoding') == 'gzip':
                import gzip
                raw_data = gzip.decompress(raw_data)
            elif resp.headers.get('Content-Encoding') == 'br':
                try:
                    import brotli
                    raw_data = brotli.decompress(raw_data)
                except ImportError:
                    pass
            
            html_content = raw_data.decode(encoding, errors='replace')
            return html_content, None
            
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:
        return None, f"Fetch error: {str(e)}"


def is_excluded_url(url: str) -> bool:
    """Check if URL matches exclusion patterns"""
    url_lower = url.lower()
    for pattern in EXCLUDE_URL_PATTERNS:
        if re.search(pattern, url_lower):
            return True
    return False


def extract_image_urls_from_html(html: str, base_url: str) -> List[str]:
    """
    Extract potential product image URLs from HTML
    
    Args:
        html: HTML content
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of image URLs found
    """
    if not BS4_AVAILABLE:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    found_urls: Set[str] = set()
    
    for selector in PRODUCT_GALLERY_SELECTORS:
        try:
            elements = soup.select(selector)
            for elem in elements:
                urls = extract_urls_from_element(elem, base_url)
                found_urls.update(urls)
        except Exception:
            continue
    
    meta_og = soup.find('meta', property='og:image')
    if meta_og and meta_og.get('content'):
        url = urljoin(base_url, meta_og['content'])
        if not is_excluded_url(url):
            found_urls.add(url)
    
    product_containers = soup.select('.product, .pdp, [itemtype*="Product"], #product, .product-detail')
    for container in product_containers:
        for img in container.find_all('img'):
            urls = extract_urls_from_element(img, base_url)
            found_urls.update(urls)
    
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            import json
            data = json.loads(script.string)
            urls = extract_urls_from_json_ld(data, base_url)
            found_urls.update(urls)
        except:
            continue
    
    return list(found_urls)


def extract_urls_from_element(elem, base_url: str) -> List[str]:
    """Extract image URLs from an HTML element"""
    urls = []
    
    url_attrs = [
        'src', 'data-src', 'data-lazy-src', 'data-original',
        'data-zoom-image', 'data-large-image', 'data-full-image',
        'data-src-large', 'data-high-res', 'data-image',
        'srcset', 'data-srcset', 'content', 'href'
    ]
    
    for attr in url_attrs:
        value = elem.get(attr)
        if value:
            if attr in ['srcset', 'data-srcset']:
                srcset_urls = parse_srcset(value)
                for url in srcset_urls:
                    full_url = urljoin(base_url, url)
                    if not is_excluded_url(full_url):
                        urls.append(full_url)
            else:
                full_url = urljoin(base_url, value)
                if not is_excluded_url(full_url):
                    urls.append(full_url)
    
    return urls


def parse_srcset(srcset: str) -> List[str]:
    """Parse srcset attribute and extract URLs, preferring larger sizes"""
    urls_with_size = []
    
    parts = srcset.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        tokens = part.split()
        if tokens:
            url = tokens[0]
            size = 0
            if len(tokens) > 1:
                size_str = tokens[1]
                match = re.search(r'(\d+)', size_str)
                if match:
                    size = int(match.group(1))
            urls_with_size.append((url, size))
    
    urls_with_size.sort(key=lambda x: x[1], reverse=True)
    return [url for url, _ in urls_with_size]


def extract_urls_from_json_ld(data, base_url: str) -> List[str]:
    """Extract image URLs from JSON-LD structured data"""
    urls = []
    
    if isinstance(data, list):
        for item in data:
            urls.extend(extract_urls_from_json_ld(item, base_url))
    elif isinstance(data, dict):
        item_type = data.get('@type', '')
        if 'Product' in str(item_type):
            image = data.get('image')
            if image:
                if isinstance(image, str):
                    urls.append(urljoin(base_url, image))
                elif isinstance(image, list):
                    for img in image:
                        if isinstance(img, str):
                            urls.append(urljoin(base_url, img))
                        elif isinstance(img, dict) and img.get('url'):
                            urls.append(urljoin(base_url, img['url']))
                elif isinstance(image, dict) and image.get('url'):
                    urls.append(urljoin(base_url, image['url']))
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                urls.extend(extract_urls_from_json_ld(value, base_url))
    
    return urls


def get_image_dimensions(url: str, referer_url: Optional[str] = None, timeout: int = 10) -> Tuple[int, int]:
    """
    Get image dimensions by downloading headers or partial content
    
    Returns:
        Tuple of (width, height) or (0, 0) if cannot determine
    """
    if not URLLIB_AVAILABLE:
        return 0, 0
    
    try:
        parsed = urlparse(url)
        headers = DEFAULT_HEADERS.copy()
        headers['Host'] = parsed.netloc
        headers['Referer'] = referer_url or f"{parsed.scheme}://{parsed.netloc}/"
        headers['Range'] = 'bytes=0-65535'
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            
            if len(data) < 24:
                return 0, 0
            
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                if len(data) >= 24:
                    width = int.from_bytes(data[16:20], 'big')
                    height = int.from_bytes(data[20:24], 'big')
                    return width, height
            
            elif data[:2] == b'\xff\xd8':
                return parse_jpeg_dimensions(data)
            
            elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
                return parse_webp_dimensions(data)
            
            return 0, 0
            
    except Exception:
        return 0, 0


def parse_jpeg_dimensions(data: bytes) -> Tuple[int, int]:
    """Parse JPEG dimensions from header data"""
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        
        marker = data[i + 1]
        
        if marker in (0xC0, 0xC1, 0xC2):
            if i + 9 < len(data):
                height = int.from_bytes(data[i + 5:i + 7], 'big')
                width = int.from_bytes(data[i + 7:i + 9], 'big')
                return width, height
        
        if marker == 0xD9 or marker == 0xDA:
            break
        
        if i + 4 < len(data):
            length = int.from_bytes(data[i + 2:i + 4], 'big')
            i += 2 + length
        else:
            break
    
    return 0, 0


def parse_webp_dimensions(data: bytes) -> Tuple[int, int]:
    """Parse WebP dimensions from header data"""
    if len(data) < 30:
        return 0, 0
    
    if data[12:16] == b'VP8 ':
        if len(data) >= 30:
            width = int.from_bytes(data[26:28], 'little') & 0x3FFF
            height = int.from_bytes(data[28:30], 'little') & 0x3FFF
            return width, height
    
    elif data[12:16] == b'VP8L':
        if len(data) >= 25:
            bits = int.from_bytes(data[21:25], 'little')
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
    
    elif data[12:16] == b'VP8X':
        if len(data) >= 30:
            width = int.from_bytes(data[24:27], 'little') + 1
            height = int.from_bytes(data[27:30], 'little') + 1
            return width, height
    
    return 0, 0


def extract_product_code(product_url: str) -> Optional[str]:
    """
    Extract product code/identifier from URL
    
    Supports:
    - Amazon ASIN (10 alphanumeric chars)
    - eBay item ID (numeric)
    - Shopify/generic slug patterns
    - SKU patterns in URL
    
    Returns:
        Product code if found, None otherwise
    """
    path = urlparse(product_url).path
    
    # Amazon ASIN: /dp/B0XXXXXX or /gp/product/B0XXXXXX
    asin_match = re.search(r'/(?:dp|gp/product)/([A-Z0-9]{10})', product_url, re.IGNORECASE)
    if asin_match:
        return asin_match.group(1).upper()
    
    # eBay item ID: /itm/TITLE/123456789
    ebay_match = re.search(r'/itm/[^/]+/(\d{10,14})', product_url)
    if ebay_match:
        return ebay_match.group(1)
    
    # Generic product ID patterns: product_id=XXX, item=XXX, sku=XXX
    param_match = re.search(r'[?&](?:product_id|item|sku|id|pid)=([A-Za-z0-9_-]+)', product_url, re.IGNORECASE)
    if param_match:
        return param_match.group(1)
    
    # Shopify-style: /products/product-name-sku123
    shopify_match = re.search(r'/products?/([a-z0-9][-a-z0-9]*)', path, re.IGNORECASE)
    if shopify_match:
        slug = shopify_match.group(1)
        # Extract trailing alphanumeric code if present
        code_match = re.search(r'[-_]([a-z0-9]{4,12})$', slug, re.IGNORECASE)
        if code_match:
            return code_match.group(1)
        # Use last segment if it looks like a code
        if len(slug) <= 20 and re.match(r'^[a-z0-9-]+$', slug, re.IGNORECASE):
            return slug
    
    # Generic path segment that looks like a product code
    segments = [s for s in path.split('/') if s]
    for seg in reversed(segments):
        # Alphanumeric code 4-15 chars
        if re.match(r'^[A-Za-z0-9]{4,15}$', seg):
            return seg
        # Code with dashes/underscores
        if re.match(r'^[A-Za-z0-9][-_A-Za-z0-9]{3,20}$', seg) and not seg.startswith(('www', 'http')):
            return seg
    
    return None


def find_longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring between two strings.
    Uses dynamic programming for efficiency.
    
    Returns:
        The longest common substring
    """
    if not s1 or not s2:
        return ""
    
    m, n = len(s1), len(s2)
    
    # Optimize: use shorter string for rows
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    
    # Only keep current and previous row for memory efficiency
    prev_row = [0] * (n + 1)
    curr_row = [0] * (n + 1)
    
    max_length = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr_row[j] = prev_row[j-1] + 1
                if curr_row[j] > max_length:
                    max_length = curr_row[j]
                    end_pos = i
            else:
                curr_row[j] = 0
        prev_row, curr_row = curr_row, prev_row
    
    return s1[end_pos - max_length:end_pos]


def extract_alphanumeric(url: str) -> str:
    """
    Extract only alphanumeric characters from URL path, lowercased.
    Ignores protocol, domain, and query parameters.
    """
    path = urlparse(url).path.lower()
    return re.sub(r'[^a-z0-9]', '', path)


def score_image_url_match(image_url: str, product_url: str, min_common_chars: int = 6) -> Tuple[int, int]:
    """
    Score how well an image URL matches the product URL using longest common substring.
    
    Args:
        image_url: URL of the image
        product_url: URL of the product page
        min_common_chars: Minimum characters in common to consider a match (default 6)
    
    Returns:
        Tuple of (score, common_length):
        - score: 2 = strong match (8+ chars), 1 = weak match (6-7 chars), 0 = no match
        - common_length: length of the longest common substring
    """
    # Extract alphanumeric parts from paths
    product_alphanum = extract_alphanumeric(product_url)
    image_alphanum = extract_alphanumeric(image_url)
    
    if not product_alphanum or not image_alphanum:
        return (0, 0)
    
    # Find longest common substring
    lcs = find_longest_common_substring(product_alphanum, image_alphanum)
    common_length = len(lcs)
    
    # Score based on common length
    if common_length >= 8:
        return (2, common_length)  # Strong match
    elif common_length >= min_common_chars:
        return (1, common_length)  # Weak match
    else:
        return (0, common_length)  # No match (below threshold)


def scrape_product_images(
    product_url: str,
    min_dimension: int = 1000,
    max_images: int = 4,
    timeout: int = 30
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """
    Scrape product images from an e-commerce page
    
    Args:
        product_url: URL of the product page
        min_dimension: Minimum width or height in pixels (default 1000)
        max_images: Maximum number of images to return (default 4)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (valid_image_urls, filtered_urls, error_message, product_code)
    """
    if not BS4_AVAILABLE:
        return [], [], "BeautifulSoup not installed (pip install beautifulsoup4)", None
    
    # Extract product code for matching
    product_code = extract_product_code(product_url)
    
    html, error = fetch_page_html(product_url, timeout)
    if error:
        return [], [], f"Failed to fetch page: {error}", product_code
    
    all_urls = extract_image_urls_from_html(html, product_url)
    
    if not all_urls:
        return [], [], "No product images found in page HTML", product_code
    
    # Deduplicate
    seen = set()
    unique_urls = []
    for url in all_urls:
        normalized = url.split('?')[0]
        if normalized not in seen:
            seen.add(normalized)
            unique_urls.append(url)
    
    # Check dimensions and score by longest common substring with product URL
    scored_urls = []
    filtered_urls = []
    no_match_urls = []
    
    for url in unique_urls:
        width, height = get_image_dimensions(url, product_url, timeout=10)
        
        if width >= min_dimension or height >= min_dimension:
            match_score, common_length = score_image_url_match(url, product_url, min_common_chars=6)
            if match_score > 0:
                # Image URL has enough characters in common with product URL
                scored_urls.append((url, match_score, common_length, width, height))
            else:
                # Below threshold - track but don't use
                no_match_urls.append(f"{url} (common={common_length})")
        else:
            filtered_urls.append(f"{url} ({width}x{height})")
    
    # Sort by match score (descending), then common length, then by dimension
    scored_urls.sort(key=lambda x: (x[1], x[2], max(x[3], x[4])), reverse=True)
    
    # Take top max_images
    valid_urls = [url for url, score, common, w, h in scored_urls[:max_images]]
    
    # Add no-match URLs to filtered list for logging
    filtered_urls.extend(no_match_urls)
    
    return valid_urls, filtered_urls, None, product_code
