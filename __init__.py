"""
Product to Ads - ComfyUI Custom Node

Generates professional advertising images from product page URLs.
Uses Gemini API for product scraping and prompt generation,
and Nano Banana Pro for image generation.

Nodes:
- ProductToAds_Manual: User provides all images, generates 1 output
- ProductToAds_Auto: Auto-downloads product images, generates 4 formats
- BrandIdentityAnalyzer: Analyzes brand identity using Gemini with Google Search
"""

from .product_to_ads_node import ProductToAdsManualNode, ProductToAdsAutoNode
from .brand_analyzer_node import BrandIdentityAnalyzerNode

NODE_CLASS_MAPPINGS = {
    "ProductToAds_Manual": ProductToAdsManualNode,
    "ProductToAds_Auto": ProductToAdsAutoNode,
    "BrandIdentityAnalyzer": BrandIdentityAnalyzerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductToAds_Manual": "Product to Ads (Manual)",
    "ProductToAds_Auto": "Product to Ads (Auto 4-Format)",
    "BrandIdentityAnalyzer": "Brand Identity Analyzer",
}

WEB_DIRECTORY = "./web/extensions/product_to_ads"

try:
    from . import api_routes
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
