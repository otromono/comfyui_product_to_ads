"""
ComfyUI Custom Node - Product to Ads API Routes
Backend routes for API verification
"""

import os
from server import PromptServer
from aiohttp import web
from google import genai


@PromptServer.instance.routes.post("/product_to_ads/verify_api")
async def verify_api_key(request):
    try:
        data = await request.json()
        api_key = data.get("api_key", "").strip()
        
        if not api_key:
            return web.json_response({"status": "error", "message": "API Key Missing"})
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Test"
            )
            if response and response.text:
                return web.json_response({"status": "success", "message": "API Key Valid"})
            else:
                return web.json_response({"status": "error", "message": "No response from API"})
        except Exception as e:
            error_msg = str(e)[:80]
            return web.json_response({"status": "error", "message": f"API Error: {error_msg}"})

    except Exception as e:
        return web.json_response({"status": "error", "message": f"Server Error: {str(e)[:50]}"})


print("[Product to Ads] API Routes registered successfully")
