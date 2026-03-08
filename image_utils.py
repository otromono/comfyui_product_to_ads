"""
Image utility functions for Product to Ads Node
Handles image conversion between tensors, bytes, and PIL images
"""

import io
from typing import Optional, Dict, List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


def tensor_to_bytes(tensor, format: str = "PNG") -> Optional[bytes]:
    """
    Convert a ComfyUI image tensor to bytes
    
    Args:
        tensor: Image tensor in ComfyUI format (B, H, W, C) with values 0-1
        format: Output image format (PNG, JPEG)
        
    Returns:
        Image bytes or None if conversion fails
    """
    if not TORCH_AVAILABLE or not PIL_AVAILABLE:
        return None
    
    if tensor is None:
        return None
    
    try:
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255)
        
        numpy_image = tensor.cpu().numpy().astype('uint8')
        
        pil_image = Image.fromarray(numpy_image)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()
        
    except Exception as e:
        print(f"Error converting tensor to bytes: {e}")
        return None


def bytes_to_tensor(image_bytes: bytes):
    """
    Convert image bytes to a ComfyUI image tensor
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Image tensor in ComfyUI format (B, H, W, C) with values 0-1
    """
    if not TORCH_AVAILABLE or not PIL_AVAILABLE:
        return None
    
    if image_bytes is None:
        return None
    
    try:
        buffer = io.BytesIO(image_bytes)
        pil_image = Image.open(buffer)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        import numpy as np
        numpy_image = np.array(pil_image).astype('float32') / 255.0
        
        tensor = torch.from_numpy(numpy_image)
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        print(f"Error converting bytes to tensor: {e}")
        return None


def get_image_dimensions(tensor) -> str:
    """Get image dimensions as a string"""
    if not TORCH_AVAILABLE or tensor is None:
        return "Unknown"
    
    try:
        if len(tensor.shape) == 4:
            b, h, w, c = tensor.shape
            return f"{w}x{h} (batch: {b})"
        elif len(tensor.shape) == 3:
            h, w, c = tensor.shape
            return f"{w}x{h}"
        else:
            return f"Unknown shape: {tensor.shape}"
    except:
        return "Unknown"


def collect_provided_images(
    talent=None,
    product_1=None,
    product_2=None,
    logo=None
) -> Tuple[Dict[str, Optional[bytes]], List[str]]:
    """
    Collect and convert provided images to bytes
    
    Args:
        talent: Talent/model image tensor
        product_1: First product image tensor
        product_2: Second product image tensor
        logo: Brand logo image tensor
        
    Returns:
        Tuple of (images dict, log messages)
    """
    images = {}
    logs = []
    
    image_inputs = [
        ("talent", talent),
        ("product_1", product_1),
        ("product_2", product_2),
        ("logo", logo),
    ]
    
    for name, tensor in image_inputs:
        if tensor is not None:
            img_bytes = tensor_to_bytes(tensor)
            if img_bytes:
                images[name] = img_bytes
                logs.append(f"{name}: provided ({get_image_dimensions(tensor)}, {len(img_bytes)} bytes)")
            else:
                images[name] = None
                logs.append(f"{name}: conversion failed")
        else:
            images[name] = None
            logs.append(f"{name}: not provided")
    
    return images, logs


def create_empty_tensor(width: int = 64, height: int = 64):
    """Create an empty black image tensor"""
    if not TORCH_AVAILABLE:
        return None
    
    return torch.zeros((1, height, width, 3))


def resize_image_bytes(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """
    Resize image if larger than max_size while maintaining aspect ratio
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum dimension size
        
    Returns:
        Resized image bytes
    """
    if not PIL_AVAILABLE:
        return image_bytes
    
    try:
        buffer = io.BytesIO(image_bytes)
        img = Image.open(buffer)
        
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_bytes
