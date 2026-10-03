"""
Backend package for AI Object Remover.
Exposes the FastAPI app and inpainting services.
"""

from .app import app
from .inpainting import (
    MaskService,
    MaskValidationError,
    InpaintingService,
    inpaint_image,
    ProviderNotConfiguredError,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
    LamaInpaintingProvider,
)

__all__ = [
    "app",
    "MaskService",
    "MaskValidationError",
    "InpaintingService",
    "inpaint_image",
    "ProviderNotConfiguredError",
    "InvalidAPIKeyError",
    "RateLimitError",
    "ProviderTimeoutError",
    "InpaintingAPIError",
    "LamaInpaintingProvider",
]
