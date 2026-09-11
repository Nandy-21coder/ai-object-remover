"""
base.py
Base classes and exception hierarchy for AI Inpainting providers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ProviderNotConfiguredError(Exception):
    """Raised when an inpainting request is made without an active AI provider key."""
    def __init__(self, message: str, provider: str, instructions: str):
        super().__init__(message)
        self.provider = provider
        self.instructions = instructions


class InvalidAPIKeyError(Exception):
    """Raised when the AI provider rejects the API credentials (401/403)."""
    pass


class RateLimitError(Exception):
    """Raised when the AI provider rate limit is exceeded or credits are exhausted (429)."""
    pass


class ProviderTimeoutError(Exception):
    """Raised when the AI provider times out."""
    pass


class InpaintingAPIError(Exception):
    """Raised when the real AI provider returns an unexpected error or fails."""
    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class BaseInpaintingProvider(ABC):
    """Abstract base class for all AI inpainting providers."""

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        """
        Executes real inpainting against the provider's official API.
        Must return generated image bytes in PNG format.
        """
        pass
