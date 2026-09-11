"""
clipdrop.py
Clipdrop Cleanup API inpainting provider implementation.
"""

from typing import Optional
import httpx

from services.providers.base import (
    BaseInpaintingProvider,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)


class ClipdropInpaintingProvider(BaseInpaintingProvider):
    """Clipdrop Cleanup API Provider."""

    def __init__(self, api_key: str, timeout: float = 60.0):
        super().__init__(api_key, timeout)
        self.url = "https://clipdrop-api.co/cleanup/v1"

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        headers = {"x-api-key": self.api_key}
        files = {
            "image_file": ("image.png", image_bytes, "image/png"),
            "mask_file": ("mask.png", mask_bytes, "image/png"),
        }
        # Clipdrop supports 'quality' (HD mode, best result) or 'fast'
        clean_mode = "fast" if model and model.strip().lower() == "fast" else "quality"
        data = {"mode": clean_mode}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, headers=headers, files=files, data=data)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Clipdrop API request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Clipdrop: {str(e)}", retryable=True)

        if response.status_code in (401, 403):
            raise InvalidAPIKeyError("Invalid Clipdrop API key. Please check credentials in .env.")
        elif response.status_code == 429:
            raise RateLimitError("Clipdrop rate limit or credit limit exceeded.")
        elif response.status_code >= 500:
            raise InpaintingAPIError(
                f"Clipdrop server error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                retryable=True,
            )
        elif response.status_code != 200:
            raise InpaintingAPIError(f"Clipdrop error ({response.status_code}): {response.text}", status_code=response.status_code)

        return response.content
