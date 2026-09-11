"""
stability.py
Stability AI inpainting provider implementation using Stable Image Inpaint API.
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


class StabilityInpaintingProvider(BaseInpaintingProvider):
    """Official Stability AI Stable Image Inpainting Provider."""

    def __init__(self, api_key: str, timeout: float = 60.0):
        super().__init__(api_key, timeout)
        self.url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "accept": "image/*",
        }
        files = {
            "image": ("image.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        }
        data = {
            "prompt": prompt or "seamless natural background fill, photorealistic, clean texture",
            "output_format": "png",
        }
        if model:
            data["model"] = model

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, headers=headers, files=files, data=data)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Stability AI inpainting request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Stability AI: {str(e)}", retryable=True)

        if response.status_code in (401, 403):
            raise InvalidAPIKeyError("Invalid Stability AI API key. Please check your credentials in .env.")
        elif response.status_code == 429:
            raise RateLimitError("Stability AI rate limit exceeded or insufficient credits. Please try again later.")
        elif response.status_code >= 500:
            raise InpaintingAPIError(
                f"Stability AI server error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                retryable=True,
            )
        elif response.status_code != 200:
            err_msg = "Unknown error"
            try:
                err_data = response.json()
                err_msg = err_data.get("message") or str(err_data.get("errors", [response.text])[0])
            except Exception:
                err_msg = response.text or f"HTTP {response.status_code}"
            raise InpaintingAPIError(f"Stability AI error ({response.status_code}): {err_msg}", status_code=response.status_code)

        return response.content
