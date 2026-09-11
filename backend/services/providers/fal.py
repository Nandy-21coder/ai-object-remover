"""
fal.py
Fal.ai inpainting provider implementation.
"""

import base64
from typing import Optional
import httpx

from services.providers.base import (
    BaseInpaintingProvider,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)


class FalInpaintingProvider(BaseInpaintingProvider):
    """Fal.ai Inpainting Provider."""

    def __init__(self, api_key: str, timeout: float = 60.0):
        super().__init__(api_key, timeout)

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        img_b64 = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        mask_b64 = f"data:image/png;base64,{base64.b64encode(mask_bytes).decode('utf-8')}"

        endpoint = model or "fal-ai/fast-sd-inpaint"
        url = f"https://fal.run/{endpoint}"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "image_url": img_b64,
            "mask_url": mask_b64,
            "prompt": prompt or "natural seamless background infill, photorealistic, clean texture",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Fal.ai request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Fal.ai: {str(e)}", retryable=True)

            if res.status_code in (401, 403):
                raise InvalidAPIKeyError("Invalid Fal.ai API key. Please check credentials in .env.")
            elif res.status_code == 429:
                raise RateLimitError("Fal.ai rate limit exceeded.")
            elif res.status_code >= 500:
                raise InpaintingAPIError(
                    f"Fal.ai server error ({res.status_code}): {res.text}",
                    status_code=res.status_code,
                    retryable=True,
                )
            elif res.status_code != 200:
                raise InpaintingAPIError(f"Fal.ai error ({res.status_code}): {res.text}", status_code=res.status_code)

            data = res.json()
            images = data.get("images", [])
            if not images or not images[0].get("url"):
                raise InpaintingAPIError("Fal.ai returned no output image.")
            img_res = await client.get(images[0]["url"])
            return img_res.content
