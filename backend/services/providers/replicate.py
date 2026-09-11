"""
replicate.py
Replicate inpainting provider implementation.
"""

import asyncio
import base64
import time
from typing import Optional
import httpx

from services.providers.base import (
    BaseInpaintingProvider,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)


class ReplicateInpaintingProvider(BaseInpaintingProvider):
    """Official Replicate Inpainting Provider."""

    def __init__(self, api_key: str, timeout: float = 75.0):
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

        # Default model: stability-ai/stable-diffusion-inpainting
        version_id = model or "c28b92a7ecd66eee4aefcd8a94eb9e7f669052157e8d32afe8b3d1fb901e8f4d"
        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "version": version_id,
            "input": {
                "image": img_b64,
                "mask": mask_b64,
                "prompt": prompt or "photorealistic seamless background fill, natural lighting, clean texture",
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Replicate request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Replicate: {str(e)}", retryable=True)

            if res.status_code in (401, 403):
                raise InvalidAPIKeyError("Invalid Replicate API token. Please check credentials in .env.")
            elif res.status_code == 429:
                raise RateLimitError("Replicate rate limit exceeded.")
            elif res.status_code >= 500:
                raise InpaintingAPIError(
                    f"Replicate server error ({res.status_code}): {res.text}",
                    status_code=res.status_code,
                    retryable=True,
                )
            elif res.status_code not in (200, 201):
                raise InpaintingAPIError(f"Replicate API error ({res.status_code}): {res.text}", status_code=res.status_code)

            prediction = res.json()
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                raise InpaintingAPIError("Replicate prediction missing polling URL.")

            start_time = time.time()
            while time.time() - start_time < self.timeout:
                poll_res = await client.get(poll_url, headers=headers)
                data = poll_res.json()
                stat = data.get("status")
                if stat == "succeeded":
                    output_url = data.get("output")
                    if isinstance(output_url, list):
                        output_url = output_url[0]
                    if not output_url:
                        raise InpaintingAPIError("Replicate returned empty output.")
                    img_res = await client.get(output_url)
                    return img_res.content
                elif stat in ("failed", "canceled"):
                    raise InpaintingAPIError(f"Replicate inpainting {stat}: {data.get('error')}")
                await asyncio.sleep(1.2)

            raise ProviderTimeoutError("Replicate inpainting timed out.")
