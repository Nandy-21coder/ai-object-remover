"""
huggingface.py
Hugging Face Inference API inpainting provider implementation.
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


class HuggingFaceInpaintingProvider(BaseInpaintingProvider):
    """Hugging Face Inference API Provider."""

    def __init__(self, api_key: str, timeout: float = 60.0):
        super().__init__(api_key, timeout)

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        endpoint = model or "runwayml/stable-diffusion-inpainting"
        url = f"https://api-inference.huggingface.co/models/{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {
            "image": ("image.png", image_bytes, "image/png"),
            "mask_image": ("mask.png", mask_bytes, "image/png"),
        }
        data = {"prompt": prompt or "seamless natural background fill, photorealistic, clean texture"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(url, headers=headers, files=files, data=data)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Hugging Face request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Hugging Face: {str(e)}", retryable=True)

            if res.status_code in (401, 403):
                raise InvalidAPIKeyError("Invalid Hugging Face token. Please check credentials in .env.")
            elif res.status_code == 429:
                raise RateLimitError("Hugging Face rate limit exceeded.")
            elif res.status_code >= 500:
                raise InpaintingAPIError(
                    f"Hugging Face server error ({res.status_code}): {res.text}",
                    status_code=res.status_code,
                    retryable=True,
                )
            elif res.status_code != 200:
                raise InpaintingAPIError(f"Hugging Face error ({res.status_code}): {res.text}", status_code=res.status_code)

            return res.content
