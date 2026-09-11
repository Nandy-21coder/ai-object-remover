"""
local.py
High-performance local inpainting provider using OpenCV.
Runs entirely offline and free on CPU with zero external API credentials required.
"""

import asyncio
from typing import Optional
import numpy as np
import cv2

from services.providers.base import BaseInpaintingProvider, InpaintingAPIError


class LocalInpaintingProvider(BaseInpaintingProvider):
    """Local offline inpainting provider powered by OpenCV Telea / Navier-Stokes algorithms."""

    def __init__(self, api_key: str = "local", timeout: float = 60.0):
        super().__init__(api_key=api_key or "local", timeout=timeout)

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        # Offload CPU-bound image processing to worker thread
        return await asyncio.to_thread(
            self._process_local_inpaint,
            image_bytes,
            mask_bytes,
            model,
        )

    def _process_local_inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        model: Optional[str] = None,
    ) -> bytes:
        try:
            # If LaMa neural model is available and user didn't explicitly request telea or ns, use LaMa AI!
            from pathlib import Path
            lama_model_file = Path(__file__).resolve().parent.parent.parent / "models" / "inpainting_lama.onnx"
            use_lama = (not model or "lama" in model.lower() or model == "default") and lama_model_file.exists() and lama_model_file.stat().st_size > 50 * 1024 * 1024
            if use_lama:
                from services.providers.lama import LamaInpaintingProvider
                lama_provider = LamaInpaintingProvider()
                return lama_provider._process_lama_inpaint(image_bytes, mask_bytes)

            # 1. Decode image
            img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InpaintingAPIError("Failed to decode image in Local Inpainting engine.")

            # 2. Decode mask
            mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise InpaintingAPIError("Failed to decode mask in Local Inpainting engine.")

            h, w = img.shape[:2]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # 3. Ensure binary mask (255 = inpaint area, 0 = keep)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Choose algorithm: Telea (fast, smooth) or Navier-Stokes
            flag = cv2.INPAINT_NS if (model and "ns" in model.lower()) else cv2.INPAINT_TELEA
            radius = 5

            # 4. Perform inpainting
            inpainted = cv2.inpaint(img, mask_binary, inpaintRadius=radius, flags=flag)

            # 5. Encode result to PNG
            success, encoded_buf = cv2.imencode(".png", inpainted)
            if not success or encoded_buf is None:
                raise InpaintingAPIError("Failed to encode inpainted result to PNG.")

            return encoded_buf.tobytes()
        except Exception as e:
            if isinstance(e, InpaintingAPIError):
                raise
            raise InpaintingAPIError(f"Local inpainting engine encountered an error: {str(e)}")
