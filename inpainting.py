"""
inpainting.py
Self-contained AI Inpainting pipeline and provider orchestration.
Includes:
- MaskService: Binary mask binarization, dilation, feathering, and safe compositing.
- Provider Architecture: Pluggable adapters for Stability AI, Replicate, Clipdrop,
  Fal.ai, Hugging Face, and local offline LaMa ONNX inference.
- InpaintingService: Context preservation, automatic retries, and format validation.
"""

import os
import io
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from dotenv import load_dotenv

try:
    import httpx
except ImportError:
    httpx = None

import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logger = logging.getLogger("ai_object_remover.inpainting")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [InpaintingService] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================================
# Exceptions
# ==============================================================================

class MaskValidationError(Exception):
    """Raised when an image or mask fails format, dimension, or content validation."""
    pass


class InpaintingAPIError(Exception):
    """Base exception for provider API failures."""
    def __init__(self, message: str, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class ProviderNotConfiguredError(InpaintingAPIError):
    """Raised when an inpainting request is made without an active AI provider key."""
    def __init__(self, provider: str, instructions: str):
        msg = f"AI provider '{provider}' is not configured. {instructions}"
        super().__init__(msg, status_code=503, retryable=False)
        self.provider = provider
        self.instructions = instructions


class InvalidAPIKeyError(InpaintingAPIError):
    """Raised when the provider rejects credentials with 401/403."""
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401, retryable=False)


class RateLimitError(InpaintingAPIError):
    """Raised when provider returns HTTP 429."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, retryable=True)


class ProviderTimeoutError(InpaintingAPIError):
    """Raised when provider call times out."""
    def __init__(self, message: str = "Provider timed out"):
        super().__init__(message, status_code=504, retryable=True)


# ==============================================================================
# Mask Service
# ==============================================================================

class MaskService:
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

    @classmethod
    def load_image(cls, image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise MaskValidationError("No image data provided. Uploaded image is empty.")
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as e:
            raise MaskValidationError(f"Invalid or corrupted image format: {str(e)}")

        img_format = (image.format or "").upper()
        if img_format and img_format not in cls.SUPPORTED_FORMATS:
            raise MaskValidationError(
                f"Unsupported image format: '{img_format}'. Supported formats: JPG, JPEG, PNG, WebP."
            )

        try:
            transposed = ImageOps.exif_transpose(image)
            if transposed is not None:
                image = transposed
        except Exception:
            pass

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        if image.width <= 0 or image.height <= 0:
            raise MaskValidationError("Invalid image dimensions.")

        return image

    @classmethod
    def load_and_binarize_mask(cls, mask_bytes: bytes, target_size: Tuple[int, int]) -> Image.Image:
        if not mask_bytes:
            raise MaskValidationError("No mask data provided. Uploaded mask is empty.")
        try:
            raw_mask = Image.open(io.BytesIO(mask_bytes))
            raw_mask.load()
        except Exception as e:
            raise MaskValidationError(f"Invalid or corrupted mask format: {str(e)}")

        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            raise MaskValidationError("Target image has invalid dimensions.")

        if raw_mask.size != (target_w, target_h):
            raw_mask = raw_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

        if raw_mask.mode == "L":
            binary_mask = raw_mask.point(lambda p: 255 if p > 50 else 0, mode="L")
        else:
            rgba = raw_mask.convert("RGBA")
            arr = np.array(rgba)
            r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            max_rgb = np.maximum(np.maximum(r, g), b)
            all_opaque = (a > 200).mean() > 0.95
            if all_opaque:
                is_mask = max_rgb > 50
            else:
                is_mask = (a > 30) & (max_rgb > 20)
                if not is_mask.any():
                    is_mask = a > 30

            out = np.zeros(arr.shape[:2], dtype=np.uint8)
            out[is_mask] = 255
            binary_mask = Image.fromarray(out, mode="L")

        extrema = binary_mask.getextrema()
        if extrema == (0, 0) or extrema is None or extrema[1] == 0:
            raise MaskValidationError(
                "The selection mask is empty. Please brush over the object or text you want to remove."
            )

        metrics = cls.get_mask_metrics(binary_mask)
        if metrics.get("coverage_percentage", 0) > 95.0:
            raise MaskValidationError(
                f"The selection mask covers {metrics['coverage_percentage']}% of the photo. "
                "Please select only the specific object or region you want to remove."
            )

        return binary_mask

    @classmethod
    def get_mask_bounding_box(cls, mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Returns (min_x, min_y, max_x, max_y) of non-zero pixels, or None if empty."""
        if mask.mode != "L":
            mask = mask.convert("L")
        return mask.getbbox()

    @classmethod
    def calculate_adaptive_mask_params(
        cls,
        image_size: Tuple[int, int],
        base_expansion: int = 4,
        base_feather: float = 2.0,
    ) -> Tuple[int, float]:
        """Calculates resolution-aware dilation and feathering parameters."""
        orig_w, orig_h = image_size
        max_dim = max(orig_w, orig_h)
        res_factor = max_dim / 1024.0
        adaptive_expansion = max(3, min(24, round(base_expansion * res_factor)))
        adaptive_feather = max(1.5, min(10.0, round(base_feather * res_factor, 1)))
        return adaptive_expansion, adaptive_feather

    @classmethod
    def expand_mask(cls, mask: Image.Image, expansion_pixels: int = 4) -> Image.Image:
        if expansion_pixels <= 0:
            return mask.copy()
        filter_size = max(3, 2 * expansion_pixels + 1)
        if filter_size % 2 == 0:
            filter_size += 1
        return mask.filter(ImageFilter.MaxFilter(size=filter_size))

    @classmethod
    def feather_mask(cls, mask: Image.Image, feather_radius: float = 2.0) -> Image.Image:
        if feather_radius <= 0:
            return mask.copy()
        return mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))

    @classmethod
    def get_mask_metrics(cls, mask: Image.Image) -> Dict[str, Any]:
        w, h = mask.size
        total = w * h
        histogram = mask.histogram()
        selected = total - histogram[0]
        coverage_pct = round((selected / total) * 100, 2) if total > 0 else 0.0
        return {
            "width": w,
            "height": h,
            "selected_pixels": selected,
            "total_pixels": total,
            "coverage_percentage": coverage_pct,
        }

    @classmethod
    def composite_inpainted_result(
        cls,
        original_image: Image.Image,
        inpainted_image: Image.Image,
        processed_mask: Image.Image,
    ) -> Image.Image:
        orig_w, orig_h = original_image.size

        if inpainted_image.size != (orig_w, orig_h):
            inpainted_image = inpainted_image.resize((orig_w, orig_h), resample=Image.Resampling.LANCZOS)

        if processed_mask.size != (orig_w, orig_h):
            processed_mask = processed_mask.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

        if original_image.mode == "RGBA":
            orig_rgb = original_image.convert("RGB")
            inpaint_rgb = inpainted_image.convert("RGB") if inpainted_image.mode != "RGB" else inpainted_image
            composite_rgb = Image.composite(inpaint_rgb, orig_rgb, processed_mask)
            orig_alpha = original_image.split()[3]
            composite_rgba = composite_rgb.convert("RGBA")
            composite_rgba.putalpha(orig_alpha)
            return composite_rgba
        else:
            inpaint_rgb = inpainted_image.convert("RGB") if inpainted_image.mode != "RGB" else inpainted_image
            orig_rgb = original_image.convert("RGB") if original_image.mode != "RGB" else original_image
            return Image.composite(inpaint_rgb, orig_rgb, processed_mask)

    @classmethod
    def validate_and_process(
        cls, image_bytes: bytes, mask_bytes: bytes
    ) -> Tuple[Image.Image, Image.Image]:
        image = cls.load_image(image_bytes)
        mask = cls.load_and_binarize_mask(mask_bytes, image.size)
        return image, mask

    @staticmethod
    def to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf.getvalue()


# ==============================================================================
# Providers
# ==============================================================================

class BaseInpaintingProvider:
    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        raise NotImplementedError("Subclasses must implement inpaint()")


class StabilityInpaintingProvider(BaseInpaintingProvider):
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
        if not httpx:
            raise InpaintingAPIError("httpx package required for cloud providers.")
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
            raise RateLimitError("Stability AI rate limit exceeded or insufficient credits.")
        elif response.status_code >= 500:
            raise InpaintingAPIError(f"Stability AI server error ({response.status_code}): {response.text}", retryable=True)
        elif response.status_code != 200:
            raise InpaintingAPIError(f"Stability AI error ({response.status_code}): {response.text}")

        return response.content


class ReplicateInpaintingProvider(BaseInpaintingProvider):
    DEFAULT_MODEL = "allen/lama:cd497ec1858546de54abeb1dcf4bb9ab7233ca4b4c73060cf0b62e4303387b99"

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        if not httpx:
            raise InpaintingAPIError("httpx package required.")
        target_model = model or self.DEFAULT_MODEL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        import base64
        img_b64 = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
        mask_b64 = f"data:image/png;base64,{base64.b64encode(mask_bytes).decode()}"

        payload = {
            "version": target_model.split(":")[-1] if ":" in target_model else target_model,
            "input": {
                "image": img_b64,
                "mask": mask_b64,
                "prompt": prompt or "clean seamless texture fill, photo restoration",
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Replicate inpainting request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Replicate: {str(e)}", retryable=True)

            if res.status_code in (401, 403):
                raise InvalidAPIKeyError("Invalid Replicate API token.")
            elif res.status_code != 200 and res.status_code != 201:
                raise InpaintingAPIError(f"Replicate error ({res.status_code}): {res.text}")

            pred = res.json()
            out_url = pred.get("output")
            if isinstance(out_url, list):
                out_url = out_url[0]
            if not out_url:
                raise InpaintingAPIError("Replicate did not return an output URL.")

            img_res = await client.get(out_url)
            return img_res.content


class ClipdropInpaintingProvider(BaseInpaintingProvider):
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
        if not httpx:
            raise InpaintingAPIError("httpx package required.")
        headers = {"x-api-key": self.api_key}
        files = {
            "image_file": ("image.png", image_bytes, "image/png"),
            "mask_file": ("mask.png", mask_bytes, "image/png"),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(self.url, headers=headers, files=files)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Clipdrop request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Clipdrop: {str(e)}", retryable=True)

        if res.status_code in (401, 403):
            raise InvalidAPIKeyError("Invalid Clipdrop API key.")
        elif res.status_code == 429:
            raise RateLimitError("Clipdrop rate limit reached.")
        elif res.status_code != 200:
            raise InpaintingAPIError(f"Clipdrop error ({res.status_code}): {res.text}")

        return res.content


class FalInpaintingProvider(BaseInpaintingProvider):
    def __init__(self, api_key: str, timeout: float = 60.0):
        super().__init__(api_key, timeout)
        self.url = "https://fal.run/fal-ai/fast-sd-inpaint"

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        if not httpx:
            raise InpaintingAPIError("httpx package required.")
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }
        import base64
        payload = {
            "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
            "mask_url": f"data:image/png;base64,{base64.b64encode(mask_bytes).decode()}",
            "prompt": prompt or "clean photorealistic seamless fill",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(self.url, headers=headers, json=payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Fal.ai request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Fal.ai: {str(e)}", retryable=True)

        if res.status_code in (401, 403):
            raise InvalidAPIKeyError("Invalid Fal.ai API key.")
        elif res.status_code != 200:
            raise InpaintingAPIError(f"Fal.ai error ({res.status_code}): {res.text}")

        data = res.json()
        images = data.get("images", [])
        if not images:
            raise InpaintingAPIError("Fal.ai returned no image.")
        img_url = images[0].get("url")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            img_res = await client.get(img_url)
            return img_res.content


class HuggingFaceInpaintingProvider(BaseInpaintingProvider):
    DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        if not httpx:
            raise InpaintingAPIError("httpx package required.")
        target_model = model or self.DEFAULT_MODEL
        url = f"https://api-inference.huggingface.co/models/{target_model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        import base64
        payload = {
            "inputs": {
                "image": base64.b64encode(image_bytes).decode("utf-8"),
                "mask_image": base64.b64encode(mask_bytes).decode("utf-8"),
                "prompt": prompt or "seamless photorealistic background fill",
            }
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("Hugging Face request timed out.")
            except Exception as e:
                raise InpaintingAPIError(f"Network error contacting Hugging Face: {str(e)}", retryable=True)

        if res.status_code in (401, 403):
            raise InvalidAPIKeyError("Invalid Hugging Face token.")
        elif res.status_code == 503:
            raise InpaintingAPIError("Hugging Face model is currently loading. Please retry.", retryable=True)
        elif res.status_code != 200:
            raise InpaintingAPIError(f"Hugging Face error ({res.status_code}): {res.text}")

        return res.content


class LamaInpaintingProvider(BaseInpaintingProvider):
    """Local offline Deep Learning Inpainting Provider using LaMa ONNX model."""
    _session = None
    _net = None
    _lock = threading.Lock()

    def __init__(self, api_key: str = "local", timeout: float = 300.0):
        super().__init__(api_key=api_key or "local", timeout=timeout)
        root_dir = Path(__file__).resolve().parent
        candidates = [
            root_dir / "inpainting_lama.onnx",
            Path.cwd() / "inpainting_lama.onnx",
            root_dir / "models" / "inpainting_lama.onnx",
            Path.cwd() / "models" / "inpainting_lama.onnx",
        ]
        chosen = candidates[0]
        for c in candidates:
            if c.exists():
                chosen = c
                break
        self.model_path = chosen

    @classmethod
    def get_inference_engine(cls, model_path: Path) -> Tuple[str, Any]:
        if not model_path.exists() or model_path.stat().st_size < 10 * 1024 * 1024:
            raise InpaintingAPIError(
                f"LaMa AI model not found at {model_path}. Please place 'inpainting_lama.onnx' in the project root."
            )

        if HAS_ORT:
            if cls._session is None:
                with cls._lock:
                    if cls._session is None:
                        opts = ort.SessionOptions()
                        opts.log_severity_level = 3
                        cpu_total = os.cpu_count() or 4
                        threads = max(2, min(6, cpu_total - 2 if cpu_total > 4 else cpu_total))
                        opts.intra_op_num_threads = threads
                        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        cls._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            return "ort", cls._session
        else:
            if cls._net is None:
                with cls._lock:
                    if cls._net is None and cv2:
                        cls._net = cv2.dnn.readNetFromONNX(str(model_path))
            if not cls._net:
                raise InpaintingAPIError("Neither onnxruntime nor opencv-python DNN is available.")
            return "dnn", cls._net

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        return await asyncio.to_thread(self._process_lama_inpaint, image_bytes, mask_bytes)

    def _process_lama_inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
    ) -> bytes:
        if cv2 is None:
            raise InpaintingAPIError("opencv-python (cv2) is required for LaMa inpainting but is not installed.")
        try:
            img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InpaintingAPIError("Failed to decode image.")

            mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise InpaintingAPIError("Failed to decode mask.")

            orig_h, orig_w = img.shape[:2]
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Aspect-Ratio-Preserving Inference with Reflection Padding
            target_len = 512
            scale = target_len / max(orig_w, orig_h)
            new_w = max(1, min(target_len, round(orig_w * scale)))
            new_h = max(1, min(target_len, round(orig_h * scale)))

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            img_scaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)
            mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            pad_top = (target_len - new_h) // 2
            pad_bottom = target_len - new_h - pad_top
            pad_left = (target_len - new_w) // 2
            pad_right = target_len - new_w - pad_left

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                img_padded = cv2.copyMakeBorder(
                    img_scaled, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
                )
                mask_padded = cv2.copyMakeBorder(
                    mask_scaled, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0.0]
                )
            else:
                img_padded = img_scaled
                mask_padded = mask_scaled

            img_tensor = img_padded.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)

            mask_arr = np.asarray(mask_padded, dtype=np.uint8)
            mask_binary = (mask_arr > 127).astype(np.float32)
            mask_tensor = np.expand_dims(np.expand_dims(mask_binary, axis=0), axis=0)

            engine_type, engine_obj = self.get_inference_engine(self.model_path)
            engine: Any = engine_obj
            if engine_type == "ort":
                inputs = {
                    engine.get_inputs()[0].name: img_tensor,
                    engine.get_inputs()[1].name: mask_tensor,
                }
                outputs = engine.run(None, inputs)
                raw_out = outputs[0]
            else:
                engine.setInput(img_tensor, "image")
                engine.setInput(mask_tensor, "mask")
                raw_out = engine.forward()

            out = raw_out[0]
            out = np.transpose(out, (1, 2, 0))
            out = np.clip(out, 0, 255).astype(np.uint8)

            # Remove padding to restore original aspect ratio exactly
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                out_unpadded = out[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
            else:
                out_unpadded = out

            out_bgr = cv2.cvtColor(out_unpadded, cv2.COLOR_RGB2BGR)
            result = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            success, encoded_buf = cv2.imencode(".png", result)
            if not success or encoded_buf is None:
                raise InpaintingAPIError("Failed to encode LaMa inpainted result.")

            return encoded_buf.tobytes()

        except Exception as e:
            if isinstance(e, InpaintingAPIError):
                raise
            raise InpaintingAPIError(f"LaMa AI inpainting engine error: {str(e)}")


class LocalInpaintingProvider(LamaInpaintingProvider):
    """Alias for local offline inpainting provider."""
    pass


PROVIDER_REGISTRY: Dict[str, Any] = {
    "stability": StabilityInpaintingProvider,
    "replicate": ReplicateInpaintingProvider,
    "clipdrop": ClipdropInpaintingProvider,
    "fal": FalInpaintingProvider,
    "huggingface": HuggingFaceInpaintingProvider,
    "lama": LamaInpaintingProvider,
    "local": LocalInpaintingProvider,
}


# ==============================================================================
# Inpainting Orchestrator Service
# ==============================================================================

class InpaintingService:
    """Service orchestrating AI provider resolution, mask enhancement, retries, and compositing."""

    @staticmethod
    def _load_env():
        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent.parent
        candidates = [
            project_root / ".env",
            backend_dir / ".env",
            Path.cwd() / ".env",
        ]
        for env_path in candidates:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                break

    def __init__(self):
        self._load_env()
        self.timeout = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
        self.max_dimension = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
        self.mask_expansion = int(os.getenv("MASK_EXPANSION_PIXELS", "4"))
        self.mask_feather = float(os.getenv("MASK_FEATHER_RADIUS", "2.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))

    @classmethod
    def get_provider_config(cls) -> Tuple[str, str, str]:
        cls._load_env()
        provider = (os.getenv("AI_PROVIDER") or os.getenv("INPAINTING_PROVIDER") or "stability").strip().lower()
        api_key = (os.getenv("AI_API_KEY") or "").strip()
        model = (os.getenv("AI_MODEL") or "").strip()

        if provider in ("local", "lama"):
            return provider, "local", model or "default"

        if not api_key:
            provider_keys = [
                ("stability", "STABILITY_API_KEY"),
                ("replicate", "REPLICATE_API_TOKEN"),
                ("clipdrop", "CLIPDROP_API_KEY"),
                ("fal", "FAL_KEY"),
                ("huggingface", "HF_TOKEN"),
            ]
            for p, k in provider_keys:
                if provider == p and os.getenv(k):
                    api_key = os.getenv(k, "").strip()
                    break

            if not api_key:
                for p, k in provider_keys:
                    val = os.getenv(k, "").strip()
                    if val:
                        return p, val, model

        return provider, api_key, model

    def get_provider_instance(self) -> Tuple[BaseInpaintingProvider, str, str]:
        provider_name, api_key, model = self.get_provider_config()

        if provider_name in ("local", "lama"):
            api_key = "local"
        elif not api_key:
            raise ProviderNotConfiguredError(
                provider=provider_name,
                instructions=(
                    f"To enable real AI object removal, add your {provider_name.upper()} API credentials in .env:\n"
                    "AI_PROVIDER=" + provider_name + "\n"
                    "AI_API_KEY=your_actual_api_key_here\n"
                    "Or use local offline model with AI_PROVIDER=lama"
                ),
            )

        provider_cls = PROVIDER_REGISTRY.get(provider_name)
        if not provider_cls:
            raise InpaintingAPIError(
                f"Unsupported AI provider: '{provider_name}'. Supported: {list(PROVIDER_REGISTRY.keys())}"
            )

        return provider_cls(api_key=api_key, timeout=self.timeout), provider_name, model

    def get_status(self) -> Dict[str, Any]:
        provider_name, api_key, model = self.get_provider_config()
        configured = bool(api_key) or provider_name in ("local", "lama")
        return {
            "status": "ok",
            "provider_configured": configured,
            "provider": provider_name,
            "model": model or "default",
            "mask_expansion_pixels": self.mask_expansion,
            "mask_feather_radius": self.mask_feather,
            "max_retries": self.max_retries,
            "message": (
                f"AI Inpainting engine connected via {provider_name.upper()}."
                if configured
                else "No AI provider configured. Set AI_API_KEY or AI_PROVIDER=lama in .env."
            ),
        }

    async def inpaint(
        self,
        original_image: Image.Image,
        selection_mask: Image.Image,
        prompt: str = "",
    ) -> bytes:
        provider_instance, provider_name, model_override = self.get_provider_instance()

        orig_w, orig_h = original_image.size

        # Step 1: Preprocess mask with resolution-adaptive dilation and feathering
        adaptive_expansion, adaptive_feather = MaskService.calculate_adaptive_mask_params(
            (orig_w, orig_h),
            base_expansion=self.mask_expansion,
            base_feather=self.mask_feather,
        )
        expanded_mask = MaskService.expand_mask(selection_mask, expansion_pixels=adaptive_expansion)
        feathered_mask = MaskService.feather_mask(expanded_mask, feather_radius=adaptive_feather)

        # Step 2: Check for localized bounding box to perform Crop-and-Inpaint
        bbox = MaskService.get_mask_bounding_box(expanded_mask)
        use_crop = False
        crop_box = None
        crop_feathered: Optional[Image.Image] = None

        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            bw = max_x - min_x
            bh = max_y - min_y

            # Use Crop-and-Inpaint if selection doesn't cover the full canvas and image is > 512px
            if (bw < orig_w * 0.85 or bh < orig_h * 0.85) and (orig_w > 512 or orig_h > 512):
                use_crop = True
                context_x = max(int(bw * 0.5), int(orig_w * 0.05), 48)
                context_y = max(int(bh * 0.5), int(orig_h * 0.05), 48)

                crop_size = max(bw + 2 * context_x, bh + 2 * context_y)
                crop_size = min(crop_size, max(orig_w, orig_h))

                cx = (min_x + max_x) // 2
                cy = (min_y + max_y) // 2

                c_x0 = max(0, cx - crop_size // 2)
                c_y0 = max(0, cy - crop_size // 2)
                c_x1 = min(orig_w, c_x0 + crop_size)
                c_y1 = min(orig_h, c_y0 + crop_size)

                if c_x1 - c_x0 < crop_size:
                    c_x0 = max(0, c_x1 - crop_size)
                if c_y1 - c_y0 < crop_size:
                    c_y0 = max(0, c_y1 - crop_size)

                crop_box = (c_x0, c_y0, c_x1, c_y1)
                work_image = original_image.crop(crop_box)
                work_mask = expanded_mask.crop(crop_box)
                crop_feathered = feathered_mask.crop(crop_box)
            else:
                work_image = original_image
                work_mask = expanded_mask
        else:
            work_image = original_image
            work_mask = expanded_mask

        # Step 3: Handle max dimension scaling if crop or full image exceeds limit
        work_w, work_h = work_image.size
        needs_downscale = max(work_w, work_h) > self.max_dimension

        if needs_downscale:
            scale_factor = self.max_dimension / max(work_w, work_h)
            scaled_w = round(work_w * scale_factor)
            scaled_h = round(work_h * scale_factor)
            send_image = work_image.resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
            send_mask = work_mask.resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST)
        else:
            send_image = work_image
            send_mask = work_mask

        work_img_bytes = MaskService.to_bytes(send_image, format="PNG")
        work_mask_bytes = MaskService.to_bytes(send_mask, format="PNG")

        # Step 4: Call AI Provider with automatic retry logic
        last_error = None
        attempt = 0
        raw_result_bytes = None
        start_time = time.time()

        while attempt <= self.max_retries:
            try:
                raw_result_bytes = await provider_instance.inpaint(
                    image_bytes=work_img_bytes,
                    mask_bytes=work_mask_bytes,
                    prompt=prompt,
                    model=model_override or None,
                )
                break
            except (InvalidAPIKeyError, RateLimitError):
                raise
            except (ProviderTimeoutError, InpaintingAPIError) as e:
                last_error = e
                attempt += 1
                if attempt <= self.max_retries and getattr(e, "retryable", True):
                    backoff = 1.5 * (2 ** (attempt - 1))
                    logger.warning(
                        f"AI provider request failed (attempt {attempt}/{self.max_retries + 1}). "
                        f"Retrying in {backoff:.1f}s... Details: {str(e)}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    break
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.max_retries:
                    await asyncio.sleep(1.5)
                else:
                    break

        if raw_result_bytes is None:
            if last_error:
                raise last_error
            raise InpaintingAPIError("Inpainting failed: No image data returned by AI provider.")

        # Step 5: Validate output format and decode
        try:
            inpainted_candidate = Image.open(io.BytesIO(raw_result_bytes))
            inpainted_candidate.load()
        except Exception as e:
            raise InpaintingAPIError(f"AI provider returned corrupted or unreadable image data: {str(e)}")

        # Step 6: Safe compositing - untouched areas guaranteed bit-exact to original
        if use_crop and crop_box is not None and crop_feathered is not None:
            patch_composited = MaskService.composite_inpainted_result(
                original_image=work_image,
                inpainted_image=inpainted_candidate,
                processed_mask=crop_feathered,
            )
            final_composited = original_image.copy()
            final_composited.paste(patch_composited, (crop_box[0], crop_box[1]))
        else:
            final_composited = MaskService.composite_inpainted_result(
                original_image=original_image,
                inpainted_image=inpainted_candidate,
                processed_mask=feathered_mask,
            )

        elapsed = round(time.time() - start_time, 3)
        metrics = MaskService.get_mask_metrics(selection_mask)
        logger.info(
            f"Inpainting succeeded in {elapsed}s | Provider: {provider_name} | "
            f"CropMode: {use_crop} | Size: {orig_w}x{orig_h} | Coverage: {metrics['coverage_percentage']}%"
        )

        return MaskService.to_bytes(final_composited, format="PNG")


async def inpaint_image(
    original_image: Image.Image,
    selection_mask: Image.Image,
    prompt: str = "",
) -> bytes:
    """Convenience top-level async entry point for inpainting."""
    service = InpaintingService()
    return await service.inpaint(original_image, selection_mask, prompt=prompt)




class SmartSegmentationService:
    """Provides computer vision and AI-assisted segmentation for rough user loops and circles."""

    @classmethod
    def segment_enclosed_region(
        cls,
        image_bytes: bytes,
        points: List[Dict[str, float]],
        mode: str = "smart_object",
        padding: int = 8,
    ) -> bytes:
        """
        Segments the foreground object inside the polygon defined by `points`.
        
        Args:
            image_bytes: Raw bytes of the source photo.
            points: List of {'x': float/int, 'y': float/int} image-space coordinates.
            mode: 'smart_object' (GrabCut + morphological boundaries) or 'fill_interior' (direct polygon fill).
            padding: Pixel padding around the bounding box for background context.

        Returns:
            PNG image bytes of the binary mask (255 = object to remove, 0 = keep).
        """
        if not points or len(points) < 3:
            raise ValueError("Smart circle requires at least 3 points to form an enclosed loop.")

        # 1. Decode source image to numpy BGR
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image for smart circle segmentation.")

        h, w = img.shape[:2]

        # 2. Convert points to integer numpy polygon and clamp to image dimensions
        pts = []
        for p in points:
            px = max(0, min(w - 1, round(p.get("x", 0))))
            py = max(0, min(h - 1, round(p.get("y", 0))))
            pts.append([px, py])
        poly_pts = np.array(pts, dtype=np.int32)

        # 3. Create full-size polygon binary mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_pts], (255,))
        poly_area = cv2.countNonZero(poly_mask)

        # Direct fill fallback if requested or if polygon is tiny (< 100 pixels)
        if mode == "fill_interior" or poly_area < 100:
            return cls._encode_mask_png(poly_mask)

        # 4. Compute bounding box around polygon with context padding
        bx, by, bw, bh = cv2.boundingRect(poly_pts)
        bx0 = max(0, bx - padding)
        by0 = max(0, by - padding)
        bx1 = min(w, bx + bw + padding)
        by1 = min(h, by + bh + padding)

        # 5. Initialize GrabCut mask
        # cv2.GC_BGD (0): Definite background
        # cv2.GC_FGD (1): Definite foreground
        # cv2.GC_PR_BGD (2): Probable background
        # cv2.GC_PR_FGD (3): Probable foreground
        grabcut_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

        # Surrounding bounding box is initialized as probable background
        grabcut_mask[by0:by1, bx0:bx1] = cv2.GC_PR_BGD

        # Interior of user loop is initialized as probable foreground
        grabcut_mask[poly_mask == 255] = cv2.GC_PR_FGD

        # Compute centroid / inner core of polygon as definite foreground seeds
        moments = cv2.moments(poly_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            core_radius = max(3, int(min(bw, bh) * 0.18))
            core_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(core_mask, (cx, cy), core_radius, (255,), -1)
            # Only set core where inside the polygon
            grabcut_mask[(core_mask == 255) & (poly_mask == 255)] = cv2.GC_FGD

        # 6. Run GrabCut optimization within bounding box
        rect = (bx0, by0, bx1 - bx0, by1 - by0)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                img,
                grabcut_mask,
                rect,
                bgd_model,
                fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_MASK,
            )
            # Extract foreground pixels (definite or probable)
            fg_mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            # Restrict foreground strictly within or near the user's circle
            dilated_poly = cv2.dilate(poly_mask, np.ones((5, 5), np.uint8), iterations=1)
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=dilated_poly)

            fg_area = cv2.countNonZero(fg_mask)
            # If GrabCut found a reasonable object (> 15% of circle area), use it
            if fg_area >= 0.15 * poly_area:
                # Apply morphological close to bridge internal object texture gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

                # Light 2px dilation ensures complete edge coverage for inpainting
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                final_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)
            else:
                # Fallback to the smoothed polygon mask
                final_mask = poly_mask
        except Exception:
            # Fallback to direct polygon fill if GrabCut encounters degenerate color distributions
            final_mask = poly_mask

        return cls._encode_mask_png(final_mask)

    @staticmethod
    def _encode_mask_png(mask: np.ndarray) -> bytes:
        """
        Encodes an 8-bit binary mask as a 4-channel BGRA PNG with full alpha transparency.
        Foreground (object to remove): White (255, 255, 255) with Alpha 255.
        Background (unselected context): Completely transparent (0, 0, 0, 0).
        """
        h, w = mask.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        fg_indices = mask > 127
        bgra[fg_indices] = [255, 255, 255, 255]
        success, encoded = cv2.imencode(".png", bgra)
        if not success:
            raise RuntimeError("Failed to encode binary mask to PNG format.")
        return encoded.tobytes()


__all__ = [
    "MaskService",
    "MaskValidationError",
    "BaseInpaintingProvider",
    "StabilityInpaintingProvider",
    "ReplicateInpaintingProvider",
    "ClipdropInpaintingProvider",
    "FalInpaintingProvider",
    "HuggingFaceInpaintingProvider",
    "LamaInpaintingProvider",
    "LocalInpaintingProvider",
    "InpaintingService",
    "inpaint_image",
    "ProviderNotConfiguredError",
    "InvalidAPIKeyError",
    "RateLimitError",
    "ProviderTimeoutError",
    "InpaintingAPIError",
    "SmartSegmentationService",
]
