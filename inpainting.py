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
import sys
import time
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from dotenv import load_dotenv

try:
    import httpx
except ImportError:
    httpx = None

try:
    import cv2
except ImportError:
    cv2 = None

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
            image = ImageOps.exif_transpose(image)
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

        return binary_mask

    @classmethod
    def expand_mask(cls, mask: Image.Image, expansion_pixels: int = 4) -> Image.Image:
        if expansion_pixels <= 0:
            return mask.copy()
        filter_size = max(3, 2 * expansion_pixels + 1)
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
        p1 = Path(__file__).resolve().parent / "inpainting_lama.onnx"
        p2 = Path(__file__).resolve().parent / "models" / "inpainting_lama.onnx"
        self.model_path = p1 if p1.exists() else p2

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

            target_size = (512, 512)
            img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            img_tensor = img_resized.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))
            img_tensor = np.expand_dims(img_tensor, axis=0)

            mask_binary = (mask_resized > 127).astype(np.float32)
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

            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
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

    def __init__(self):
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
        self.timeout = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
        self.max_dimension = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
        self.mask_expansion = int(os.getenv("MASK_EXPANSION_PIXELS", "4"))
        self.mask_feather = float(os.getenv("MASK_FEATHER_RADIUS", "2.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))

    @staticmethod
    def get_provider_config() -> Tuple[str, str, str]:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
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

        # Step 1: Preprocess mask with dilation and feathering
        expanded_mask = MaskService.expand_mask(selection_mask, expansion_pixels=self.mask_expansion)
        feathered_mask = MaskService.feather_mask(expanded_mask, feather_radius=self.mask_feather)

        # Step 2: Handle max dimension scaling if image exceeds limit
        orig_w, orig_h = original_image.size
        needs_downscale = max(orig_w, orig_h) > self.max_dimension

        if needs_downscale:
            scale_factor = self.max_dimension / max(orig_w, orig_h)
            work_w = int(round(orig_w * scale_factor))
            work_h = int(round(orig_h * scale_factor))
            work_image = original_image.resize((work_w, work_h), resample=Image.Resampling.LANCZOS)
            work_mask = expanded_mask.resize((work_w, work_h), resample=Image.Resampling.NEAREST)
        else:
            work_image = original_image
            work_mask = expanded_mask

        # Convert to PNG bytes
        work_img_bytes = MaskService.to_bytes(work_image, format="PNG")
        work_mask_bytes = MaskService.to_bytes(work_mask, format="PNG")

        # Step 3: Call AI Provider with automatic retry logic
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

        # Step 4: Validate output format and decode
        try:
            inpainted_candidate = Image.open(io.BytesIO(raw_result_bytes))
            inpainted_candidate.load()
        except Exception as e:
            raise InpaintingAPIError(f"AI provider returned corrupted or unreadable image data: {str(e)}")

        # Step 5: Safe compositing - untouched areas guaranteed bit-exact to original
        final_composited = MaskService.composite_inpainted_result(
            original_image=original_image,
            inpainted_image=inpainted_candidate,
            processed_mask=feathered_mask,
        )

        elapsed = round(time.time() - start_time, 3)
        metrics = MaskService.get_mask_metrics(selection_mask)
        logger.info(
            f"Inpainting succeeded in {elapsed}s | Provider: {provider_name} | "
            f"Size: {orig_w}x{orig_h} | Coverage: {metrics['coverage_percentage']}%"
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
]
