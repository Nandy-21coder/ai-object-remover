"""
inpainting_service.py
Orchestration layer for real AI inpainting pipelines.
Implements:
- Pluggable AI provider abstraction (Stability, Replicate, Clipdrop, Fal, Hugging Face).
- Mask preprocessing (configurable dilation and edge feathering).
- Contextual preservation and bit-exact untouched area compositing.
- Automatic retry handling with backoff for transient failures (max 2 retries).
- Strict output validation and dimension preservation.
- Diagnostics logging without logging any API keys or secrets.
"""

import os
import io
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from dotenv import load_dotenv

from services.mask_service import MaskService
from services.providers import (
    BaseInpaintingProvider,
    PROVIDER_REGISTRY,
    ProviderNotConfiguredError,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)

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


class InpaintingService:
    """Service orchestrating AI provider resolution, mask enhancement, retries, and compositing."""

    def __init__(self):
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
        self.timeout = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
        self.max_dimension = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
        self.mask_expansion = int(os.getenv("MASK_EXPANSION_PIXELS", "4"))
        self.mask_feather = float(os.getenv("MASK_FEATHER_RADIUS", "2.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "2"))

    @staticmethod
    def get_provider_config() -> Tuple[str, str, str]:
        """
        Reads and normalizes provider configuration from environment variables.
        Returns: (provider_name, api_key, model)
        """
        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
        provider = (os.getenv("AI_PROVIDER") or os.getenv("INPAINTING_PROVIDER") or "stability").strip().lower()
        api_key = (os.getenv("AI_API_KEY") or "").strip()
        model = (os.getenv("AI_MODEL") or "").strip()

        # Local or LaMa offline provider doesn't require an external API key
        if provider in ("local", "lama"):
            return provider, "local", model or "default"

        # Fallbacks for specific provider env keys if generic AI_API_KEY is unset
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

            # If still unset, check if any provider key is present
            if not api_key:
                for p, k in provider_keys:
                    val = os.getenv(k, "").strip()
                    if val:
                        return p, val, model

        return provider, api_key, model

    def get_provider_instance(self) -> Tuple[BaseInpaintingProvider, str, str]:
        """Resolves active provider instance or raises ProviderNotConfiguredError."""
        provider_name, api_key, model = self.get_provider_config()

        if provider_name in ("local", "lama"):
            api_key = "local"
        elif not api_key:
            raise ProviderNotConfiguredError(
                message=f"AI provider '{provider_name}' is not configured with an API key.",
                provider=provider_name,
                instructions=(
                    "To enable real AI object removal, add your AI provider credentials to backend/.env:\n"
                    f"AI_PROVIDER={provider_name}\n"
                    "AI_API_KEY=your_actual_api_key_here\n"
                    "AI_MODEL=\n\n"
                    "Or use offline mode:\nAI_PROVIDER=local\n\n"
                    "Supported providers: local, stability, replicate, clipdrop, fal, huggingface."
                ),
            )

        provider_cls = PROVIDER_REGISTRY.get(provider_name)
        if not provider_cls:
            raise InpaintingAPIError(
                f"Unsupported AI provider: '{provider_name}'. Supported: {list(PROVIDER_REGISTRY.keys())}"
            )

        return provider_cls(api_key=api_key, timeout=self.timeout), provider_name, model

    def get_status(self) -> Dict[str, Any]:
        """Returns current status of AI provider configuration."""
        provider_name, api_key, model = self.get_provider_config()
        configured = bool(api_key) or provider_name == "local"

        return {
            "status": "ok",
            "provider_configured": configured,
            "provider": provider_name,
            "model": model or ("telea" if provider_name == "local" else "default"),
            "mask_expansion_pixels": self.mask_expansion,
            "mask_feather_radius": self.mask_feather,
            "max_retries": self.max_retries,
            "message": (
                f"AI Inpainting engine connected via {provider_name.upper()}."
                if configured
                else "No AI provider configured. Set AI_API_KEY or AI_PROVIDER=local in backend/.env."
            ),
        }

    async def execute_provider_with_retry(
        self,
        provider: BaseInpaintingProvider,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        model: Optional[str],
    ) -> bytes:
        """
        Executes real inpainting against the provider with safe retry handling.
        Retries up to self.max_retries only for transient network or server errors.
        Does NOT retry for client errors (401/403/429).
        """
        attempts = 0
        last_exception = None

        while attempts <= self.max_retries:
            attempts += 1
            try:
                return await provider.inpaint(
                    image_bytes=image_bytes,
                    mask_bytes=mask_bytes,
                    prompt=prompt,
                    model=model,
                )
            except (InvalidAPIKeyError, RateLimitError):
                # Never retry authentication or quota failures
                raise
            except (ProviderTimeoutError, InpaintingAPIError) as e:
                last_exception = e
                # Check if retryable
                is_retryable = getattr(e, "retryable", False) or isinstance(e, ProviderTimeoutError)
                if is_retryable and attempts <= self.max_retries:
                    backoff = 1.0 * (2 ** (attempts - 1))
                    logger.warning(
                        f"Transient provider failure on attempt {attempts}/{self.max_retries + 1}. "
                        f"Retrying in {backoff:.1f}s... Error: {str(e)}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise e
            except Exception as e:
                last_exception = e
                if attempts <= self.max_retries:
                    backoff = 1.0 * (2 ** (attempts - 1))
                    logger.warning(f"Unexpected connection error. Retrying in {backoff:.1f}s: {str(e)}")
                    await asyncio.sleep(backoff)
                    continue
                raise InpaintingAPIError(f"AI provider failed after {attempts} attempts: {str(e)}")

        raise last_exception or InpaintingAPIError("AI inpainting failed after maximum retries.")

    async def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
    ) -> bytes:
        """
        Complete Context-Aware AI Inpainting Pipeline:
        1. Validates and preprocesses user mask.
        2. Applies configurable mask dilation/expansion (eliminates edge halos).
        3. Applies subtle mask edge feathering (creates soft alpha ramp).
        4. Maintains full visual context for the model.
        5. Preserves aspect ratio and scales proportionally if exceeding limits.
        6. Submits to real AI provider with automatic transient retries.
        7. Validates decoded AI output (dimensions, format, corruption check).
        8. Restores AI output to original dimensions.
        9. Safely composites AI result with original image:
           Final = AI_Result (inside mask) + Original_Image (outside mask)
           Guarantees untouched pixels outside mask remain 100% bit-exact identical.
        10. Logs diagnostics without exposing secrets.
        """
        start_time = time.time()
        provider_instance, provider_name, model = self.get_provider_instance()
        orig_w, orig_h = image.size

        # Diagnostic metrics before processing
        mask_metrics = MaskService.get_mask_metrics(mask)

        # 1. Mask Expansion (Dilation)
        expanded_mask = MaskService.expand_mask(mask, self.mask_expansion)

        # 2. Mask Edge Feathering for soft compositing boundary
        feathered_mask = MaskService.feather_mask(expanded_mask, self.mask_feather)

        # 3. Contextual Resolution Handling
        send_image = image
        send_mask = expanded_mask  # Send expanded binary mask to provider for full coverage

        if orig_w > self.max_dimension or orig_h > self.max_dimension:
            scale = min(self.max_dimension / orig_w, self.max_dimension / orig_h)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            # If RGBA, convert to RGB cleanly on white background for provider
            rgb_source = image.convert("RGB") if image.mode == "RGBA" else image
            send_image = rgb_source.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            send_mask = expanded_mask.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
        elif image.mode == "RGBA":
            # Normalization for providers that expect RGB
            send_image = image.convert("RGB")

        # Encode payloads
        img_buf = io.BytesIO()
        send_image.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        mask_buf = io.BytesIO()
        send_mask.save(mask_buf, format="PNG")
        mask_bytes = mask_buf.getvalue()

        # 4. Invoke Real AI Provider with automatic retry
        try:
            raw_ai_bytes = await self.execute_provider_with_retry(
                provider=provider_instance,
                image_bytes=img_bytes,
                mask_bytes=mask_bytes,
                prompt=prompt,
                model=model if model else None,
            )
        except Exception as e:
            duration = round(time.time() - start_time, 3)
            logger.error(
                f"Inpainting failed after {duration}s | Provider: {provider_name} | "
                f"Image: {orig_w}x{orig_h} | Mask coverage: {mask_metrics['coverage_percentage']}% | "
                f"Error: {type(e).__name__} - {str(e)}"
            )
            raise

        # 5. Output Validation
        if not raw_ai_bytes or len(raw_ai_bytes) == 0:
            raise InpaintingAPIError("AI provider returned an empty response.")

        try:
            ai_image = Image.open(io.BytesIO(raw_ai_bytes))
            ai_image.load()
            if ai_image.width <= 0 or ai_image.height <= 0:
                raise InpaintingAPIError("AI provider returned an image with invalid dimensions.")
        except Exception as e:
            raise InpaintingAPIError(f"AI provider returned corrupted or unreadable image data: {str(e)}")

        # 6. Restore to original dimensions if resized
        if ai_image.size != (orig_w, orig_h):
            ai_image = ai_image.resize((orig_w, orig_h), resample=Image.Resampling.LANCZOS)

        # 7. Safe Compositing: Final = AI_Result (inside mask) + Original (outside mask)
        # Guarantees that pixels outside the expanded/feathered mask remain bit-exact identical.
        final_image = MaskService.composite_inpainted_result(
            original_image=image,
            inpainted_image=ai_image,
            processed_mask=feathered_mask,
        )

        final_bytes = MaskService.to_bytes(final_image, format="PNG")
        duration = round(time.time() - start_time, 3)

        self.last_diagnostics = {
            "provider": provider_name,
            "model": model or "default",
            "original_width": orig_w,
            "original_height": orig_h,
            "mask_coverage_percentage": mask_metrics["coverage_percentage"],
            "mask_expansion_pixels": self.mask_expansion,
            "mask_feather_radius": self.mask_feather,
            "duration_seconds": duration,
        }

        logger.info(
            f"Inpainting succeeded in {duration}s | Provider: {provider_name} | "
            f"Size: {orig_w}x{orig_h} | Coverage: {mask_metrics['coverage_percentage']}%"
        )

        return final_bytes


# Conceptual entry point helper for backward compatibility
async def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    prompt: str = "seamless background fill",
) -> bytes:
    """
    Conceptual entry point: inpaint_image(image, mask).
    Returns generated PNG bytes.
    """
    service = InpaintingService()
    final_bytes = await service.inpaint(image, mask, prompt=prompt)
    return final_bytes
