"""
mask_service.py
Provides processing, verification, dilation, feathering, and safe compositing
for images and selection masks.

Mask specification:
- Dimensions must match original image dimensions.
- Strict binary mask:
    - 255 (White) = Area to remove (inpaint target)
    - 0 (Black) = Area to preserve (background / context)
- Mask Expansion (Dilation): Expands the removal boundary slightly to eliminate leftover outlines.
- Mask Edge Feathering: Subtle Gaussian smoothing of the mask edge for seamless compositing.
- Safe Compositing: Guarantees untouched areas (where mask == 0) remain 100% bit-exact identical to the original image.
"""

import io
import os
from typing import Tuple, Dict, Any
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class MaskValidationError(Exception):
    """Raised when an image or mask fails format, dimension, or content validation."""
    pass


class MaskService:
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

    @classmethod
    def load_image(cls, image_bytes: bytes) -> Image.Image:
        """Loads and validates an image from raw bytes."""
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

        # Respect EXIF orientation if present
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        # Maintain RGB or RGBA mode; convert palletized or CMYK safely
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        if image.width <= 0 or image.height <= 0:
            raise MaskValidationError("Invalid image dimensions.")

        return image

    @classmethod
    def load_and_binarize_mask(cls, mask_bytes: bytes, target_size: Tuple[int, int]) -> Image.Image:
        """
        Loads mask bytes, ensures strict alignment with target_size,
        and converts to a strict binary grayscale mask ('L' mode) where:
        255 = White (Area to remove)
        0 = Black (Area to preserve)
        """
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

        # Ensure mask dimensions match the target image dimensions exactly
        if raw_mask.size != (target_w, target_h):
            raw_mask = raw_mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        # Handle various mask input formats (RGBA from canvas, Grayscale, etc.)
        if raw_mask.mode == "L":
            binary_mask = raw_mask.point(lambda p: 255 if p > 50 else 0, mode="L")
        else:
            rgba = raw_mask.convert("RGBA")
            arr = np.array(rgba)
            r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            max_rgb = np.maximum(np.maximum(r, g), b)
            # If mask is opaque black/white (canvas export: background black, selection white)
            all_opaque = (a > 200).mean() > 0.95
            if all_opaque:
                is_mask = max_rgb > 50
            else:
                # Transparent canvas: stroke pixels have alpha > 30 and non-black color, or simply alpha > 30
                is_mask = (a > 30) & (max_rgb > 20)
                if not is_mask.any():
                    is_mask = a > 30

            out = np.zeros(arr.shape[:2], dtype=np.uint8)
            out[is_mask] = 255
            binary_mask = Image.fromarray(out, mode="L")

        # Verify that the mask is not completely black
        extrema = binary_mask.getextrema()
        if extrema == (0, 0) or extrema is None or extrema[1] == 0:
            raise MaskValidationError(
                "The selection mask is empty. Please brush over the object or text you want to remove."
            )

        return binary_mask

    @classmethod
    def expand_mask(cls, mask: Image.Image, expansion_pixels: int = 4) -> Image.Image:
        """
        Applies morphological dilation to expand the selection area by expansion_pixels.
        This ensures object boundaries, edge halos, and slight user under-painting
        are completely encompassed.
        Does not alter the original mask in-place.
        """
        if expansion_pixels <= 0:
            return mask.copy()

        # MaxFilter with kernel size = 2 * r + 1 expands white pixels by r
        filter_size = max(3, 2 * expansion_pixels + 1)
        expanded = mask.filter(ImageFilter.MaxFilter(size=filter_size))
        return expanded

    @classmethod
    def feather_mask(cls, mask: Image.Image, feather_radius: float = 2.0) -> Image.Image:
        """
        Applies subtle Gaussian feathering to mask edges for seamless alpha blending.
        Feathering is strictly controlled and subtle so objects are not partially preserved.
        """
        if feather_radius <= 0:
            return mask.copy()

        feathered = mask.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
        return feathered

    @classmethod
    def get_mask_metrics(cls, mask: Image.Image) -> Dict[str, Any]:
        """
        Calculates diagnostic metrics for the mask without logging sensitive data:
        - width, height
        - selected_pixel_count
        - total_pixel_count
        - coverage_percentage
        """
        w, h = mask.size
        total = w * h
        # Count non-zero pixels
        histogram = mask.histogram()
        # histogram for 'L' has 256 entries; index 0 is black (keep)
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
        """
        Safely composites the AI inpainting result back onto the original image.

        Guarantees:
        - Where processed_mask == 0 (untouched areas): Final image is 100% bit-exact identical to original_image.
        - Where processed_mask == 255 (inpainted area): Final image uses inpainted_image.
        - Where 0 < processed_mask < 255 (feathered edge): Seamless alpha ramp transition.
        - Preserves alpha transparency for PNGs without creating black borders.
        """
        orig_w, orig_h = original_image.size

        # Ensure inpainted_image matches original dimensions
        if inpainted_image.size != (orig_w, orig_h):
            inpainted_image = inpainted_image.resize((orig_w, orig_h), resample=Image.Resampling.LANCZOS)

        # Ensure mask matches original dimensions
        if processed_mask.size != (orig_w, orig_h):
            processed_mask = processed_mask.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

        # Handle RGBA images
        if original_image.mode == "RGBA":
            orig_rgb = original_image.convert("RGB")
            inpaint_rgb = inpainted_image.convert("RGB") if inpainted_image.mode != "RGB" else inpainted_image
            composite_rgb = Image.composite(inpaint_rgb, orig_rgb, processed_mask)

            # Re-attach original alpha channel
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
        """
        Validates original image and selection mask, returning validated PIL Image
        and strict binary mask.
        """
        image = cls.load_image(image_bytes)
        mask = cls.load_and_binarize_mask(mask_bytes, image.size)
        return image, mask

    @staticmethod
    def to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """Encodes a PIL Image into bytes in the specified format."""
        buf = io.BytesIO()
        image.save(buf, format=format)
        return buf.getvalue()
