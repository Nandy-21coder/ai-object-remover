"""
tests/test_inpainting.py
Automated unit and integration tests for inpainting.py components:
- MaskService: image loading, format verification, mask binarization, dilation, feathering, compositing
- InpaintingService: provider configuration, retry handling, dimension preservation
"""

import io
import sys
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.backend.inpainting import (
    MaskService,
    MaskValidationError,
    InpaintingService,
    ProviderNotConfiguredError,
    inpaint_image,
)


def create_sample_image(w=100, h=100, color=(50, 100, 150)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def create_sample_mask(w=100, h=100, select_box=(20, 20, 60, 60)) -> Image.Image:
    mask = Image.new("L", (w, h), color=0)
    if select_box:
        draw = ImageDraw.Draw(mask)
        draw.rectangle(select_box, fill=255)
    return mask


# ==============================================================================
# 1. MaskService Validation & Binarization Tests
# ==============================================================================

def test_mask_service_load_image_valid():
    """Verify MaskService.load_image loads valid RGB image."""
    img = create_sample_image(120, 80)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    loaded = MaskService.load_image(buf.getvalue())
    assert loaded.size == (120, 80)
    assert loaded.mode in ("RGB", "RGBA")


def test_mask_service_load_image_empty():
    """Verify MaskService.load_image raises MaskValidationError on empty bytes."""
    with pytest.raises(MaskValidationError):
        MaskService.load_image(b"")


def test_mask_service_load_image_corrupted():
    """Verify MaskService.load_image raises MaskValidationError on corrupt data."""
    with pytest.raises(MaskValidationError):
        MaskService.load_image(b"not-an-image-data")


def test_mask_service_load_mask_empty_selection():
    """Verify empty mask raises MaskValidationError."""
    empty_mask = Image.new("L", (100, 100), color=0)
    buf = io.BytesIO()
    empty_mask.save(buf, format="PNG")

    with pytest.raises(MaskValidationError) as exc:
        MaskService.load_and_binarize_mask(buf.getvalue(), target_size=(100, 100))
    assert "mask is empty" in str(exc.value).lower()


def test_mask_service_load_mask_resizes_to_target():
    """Verify mask is automatically resized to match target image dimensions."""
    mask = create_sample_mask(50, 50)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")

    binarized = MaskService.load_and_binarize_mask(buf.getvalue(), target_size=(200, 200))
    assert binarized.size == (200, 200)
    # Check that mask has active 255 values
    extrema = binarized.getextrema()
    assert extrema == (0, 255)


def test_mask_service_expand_and_feather():
    """Verify dilation increases mask area and feathering preserves bounds."""
    mask = create_sample_mask(100, 100, select_box=(40, 40, 60, 60))
    original_metrics = MaskService.get_mask_metrics(mask)

    expanded = MaskService.expand_mask(mask, expansion_pixels=4)
    expanded_metrics = MaskService.get_mask_metrics(expanded)

    assert expanded_metrics["selected_pixels"] > original_metrics["selected_pixels"]

    feathered = MaskService.feather_mask(expanded, feather_radius=2.0)
    assert feathered.size == mask.size


def test_mask_service_compositing_preserves_untouched_pixels():
    """
    Verify that untouched pixels outside the mask are strictly preserved bit-exact.
    """
    orig = Image.new("RGB", (100, 100), color=(10, 20, 30))
    # Synthetic candidate where everything was changed to yellow
    candidate = Image.new("RGB", (100, 100), color=(255, 255, 0))

    # Mask modifying only a 10x10 corner
    mask = Image.new("L", (100, 100), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, 10, 10], fill=255)

    composited = MaskService.composite_inpainted_result(orig, candidate, mask)

    # Check a point outside the mask (e.g. at 50, 50)
    orig_pixel = orig.getpixel((50, 50))
    comp_pixel = composited.getpixel((50, 50))
    assert comp_pixel == orig_pixel == (10, 20, 30)

    # Check a point inside the mask (e.g. at 5, 5)
    inpaint_pixel = composited.getpixel((5, 5))
    assert inpaint_pixel == (255, 255, 0)


# ==============================================================================
# 2. InpaintingService Provider & Execution Tests
# ==============================================================================

def test_inpainting_service_status():
    """Verify InpaintingService returns expected status metadata."""
    service = InpaintingService()
    status_dict = service.get_status()
    assert "status" in status_dict
    assert "provider" in status_dict
    assert "mask_expansion_pixels" in status_dict
    assert "mask_feather_radius" in status_dict


def test_end_to_end_inpaint_service_execution():
    """
    Tests end-to-end inpaint_image() call with active provider.
    Skips if provider is not configured.
    """
    import asyncio

    service = InpaintingService()
    status_dict = service.get_status()
    if not status_dict.get("provider_configured"):
        pytest.skip("AI provider not configured; skipping integration inpainting test.")

    test_img = create_sample_image(64, 64, color=(80, 120, 160))
    test_mask = create_sample_mask(64, 64, select_box=(16, 16, 32, 32))

    try:
        result_bytes = asyncio.run(inpaint_image(test_img, test_mask))
    except ProviderNotConfiguredError:
        pytest.skip("Provider credentials missing.")

    assert len(result_bytes) > 0
    res_img = Image.open(io.BytesIO(result_bytes))
    assert res_img.size == (64, 64)
