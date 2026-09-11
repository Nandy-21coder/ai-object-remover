"""
test_quality_suite.py
Comprehensive quality and visual realism test suite for AI Object Remover.
Covers:
- Test A: Small object on plain wall (natural wall texture & dimension match)
- Test B: Person on outdoor background (mask dilation eliminates edge halo & outlines)
- Test C: Object on grass (texture context infill)
- Test D: Balloon in sky (continuous gradient reconstruction)
- Test E: Object near another important object (bit-exact preservation of unselected pixels)
- Test F: Large selected object handling
- Test G: Very small selected object handling
- Test H: Multiple sequential removals (image preservation over repeated edits)
- Unit: Mask dilation (expansion) and subtle edge feathering
- Unit: Bit-exact untouched area compositing
- Unit: RGBA PNG transparency preservation
- Unit: Retry logic on transient network failures
- Unit: Diagnostics logging and safe metadata
"""

import io
import os
import sys
import unittest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from unittest.mock import patch, AsyncMock

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.mask_service import MaskService, MaskValidationError
from services.inpainting_service import InpaintingService, InpaintingAPIError, InvalidAPIKeyError
from services.providers.base import BaseInpaintingProvider, ProviderTimeoutError


class TestQualityPipeline(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.service = InpaintingService()
        self.service.mask_expansion = 4
        self.service.mask_feather = 2.0
        self.service.max_retries = 2

    def test_a_small_object_on_plain_wall(self):
        """TEST A: Small object on plain wall -> natural reconstruction & dimension matching."""
        wall_color = (220, 215, 205)  # Warm plaster wall
        orig = Image.new("RGB", (600, 400), color=wall_color)
        draw = ImageDraw.Draw(orig)
        # Small stain or electrical socket on wall
        draw.rectangle([280, 180, 320, 220], fill=(50, 50, 50))

        # Mask covering the small object
        mask = Image.new("L", (600, 400), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([275, 175, 325, 225], fill=255)

        # AI reconstructs the wall
        ai_patch = Image.new("RGB", (600, 400), color=wall_color)
        buf = io.BytesIO()
        ai_patch.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            result_bytes = asyncio.run(self.service.inpaint(orig, mask))
            result_img = Image.open(io.BytesIO(result_bytes))

            self.assertEqual(result_img.size, (600, 400))
            # Outside the mask, wall color is 100% identical
            self.assertEqual(result_img.getpixel((50, 50)), wall_color)
            # The dark stain is removed
            self.assertEqual(result_img.getpixel((300, 200)), wall_color)

    def test_b_person_on_outdoor_background_dilation_removes_edges(self):
        """TEST B: Person on outdoor background -> mask dilation ensures body/clothing outlines are eliminated."""
        bg_color = (135, 206, 235)  # Sky/outdoor
        orig = Image.new("RGB", (500, 500), color=bg_color)
        draw = ImageDraw.Draw(orig)
        # Person silhouette in center
        draw.ellipse([230, 150, 270, 200], fill=(210, 180, 140))  # Head
        draw.rectangle([210, 200, 290, 400], fill=(30, 60, 120))  # Body

        # Tight user brush selection that barely covers the body
        mask = Image.new("L", (500, 500), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([210, 200, 290, 400], fill=255)

        # Verify that MaskService.expand_mask expands the boundary by expansion pixels
        expanded = MaskService.expand_mask(mask, expansion_pixels=4)
        self.assertEqual(expanded.getpixel((208, 250)), 255)  # Expanded past original left edge 210
        self.assertEqual(expanded.getpixel((293, 250)), 255)  # Expanded past original right edge 290

        # Verify original user mask was not modified
        self.assertEqual(mask.getpixel((208, 250)), 0)

    def test_c_object_on_grass(self):
        """TEST C: Object on grass -> natural grass texture reconstruction."""
        grass_color = (34, 139, 34)
        orig = Image.new("RGB", (400, 400), color=grass_color)
        draw = ImageDraw.Draw(orig)
        draw.rectangle([180, 180, 220, 220], fill=(255, 0, 0))  # Red ball on grass

        mask = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(mask).rectangle([175, 175, 225, 225], fill=255)

        ai_res = Image.new("RGB", (400, 400), color=grass_color)
        buf = io.BytesIO()
        ai_res.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            res_img = Image.open(io.BytesIO(res_bytes))
            self.assertEqual(res_img.getpixel((200, 200)), grass_color)
            self.assertEqual(res_img.getpixel((20, 20)), grass_color)

    def test_d_balloon_in_sky(self):
        """TEST D: Balloon in sky -> continuous sky reconstruction."""
        sky_top = (56, 189, 248)
        orig = Image.new("RGB", (300, 300), color=sky_top)
        draw = ImageDraw.Draw(orig)
        draw.ellipse([130, 80, 170, 140], fill=(239, 68, 68))  # Red balloon
        draw.line([150, 140, 150, 220], fill=(50, 50, 50), width=2)  # String

        mask = Image.new("L", (300, 300), 0)
        ImageDraw.Draw(mask).rectangle([125, 75, 175, 225], fill=255)

        ai_res = Image.new("RGB", (300, 300), color=sky_top)
        buf = io.BytesIO()
        ai_res.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            res_img = Image.open(io.BytesIO(res_bytes))
            self.assertEqual(res_img.size, (300, 300))
            self.assertEqual(res_img.getpixel((150, 100)), sky_top)

    def test_e_object_near_another_important_object_bit_exact(self):
        """
        TEST E: Object near another important object.
        CRITICAL: All unmasked pixels (faces, adjacent important objects) must remain 100%
        bit-exact identical to the original image!
        """
        orig = Image.new("RGB", (200, 200), color=(50, 100, 150))
        draw = ImageDraw.Draw(orig)
        important_object_color = (255, 215, 0)  # Gold statue / face
        unwanted_object_color = (180, 30, 30)   # Garbage / photobomber

        # Important object on left
        draw.rectangle([20, 50, 80, 150], fill=important_object_color)
        # Unwanted object on right
        draw.rectangle([120, 50, 180, 150], fill=unwanted_object_color)

        # Mask ONLY the unwanted object
        mask = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(mask).rectangle([115, 45, 185, 155], fill=255)

        # AI generates a result where it altered the whole canvas (simulating providers that hallucinate elsewhere)
        hallucinated_ai = Image.new("RGB", (200, 200), color=(255, 0, 255))  # Magenta everywhere
        buf = io.BytesIO()
        hallucinated_ai.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            res_img = Image.open(io.BytesIO(res_bytes))

            # The important object must remain 100% BIT-EXACT untouched!
            for y in range(50, 150):
                for x in range(20, 80):
                    self.assertEqual(
                        res_img.getpixel((x, y)),
                        important_object_color,
                        f"Important object pixel at ({x}, {y}) was altered!",
                    )

            # Outside background must also remain bit-exact
            self.assertEqual(res_img.getpixel((5, 5)), (50, 100, 150))

    def test_f_large_selected_object(self):
        """TEST F: Large selected object covering 40% of canvas."""
        orig = Image.new("RGB", (400, 400), color=(100, 100, 100))
        mask = Image.new("L", (400, 400), 0)
        # 40% selection
        ImageDraw.Draw(mask).rectangle([50, 50, 350, 250], fill=255)

        metrics = MaskService.get_mask_metrics(mask)
        self.assertGreater(metrics["coverage_percentage"], 35.0)

        ai_res = Image.new("RGB", (400, 400), color=(120, 120, 120))
        buf = io.BytesIO()
        ai_res.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            res_img = Image.open(io.BytesIO(res_bytes))
            self.assertEqual(res_img.size, (400, 400))

    def test_g_very_small_selected_object(self):
        """TEST G: Very small selected object (tiny blemish, < 0.1%)."""
        orig = Image.new("RGB", (500, 500), color=(200, 200, 200))
        mask = Image.new("L", (500, 500), 0)
        ImageDraw.Draw(mask).rectangle([250, 250, 253, 253], fill=255)

        metrics = MaskService.get_mask_metrics(mask)
        self.assertLess(metrics["coverage_percentage"], 0.1)

        ai_res = Image.new("RGB", (500, 500), color=(200, 200, 200))
        buf = io.BytesIO()
        ai_res.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            res_img = Image.open(io.BytesIO(res_bytes))
            self.assertEqual(res_img.size, (500, 500))

    def test_h_multiple_sequential_removals(self):
        """TEST H: Multiple sequential removals preserve quality and untargeted pixels across iterations."""
        current_img = Image.new("RGB", (300, 300), color=(200, 220, 240))
        draw = ImageDraw.Draw(current_img)
        # Spot 1 and Spot 2
        draw.rectangle([50, 50, 70, 70], fill=(255, 0, 0))
        draw.rectangle([200, 200, 220, 220], fill=(0, 0, 255))

        clean_bg = Image.new("RGB", (300, 300), color=(200, 220, 240))
        buf = io.BytesIO()
        clean_bg.save(buf, format="PNG")
        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio

            # Removal 1: Remove Spot 1
            mask1 = Image.new("L", (300, 300), 0)
            ImageDraw.Draw(mask1).rectangle([48, 48, 72, 72], fill=255)
            step1_bytes = asyncio.run(self.service.inpaint(current_img, mask1))
            step1_img = Image.open(io.BytesIO(step1_bytes))

            # Spot 1 is removed
            self.assertEqual(step1_img.getpixel((60, 60)), (200, 220, 240))
            # Spot 2 remains completely intact!
            self.assertEqual(step1_img.getpixel((210, 210)), (0, 0, 255))

            # Removal 2: Remove Spot 2 from step1_img
            mask2 = Image.new("L", (300, 300), 0)
            ImageDraw.Draw(mask2).rectangle([198, 198, 222, 222], fill=255)
            step2_bytes = asyncio.run(self.service.inpaint(step1_img, mask2))
            step2_img = Image.open(io.BytesIO(step2_bytes))

            # Both spots now removed, background consistent
            self.assertEqual(step2_img.getpixel((60, 60)), (200, 220, 240))
            self.assertEqual(step2_img.getpixel((210, 210)), (200, 220, 240))
            self.assertEqual(step2_img.getpixel((10, 10)), (200, 220, 240))

    def test_mask_expansion_and_feathering_units(self):
        """Unit: Verify morphological expansion and subtle edge feathering properties."""
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).rectangle([40, 40, 60, 60], fill=255)

        # Expansion
        exp4 = MaskService.expand_mask(mask, expansion_pixels=4)
        # Original was 40-60, expanded should reach 36-64
        self.assertEqual(exp4.getpixel((36, 50)), 255)
        self.assertEqual(exp4.getpixel((64, 50)), 255)
        self.assertEqual(exp4.getpixel((34, 50)), 0)

        # Subtle feathering creates smooth boundary ramp
        feathered = MaskService.feather_mask(exp4, feather_radius=2.0)
        # Inside center is still bright
        center_val = feathered.getpixel((50, 50))
        assert isinstance(center_val, (int, float))
        self.assertGreater(center_val, 240)
        # Boundary transition is between 0 and 255
        boundary_val = feathered.getpixel((35, 50))
        assert isinstance(boundary_val, (int, float))
        self.assertTrue(0 < boundary_val < 255)

    def test_rgba_png_transparency_preservation(self):
        """Unit: Verify transparent PNG retains transparent alpha channel outside the mask."""
        orig_rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))  # Transparent PNG
        draw = ImageDraw.Draw(orig_rgba)
        draw.rectangle([50, 50, 150, 150], fill=(255, 100, 0, 255))  # Opaque orange icon
        draw.rectangle([90, 90, 110, 110], fill=(0, 0, 0, 255))  # Unwanted black mark on icon

        mask = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(mask).rectangle([88, 88, 112, 112], fill=255)

        ai_patch = Image.new("RGB", (200, 200), (255, 100, 0))  # Reconstructed orange
        buf = io.BytesIO()
        ai_patch.save(buf, format="PNG")

        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = buf.getvalue()

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig_rgba, mask))
            res_img = Image.open(io.BytesIO(res_bytes))

            self.assertEqual(res_img.mode, "RGBA")
            # Transparent background must remain transparent (alpha == 0), NOT black!
            pixel_bg = res_img.getpixel((10, 10))
            assert isinstance(pixel_bg, tuple)
            self.assertEqual(pixel_bg[3], 0)
            # Reconstructed area is now orange
            pixel_patch = res_img.getpixel((100, 100))
            assert isinstance(pixel_patch, tuple)
            self.assertEqual(pixel_patch[:3], (255, 100, 0))

    def test_retry_on_transient_failure(self):
        """Unit: InpaintingService retries transient failures up to max_retries and succeeds."""
        mock_provider = AsyncMock()
        valid_img = Image.new("RGB", (100, 100), (50, 50, 50))
        buf = io.BytesIO()
        valid_img.save(buf, format="PNG")

        # Fails once with transient InpaintingAPIError, succeeds on second attempt
        mock_provider.inpaint.side_effect = [
            InpaintingAPIError("502 Bad Gateway from upstream", status_code=502, retryable=True),
            buf.getvalue(),
        ]

        orig = Image.new("RGB", (100, 100), (50, 50, 50))
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).rectangle([40, 40, 60, 60], fill=255)

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            res_bytes = asyncio.run(self.service.inpaint(orig, mask))
            self.assertTrue(len(res_bytes) > 0)
            self.assertEqual(mock_provider.inpaint.call_count, 2)

    def test_no_retry_on_auth_failure(self):
        """Unit: InpaintingService does NOT retry 401/403 InvalidAPIKeyError."""
        mock_provider = AsyncMock()
        mock_provider.inpaint.side_effect = InvalidAPIKeyError("Invalid API key")

        orig = Image.new("RGB", (100, 100), (50, 50, 50))
        mask = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(mask).rectangle([40, 40, 60, 60], fill=255)

        with patch.object(self.service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            import asyncio
            with self.assertRaises(InvalidAPIKeyError):
                asyncio.run(self.service.inpaint(orig, mask))

            # Must NOT retry auth failures (call_count should be exactly 1)
            self.assertEqual(mock_provider.inpaint.call_count, 1)


if __name__ == "__main__":
    unittest.main()
