"""
test_api.py
Automated test suite for AI Object Remover backend API and InpaintingService.
Uses Python's standard library unittest framework (no external pytest dependency required).
Covers:
- Test 1: Upload valid JPG + mask -> inpainting pipeline
- Test 2: Upload PNG + mask -> inpainting pipeline
- Test 3: Upload with empty/all-black mask -> 422 Unprocessable Entity
- Test 4: Invalid/corrupt image bytes -> 400 Bad Request
- Test 5: AI API unconfigured or provider failure -> proper HTTP error and JSON
- Test 6: Successful result -> valid generated PNG returned with dimensions
- Test 7: Dimension preservation -> output strictly matches original dimensions
- Test 8: Health check endpoint -> /api/health and /api/status
"""

import os
import io
import sys
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import app
from services.mask_service import MaskService, MaskValidationError
from services.inpainting_service import (
    InpaintingService,
    inpaint_image,
    ProviderNotConfiguredError,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
)

client = TestClient(app)


def create_test_image(format="JPEG", size=(400, 300), color=(100, 150, 200)) -> bytes:
    """Helper to create test image bytes."""
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 120, 120], fill=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def create_test_mask(size=(400, 300), has_selection=True) -> bytes:
    """Helper to create binary mask bytes (White = remove, Black = keep)."""
    mask = Image.new("L", size, color=0)
    if has_selection:
        draw = ImageDraw.Draw(mask)
        draw.ellipse([45, 45, 125, 125], fill=255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


class TestAIObjectRemoverAPI(unittest.TestCase):

    def test_health_endpoint(self):
        """Test 8: Verify health check endpoints."""
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("provider_configured", data)
        self.assertIn("provider", data)

        # Backward compatibility endpoint
        status_response = client.get("/api/status")
        self.assertEqual(status_response.status_code, 200)
        self.assertIn("provider_configured", status_response.json())

    def test_empty_mask_selection_rejected(self):
        """Test 3: Empty selection (completely black mask) must be rejected with 422."""
        img_bytes = create_test_image(format="JPEG")
        empty_mask = create_test_mask(has_selection=False)

        response = client.post(
            "/api/remove-object",
            files={
                "image": ("test.jpg", img_bytes, "image/jpeg"),
                "mask": ("mask.png", empty_mask, "image/png"),
            },
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["detail"]["error"], "EMPTY_SELECTION")
        self.assertTrue("empty" in data["detail"]["message"].lower() or "select" in data["detail"]["message"].lower())

    def test_invalid_image_data_rejected(self):
        """Test 4: Invalid/corrupt image bytes must be rejected with 400 Bad Request."""
        corrupt_bytes = b"NOT_AN_IMAGE_FILE_DATA_CORRUPT"
        mask_bytes = create_test_mask()

        response = client.post(
            "/api/remove-object",
            files={
                "image": ("corrupt.jpg", corrupt_bytes, "image/jpeg"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            },
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data["detail"])

    def test_unsupported_format_rejected(self):
        """Test unsupported formats (e.g. GIF) return 415 or 400."""
        gif_img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        gif_img.save(buf, format="GIF")
        mask_bytes = create_test_mask(size=(100, 100))

        response = client.post(
            "/api/remove-object",
            files={
                "image": ("test.gif", buf.getvalue(), "image/gif"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            },
        )
        self.assertIn(response.status_code, (400, 415))

    def test_unconfigured_provider_returns_503(self):
        """Test 5: When AI API key is not configured, returns clean HTTP 503."""
        img_bytes = create_test_image(format="JPEG")
        mask_bytes = create_test_mask()

        with patch.object(InpaintingService, "get_provider_config", return_value=("stability", "", "")):
            response = client.post(
                "/api/remove-object",
                files={
                    "image": ("test.jpg", img_bytes, "image/jpeg"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                },
            )
            self.assertEqual(response.status_code, 503)
            data = response.json()
            self.assertEqual(data["detail"]["error"], "AI_PROVIDER_NOT_CONFIGURED")
            self.assertIn("instructions", data["detail"])

    def test_successful_jpg_inpainting_preserves_dimensions(self):
        """Test 1 & 6: Upload valid JPG + mask -> inpaint -> returns generated PNG with original dimensions."""
        orig_w, orig_h = 640, 480
        img_bytes = create_test_image(format="JPEG", size=(orig_w, orig_h))
        mask_bytes = create_test_mask(size=(orig_w, orig_h))

        mock_result_img = Image.new("RGB", (orig_w, orig_h), color=(100, 150, 200))
        mock_buf = io.BytesIO()
        mock_result_img.save(mock_buf, format="PNG")
        mock_png_bytes = mock_buf.getvalue()

        with patch("app.inpaint_image", new_callable=AsyncMock) as mock_inpaint:
            mock_inpaint.return_value = mock_png_bytes

            response = client.post(
                "/api/remove-object",
                files={
                    "image": ("photo.jpg", img_bytes, "image/jpeg"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/png")
            self.assertEqual(response.headers["x-original-width"], str(orig_w))
            self.assertEqual(response.headers["x-original-height"], str(orig_h))

            result_image = Image.open(io.BytesIO(response.content))
            self.assertEqual(result_image.size, (orig_w, orig_h))

    def test_successful_png_inpainting(self):
        """Test 2: Upload valid PNG + mask -> inpaint -> returns valid image."""
        orig_w, orig_h = 500, 500
        img_bytes = create_test_image(format="PNG", size=(orig_w, orig_h))
        mask_bytes = create_test_mask(size=(orig_w, orig_h))

        mock_result_img = Image.new("RGB", (orig_w, orig_h), color=(240, 240, 240))
        mock_buf = io.BytesIO()
        mock_result_img.save(mock_buf, format="PNG")
        mock_png_bytes = mock_buf.getvalue()

        with patch("app.inpaint_image", new_callable=AsyncMock) as mock_inpaint:
            mock_inpaint.return_value = mock_png_bytes

            response = client.post(
                "/api/remove-object",
                files={
                    "image": ("photo.png", img_bytes, "image/png"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/png")

    def test_provider_error_handling_statuses(self):
        """Test 5: Granular HTTP statuses for provider errors."""
        img_bytes = create_test_image()
        mask_bytes = create_test_mask()

        # 401 Unauthorized
        with patch("app.inpaint_image", side_effect=InvalidAPIKeyError("Invalid API key")):
            res = client.post("/api/remove-object", files={"image": ("t.jpg", img_bytes, "image/jpeg"), "mask": ("m.png", mask_bytes, "image/png")})
            self.assertEqual(res.status_code, 401)
            self.assertEqual(res.json()["detail"]["error"], "INVALID_API_KEY")

        # 429 Rate limit
        with patch("app.inpaint_image", side_effect=RateLimitError("Rate limit exceeded")):
            res = client.post("/api/remove-object", files={"image": ("t.jpg", img_bytes, "image/jpeg"), "mask": ("m.png", mask_bytes, "image/png")})
            self.assertEqual(res.status_code, 429)
            self.assertEqual(res.json()["detail"]["error"], "RATE_LIMIT_EXCEEDED")

        # 504 Timeout
        with patch("app.inpaint_image", side_effect=ProviderTimeoutError("Timed out")):
            res = client.post("/api/remove-object", files={"image": ("t.jpg", img_bytes, "image/jpeg"), "mask": ("m.png", mask_bytes, "image/png")})
            self.assertEqual(res.status_code, 504)
            self.assertEqual(res.json()["detail"]["error"], "PROVIDER_TIMEOUT")

        # 502 Bad Gateway
        with patch("app.inpaint_image", side_effect=InpaintingAPIError("Provider down")):
            res = client.post("/api/remove-object", files={"image": ("t.jpg", img_bytes, "image/jpeg"), "mask": ("m.png", mask_bytes, "image/png")})
            self.assertEqual(res.status_code, 502)
            self.assertEqual(res.json()["detail"]["error"], "AI_INPAINTING_ERROR")


class TestDimensionPreservation(unittest.IsolatedAsyncioTestCase):

    async def test_dimension_preservation_and_restoration(self):
        """
        Test 7: If provider output dimension differs (e.g. provider returned 512x512 for a 750x450 image),
        InpaintingService automatically restores exact original dimensions without distortion.
        """
        orig_size = (750, 450)
        orig_img = Image.new("RGB", orig_size, color="green")
        mask_img = Image.new("L", orig_size, color=0)
        ImageDraw.Draw(mask_img).rectangle([50, 50, 100, 100], fill=255)

        # Provider returns 512x512
        provider_out_img = Image.new("RGB", (512, 512), color="green")
        buf = io.BytesIO()
        provider_out_img.save(buf, format="PNG")
        provider_bytes = buf.getvalue()

        service = InpaintingService()
        mock_provider = AsyncMock()
        mock_provider.inpaint.return_value = provider_bytes

        with patch.object(service, "get_provider_instance", return_value=(mock_provider, "stability", "")):
            final_bytes = await service.inpaint(orig_img, mask_img)
            final_img = Image.open(io.BytesIO(final_bytes))

            # Must be restored strictly to original (750, 450)
            self.assertEqual(final_img.size, orig_size)


if __name__ == "__main__":
    unittest.main()
