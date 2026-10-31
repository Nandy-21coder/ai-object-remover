"""
tests/test_api.py
=================
Automated integration tests for the AI Object Remover FastAPI backend.

Tests use the HTTPX AsyncClient with FastAPI's TestClient to call real
application routes without a live server process. All image and mask
fixtures are generated programmatically in-memory so the test suite has
zero external file dependencies.

Running the tests:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short

Test modules:
    - Health check endpoints
    - Auth config endpoint
    - /api/remove-object (happy path, validation failures, size limits)
    - /api/smart-circle-segment (happy path, bad points, empty image)
    - Static file serving routes
    - Response header validation (X-Process-Time-Ms, etc.)
"""

import io
import json
import sys
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of how pytest is invoked
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Patch AI_PROVIDER to "lama" before importing the app so that the test
# environment does not require a live cloud API key. The LaMa ONNX model
# must be present at PROJECT_ROOT / "inpainting_lama.onnx" for inpainting
# tests to pass; tests that do not call the inpainting pipeline will pass
# without the model file.
# ---------------------------------------------------------------------------
os.environ.setdefault("AI_PROVIDER", "lama")

from fastapi.testclient import TestClient  # noqa: E402
from app import app  # noqa: E402

# ---------------------------------------------------------------------------
# Test client — synchronous TestClient wraps the async ASGI app
# ---------------------------------------------------------------------------
client = TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# Fixture helpers
# ===========================================================================

def _make_png_bytes(width: int = 64, height: int = 64, color: tuple = (100, 150, 200)) -> bytes:
    """
    Creates a minimal in-memory PNG image using Pillow.
    Returns raw PNG bytes suitable for multipart upload.

    Args:
        width:  Image width in pixels.
        height: Image height in pixels.
        color:  RGB tuple for the solid fill color.
    Returns:
        PNG-encoded bytes.
    """
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_white_mask_png(width: int = 64, height: int = 64, coverage: float = 0.3) -> bytes:
    """
    Creates a binary mask where the center `coverage` fraction of pixels
    are white (removal region) and the remainder are black (preserve).

    Args:
        width:    Mask width in pixels — must match the source image.
        height:   Mask height in pixels — must match the source image.
        coverage: Fraction of image area to mark as white (0.0 – 1.0).
    Returns:
        PNG-encoded bytes of an L-mode (grayscale) mask.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((height, width), dtype=np.uint8)
    # Fill the center region with 255 (white = remove)
    cy, cx = height // 2, width // 2
    half_h = max(1, int(height * coverage ** 0.5 / 2))
    half_w = max(1, int(width * coverage ** 0.5 / 2))
    arr[cy - half_h: cy + half_h, cx - half_w: cx + half_w] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_empty_mask_png(width: int = 64, height: int = 64) -> bytes:
    """
    Creates a fully-black (all-zero) mask — simulates user submitting
    without brushing anything over the image.
    """
    from PIL import Image
    img = Image.new("L", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Creates a minimal in-memory JPEG image."""
    from PIL import Image
    img = Image.new("RGB", (width, height), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ===========================================================================
# TC-API-001 — TC-API-003: Health / Status Endpoints
# ===========================================================================

class TestHealthEndpoints:
    """Tests for the /health, /api/health, and /api/status endpoints."""

    def test_api_health_returns_200(self):
        """
        TC-API-001
        Endpoint: GET /api/health
        Expected: HTTP 200 with JSON body containing 'status' key.
        """
        response = client.get("/api/health")
        assert response.status_code == 200, (
            f"Expected 200 OK, got {response.status_code}: {response.text}"
        )

    def test_api_health_response_has_required_keys(self):
        """
        TC-API-002
        Endpoint: GET /api/health
        Expected: JSON body contains 'status' and 'model' keys as specified
                  in the API contract (status='ok'/'error', model='ready'/'not_ready').
        """
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body, f"'status' key missing from health response: {body}"
        assert "model" in body, f"'model' key missing from health response: {body}"
        assert body["status"] in ("ok", "error"), (
            f"Unexpected status value: {body['status']}"
        )

    def test_api_status_alias_returns_200(self):
        """
        TC-API-003
        Endpoint: GET /api/status (alias of /api/health)
        Expected: HTTP 200.
        """
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_health_root_alias_returns_200(self):
        """
        TC-API-004
        Endpoint: GET /health (root alias)
        Expected: HTTP 200.
        """
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_memory_rss_field_present(self):
        """
        TC-API-005
        Endpoint: GET /api/health
        Expected: JSON response includes 'memory_rss_mb' as a float (or 0.0
                  when psutil is not installed).
        """
        response = client.get("/api/health")
        body = response.json()
        assert "memory_rss_mb" in body, "memory_rss_mb missing from health response"
        assert isinstance(body["memory_rss_mb"], (int, float))


# ===========================================================================
# TC-API-006: Auth Config Endpoint
# ===========================================================================

class TestAuthConfigEndpoint:
    """Tests for the /api/auth/config endpoint."""

    def test_auth_config_returns_200(self):
        """
        TC-API-006
        Endpoint: GET /api/auth/config
        Expected: HTTP 200 with JSON body containing 'supabase_url' and
                  'supabase_anon_key' keys. Values may be empty strings when
                  environment variables are not configured.
        """
        response = client.get("/api/auth/config")
        assert response.status_code == 200
        body = response.json()
        assert "supabase_url" in body
        assert "supabase_anon_key" in body
        # Values must be strings (empty is acceptable when not configured)
        assert isinstance(body["supabase_url"], str)
        assert isinstance(body["supabase_anon_key"], str)


# ===========================================================================
# TC-API-007 — TC-API-010: Remove Object — Validation Failures
# ===========================================================================

class TestRemoveObjectValidation:
    """
    Tests for /api/remove-object request validation.
    These tests exercise error branches in app.py and MaskService.validate_and_process.
    None of these calls reach the ONNX inference engine.
    """

    def test_missing_both_files_returns_422(self):
        """
        TC-API-007
        Input:  POST /api/remove-object with no multipart fields.
        Expected: HTTP 422 Unprocessable Entity (FastAPI validates required fields).
        """
        response = client.post("/api/remove-object")
        assert response.status_code == 422, (
            f"Expected 422 for missing files, got {response.status_code}"
        )

    def test_missing_mask_returns_422(self):
        """
        TC-API-008
        Input:  POST /api/remove-object with only 'image', no 'mask'.
        Expected: HTTP 422 — mask is a required field.
        """
        png_bytes = _make_png_bytes()
        response = client.post(
            "/api/remove-object",
            files={"image": ("photo.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 422

    def test_unsupported_image_format_returns_415(self):
        """
        TC-API-009
        Input:  POST /api/remove-object with a GIF file as 'image'.
        Expected: HTTP 415 Unsupported Media Type.
        Note: GIF format is not in MaskService.SUPPORTED_FORMATS.
        """
        # Minimal GIF87a header bytes
        gif_bytes = (
            b"GIF87a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff"
            b"\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,"
            b"\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
        )
        mask_bytes = _make_white_mask_png(1, 1)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("image.gif", io.BytesIO(gif_bytes), "image/gif"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code in (400, 415), (
            f"Expected 400 or 415 for GIF format, got {response.status_code}: {response.text}"
        )

    def test_empty_image_bytes_returns_400(self):
        """
        TC-API-010
        Input:  POST /api/remove-object with a zero-byte 'image' file.
        Expected: HTTP 400 Bad Request with error code EMPTY_IMAGE.
        """
        mask_bytes = _make_white_mask_png()
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(b""), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert body.get("detail", {}).get("error") == "EMPTY_IMAGE"

    def test_empty_mask_bytes_returns_400(self):
        """
        TC-API-011
        Input:  POST /api/remove-object with valid image but zero-byte 'mask'.
        Expected: HTTP 400 Bad Request with error code EMPTY_MASK.
        """
        image_bytes = _make_png_bytes()
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(image_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(b""), "image/png"),
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert body.get("detail", {}).get("error") == "EMPTY_MASK"

    def test_empty_selection_mask_returns_422(self):
        """
        TC-API-012
        Input:  POST /api/remove-object with valid image and all-black (empty) mask.
        Expected: HTTP 422 Unprocessable Entity with error code EMPTY_SELECTION.
                  An all-black mask has no pixels set; MaskService raises
                  MaskValidationError('The selection mask is empty.').
        """
        image_bytes = _make_png_bytes(64, 64)
        mask_bytes = _make_empty_mask_png(64, 64)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(image_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 422, (
            f"Expected 422 for empty selection mask, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("detail", {}).get("error") == "EMPTY_SELECTION"

    def test_oversized_image_returns_413(self):
        """
        TC-API-013
        Input:  POST /api/remove-object with 'image' file larger than 25 MB.
        Expected: HTTP 413 Request Entity Too Large with error code FILE_TOO_LARGE.
        Note: 25 MB limit is enforced in app.py before any image decoding.
        """
        oversized_bytes = b"X" * (26 * 1024 * 1024)  # 26 MB of dummy data
        mask_bytes = _make_white_mask_png()
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("huge.png", io.BytesIO(oversized_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 413, (
            f"Expected 413 for oversized file, got {response.status_code}"
        )
        body = response.json()
        assert body.get("detail", {}).get("error") == "FILE_TOO_LARGE"

    def test_corrupted_image_bytes_returns_400(self):
        """
        TC-API-014
        Input:  POST /api/remove-object with malformed (non-image) bytes as 'image'.
        Expected: HTTP 400 Bad Request (IMAGE_DECODE_ERROR or VALIDATION_FAILED).
        """
        corrupt_bytes = b"\x00\x01\x02garbage_not_an_image\xff\xfe"
        mask_bytes = _make_white_mask_png()
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(corrupt_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code in (400, 415), (
            f"Expected 400 or 415 for corrupted image, got {response.status_code}: {response.text}"
        )


# ===========================================================================
# TC-API-015 — TC-API-017: Remove Object — Success Path (LaMa required)
# ===========================================================================

class TestRemoveObjectSuccess:
    """
    Tests for /api/remove-object successful inpainting responses.
    These tests call the full pipeline including LaMa ONNX inference.
    They will be SKIPPED automatically if the model file is absent.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Auto-skip all tests in this class if inpainting_lama.onnx is absent."""
        model_path = PROJECT_ROOT / "inpainting_lama.onnx"
        if not model_path.exists():
            pytest.skip(
                "inpainting_lama.onnx not present — skipping live inference tests"
            )

    def test_remove_object_png_returns_200_png(self):
        """
        TC-API-015
        Input:  POST /api/remove-object with valid 64×64 PNG image and 30% coverage mask.
        Expected: HTTP 200, Content-Type image/png, non-empty body.
        """
        image_bytes = _make_png_bytes(64, 64)
        mask_bytes = _make_white_mask_png(64, 64, coverage=0.3)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(image_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:300]}"
        )
        assert "image/png" in response.headers.get("content-type", ""), (
            "Expected Content-Type image/png in response"
        )
        assert len(response.content) > 0, "Response body is empty"

    def test_remove_object_jpeg_source_returns_200(self):
        """
        TC-API-016
        Input:  POST /api/remove-object with a JPEG image and a PNG mask.
        Expected: HTTP 200 (JPEG images are accepted and internally converted to PNG).
        """
        image_bytes = _make_jpeg_bytes(64, 64)
        mask_bytes = _make_white_mask_png(64, 64, coverage=0.3)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.jpg", io.BytesIO(image_bytes), "image/jpeg"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 200, (
            f"Expected 200 for JPEG source, got {response.status_code}: {response.text[:300]}"
        )

    def test_remove_object_result_has_timing_headers(self):
        """
        TC-API-017
        Input:  POST /api/remove-object with valid PNG + mask.
        Expected: Response headers include 'X-Process-Time-Ms' and
                  'X-Inference-Time-Ms' (instrumentation headers).
        """
        image_bytes = _make_png_bytes(64, 64)
        mask_bytes = _make_white_mask_png(64, 64, coverage=0.3)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(image_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers, (
            "X-Process-Time-Ms header missing"
        )
        assert "x-inference-time-ms" in response.headers, (
            "X-Inference-Time-Ms header missing"
        )

    def test_remove_object_result_preserves_dimensions(self):
        """
        TC-API-018
        Input:  POST /api/remove-object with 100×80 PNG image.
        Expected: Response headers 'X-Original-Width' = 100 and
                  'X-Original-Height' = 80 (dimensions preserved end-to-end).
        """
        image_bytes = _make_png_bytes(100, 80)
        mask_bytes = _make_white_mask_png(100, 80, coverage=0.25)
        response = client.post(
            "/api/remove-object",
            files={
                "image": ("photo.png", io.BytesIO(image_bytes), "image/png"),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
        )
        assert response.status_code == 200
        assert response.headers.get("x-original-width") == "100"
        assert response.headers.get("x-original-height") == "80"


# ===========================================================================
# TC-API-019 — TC-API-022: Smart Circle Segmentation Endpoint
# ===========================================================================

class TestSmartCircleSegmentation:
    """Tests for POST /api/smart-circle-segment."""

    def test_valid_points_returns_200_png(self):
        """
        TC-API-019
        Input:  POST /api/smart-circle-segment with a 64×64 image and 5 polygon
                points forming a rough circle around the center.
        Expected: HTTP 200, Content-Type image/png.
        """
        image_bytes = _make_png_bytes(64, 64)
        points = [
            {"x": 25, "y": 15},
            {"x": 45, "y": 20},
            {"x": 50, "y": 40},
            {"x": 30, "y": 50},
            {"x": 15, "y": 35},
        ]
        response = client.post(
            "/api/smart-circle-segment",
            files={"image": ("photo.png", io.BytesIO(image_bytes), "image/png")},
            data={"points": json.dumps(points), "mode": "smart_object"},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        assert "image/png" in response.headers.get("content-type", "")

    def test_fewer_than_3_points_returns_400(self):
        """
        TC-API-020
        Input:  POST /api/smart-circle-segment with only 2 points (below minimum).
        Expected: HTTP 400 Bad Request with error code INSUFFICIENT_POINTS.
        """
        image_bytes = _make_png_bytes(64, 64)
        points = [{"x": 10, "y": 10}, {"x": 50, "y": 50}]
        response = client.post(
            "/api/smart-circle-segment",
            files={"image": ("photo.png", io.BytesIO(image_bytes), "image/png")},
            data={"points": json.dumps(points)},
        )
        assert response.status_code == 400
        body = response.json()
        assert body.get("detail", {}).get("error") == "INSUFFICIENT_POINTS"

    def test_invalid_points_json_returns_400(self):
        """
        TC-API-021
        Input:  POST /api/smart-circle-segment with malformed JSON for 'points'.
        Expected: HTTP 400 Bad Request with error code INVALID_POINTS.
        """
        image_bytes = _make_png_bytes(64, 64)
        response = client.post(
            "/api/smart-circle-segment",
            files={"image": ("photo.png", io.BytesIO(image_bytes), "image/png")},
            data={"points": "not-valid-json"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body.get("detail", {}).get("error") == "INVALID_POINTS"

    def test_empty_image_returns_400(self):
        """
        TC-API-022
        Input:  POST /api/smart-circle-segment with zero-byte image.
        Expected: HTTP 400 Bad Request with error code EMPTY_IMAGE.
        """
        points = [{"x": 10, "y": 10}, {"x": 40, "y": 10}, {"x": 25, "y": 40}]
        response = client.post(
            "/api/smart-circle-segment",
            files={"image": ("photo.png", io.BytesIO(b""), "image/png")},
            data={"points": json.dumps(points)},
        )
        assert response.status_code == 400
        body = response.json()
        assert body.get("detail", {}).get("error") == "EMPTY_IMAGE"


# ===========================================================================
# TC-API-023 — TC-API-025: Static File Routes
# ===========================================================================

class TestStaticFileRoutes:
    """Tests that the static asset serving routes respond correctly."""

    def test_root_returns_200_or_html(self):
        """
        TC-API-023
        Endpoint: GET /
        Expected: HTTP 200. When index.html is present it is served as
                  text/html; otherwise a JSON status object is returned.
        """
        response = client.get("/")
        assert response.status_code == 200

    def test_style_css_content_type(self):
        """
        TC-API-024
        Endpoint: GET /style.css
        Expected: HTTP 200 with Content-Type text/css.
        """
        response = client.get("/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")

    def test_script_js_content_type(self):
        """
        TC-API-025
        Endpoint: GET /script.js
        Expected: HTTP 200 with Content-Type application/javascript.
        """
        response = client.get("/script.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")


# ===========================================================================
# TC-API-026: Middleware — Process-Time Header on any request
# ===========================================================================

class TestMiddlewareHeaders:
    """Ensures the timing middleware injects headers on every response."""

    def test_health_response_has_process_time_header(self):
        """
        TC-API-026
        Endpoint: GET /api/health
        Expected: Response header 'X-Process-Time-Ms' is present and parseable
                  as a float value.
        """
        response = client.get("/api/health")
        header_val = response.headers.get("x-process-time-ms")
        assert header_val is not None, "X-Process-Time-Ms header missing"
        try:
            float(header_val)
        except ValueError:
            pytest.fail(f"X-Process-Time-Ms is not a valid float: {header_val!r}")
