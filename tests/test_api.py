"""
tests/test_api.py
Comprehensive automated API tests for AI Object Remover FastAPI endpoints.
Tests real endpoints defined in app.py:
- GET /health, /api/health, /api/status
- GET /, /index.html, /style.css, /script.js
- POST /api/remove-object, /remove-object, /api/inpaint
"""

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.backend.app import app
from app.backend.inpainting import InpaintingService

client = TestClient(app)


def create_test_image_bytes(width: int = 128, height: int = 128, color=(200, 100, 100), fmt="PNG") -> bytes:
    """Helper to generate in-memory valid image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_test_mask_bytes(width: int = 128, height: int = 128, has_selection: bool = True, fmt="PNG") -> bytes:
    """Helper to generate in-memory mask bytes."""
    mask = Image.new("L", (width, height), color=0)
    if has_selection:
        draw = ImageDraw.Draw(mask)
        # Select central 32x32 area
        draw.rectangle([32, 32, 64, 64], fill=255)
    buf = io.BytesIO()
    mask.save(buf, format=fmt)
    return buf.getvalue()


# ==============================================================================
# 1. Health & Server Availability Endpoints
# ==============================================================================

def test_health_check_endpoint():
    """Verify GET /api/health returns 200 with online status and provider details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ("ok", "online")
    assert data.get("model") == "ready"
    assert "provider" in data
    assert "provider_configured" in data


def test_health_check_aliases():
    """Verify aliases GET /health and GET /api/status match /api/health."""
    res_health = client.get("/health")
    res_status = client.get("/api/status")
    assert res_health.status_code == 200
    assert res_status.status_code == 200
    assert res_health.json()["status"] in ("online", "ok")
    assert res_status.json()["status"] in ("online", "ok")


# ==============================================================================
# 2. Static Assets Serving Endpoints
# ==============================================================================

def test_root_index_serving():
    """Verify GET / serves HTML index page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in response.text or "<html" in response.text


def test_static_files_serving():
    """Verify GET /style.css, GET /script.js, and GET /auth.js return 200 with correct media types."""
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers.get("content-type", "")

    res_js = client.get("/script.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers.get("content-type", "")

    res_auth = client.get("/auth.js")
    assert res_auth.status_code == 200
    assert "javascript" in res_auth.headers.get("content-type", "")


def test_auth_config_endpoint():
    """Verify GET /api/auth/config returns 200 with expected Supabase config structure."""
    response = client.get("/api/auth/config")
    assert response.status_code == 200
    data = response.json()
    assert "supabase_url" in data
    assert "supabase_anon_key" in data


# ==============================================================================
# 3. Input Validation & Error Handling
# ==============================================================================

def test_remove_object_missing_files():
    """Verify POST /api/remove-object fails with 422 if files are missing."""
    response = client.post("/api/remove-object", data={"prompt": "clean background"})
    assert response.status_code == 422


def test_remove_object_empty_image_bytes():
    """Verify POST /api/remove-object fails with 400 when uploaded image is 0 bytes."""
    mask_bytes = create_test_mask_bytes(64, 64, has_selection=True)
    files = {
        "image": ("test.png", b"", "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    response = client.post("/api/remove-object", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "EMPTY_IMAGE"


def test_remove_object_empty_mask_bytes():
    """Verify POST /api/remove-object fails with 400 when uploaded mask is 0 bytes."""
    img_bytes = create_test_image_bytes(64, 64)
    files = {
        "image": ("test.png", img_bytes, "image/png"),
        "mask": ("mask.png", b"", "image/png"),
    }
    response = client.post("/api/remove-object", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "EMPTY_MASK"


def test_remove_object_unsupported_image_format():
    """Verify POST /api/remove-object fails with 415 on unsupported format."""
    # Create a dummy text file as image
    files = {
        "image": ("test.txt", b"not-an-image-file", "text/plain"),
        "mask": ("mask.png", create_test_mask_bytes(64, 64), "image/png"),
    }
    response = client.post("/api/remove-object", files=files)
    assert response.status_code in (400, 415)


def test_remove_object_empty_selection_mask():
    """Verify POST /api/remove-object returns 422 when mask has zero selected pixels."""
    img_bytes = create_test_image_bytes(64, 64)
    empty_mask_bytes = create_test_mask_bytes(64, 64, has_selection=False)
    files = {
        "image": ("test.png", img_bytes, "image/png"),
        "mask": ("mask.png", empty_mask_bytes, "image/png"),
    }
    response = client.post("/api/remove-object", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "EMPTY_SELECTION"


# ==============================================================================
# 4. Successful Inpainting Execution & Output Image Validity
# ==============================================================================

def test_successful_object_removal_execution():
    """
    Test real inpainting execution through POST /api/remove-object.
    Verifies 200 response, image/png media type, valid decodable image, and response headers.
    """
    service = InpaintingService()
    status_info = service.get_status()
    if not status_info.get("provider_configured"):
        pytest.skip("No AI inpainting provider configured in environment; skipping integration test.")

    # Small image for fast test execution
    w, h = 128, 128
    img_bytes = create_test_image_bytes(w, h, color=(100, 150, 200))
    mask_bytes = create_test_mask_bytes(w, h, has_selection=True)

    files = {
        "image": ("photo.png", img_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    data = {"prompt": "seamless background fill"}

    response = client.post("/api/remove-object", files=files, data=data)

    if response.status_code == 503:
        pytest.skip(f"Provider not available or model weight missing: {response.text}")

    assert response.status_code == 200
    assert response.headers.get("content-type") == "image/png"
    assert "X-Original-Width" in response.headers
    assert response.headers["X-Original-Width"] == str(w)
    assert response.headers["X-Original-Height"] == str(h)

    # Validate output image byte stream
    result_img = Image.open(io.BytesIO(response.content))
    result_img.load()
    assert result_img.size == (w, h)
    assert result_img.mode in ("RGB", "RGBA")

    # Verify timing header is present on success
    assert "X-Process-Time-Ms" in response.headers
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


# ==============================================================================
# 5. Live Metrics & Timing Instrumentation Tests
# ==============================================================================

def test_remove_object_timing_and_memory_headers():
    """
    Verify /api/remove-object returns valid numeric X-Process-Time-Ms and memory headers:
    1. Header is present in successful response.
    2. Header value is numeric and non-negative.
    3. Server-Timing header contains dur metric.
    4. Memory headers (X-Memory-Rss-Mb, X-Memory-Delta-Mb) are populated.
    """
    service = InpaintingService()
    if not service.get_status().get("provider_configured"):
        pytest.skip("AI provider not configured; skipping metrics test.")

    w, h = 64, 64
    img_bytes = create_test_image_bytes(w, h, color=(120, 180, 220))
    mask_bytes = create_test_mask_bytes(w, h, has_selection=True)

    files = {
        "image": ("test.png", img_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    response = client.post("/api/remove-object", files=files, data={"prompt": "seamless"})
    if response.status_code == 503:
        pytest.skip("Provider offline.")

    assert response.status_code == 200

    # 1 & 2. Verify X-Process-Time-Ms is present, numeric, and >= 0
    assert "X-Process-Time-Ms" in response.headers
    process_time_val = float(response.headers["X-Process-Time-Ms"])
    assert process_time_val >= 0.0

    # 3. Verify Server-Timing header
    assert "Server-Timing" in response.headers
    assert "total;dur=" in response.headers["Server-Timing"]

    # 4. Verify Memory Headers
    assert "X-Memory-Rss-Mb" in response.headers
    rss_val = float(response.headers["X-Memory-Rss-Mb"])
    assert rss_val >= 0.0

    assert "X-Memory-Delta-Mb" in response.headers
    delta_val = float(response.headers["X-Memory-Delta-Mb"])
    assert isinstance(delta_val, float)


def test_timing_middleware_on_error_responses():
    """
    Verify error responses are not broken by the timing middleware:
    1. X-Process-Time-Ms header is present on error responses (400, 422).
    2. Header value is numeric and >= 0.
    3. Error status codes and JSON detail payloads are completely preserved.
    """
    # Test 400 Bad Request (empty image)
    mask_bytes = create_test_mask_bytes(64, 64, has_selection=True)
    res_empty_img = client.post(
        "/api/remove-object",
        files={"image": ("empty.png", b"", "image/png"), "mask": ("mask.png", mask_bytes, "image/png")},
    )
    assert res_empty_img.status_code == 400
    assert "X-Process-Time-Ms" in res_empty_img.headers
    assert float(res_empty_img.headers["X-Process-Time-Ms"]) >= 0.0
    assert res_empty_img.json()["detail"]["error"] == "EMPTY_IMAGE"

    # Test 422 Unprocessable Entity (empty mask selection)
    img_bytes = create_test_image_bytes(64, 64)
    empty_mask = create_test_mask_bytes(64, 64, has_selection=False)
    res_empty_mask = client.post(
        "/api/remove-object",
        files={"image": ("test.png", img_bytes, "image/png"), "mask": ("mask.png", empty_mask, "image/png")},
    )
    assert res_empty_mask.status_code == 422
    assert "X-Process-Time-Ms" in res_empty_mask.headers
    assert float(res_empty_mask.headers["X-Process-Time-Ms"]) >= 0.0
    assert res_empty_mask.json()["detail"]["error"] == "EMPTY_SELECTION"


def test_health_check_returns_memory_metric():
    """Verify /api/health includes safe memory_rss_mb metric."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "memory_rss_mb" in data
    assert isinstance(data["memory_rss_mb"], (int, float))
    assert data["memory_rss_mb"] >= 0.0

