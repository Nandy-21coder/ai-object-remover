"""
app.py
FastAPI application backend for AI Object Remover.
Provides endpoints for AI inpainting (/api/remove-object), health check (/api/health),
and direct root static file serving (index.html, style.css, script.js).
"""

import os
import sys
from pathlib import Path
from typing import Optional
import time
import uuid
import logging

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("app.api")
if not logger.handlers:
    _log_handler = logging.StreamHandler(sys.stdout)
    _log_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [API] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_log_handler)
    logger.setLevel(logging.INFO)


def get_process_rss_mb() -> float:
    """Safe process-level Resident Set Size (RSS) memory measurement using psutil."""
    if psutil is not None:
        try:
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0
    return 0.0

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from dotenv import load_dotenv

# Define directory root reliably
PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure PROJECT_ROOT is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
def _load_env_file():
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            return

_load_env_file()

def get_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "inpainting_lama.onnx",
        Path.cwd() / "inpainting_lama.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "inpainting_lama.onnx"

from inpainting import (
    MaskService,
    MaskValidationError,
    InpaintingService,
    inpaint_image,
    ProviderNotConfiguredError,
    InvalidAPIKeyError,
    RateLimitError,
    ProviderTimeoutError,
    InpaintingAPIError,
    LamaInpaintingProvider,
    SmartSegmentationService,
)

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background warm-up of active offline model if present
    async def _warmup():
        try:
            model_path = get_model_path()
            if model_path.exists():
                await asyncio.to_thread(LamaInpaintingProvider.get_inference_engine, model_path)
        except Exception:
            pass

    asyncio.create_task(_warmup())
    yield


app = FastAPI(
    title="AI Object Remover API",
    description="Real AI photo editing & object removal pipeline.",
    version="1.2.0",
    lifespan=lifespan,
)

# Concurrency throttle to prevent CPU/memory exhaustion under load
_inference_semaphore = asyncio.Semaphore(2)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Process-Time-Ms",
        "X-Inference-Time-Ms",
        "X-Memory-Rss-Mb",
        "X-Memory-Delta-Mb",
        "Server-Timing",
    ],
)


@app.middleware("http")
async def add_timing_and_metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    if "X-Process-Time-Ms" not in response.headers:
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    if "Server-Timing" not in response.headers:
        response.headers["Server-Timing"] = f"total;dur={duration_ms:.2f}"
    return response


@app.get("/health")
@app.get("/api/health")
@app.get("/api/status")
async def health_check():
    """Health check endpoint returning active provider and model readiness status."""
    _load_env_file()
    service = InpaintingService()
    status_dict = service.get_status()

    provider = status_dict.get("provider", "lama")
    model_status = "ready"
    error_detail = None

    if provider in ("lama", "local"):
        model_path = get_model_path()
        if not model_path.exists():
            model_status = "missing"
            error_detail = f"Model file '{model_path.name}' is missing from models directory."
        elif model_path.stat().st_size < 10 * 1024 * 1024:
            model_status = "corrupted"
            error_detail = f"Model file '{model_path.name}' is incomplete or corrupted."
        else:
            try:
                engine_type, engine = LamaInpaintingProvider.get_inference_engine(model_path)
                if engine is None:
                    model_status = "load_error"
                    error_detail = "Inference engine failed to initialize."
                else:
                    model_status = "ready"
            except Exception as e:
                model_status = "load_error"
                error_detail = f"Error initializing model: {str(e)}"
    elif status_dict.get("provider_configured"):
        model_status = "ready"
    else:
        model_status = "not_configured"
        error_detail = status_dict.get("message", "AI provider is not configured.")

    # Conforms to API specification & test requirements: {"status": "ok", "model": "ready"}
    is_ready = model_status == "ready"
    status_dict["status"] = "ok" if is_ready else "error"
    status_dict["model"] = "ready" if is_ready else "not_ready"
    status_dict["model_state"] = model_status
    status_dict["app"] = "ai-object-remover"
    status_dict["memory_rss_mb"] = get_process_rss_mb()
    if error_detail:
        status_dict["error"] = error_detail

    return status_dict


@app.get("/api/auth/config")
async def get_auth_config():
    """Returns public client auth configuration (Supabase URL and anon key)."""
    _load_env_file()
    return {
        "supabase_url": os.getenv("SUPABASE_URL", "").strip(),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY", "").strip(),
    }


@app.post("/api/smart-circle-segment")
async def smart_circle_segment(
    image: UploadFile = File(..., description="Uploaded image file"),
    points: str = Form(..., description="JSON array of {'x': int, 'y': int} polygon points"),
    mode: Optional[str] = Form(default="smart_object"),
):
    """
    AI Smart Circle Selection Endpoint:
    Segments the primary object encircled by the user's rough loop.
    """
    import json
    try:
        points_list = json.loads(points)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_POINTS", "message": "Points must be a valid JSON array of {x, y} coordinates."}
        )

    if not isinstance(points_list, list) or len(points_list) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INSUFFICIENT_POINTS", "message": "At least 3 points required to define an enclosed circle."}
        )

    image_bytes = await image.read()
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "EMPTY_IMAGE", "message": "Uploaded image file is empty."}
        )

    try:
        mask_bytes = await asyncio.to_thread(
            SmartSegmentationService.segment_enclosed_region,
            image_bytes,
            points_list,
            mode or "smart_object",
        )
        return Response(content=mask_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Smart circle segmentation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "SEGMENTATION_ERROR", "message": str(e)}
        )


@app.post("/remove-object")
@app.post("/api/remove-object")
@app.post("/api/inpaint")
async def remove_object(
    image: UploadFile = File(..., description="Uploaded image file (JPG, JPEG, PNG, WebP)"),
    mask: UploadFile = File(..., description="Binary selection mask file (White = remove, Black = keep)"),
    prompt: Optional[str] = Form(default="seamless background fill, photorealistic, clean texture"),
):
    """
    Removes unwanted objects from an image using a real AI inpainting provider.
    1. Validates the uploaded image (format, content).
    2. Validates the mask and checks that selection is not empty.
    3. Ensures image and mask dimensions match.
    4. Converts mask to strict binary representation (255 remove, 0 preserve).
    5. Preserves original dimensions and aspect ratio without stretching or cropping.
    6. Sends to real AI provider.
    7. Returns the generated inpainted image to the frontend.
    """
    req_start_perf = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    # 1. Read files
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "READ_ERROR",
                "message": f"Failed to read uploaded files: {str(e)}",
            },
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "EMPTY_IMAGE",
                "message": "The uploaded image file is empty.",
            },
        )
    if not mask_bytes or len(mask_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "EMPTY_MASK",
                "message": "The uploaded mask file is empty.",
            },
        )

    # Check file size limit (25MB)
    MAX_FILE_SIZE = 25 * 1024 * 1024
    if len(image_bytes) > MAX_FILE_SIZE or len(mask_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "FILE_TOO_LARGE",
                "message": "Image size exceeds 25MB limit. Please upload a smaller image.",
            },
        )

    # 2. Validate format and content
    try:
        pil_image, pil_mask = MaskService.validate_and_process(image_bytes, mask_bytes)
    except MaskValidationError as e:
        err_str = str(e)
        if "Unsupported image format" in err_str:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={"error": "UNSUPPORTED_FORMAT", "message": "Please upload a JPG, JPEG or PNG image."},
            )
        elif "empty" in err_str.lower() or "brush over" in err_str.lower():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "EMPTY_SELECTION", "message": "Please select the object you want to remove."},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "VALIDATION_FAILED", "message": err_str},
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "IMAGE_DECODE_ERROR", "message": "Failed to decode image data. Please upload a valid image."},
        )

    # 3. Call real AI Inpainting Service with concurrency control and live memory profiling
    mask_metrics = MaskService.get_mask_metrics(pil_mask)
    rss_before_mb = get_process_rss_mb()
    inference_start_perf = time.perf_counter()

    try:
        async with _inference_semaphore:
            result_bytes = await inpaint_image(pil_image, pil_mask, prompt=prompt or "")
    except ProviderNotConfiguredError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "AI_PROVIDER_NOT_CONFIGURED",
                "message": "AI service configuration is incomplete.",
                "provider": e.provider,
                "instructions": e.instructions,
            },
        )
    except InvalidAPIKeyError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_API_KEY",
                "message": "AI service configuration is incomplete. Invalid API credentials.",
            },
        )
    except RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please try again later.",
            },
        )
    except ProviderTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "PROVIDER_TIMEOUT",
                "message": "Unable to connect to the AI service. Please try again.",
            },
        )
    except InpaintingAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "AI_INPAINTING_ERROR",
                "message": str(e) or "AI processing failed. Please try again.",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"Unexpected server error: {str(e)}",
            },
        )

    if not result_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "GENERATED_IMAGE_UNAVAILABLE",
                "message": "AI provider succeeded but returned no image data.",
            },
        )

    inference_duration_ms = round((time.perf_counter() - inference_start_perf) * 1000, 2)
    rss_after_mb = get_process_rss_mb()
    rss_delta_mb = round(rss_after_mb - rss_before_mb, 2)
    total_duration_ms = round((time.perf_counter() - req_start_perf) * 1000, 2)

    service = InpaintingService()
    provider_name, _, _ = service.get_provider_config()

    # Structured request logging
    logger.info(
        f"[REQUEST {request_id}] provider={provider_name} "
        f"dimensions={pil_image.width}x{pil_image.height} "
        f"mask_coverage={mask_metrics.get('coverage_percentage', 0.0)}% "
        f"latency_ms={total_duration_ms} inference_ms={inference_duration_ms} "
        f"rss_before_mb={rss_before_mb} rss_after_mb={rss_after_mb} rss_delta_mb={rss_delta_mb}"
    )

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "X-Original-Width": str(pil_image.width),
            "X-Original-Height": str(pil_image.height),
            "X-Mask-Coverage": f"{mask_metrics.get('coverage_percentage', 0.0)}%",
            "X-Process-Time-Ms": f"{total_duration_ms:.2f}",
            "X-Inference-Time-Ms": f"{inference_duration_ms:.2f}",
            "X-Memory-Rss-Mb": f"{rss_after_mb:.2f}",
            "X-Memory-Delta-Mb": f"{rss_delta_mb:.2f}",
            "Server-Timing": f"total;dur={total_duration_ms:.2f}, inpaint;dur={inference_duration_ms:.2f}",
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )


# ------------------------------------------------------------------------------
# Direct Static Serving from Frontend Directory
# ------------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "AI Object Remover API running"})


@app.get("/index.html")
async def serve_index_direct():
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/style.css")
async def serve_css():
    css_file = PROJECT_ROOT / "style.css"
    return FileResponse(css_file, media_type="text/css")


@app.get("/script.js")
async def serve_js():
    js_file = PROJECT_ROOT / "script.js"
    return FileResponse(js_file, media_type="application/javascript")


@app.get("/auth.js")
async def serve_auth_js():
    auth_file = PROJECT_ROOT / "auth.js"
    return FileResponse(auth_file, media_type="application/javascript")


@app.get("/{filename}.jpg")
async def serve_jpg(filename: str):
    img_file = PROJECT_ROOT / f"{filename}.jpg"
    if img_file.exists() and img_file.is_file():
        return FileResponse(img_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/assets/{filename}")
async def serve_frontend_asset(filename: str):
    asset_file = PROJECT_ROOT / filename
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(asset_file)
    raise HTTPException(status_code=404, detail="Asset not found")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Starting AI Object Remover backend on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
