"""
app.py
FastAPI application backend for AI Object Remover.
Provides endpoints for AI inpainting (/api/remove-object), health check (/api/health),
and static frontend file serving.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Ensure services are importable
sys.path.insert(0, str(Path(__file__).parent))

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

import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Background warm-up of active offline model if present
    async def _warmup():
        try:
            from services.providers.lama import LamaInpaintingProvider
            model_path = Path(__file__).resolve().parent / "models" / "inpainting_lama.onnx"
            if model_path.exists():
                await asyncio.to_thread(LamaInpaintingProvider.get_inference_engine, model_path)
        except Exception:
            pass

    asyncio.create_task(_warmup())
    yield

app = FastAPI(
    title="AI Object Remover API",
    description="Real AI photo editing & object removal pipeline.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
@app.get("/api/status")
async def health_check():
    """
    Health check endpoint.
    Returns JSON showing that the backend is running and whether the AI provider is configured.
    """
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
    service = InpaintingService()
    return service.get_status()


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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "IMAGE_DECODE_ERROR", "message": "Failed to decode image data. Please upload a valid image."},
        )

    # 3. Call real AI Inpainting Service
    mask_metrics = MaskService.get_mask_metrics(pil_mask)
    try:
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

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers={
            "X-Original-Width": str(pil_image.width),
            "X-Original-Height": str(pil_image.height),
            "X-Mask-Coverage": f"{mask_metrics.get('coverage_percentage', 0.0)}%",
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )


# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    assets_dir = frontend_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")

    @app.get("/{filename}")
    async def serve_frontend_root(filename: str):
        target = frontend_dir / filename
        if target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(frontend_dir / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"Starting AI Object Remover backend on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)
