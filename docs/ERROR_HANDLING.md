# Error Boundary & Exception Handling Architecture

Technical reference documenting the error boundary architecture, fault-isolation barriers, HTTP status translations, and recovery mechanisms across the **AI-Powered Photo Editing Assistant**.

---

## 1. Architectural Error Philosophy

The system enforces a **fail-safe, non-destructive, and sanitized error architecture**:
1. **Client-Side Pre-Flight Checks**: Rejects invalid actions (empty canvas, unsupported MIME, missing files) before triggering network I/O.
2. **Stateless Sanitization**: The FastAPI backend catches all low-level library exceptions (PIL, OpenCV, ONNX Runtime) and maps them into uniform JSON responses without leaking file paths, system directories, or stack traces.
3. **Canvas State Isolation**: Errors never discard the user's uploaded image or drawn mask strokes. The canvas remains interactive and reverts processing animations immediately upon receiving an error.
4. **Concurrency Safety**: Global execution semaphores (`asyncio.Semaphore(2)`) prevent out-of-memory crashes on concurrent load.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Error Boundary Hierarchy                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Frontend UI Boundaries (script.js)                                  │
│    • Drag & Drop validation (MIME check)                               │
│    • Client pre-flight checks (hasMaskSelection, state.currentImage)   │
│    • Network fetch rejection isolation & AbortController               │
│    • Non-destructive toast notification & state recovery               │
├────────────────────────────────────────────────────────────────────────┤
│ 2. HTTP Transport Boundary (FastAPI app.py)                            │
│    • Concurrency throttling semaphore (asyncio.Semaphore(2))           │
│    • File size threshold limiter (MAX_FILE_SIZE = 25 MB)               │
│    • Multipart parsing & Starlette RequestValidationError handler      │
│    • Global Timing Middleware & structured exception logger            │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Computer Vision Domain Boundary (inpainting.py / MaskService)       │
│    • EXIF rotation normalization & byte decoding                       │
│    • Binary mask quantization & coverage limit enforcement (<= 95%)    │
│    • Morphological dilation & feathering bounds check                  │
├────────────────────────────────────────────────────────────────────────┤
│ 4. AI Engine / Model Boundary (LamaInpaintingProvider)                 │
│    • Model presence & SHA-256 / size integrity verification            │
│    • Dynamic reflection-padding to 512×512 to prevent tensor mismatch  │
│    • ONNX Runtime execution fault containment                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Error Boundaries (`script.js`)

| Error Condition | Detection Point | HTTP Status | Response / UI Format | Application Handling | Recovery Behavior |
|---|---|---|---|---|---|
| **No Image Selected** | `executeObjectRemoval()`, `handleDownloadResult()` | N/A (Client check) | Warning Toast: `"Please select or brush over an object to remove first."` | Halts execution immediately before network request. | Editor remains in current state; guides user to upload photo or brush canvas. |
| **Invalid File Type** | `handleFileUpload()`, Drag-and-drop listener | N/A (Client check) | Error Toast: `"Please upload a valid image file (JPG, PNG, or WebP)."` | Rejects file, resets `<input type="file">`, preserves existing canvas. | File picker is cleared; user selects a supported photo. |
| **Unsupported Image Format** | `executeObjectRemoval()` (Server response) | HTTP 415 | Error Toast: `"Please upload a JPG, JPEG or PNG image."` | Catches 415 response from backend; resets processing indicators. | User is informed of format restrictions and uploads a supported file. |
| **Invalid Selection / Empty Mask** | `executeObjectRemoval()` (Client & Server) | HTTP 422 | Warning Toast: `"Please select the object you want to remove."` | Pre-checks `state.hasMaskSelection`; if bypassed, catches backend 422 `EMPTY_SELECTION`. | Processing overlay terminates; user continues painting on the mask canvas. |
| **Network Failure** | `fetch(REMOVE_OBJECT_ENDPOINT)` `catch` block | N/A (TypeError: Failed to fetch) | Error Toast: `"Unable to connect to backend server. Ensure http://127.0.0.1:8000 is running."` | Clears interval timer (`stopProcessingStages()`), restores interactive buttons, sets `state.isProcessing = false`. | User's photo and drawn mask are preserved on canvas; user can retry when server resumes. |
| **Request Timeout** | `fetch` AbortSignal / HTTP 504 response | HTTP 504 | Error Toast: `"Unable to connect to the AI service. Please try again."` | Disables progress overlay; cleans up canvas states. | Canvas drawing stays intact; user can re-trigger inference without redrawing. |
| **Backend Unavailable** | `/api/health` status check or 503 response | HTTP 503 | Error Toast: `"AI service configuration is incomplete."` | Inspects `/api/health` payload; updates UI status badge to error/missing state. | Informs user that `inpainting_lama.onnx` must be placed in root directory. |
| **Invalid API Response** | `Image.onerror` on returned Blob URL | HTTP 200/502 | Error Toast: `"Failed to render AI processed image."` | Rejects blob rendering promise, catches error, resets result canvas. | Canvas reverts to previous original image state; user can retry. |

---

## 3. Backend Error Boundaries (`app.py` & `inpainting.py`)

All backend error responses adhere to the standard schema:
```json
{
  "detail": {
    "error": "<MACHINE_READABLE_CODE>",
    "message": "<HUMAN_READABLE_DESCRIPTION>"
  }
}
```

### 3.1 Request Validation Errors
- **Condition**: Missing `image` or `mask` multipart fields, missing points parameter.
- **Where Detected**: FastAPI / Starlette request parser (`app.py:remove_object`).
- **HTTP Status**: `422 Unprocessable Content`.
- **Response Format**:
  ```json
  {
    "detail": [
      {
        "type": "missing",
        "loc": ["body", "mask"],
        "msg": "Field required",
        "input": null
      }
    ]
  }
  ```
- **Handling**: FastAPI rejects the request prior to invoking endpoint logic or loading files into memory.
- **Recovery Behavior**: Client receives exact missing field specification and corrects multipart form payload.

### 3.2 File Size Threshold Errors
- **Condition**: Uploaded image or mask exceeds maximum threshold ($25 \text{ MB}$).
- **Where Detected**: `app.py:remove_object` size check (`len(image_bytes) > MAX_FILE_SIZE`).
- **HTTP Status**: `413 Request Entity Too Large`.
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "FILE_TOO_LARGE",
      "message": "Image size exceeds 25MB limit. Please upload a smaller image."
    }
  }
  ```
- **Handling**: Stream reading is halted immediately, preventing server memory exhaustion.
- **Recovery Behavior**: Client notifies user to resize or compress the image before retrying.

### 3.3 Image Decoding Errors
- **Condition**: Non-image bytes uploaded or header data corrupted.
- **Where Detected**: `inpainting.py:MaskService.load_image` via `PIL.Image.open(io.BytesIO(...))` and `image.load()`.
- **HTTP Status**: `400 Bad Request`.
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "IMAGE_DECODE_ERROR",
      "message": "Failed to decode image data. Please upload a valid image."
    }
  }
  ```
- **Handling**: Trapped in `except MaskValidationError` / `except Exception`; converts raw PIL `UnidentifiedImageError` into a clean HTTP 400.
- **Recovery Behavior**: Returns failure reason to frontend; no server stack trace is emitted.

### 3.4 Mask Validation Errors
- **Condition**: Empty mask bytes, zero pixels selected (all black), or selection covers $> 95\%$ of the image canvas.
- **Where Detected**: `inpainting.py:MaskService.load_and_binarize_mask`.
- **HTTP Status**: `422 Unprocessable Content` (for empty selection) or `400 Bad Request` (for empty bytes / dimension mismatch).
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "EMPTY_SELECTION",
      "message": "Please select the object you want to remove."
    }
  }
  ```
- **Handling**: Validates mask extrema via `binary_mask.getextrema()`. If `(0, 0)` or non-zero pixels are 0, raises `MaskValidationError`.
- **Recovery Behavior**: Informs frontend to prompt user to paint over the target object.

### 3.5 Model Loading Failures
- **Condition**: `inpainting_lama.onnx` is missing from disk or file size is under $10 \text{ MB}$ (truncated download).
- **Where Detected**: `LamaInpaintingProvider.get_inference_engine()` and `app.py:health_check()`.
- **HTTP Status**: `503 Service Unavailable` on inference, or `{"model": "not_ready", "model_state": "missing"}` on health check.
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "AI_PROVIDER_NOT_CONFIGURED",
      "message": "AI service configuration is incomplete.",
      "provider": "lama",
      "instructions": "Please place inpainting_lama.onnx in the project root."
    }
  }
  ```
- **Handling**: Validates model file size ($> 10\text{ MB}$) before initializing ONNX Runtime session. Thread-safe lock (`_lock`) prevents race conditions during initialization.
- **Recovery Behavior**: Server admin downloads `inpainting_lama.onnx` into project root; application picks it up without restart.

### 3.6 ONNX Inference Errors
- **Condition**: ONNX Runtime engine execution exception (e.g., out-of-memory, invalid tensor allocation).
- **Where Detected**: `LamaInpaintingProvider._process_lama_inpaint()`.
- **HTTP Status**: `502 Bad Gateway`.
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "AI_INPAINTING_ERROR",
      "message": "LaMa AI inpainting engine error: <Sanitized Message>"
    }
  }
  ```
- **Handling**: Trapped in `_process_lama_inpaint`; logged with structured stack trace to stdout/logger while sanitized message is passed to client.
- **Recovery Behavior**: The session object remains cached; transient inference failures do not crash the ASGI worker.

### 3.7 Concurrency Throttling & Resource Protection
- **Condition**: Multiple concurrent requests submitted to CPU-bound ONNX model.
- **Where Detected**: `app.py` via `_inference_semaphore = asyncio.Semaphore(2)`.
- **HTTP Status**: Handled asynchronously; requests are queued rather than aborted.
- **Handling**: Up to 2 inference tasks execute concurrently; additional requests queue in FIFO order without memory exhaustion or process thrashing.
- **Recovery Behavior**: As previous requests complete, queued requests are automatically dispatched.

### 3.8 Unexpected Internal Errors
- **Condition**: Unhandled python runtime error.
- **Where Detected**: Global route `try...except Exception as e` in `app.py:remove_object`.
- **HTTP Status**: `500 Internal Server Error`.
- **Response Format**:
  ```json
  {
    "detail": {
      "error": "INTERNAL_SERVER_ERROR",
      "message": "Unexpected server error: <Sanitized Error Description>"
    }
  }
  ```
- **Handling**: Logs detailed error with request UUID to server console; releases semaphore locks safely.
- **Recovery Behavior**: Frontend displays error toast; server remains alive and ready for subsequent requests.

---

## 4. Verification & Testing

Every error boundary documented above is verified by automated pytest test cases in `tests/test_api.py` and `tests/test_inpainting.py`:
- `TC-API-007` to `TC-API-014` (Input validation, empty files, corrupt data, oversized files)
- `TC-API-020` to `TC-API-022` (Smart circle invalid parameters)
- `TC-INP-002`, `TC-INP-003`, `TC-INP-005` (Image format and corruption)
- `TC-INP-007`, `TC-INP-010`, `TC-INP-011` (Mask binarization error limits)
- `TC-INP-026` (Smart segmentation vertex limits)

