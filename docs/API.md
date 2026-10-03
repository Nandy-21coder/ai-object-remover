# AI Object Remover - API Documentation

The AI Object Remover backend is built on **FastAPI** (`app.py`), providing an asynchronous, production-ready REST API for image inpainting, health diagnostic monitoring, and static web application serving.

---

## Base URL
```
http://127.0.0.1:8000
```

---

## Endpoints Overview

| Method | Endpoint | Aliases | Description |
|---|---|---|---|
| `GET` | `/api/health` | `/health`, `/api/status` | Diagnostic health check, active provider, and engine configuration. |
| `POST` | `/api/remove-object` | `/remove-object`, `/api/inpaint` | Core object removal pipeline via deep generative AI inpainting. |
| `GET` | `/` | `/index.html` | Serves the web workstation HTML frontend. |
| `GET` | `/style.css` | — | Serves the frontend stylesheet. |
| `GET` | `/script.js` | — | Serves the frontend canvas workstation JavaScript application. |

---

## 1. Health & Status Check

### `GET /api/health`
*(Also accessible at `GET /health` and `GET /api/status`)*

Returns operational status, model configuration, and active provider credentials status.

- **HTTP Method**: `GET`
- **Request Format**: No request body required.
- **Required Fields**: None.
- **Optional Fields**: None.

### Response
- **Status Code**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "status": "ok",
  "provider_configured": true,
  "provider": "lama",
  "model": "lama",
  "model_state": "ready",
  "memory_rss_mb": 562.57,
  "mask_expansion_pixels": 4,
  "mask_feather_radius": 2.0,
  "max_retries": 2,
  "message": "AI Inpainting engine connected via LAMA."
}
```

### Error Responses
None under normal conditions.

### Example Request (curl)
```bash
curl -X GET http://127.0.0.1:8000/api/health
```

---

## 2. Remove Object (AI Inpainting)

### `POST /api/remove-object`
*(Also accessible at `POST /remove-object` and `POST /api/inpaint`)*

Performs AI object removal on an uploaded image using the provided binary selection mask. Untouched pixels outside the mask boundary are preserved bit-exact at 100% source fidelity.

- **HTTP Method**: `POST`
- **Request Format**: `multipart/form-data`
- **Payload Limit**: Maximum file size 25 MB per file.

### Required Fields

| Field Name | Type | Content-Type | Description |
|---|---|---|---|
| `image` | Binary File | `image/jpeg`, `image/png`, `image/webp` | The source photograph containing the object to remove. |
| `mask` | Binary File | `image/png`, `image/jpeg` | Binary alpha mask. White pixels (`255` or RGB > 50) denote areas to remove; black pixels (`0`) denote preserved background. |

### Optional Fields

| Field Name | Type | Default | Description |
|---|---|---|---|
| `prompt` | Form String | `"seamless background fill, photorealistic, clean texture"` | Contextual prompt hint used by diffusion models (ignored by pure inpainting models like LaMa). |

### Successful Response
- **Status Code**: `200 OK`
- **Content-Type**: `image/png`
- **Body**: Binary PNG image stream of the reconstructed photograph.
- **Response Headers**:
  - `X-Original-Width`: Original image width in pixels.
  - `X-Original-Height`: Original image height in pixels.
  - `X-Mask-Coverage`: Percentage of the canvas area marked for removal (e.g. `2.82%`).
  - `X-Process-Time-Ms`: Total HTTP endpoint request handling latency in milliseconds (e.g. `21165.90`).
  - `X-Inference-Time-Ms`: Duration spent specifically inside deep generative inpainting in milliseconds.
  - `X-Memory-Rss-Mb`: Host process Resident Set Size (RSS) memory in megabytes after request completion.
  - `X-Memory-Delta-Mb`: Process memory delta allocated during inference in megabytes.
  - `Server-Timing`: W3C Server-Timing header (e.g. `total;dur=21165.90, inpaint;dur=20939.00`).
  - `Cache-Control`: `no-store, no-cache, must-revalidate`

### Error Responses

#### `400 Bad Request` - Corrupted or Empty Payload
Returned when file reading fails or file is 0 bytes.
```json
{
  "detail": {
    "error": "EMPTY_IMAGE",
    "message": "The uploaded image file is empty."
  }
}
```

#### `413 Request Entity Too Large` - Size Limit Exceeded
Returned when image or mask exceeds 25 MB.
```json
{
  "detail": {
    "error": "FILE_TOO_LARGE",
    "message": "Image size exceeds 25MB limit. Please upload a smaller image."
  }
}
```

#### `415 Unsupported Media Type` - Invalid Format
Returned when uploaded file is not JPG, JPEG, PNG, or WebP.
```json
{
  "detail": {
    "error": "UNSUPPORTED_FORMAT",
    "message": "Please upload a JPG, JPEG or PNG image."
  }
}
```

#### `422 Unprocessable Entity` - Empty Mask Selection
Returned when the mask contains zero painted pixels.
```json
{
  "detail": {
    "error": "EMPTY_SELECTION",
    "message": "Please select the object you want to remove."
  }
}
```

#### `503 Service Unavailable` - Missing Provider Configuration
Returned when cloud provider API keys are missing and no local model is available.
```json
{
  "detail": {
    "error": "AI_PROVIDER_NOT_CONFIGURED",
    "message": "AI service configuration is incomplete.",
    "provider": "stability",
    "instructions": "To enable real AI object removal, add your STABILITY API credentials in .env..."
  }
}
```

### Example Request (curl)
```bash
curl -X POST http://127.0.0.1:8000/api/remove-object \
  -F "image=@photo.jpg;type=image/jpeg" \
  -F "mask=@mask.png;type=image/png" \
  -F "prompt=clean background fill" \
  --output result.png
```

### Example Request (JavaScript / Fetch)
```javascript
const formData = new FormData();
formData.append("image", imageFile);
formData.append("mask", maskBlob, "mask.png");
formData.append("prompt", "seamless background fill");

const response = await fetch("http://127.0.0.1:8000/api/remove-object", {
  method: "POST",
  body: formData,
});

if (response.ok) {
  const blob = await response.blob();
  const resultUrl = URL.createObjectURL(blob);
  console.log("Inpainted image available at:", resultUrl);
} else {
  const errorJson = await response.json();
  console.error("Inpainting error:", errorJson.detail);
}
```

### Example Request (Python Requests)
```python
import requests

with open("photo.jpg", "rb") as f_img, open("mask.png", "rb") as f_mask:
    files = {
        "image": ("photo.jpg", f_img, "image/jpeg"),
        "mask": ("mask.png", f_mask, "image/png"),
    }
    data = {"prompt": "seamless background fill"}
    res = requests.post("http://127.0.0.1:8000/api/remove-object", files=files, data=data)

if res.status_code == 200:
    with open("output_clean.png", "wb") as f_out:
        f_out.write(res.content)
    print("Success! Output saved as output_clean.png")
else:
    print("Error:", res.status_code, res.json())
```

---

## 3. Static Web Application Routes

### `GET /` & `GET /index.html`
- Serves the Single-Page Application (SPA) canvas workstation.
- **Content-Type**: `text/html; charset=utf-8`

### `GET /style.css`
- Serves the UI design stylesheet.
- **Content-Type**: `text/css; charset=utf-8`

### `GET /script.js`
- Serves the canvas rendering, brush tooling, undo/redo manager, and API client scripts.
- **Content-Type**: `application/javascript; charset=utf-8`
