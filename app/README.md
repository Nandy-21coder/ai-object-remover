# Application Architecture Components

This directory serves as the application module namespace for the AI Object Remover suite.

### Core Application Entrypoints & Modules:
- **`../app.py`**: High-performance FastAPI REST API service, request validation, concurrency throttling (`asyncio.Semaphore(2)`), and telemetry instrumentation.
- **`../inpainting.py`**: Inpainting pipeline, `MaskService` (EXIF correction, binary binarization, adaptive dilation/feathering, compositing), `LamaInpaintingProvider` (offline LaMa ONNX session), and `SmartSegmentationService` (GrabCut foreground extraction).
- **`../script.js`**: HTML5 Canvas drawing workstation, coordinate projection engine, and API integration.
- **`../auth.js`**: Client authentication service with Supabase integration and offline local fallback.
- **`../style.css`**: Responsive glassmorphism studio design system.
- **`../index.html`**: Complete studio web application interface.

