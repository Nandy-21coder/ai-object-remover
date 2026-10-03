# AI Object Remover - Performance & Quality Benchmarks

This directory contains empirical performance profiling, latency decomposition, memory instrumentation, and image reconstruction fidelity benchmarks for the AI Object Remover system.

---

## 1. Benchmarking Philosophy & Latency Hierarchy

To prevent misleading claims, this repository strictly distinguishes between three different tiers of latency:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        1. BROWSER END-TO-END LATENCY (Client)                          │
│  Mask prep (Canvas toBlob) → HTTP Upload → Server Queue → Inference → Result Download  │
│  → PNG Byte Decoding → Result Canvas Render                                            │
│  Measured via performance.now() in app/frontend/script.js                              │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        2. HTTP / API RESPONSE TIME (FastAPI)                           │
│  Request arrival → Payload parsing → MIME check → Concurrency semaphore                │
│  → Service dispatch → Response header generation → Stream flush                        │
│  Measured via time.perf_counter() in app/backend/app.py (X-Process-Time-Ms)            │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3. BACKEND INFERENCE LATENCY (Engine)                           │
│  Mask dilation & feathering → Reflection padding → ONNX Runtime forward pass           │
│  → Unpadding → Bicubic resize → Bit-exact alpha compositing                            │
│  Measured via time.perf_counter() in benchmarks/benchmark.py (InpaintingService.inpaint)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **No Hardcoded Numbers**: All metrics in `results.csv` and `image_quality_results.csv` are generated strictly from live execution on the host machine.
- **Reference-Grounded Quality**: PSNR and SSIM are mathematical metrics requiring a clean ground-truth reference image ($R$). Calculating PSNR/SSIM on ungrounded single images without a reference is invalid. The test harness creates verified ground-truth reference pairs to measure true reconstruction accuracy.
- **Memory Transparency**: Resident Set Size (RSS) memory is captured before, after, and as a delta during inpainting execution. Peak RSS is tracked using process-level `psutil.Process().memory_info().rss`.

---

## 2. Technical Metrics Reference

### A. Backend Inference Latency
- **What is measured**: Wall-clock time spent inside `InpaintingService.inpaint()` covering mask preprocessing (dilation, Gaussian feathering), aspect-ratio reflection padding, ONNX model forward pass, and bit-exact alpha compositing.
- **Where measured**: `benchmarks/benchmark.py:run_single_benchmark()` and `app/backend/inpainting.py`.
- **How calculated**: High-resolution `time.perf_counter()`.
- **Environment**: Local CPU with multithreaded ONNX Runtime (`CPUExecutionProvider`).
- **Observed Values**: ~20–41s for cold start / initial model load; ~3–21s on warm model execution depending on resolution.
- **Limitations**: Excludes network transfer, HTTP server deserialization, and browser rendering.

### B. HTTP / API Response Latency
- **What is measured**: Total duration from when an HTTP request hits the FastAPI server until the response headers and streaming PNG bytes are flushed to the client.
- **Where measured**: `app/backend/app.py` in `remove_object()` and `add_timing_and_metrics_middleware`.
- **How calculated**: `time.perf_counter()`, returned via `X-Process-Time-Ms` and `Server-Timing: total;dur=...` headers.
- **Limitations**: Measures server-side lifecycle only; does not include client upload bandwidth.

### C. Browser End-to-End Latency
- **What is measured**: True client-perceived duration from the user clicking "Remove Object" through client mask export, network upload, backend execution, download, image decoding, and canvas rendering.
- **Where measured**: `app/frontend/script.js:executeObjectRemoval()`.
- **How calculated**: `performance.now()`, displayed in `#resultMetrics` UI and logged to browser console.
- **Limitations**: Influenced by client browser CPU, canvas resolution, and localhost networking.

### D. Process Memory (RAM / RSS) Usage
- **What is measured**: Host OS process Resident Set Size (RSS) memory in megabytes.
- **Where measured**: `benchmarks/benchmark.py:get_current_ram_mb()`, `app/backend/app.py:get_process_rss_mb()`.
- **How calculated**: `psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)`.
- **Logged fields**: `rss_before_mb`, `rss_after_mb`, `rss_delta_mb`, `peak_ram_mb`.

### E. Local LaMa Model Memory Usage
- **What is measured**: Memory overhead of loading and executing `models/inpainting_lama.onnx` (88.3MB weights).
- **Where measured**: `benchmarks/benchmark.py` and live in `app/backend/app.py` during inpaint requests.
- **Observed Values**: Baseline Python server sits at ~86 MB RSS. Loading the ONNX session and tensor buffers increases RSS by ~475 MB to ~562 MB. Peak inpainting operations at 512x512 reach ~730–761 MB RSS.
- **Limitations**: Measures whole-process RSS delta; internal ONNX C++ runtime heap allocations are bounded by OS process limits.

### F. PSNR (Peak Signal-to-Noise Ratio)
- **What is measured**: Objective pixel-level reconstruction accuracy in decibels (dB) between a ground-truth reference and the AI-inpainted image.
- **Where measured**: `benchmarks/image_quality.py:calculate_metrics()`.
- **How calculated**: `skimage.metrics.peak_signal_noise_ratio(reference_rgb, reconstructed_rgb, data_range=255)`.
- **Observed Values**: 53.69 dB (synthetic horizon 512x512), 54.88 dB (synthetic backdrop 256x256). High values reflect that ~98.5% of pixels outside the mask are preserved bit-exact via alpha compositing.

### G. SSIM (Structural Similarity Index)
- **What is measured**: Structural, luminance, and contrast similarity index (-1.0 to 1.0, where 1.0 = identical).
- **Where measured**: `benchmarks/image_quality.py:calculate_metrics()`.
- **How calculated**: `skimage.metrics.structural_similarity(reference_rgb, reconstructed_rgb, data_range=255, channel_axis=2)`.
- **Observed Values**: 0.9993 (horizon 512x512), 0.9997 (backdrop 256x256).

---

## 3. Benchmark Scripts & Output Schema

| Script | Purpose | Output File | Metrics |
|---|---|---|---|
| `benchmark.py` | Measures backend inference latency and process memory (RSS before, after, delta, peak). | `results.csv` | `latency_seconds`, `rss_before_mb`, `rss_after_mb`, `rss_delta_mb`, `peak_ram_mb`, `peak_gpu_memory_mb` |
| `image_quality.py` | Measures generative reconstruction fidelity against ground truth (Synthetic & Natural). | `image_quality_results.csv` | `evaluation_type`, `psnr`, `ssim`, `resolution`, `status` |

### `results.csv` Schema
```csv
test_id,image_name,width,height,model,latency_seconds,rss_before_mb,rss_after_mb,rss_delta_mb,peak_ram_mb,peak_gpu_memory_mb,status
BENCH-003,synthetic_256x256_run1,256,256,lama-lama,41.3636,86.59,562.57,475.98,562.57,N/A,SUCCESS
BENCH-004,synthetic_512x512_run1,512,512,lama-lama,21.1659,563.57,730.66,167.09,730.66,N/A,SUCCESS
```

### `image_quality_results.csv` Schema
```csv
test_id,evaluation_type,image_name,psnr,ssim,resolution,model,status
QUAL-003,Synthetic Ground Truth,synthetic_horizon_512,53.695,0.9993,512x512,lama-lama,SUCCESS
QUAL-004,Synthetic Ground Truth,synthetic_backdrop_256,54.886,0.9997,256x256,lama-lama,SUCCESS
QUAL-NAT-005,Natural Reference Evaluation,natural_photo_benchmark,PENDING,PENDING,PENDING,lama-lama,PENDING_NATURAL_REFERENCE_DATASET
```

---

## 4. Execution Commands

```bash
# Run performance and memory benchmark (preserves historical entries and appends live runs)
python benchmarks/benchmark.py

# Run quality evaluation (PSNR / SSIM on synthetic ground truth + checks natural dataset)
python benchmarks/image_quality.py
```
