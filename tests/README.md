# AI Object Remover - Test Suite & Verification Evidence

This directory contains the automated test suite for the AI Object Remover FastAPI backend, live performance metrics headers, memory instrumentation, and local LaMa neural network inpainting pipeline.

---

## 1. Test Architecture & Coverage

The automated test suite contains **22 collected tests** across 2 core test modules:

| Test Module | Tests | Target Component | Verification Scope |
|---|---|---|---|
| `tests/test_api.py` | 13 | FastAPI Application (`app.py`) | • Endpoint routing (`GET /health`, `/api/health`, `/api/status`, `/`, `/style.css`, `/script.js`)<br>• Input validation (empty image 400, empty mask 400, unselected mask 422, unsupported media 415)<br>• Live inpainting through `POST /api/remove-object` (200 OK, PNG format, dimension headers)<br>• **Timing & Metric Headers** (`X-Process-Time-Ms` numeric & >= 0, `Server-Timing`)<br>• **Live Memory Headers** (`X-Memory-Rss-Mb`, `X-Memory-Delta-Mb`)<br>• **Middleware Error Safety** (verifies timing middleware preserves 400/422 payloads)<br>• **Diagnostic Health Memory** (`GET /api/health` reports `memory_rss_mb`) |
| `tests/test_inpainting.py` | 9 | Inpainting Engine (`inpainting.py`) | • `MaskService.load_image` (EXIF normalization, RGB conversion, corruption rejection)<br>• `MaskService.load_and_binarize_mask` (strict 0/255 binarization, auto-resizing)<br>• Morphological dilation (`expand_mask`) & Gaussian feathering (`feather_mask`)<br>• **Bit-Exact Alpha Compositing** (guarantees untouched pixels outside mask are 100% bit-exact)<br>• `InpaintingService.get_status` and end-to-end local LaMa model execution |

---

## 2. Latest Empirical Execution Evidence

The test suite was executed against the active virtual environment on the host machine:

```bash
pytest tests/ -v
```

### Actual Execution Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-7.4.4, pluggy-1.0.0 -- C:\Users\user\Desktop\ai\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\user\Desktop\ai\ai-object-remover
plugins: anyio-4.2.0
collecting ... collected 22 items

tests/test_api.py::test_health_check_endpoint PASSED                     [  4%]
tests/test_api.py::test_health_check_aliases PASSED                      [  9%]
tests/test_api.py::test_root_index_serving PASSED                        [ 13%]
tests/test_api.py::test_static_files_serving PASSED                      [ 18%]
tests/test_api.py::test_remove_object_missing_files PASSED               [ 22%]
tests/test_api.py::test_remove_object_empty_image_bytes PASSED           [ 27%]
tests/test_api.py::test_remove_object_empty_mask_bytes PASSED            [ 31%]
tests/test_api.py::test_remove_object_unsupported_image_format PASSED    [ 36%]
tests/test_api.py::test_remove_object_empty_selection_mask PASSED        [ 40%]
tests/test_api.py::test_successful_object_removal_execution PASSED       [ 45%]
tests/test_api.py::test_remove_object_timing_and_memory_headers PASSED   [ 50%]
tests/test_api.py::test_timing_middleware_on_error_responses PASSED      [ 54%]
tests/test_api.py::test_health_check_returns_memory_metric PASSED        [ 59%]
tests/test_inpainting.py::test_mask_service_load_image_valid PASSED      [ 63%]
tests/test_inpainting.py::test_mask_service_load_image_empty PASSED      [ 68%]
tests/test_inpainting.py::test_mask_service_load_image_corrupted PASSED  [ 72%]
tests/test_inpainting.py::test_mask_service_load_mask_empty_selection PASSED [ 77%]
tests/test_inpainting.py::test_mask_service_load_mask_resizes_to_target PASSED [ 81%]
tests/test_inpainting.py::test_mask_service_expand_and_feather PASSED    [ 86%]
tests/test_inpainting.py::test_mask_service_compositing_preserves_untouched_pixels PASSED [ 90%]
tests/test_inpainting.py::test_inpainting_service_status PASSED          [ 95%]
tests/test_inpainting.py::test_end_to_end_inpaint_service_execution PASSED [100%]

======================= 22 passed, 3 warnings in 22.34s =======================
```

- **Total Collected Tests**: 22
- **Passing Tests**: 22 (100% Pass Rate)
- **Failing Tests**: 0
- **Execution Time**: ~22.34s

---

## 3. Local Test Instructions

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate

# Run complete test suite with verbose output
pytest tests/ -v

# Run individual test files
pytest tests/test_api.py -v
pytest tests/test_inpainting.py -v
```

---

## 4. Benchmark Execution Instructions

```bash
# Run latency & memory benchmarking suite (outputs to benchmarks/results.csv)
python benchmarks/benchmark.py

# Run quality evaluation suite (outputs to benchmarks/image_quality_results.csv)
python benchmarks/image_quality.py
```
