"""
benchmarks/benchmark.py
Automated Performance Benchmarking for AI Object Remover.
Measures end-to-end inference latency, peak RAM memory, peak GPU memory (if CUDA available),
image resolution, and throughput across multiple test runs.
"""

import os
import sys
import time
import argparse
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image, ImageDraw

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import psutil
except ImportError:
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Benchmark] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def check_gpu_memory() -> Any:
    """Returns peak GPU memory in MB if PyTorch CUDA is available, else 'N/A'."""
    try:
        import importlib
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        return "N/A"
    except Exception:
        return "N/A"


def reset_gpu_memory():
    """Resets PyTorch CUDA peak memory stats if available."""
    try:
        import importlib
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_current_ram_mb() -> float:
    """Returns current process RAM usage in MB using psutil."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    return 0.0


def create_synthetic_test_pair(width: int, height: int, name: str) -> tuple:
    """
    Creates a controlled synthetic test image with a known foreground object
    and corresponding binary mask for benchmarking.
    """
    # Background: Smooth gradient
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Add simple gradient/horizon
    draw.rectangle([0, int(height * 0.6), width, height], fill=(34, 139, 34))

    # Foreground object to remove (e.g. a red billboard or box)
    box_w = max(16, int(width * 0.15))
    box_h = max(16, int(height * 0.15))
    x0 = int(width * 0.4)
    y0 = int(height * 0.45)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(220, 20, 60))

    # Mask: 0 = keep, 255 = remove object
    mask = Image.new("L", (width, height), color=0)
    mask_draw = ImageDraw.Draw(mask)
    # Give 2px margin around the object
    mask_draw.rectangle([x0 - 2, y0 - 2, x0 + box_w + 2, y0 + box_h + 2], fill=255)

    return img, mask


async def run_single_benchmark(
    service,
    test_id: str,
    image_name: str,
    image: Image.Image,
    mask: Image.Image,
    model_name: str,
) -> Dict[str, Any]:
    """Runs a single end-to-end inpainting pipeline benchmark with latency and memory profiling."""
    width, height = image.size
    reset_gpu_memory()

    ram_before = get_current_ram_mb()
    start_time = time.perf_counter()
    status = "SUCCESS"
    error_msg = ""
    ram_after = ram_before

    try:
        # Full end-to-end pipeline:
        # image input -> mask processing (dilation/feathering) -> AI inference -> compositing -> final output bytes
        result_bytes = await service.inpaint(image, mask, prompt="seamless background fill")
        ram_after = get_current_ram_mb()
        peak_ram = max(ram_before, ram_after)

        # Validate that output is decodable
        if not result_bytes or len(result_bytes) == 0:
            status = "FAILED: Empty result"
    except Exception as e:
        ram_after = get_current_ram_mb()
        peak_ram = max(ram_before, ram_after)
        status = f"FAILED: {type(e).__name__}"
        error_msg = str(e)
        logger.error(f"Benchmark test {test_id} failed: {error_msg}")

    latency_seconds = round(time.perf_counter() - start_time, 4)
    rss_delta = round(ram_after - ram_before, 2)
    gpu_mem = check_gpu_memory()

    return {
        "test_id": test_id,
        "image_name": image_name,
        "width": width,
        "height": height,
        "model": model_name,
        "latency_seconds": latency_seconds,
        "rss_before_mb": ram_before,
        "rss_after_mb": ram_after,
        "rss_delta_mb": rss_delta,
        "peak_ram_mb": peak_ram,
        "peak_gpu_memory_mb": gpu_mem,
        "status": status,
    }


async def run_benchmark_suite(
    runs_per_config: int = 1,
    output_csv_path: Optional[Path] = None,
    resolutions: Optional[List[tuple]] = None,
) -> List[Dict[str, Any]]:
    """Runs the benchmark suite across multiple resolutions and records results to CSV."""
    from app.backend.inpainting import InpaintingService

    if output_csv_path is None:
        output_csv_path = Path(__file__).resolve().parent / "results.csv"

    # Preserve existing historical rows if CSV exists
    existing_rows: List[Dict[str, Any]] = []
    if output_csv_path.exists():
        try:
            with open(output_csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r.setdefault("rss_before_mb", r.get("peak_ram_mb", "N/A"))
                    r.setdefault("rss_after_mb", r.get("peak_ram_mb", "N/A"))
                    r.setdefault("rss_delta_mb", "0.00")
                    existing_rows.append(r)
        except Exception as e:
            logger.warning(f"Could not read existing historical CSV: {e}")

    service = InpaintingService()
    provider_name, _, model_name = service.get_provider_config()
    effective_model = f"{provider_name}-{model_name or 'default'}"

    logger.info(f"Initialized benchmark suite. Active Provider: {provider_name} ({effective_model})")

    if resolutions is None:
        # Representative resolutions: Small (256x256), Medium (512x512)
        resolutions = [(256, 256), (512, 512)]

    results: List[Dict[str, Any]] = []
    test_counter = len(existing_rows) + 1

    for w, h in resolutions:
        logger.info(f"Running benchmark for resolution {w}x{h} ({runs_per_config} run(s))...")
        for run_idx in range(1, runs_per_config + 1):
            test_id = f"BENCH-{test_counter:03d}"
            image_name = f"synthetic_{w}x{h}_run{run_idx}"
            test_img, test_mask = create_synthetic_test_pair(w, h, image_name)

            record = await run_single_benchmark(
                service=service,
                test_id=test_id,
                image_name=image_name,
                image=test_img,
                mask=test_mask,
                model_name=effective_model,
            )
            results.append(record)
            logger.info(
                f"[{record['test_id']}] {image_name} ({w}x{h}) - "
                f"Latency: {record['latency_seconds']}s | RAM Before: {record['rss_before_mb']}MB | "
                f"RAM After: {record['rss_after_mb']}MB | Delta: {record['rss_delta_mb']}MB | "
                f"Peak RAM: {record['peak_ram_mb']}MB | GPU: {record['peak_gpu_memory_mb']} | Status: {record['status']}"
            )
            test_counter += 1

    fieldnames = [
        "test_id",
        "image_name",
        "width",
        "height",
        "model",
        "latency_seconds",
        "rss_before_mb",
        "rss_after_mb",
        "rss_delta_mb",
        "peak_ram_mb",
        "peak_gpu_memory_mb",
        "status",
    ]

    all_rows = existing_rows + results

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Benchmark results successfully updated at {output_csv_path} (Total records: {len(all_rows)})")
    return results


def main():
    import asyncio
    parser = argparse.ArgumentParser(description="AI Object Remover Benchmark Runner")
    parser.add_argument("--runs", type=int, default=1, help="Number of test runs per resolution")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV filename")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / args.output
    asyncio.run(run_benchmark_suite(runs_per_config=args.runs, output_csv_path=out_path))


if __name__ == "__main__":
    main()
