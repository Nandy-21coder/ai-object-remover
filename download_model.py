#!/usr/bin/env python3
"""
download_model.py
Automated downloader for the LaMa ONNX neural inpainting model.

Fetches the verified LaMa ONNX model (88.3 MB) from the official Hugging Face
repository and verifies its SHA-256 integrity hash.
Preserves the exact local path: `inpainting_lama.onnx` in the project root.
"""

import sys
import hashlib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_FILE = PROJECT_ROOT / "inpainting_lama.onnx"

# Official model source from Hugging Face opencv/inpainting_lama
MODEL_URL = "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx"
EXPECTED_SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
EXPECTED_SIZE = 92591623  # ~88.3 MB


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_model(force: bool = False) -> bool:
    """
    Downloads and verifies the LaMa ONNX model file.
    Returns True if model is present and valid.
    """
    if TARGET_FILE.exists() and not force:
        file_size = TARGET_FILE.stat().st_size
        if file_size >= 10 * 1024 * 1024:
            print(f"[Model Check] Found existing model: {TARGET_FILE.name} ({file_size / (1024 * 1024):.2f} MB)")
            print("[Model Check] Verifying SHA-256 integrity...")
            current_hash = compute_sha256(TARGET_FILE)
            if current_hash == EXPECTED_SHA256:
                print("[Model Check] SHA-256 verified successfully. Model is ready.")
                return True
            else:
                print(f"[Model Check] Hash mismatch (expected {EXPECTED_SHA256[:8]}..., got {current_hash[:8]}...). Redownloading.")

    print("=" * 65)
    print("   AI Object Remover — Model Asset Downloader")
    print("=" * 65)
    print(f"Target:      {TARGET_FILE}")
    print(f"Source:      {MODEL_URL}")
    print(f"Target Size: {EXPECTED_SIZE / (1024 * 1024):.2f} MB")
    print("Downloading model weights... (this may take 10-30 seconds)")

    temp_file = TARGET_FILE.with_suffix(".tmp")
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, (downloaded / total_size) * 100)
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Progress: {pct:5.1f}% [{mb_down:5.1f} / {mb_tot:5.1f} MB]")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, temp_file, reporthook=reporthook)
        print()

        # Verify downloaded file
        downloaded_hash = compute_sha256(temp_file)
        if downloaded_hash != EXPECTED_SHA256:
            temp_file.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {downloaded_hash}")

        # Replace target file atomically
        if TARGET_FILE.exists():
            TARGET_FILE.unlink()
        temp_file.rename(TARGET_FILE)

        print(f"[Success] LaMa model installed at {TARGET_FILE}")
        return True

    except Exception as e:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        print(f"\n[Error] Model download failed: {e}")
        print("Manual download instructions:")
        print(f"1. Download from: {MODEL_URL}")
        print(f"2. Place at:     {TARGET_FILE}")
        return False


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    success = download_model(force=force_download)
    sys.exit(0 if success else 1)

