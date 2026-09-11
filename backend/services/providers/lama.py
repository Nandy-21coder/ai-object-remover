"""
lama.py
State-of-the-Art Local Deep Learning Inpainting Provider powered by LaMa (Large Mask Inpainting) ONNX model.
Runs 100% offline, free, CPU-optimized with multi-threading via ONNX Runtime / OpenCV DNN.
Completely eliminates smudging and blurs by reconstructing realistic textures, fabrics, and edges.
"""

import os
import asyncio
import threading
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from services.providers.base import BaseInpaintingProvider, InpaintingAPIError

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class LamaInpaintingProvider(BaseInpaintingProvider):
    """Local offline Deep Learning Inpainting Provider using LaMa ONNX model."""

    _session = None
    _net = None
    _lock = threading.Lock()

    def __init__(self, api_key: str = "local", timeout: float = 300.0):
        super().__init__(api_key=api_key or "local", timeout=timeout)
        self.model_path = Path(__file__).resolve().parent.parent.parent / "models" / "inpainting_lama.onnx"

    @classmethod
    def get_inference_engine(cls, model_path: Path) -> Tuple[str, Any]:
        """Loads and caches the ONNX Runtime session or OpenCV DNN net."""
        if not model_path.exists() or model_path.stat().st_size < 50 * 1024 * 1024:
            raise InpaintingAPIError(
                f"LaMa AI model not found at {model_path}. Please wait for download to finish."
            )

        if HAS_ORT:
            if cls._session is None:
                with cls._lock:
                    if cls._session is None:
                        opts = ort.SessionOptions()
                        opts.log_severity_level = 3  # Suppress benign graph warnings
                        cpu_total = os.cpu_count() or 4
                        threads = max(2, min(6, cpu_total - 2 if cpu_total > 4 else cpu_total))
                        opts.intra_op_num_threads = threads
                        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        cls._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            return "ort", cls._session
        else:
            if cls._net is None:
                with cls._lock:
                    if cls._net is None:
                        cls._net = cv2.dnn.readNetFromONNX(str(model_path))
            return "dnn", cls._net

    async def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "",
        model: Optional[str] = None,
    ) -> bytes:
        return await asyncio.to_thread(
            self._process_lama_inpaint,
            image_bytes,
            mask_bytes,
        )

    def _process_lama_inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
    ) -> bytes:
        try:
            # 1. Decode original image (BGR)
            img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InpaintingAPIError("Failed to decode image.")

            # 2. Decode mask (Grayscale)
            mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise InpaintingAPIError("Failed to decode mask.")

            orig_h, orig_w = img.shape[:2]
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Convert BGR to RGB for neural network
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Prepare 512x512 inputs expected by LaMa model
            target_size = (512, 512)
            img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            # Image blob: (1, 3, 512, 512), float32, range [0, 1]
            img_tensor = img_resized.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
            img_tensor = np.expand_dims(img_tensor, axis=0)  # BCHW

            # Mask blob: (1, 1, 512, 512), float32, binary 0.0 or 1.0
            mask_binary = (mask_resized > 127).astype(np.float32)
            mask_tensor = np.expand_dims(np.expand_dims(mask_binary, axis=0), axis=0)

            # Run inference
            engine_type, engine_obj = self.get_inference_engine(self.model_path)
            engine: Any = engine_obj
            if engine_type == "ort":
                inputs = {
                    engine.get_inputs()[0].name: img_tensor,
                    engine.get_inputs()[1].name: mask_tensor,
                }
                outputs = engine.run(None, inputs)
                raw_out = outputs[0]  # shape (1, 3, 512, 512)
            else:
                engine.setInput(img_tensor, "image")
                engine.setInput(mask_tensor, "mask")
                raw_out = engine.forward()

            # Output postprocessing: (1, 3, 512, 512) -> (512, 512, 3)
            out = raw_out[0]
            out = np.transpose(out, (1, 2, 0))
            out = np.clip(out, 0, 255).astype(np.uint8)

            # Convert RGB back to BGR for OpenCV
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            # Resize output back to original dimensions
            result = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # Encode result to PNG
            success, encoded_buf = cv2.imencode(".png", result)
            if not success or encoded_buf is None:
                raise InpaintingAPIError("Failed to encode LaMa inpainted result.")

            return encoded_buf.tobytes()

        except Exception as e:
            if isinstance(e, InpaintingAPIError):
                raise
            raise InpaintingAPIError(f"LaMa AI inpainting engine error: {str(e)}")
