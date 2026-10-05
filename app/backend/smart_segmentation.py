"""
app/backend/smart_segmentation.py
AI Smart Circle & Polygon Object Segmentation Service.
Uses OpenCV GrabCut and morphological boundary analysis to extract
the primary foreground object enclosed within a user's rough circle/loop.
"""

import json
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import io


class SmartSegmentationService:
    """Provides computer vision and AI-assisted segmentation for rough user loops and circles."""

    @classmethod
    def segment_enclosed_region(
        cls,
        image_bytes: bytes,
        points: List[Dict[str, float]],
        mode: str = "smart_object",
        padding: int = 8,
    ) -> bytes:
        """
        Segments the foreground object inside the polygon defined by `points`.
        
        Args:
            image_bytes: Raw bytes of the source photo.
            points: List of {'x': float/int, 'y': float/int} image-space coordinates.
            mode: 'smart_object' (GrabCut + morphological boundaries) or 'fill_interior' (direct polygon fill).
            padding: Pixel padding around the bounding box for background context.

        Returns:
            PNG image bytes of the binary mask (255 = object to remove, 0 = keep).
        """
        if not points or len(points) < 3:
            raise ValueError("Smart circle requires at least 3 points to form an enclosed loop.")

        # 1. Decode source image to numpy BGR
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image for smart circle segmentation.")

        h, w = img.shape[:2]

        # 2. Convert points to integer numpy polygon and clamp to image dimensions
        pts = []
        for p in points:
            px = max(0, min(w - 1, round(p.get("x", 0))))
            py = max(0, min(h - 1, round(p.get("y", 0))))
            pts.append([px, py])
        poly_pts = np.array(pts, dtype=np.int32)

        # 3. Create full-size polygon binary mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_pts], (255,))
        poly_area = cv2.countNonZero(poly_mask)

        # Direct fill fallback if requested or if polygon is tiny (< 100 pixels)
        if mode == "fill_interior" or poly_area < 100:
            return cls._encode_mask_png(poly_mask)

        # 4. Compute bounding box around polygon with context padding
        bx, by, bw, bh = cv2.boundingRect(poly_pts)
        bx0 = max(0, bx - padding)
        by0 = max(0, by - padding)
        bx1 = min(w, bx + bw + padding)
        by1 = min(h, by + bh + padding)

        # 5. Initialize GrabCut mask
        # cv2.GC_BGD (0): Definite background
        # cv2.GC_FGD (1): Definite foreground
        # cv2.GC_PR_BGD (2): Probable background
        # cv2.GC_PR_FGD (3): Probable foreground
        grabcut_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

        # Surrounding bounding box is initialized as probable background
        grabcut_mask[by0:by1, bx0:bx1] = cv2.GC_PR_BGD

        # Interior of user loop is initialized as probable foreground
        grabcut_mask[poly_mask == 255] = cv2.GC_PR_FGD

        # Compute centroid / inner core of polygon as definite foreground seeds
        moments = cv2.moments(poly_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            core_radius = max(3, int(min(bw, bh) * 0.18))
            core_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(core_mask, (cx, cy), core_radius, (255,), -1)
            # Only set core where inside the polygon
            grabcut_mask[(core_mask == 255) & (poly_mask == 255)] = cv2.GC_FGD

        # 6. Run GrabCut optimization within bounding box
        rect = (bx0, by0, bx1 - bx0, by1 - by0)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                img,
                grabcut_mask,
                rect,
                bgd_model,
                fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_MASK,
            )
            # Extract foreground pixels (definite or probable)
            fg_mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            # Restrict foreground strictly within or near the user's circle
            dilated_poly = cv2.dilate(poly_mask, np.ones((5, 5), np.uint8), iterations=1)
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=dilated_poly)

            fg_area = cv2.countNonZero(fg_mask)
            # If GrabCut found a reasonable object (> 15% of circle area), use it
            if fg_area >= 0.15 * poly_area:
                # Apply morphological close to bridge internal object texture gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

                # Light 2px dilation ensures complete edge coverage for inpainting
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                final_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)
            else:
                # Fallback to the smoothed polygon mask
                final_mask = poly_mask
        except Exception:
            # Fallback to direct polygon fill if GrabCut encounters degenerate color distributions
            final_mask = poly_mask

        return cls._encode_mask_png(final_mask)

    @staticmethod
    def _encode_mask_png(mask: np.ndarray) -> bytes:
        """
        Encodes an 8-bit binary mask as a 4-channel BGRA PNG with full alpha transparency.
        Foreground (object to remove): White (255, 255, 255) with Alpha 255.
        Background (unselected context): Completely transparent (0, 0, 0, 0).
        """
        h, w = mask.shape[:2]
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        fg_indices = mask > 127
        bgra[fg_indices] = [255, 255, 255, 255]
        success, encoded = cv2.imencode(".png", bgra)
        if not success:
            raise RuntimeError("Failed to encode binary mask to PNG format.")
        return encoded.tobytes()

