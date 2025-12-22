"""Depth processing utilities for RGB-D pose estimation."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .camera import CameraIntrinsics


@dataclass
class DepthConfig:
    """Configuration for depth processing."""

    # Size of region for averaging (e.g., 5 = 5x5 pixels)
    region_size: int = 5

    # Valid depth range (millimeters)
    min_valid_depth_mm: int = 200  # 20cm minimum
    max_valid_depth_mm: int = 2500  # 3m maximum

    # Temporal smoothing factor (0 = no smoothing, higher = more smoothing)
    # Smoothed = alpha * previous + (1 - alpha) * current
    temporal_alpha: float = 0.7

    # Minimum number of valid pixels required in region
    require_min_valid_pixels: int = 5


class DepthProcessor:
    """Handles depth lookup, filtering, and 3D deprojection."""

    def __init__(self, config: Optional[DepthConfig] = None):
        """Initialize depth processor.

        Args:
            config: Depth processing configuration. Uses defaults if None.
        """
        self.config = config or DepthConfig()
        self._temporal_cache: Dict[str, float] = {}

    def lookup_depth(
        self,
        depth_frame: np.ndarray,
        x_pixel: int,
        y_pixel: int,
        joint_name: Optional[str] = None,
    ) -> Optional[float]:
        """Look up depth at pixel with region averaging and temporal smoothing.

        Args:
            depth_frame: Depth image (H, W) in millimeters.
            x_pixel: X pixel coordinate.
            y_pixel: Y pixel coordinate.
            joint_name: Optional name for temporal smoothing cache.

        Returns:
            Depth in millimeters, or None if no valid depth found.
        """
        cfg = self.config
        h, w = depth_frame.shape
        half = cfg.region_size // 2

        # Clamp pixel coords to valid range
        x_pixel = max(0, min(w - 1, x_pixel))
        y_pixel = max(0, min(h - 1, y_pixel))

        # Extract region around pixel
        x0 = max(0, x_pixel - half)
        x1 = min(w, x_pixel + half + 1)
        y0 = max(0, y_pixel - half)
        y1 = min(h, y_pixel + half + 1)

        region = depth_frame[y0:y1, x0:x1].astype(np.float32)

        # Filter for valid depths
        valid_mask = (region >= cfg.min_valid_depth_mm) & (region <= cfg.max_valid_depth_mm)
        valid_depths = region[valid_mask]

        if len(valid_depths) < cfg.require_min_valid_pixels:
            return None

        # Use median for robustness to outliers
        depth_mm = float(np.median(valid_depths))

        # Apply temporal smoothing if we have a previous value
        if joint_name and joint_name in self._temporal_cache:
            prev = self._temporal_cache[joint_name]
            depth_mm = cfg.temporal_alpha * prev + (1 - cfg.temporal_alpha) * depth_mm

        # Update cache
        if joint_name:
            self._temporal_cache[joint_name] = depth_mm

        return depth_mm

    def deproject(
        self,
        x_pixel: float,
        y_pixel: float,
        depth_mm: float,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """Convert pixel coordinates + depth to 3D point in camera frame.

        Uses the standard pinhole camera model for deprojection:
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            Z = depth

        Args:
            x_pixel: X pixel coordinate.
            y_pixel: Y pixel coordinate.
            depth_mm: Depth in millimeters.
            intrinsics: Camera intrinsic parameters.

        Returns:
            np.ndarray([x, y, z]) in meters, in camera frame.
            Camera frame: x=right, y=down, z=forward (into scene).
        """
        z = depth_mm / 1000.0  # mm -> meters
        x = (x_pixel - intrinsics.cx) * z / intrinsics.fx
        y = (y_pixel - intrinsics.cy) * z / intrinsics.fy
        return np.array([x, y, z])

    def reset(self) -> None:
        """Reset temporal smoothing cache."""
        self._temporal_cache.clear()
