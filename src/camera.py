"""Camera abstraction layer for RGB and RGB-D cameras."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for 3D deprojection."""

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int
    height: int


@dataclass
class CameraFrame:
    """A single frame from the camera."""

    rgb: np.ndarray  # RGB image (H, W, 3) uint8
    depth: Optional[np.ndarray]  # Depth image (H, W) uint16, in mm; None for webcam
    timestamp_ms: int  # Frame timestamp in milliseconds
    intrinsics: Optional[CameraIntrinsics]  # Camera intrinsics; None for webcam


class CameraInterface(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        pass

    @abstractmethod
    def read(self) -> Optional[CameraFrame]:
        """Read a frame. Returns None if read fails."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""
        pass

    @abstractmethod
    def has_depth(self) -> bool:
        """Returns True if camera provides depth data."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        pass


class WebcamCamera(CameraInterface):
    """Standard webcam using OpenCV VideoCapture."""

    def __init__(self, camera_id: int = 0):
        """Initialize webcam.

        Args:
            camera_id: Camera device ID.
        """
        self._camera_id = camera_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = 0
        self._height = 0
        self._start_time = 0.0

    def open(self) -> bool:
        """Open the webcam."""
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._start_time = time.time()
        return True

    def read(self) -> Optional[CameraFrame]:
        """Read a frame from the webcam."""
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate timestamp
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        return CameraFrame(
            rgb=rgb_frame,
            depth=None,
            timestamp_ms=timestamp_ms,
            intrinsics=None,
        )

    def close(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def has_depth(self) -> bool:
        """Webcam does not provide depth data."""
        return False

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        return (self._width, self._height)


class OrbbecCamera(CameraInterface):
    """Orbbec Gemini E RGB-D camera using pyorbbecsdk.

    Supported resolutions for Gemini E:
        Depth: 1024x768 @ 5-10fps, 640x480 @ 30fps, 512x384 @ 30fps
        Color: 1920x1080 @ 30fps
    """

    def __init__(
        self,
        color_width: int = 1920,
        color_height: int = 1080,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
    ):
        """Initialize Orbbec camera.

        Args:
            color_width: Color stream width (default 1920 for 1080p).
            color_height: Color stream height (default 1080 for 1080p).
            depth_width: Depth stream width (default 640 for 30fps support).
            depth_height: Depth stream height (default 480 for 30fps support).
            fps: Target framerate (default 30).
        """
        self._color_width = color_width
        self._color_height = color_height
        self._depth_width = depth_width
        self._depth_height = depth_height
        self._fps = fps

        self._pipeline = None
        self._config = None
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._start_time = 0.0

    def open(self) -> bool:
        """Open the Orbbec camera with color and depth streams."""
        try:
            from pyorbbecsdk import (
                Pipeline,
                Config,
                OBSensorType,
                OBFormat,
                OBAlignMode,
            )
        except ImportError:
            print("[Camera] pyorbbecsdk not installed")
            return False

        try:
            self._pipeline = Pipeline()

            # Get device and sensor info
            device = self._pipeline.get_device()
            device_info = device.get_device_info()
            print(f"[Camera] Found Orbbec device: {device_info.get_name()}")

            # Create config
            self._config = Config()

            # Enable color stream
            color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = None
            # First try: exact match with resolution and fps
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_video_stream_profile(i)
                if (profile.get_width() == self._color_width and
                    profile.get_height() == self._color_height and
                    profile.get_format() == OBFormat.RGB and
                    profile.get_fps() == self._fps):
                    color_profile = profile
                    break

            # Second try: match resolution, any fps
            if color_profile is None:
                for i in range(color_profiles.get_count()):
                    profile = color_profiles.get_video_stream_profile(i)
                    if (profile.get_width() == self._color_width and
                        profile.get_height() == self._color_height and
                        profile.get_format() == OBFormat.RGB):
                        color_profile = profile
                        break

            # Fall back to first available RGB profile
            if color_profile is None:
                for i in range(color_profiles.get_count()):
                    profile = color_profiles.get_video_stream_profile(i)
                    if profile.get_format() == OBFormat.RGB:
                        color_profile = profile
                        self._color_width = profile.get_width()
                        self._color_height = profile.get_height()
                        break

            if color_profile is None:
                print("[Camera] No suitable color profile found")
                return False

            self._config.enable_stream(color_profile)
            actual_color_fps = color_profile.get_fps()
            print(f"[Camera] Color stream: {self._color_width}x{self._color_height} @ {actual_color_fps}fps")

            # Enable depth stream
            depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = None
            # First try: exact match with resolution and fps
            for i in range(depth_profiles.get_count()):
                profile = depth_profiles.get_video_stream_profile(i)
                if (profile.get_width() == self._depth_width and
                    profile.get_height() == self._depth_height and
                    profile.get_fps() == self._fps):
                    depth_profile = profile
                    break

            # Second try: match resolution, any fps
            if depth_profile is None:
                for i in range(depth_profiles.get_count()):
                    profile = depth_profiles.get_video_stream_profile(i)
                    if (profile.get_width() == self._depth_width and
                        profile.get_height() == self._depth_height):
                        depth_profile = profile
                        break

            # Fall back to first available depth profile
            if depth_profile is None:
                depth_profile = depth_profiles.get_video_stream_profile(0)
                self._depth_width = depth_profile.get_width()
                self._depth_height = depth_profile.get_height()

            self._config.enable_stream(depth_profile)
            actual_depth_fps = depth_profile.get_fps()
            print(f"[Camera] Depth stream: {self._depth_width}x{self._depth_height} @ {actual_depth_fps}fps")

            # Enable depth-to-color alignment
            self._config.set_align_mode(OBAlignMode.SW_MODE)

            # Start pipeline
            self._pipeline.start(self._config)

            # Get camera intrinsics from color stream
            camera_param = self._pipeline.get_camera_param()
            color_intrinsic = camera_param.rgb_intrinsic
            self._intrinsics = CameraIntrinsics(
                fx=color_intrinsic.fx,
                fy=color_intrinsic.fy,
                cx=color_intrinsic.cx,
                cy=color_intrinsic.cy,
                width=int(color_intrinsic.width),
                height=int(color_intrinsic.height),
            )
            print(f"[Camera] Intrinsics: fx={self._intrinsics.fx:.1f}, fy={self._intrinsics.fy:.1f}, "
                  f"cx={self._intrinsics.cx:.1f}, cy={self._intrinsics.cy:.1f}")

            self._start_time = time.time()
            return True

        except Exception as e:
            print(f"[Camera] Failed to open Orbbec camera: {e}")
            self.close()
            return False

    def read(self) -> Optional[CameraFrame]:
        """Read aligned color and depth frames."""
        if self._pipeline is None:
            return None

        try:
            # Wait for frameset with 100ms timeout
            frameset = self._pipeline.wait_for_frames(100)
            if frameset is None:
                return None

            # Get color frame
            color_frame = frameset.get_color_frame()
            if color_frame is None:
                return None

            # Get depth frame
            depth_frame = frameset.get_depth_frame()

            # Convert color frame to numpy RGB array
            color_data = np.asarray(color_frame.get_data())
            # Reshape based on format
            rgb_frame = color_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))

            # Convert depth frame to numpy array (uint16, in mm)
            depth_data = None
            if depth_frame is not None:
                depth_data = np.asarray(depth_frame.get_data())
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.uint16)

            # Calculate timestamp
            timestamp_ms = int((time.time() - self._start_time) * 1000)

            return CameraFrame(
                rgb=rgb_frame,
                depth=depth_data,
                timestamp_ms=timestamp_ms,
                intrinsics=self._intrinsics,
            )

        except Exception as e:
            print(f"[Camera] Error reading frame: {e}")
            return None

    def close(self) -> None:
        """Stop the pipeline and release resources."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._config = None
        self._intrinsics = None

    def has_depth(self) -> bool:
        """Orbbec camera provides depth data."""
        return True

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns color stream resolution (width, height)."""
        return (self._color_width, self._color_height)


def create_camera(
    prefer_rgbd: bool = True,
    camera_id: int = 0,
    **kwargs,
) -> CameraInterface:
    """Factory function to create a camera with automatic fallback.

    Args:
        prefer_rgbd: If True, try Orbbec RGB-D first, fallback to webcam.
        camera_id: Webcam device ID (used if falling back).
        **kwargs: Additional arguments for OrbbecCamera.

    Returns:
        CameraInterface instance.

    Raises:
        RuntimeError: If no camera is available.
    """
    if prefer_rgbd:
        try:
            cam = OrbbecCamera(**kwargs)
            if cam.open():
                print("[Camera] Using Orbbec RGB-D camera")
                return cam
            cam.close()
        except ImportError:
            print("[Camera] pyorbbecsdk not installed, falling back to webcam")
        except Exception as e:
            print(f"[Camera] Orbbec init failed: {e}, falling back to webcam")

    cam = WebcamCamera(camera_id)
    if cam.open():
        print(f"[Camera] Using webcam (device {camera_id})")
        return cam

    raise RuntimeError("No camera available")
