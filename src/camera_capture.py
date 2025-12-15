"""Camera capture utilities for local and network streams."""

from typing import Optional, Tuple
import cv2
import numpy as np
import urllib.request


class CameraCapture:
    """Unified interface for local and network camera capture."""

    def __init__(self, source: str = "0"):
        """Initialize camera capture.

        Args:
            source: Camera source. Can be:
                - Integer camera ID as string (e.g., "0", "1")
                - HTTP/RTSP URL (e.g., "http://localhost:5000/video_feed")
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_network_stream = source.startswith("http://") or source.startswith("https://")

    def open(self) -> bool:
        """Open the camera/stream.

        Returns:
            True if successful, False otherwise.
        """
        if self.is_network_stream:
            # For network streams, OpenCV can handle MJPEG streams directly
            self.cap = cv2.VideoCapture(self.source)
        else:
            # Local camera
            try:
                camera_id = int(self.source)
                self.cap = cv2.VideoCapture(camera_id)
            except ValueError:
                print(f"Error: Invalid camera source '{self.source}'")
                return False

        if self.cap is None or not self.cap.isOpened():
            return False

        return True

    def isOpened(self) -> bool:
        """Check if camera is opened.

        Returns:
            True if opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success, frame).
        """
        if self.cap is None:
            return False, None

        return self.cap.read()

    def get(self, prop: int) -> float:
        """Get camera property.

        Args:
            prop: OpenCV property ID.

        Returns:
            Property value.
        """
        if self.cap is None:
            return 0.0

        return self.cap.get(prop)

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def test_stream(url: str, duration: int = 5, show_display: bool = False):
    """Test a camera stream.

    Args:
        url: Stream URL or camera ID.
        duration: Test duration in seconds.
        show_display: If True, attempt to display video (requires X11/display).
    """
    print(f"Testing stream: {url}")

    cap = CameraCapture(url)

    if not cap.open():
        print(f"Error: Could not open stream")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream: {width}x{height}")

    if not show_display:
        print("Running in headless mode (no display)")

    import time
    start_time = time.time()
    frame_count = 0
    display_available = show_display

    try:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame")
                break

            frame_count += 1

            # Display frame if requested and available
            if display_available:
                try:
                    cv2.imshow('Stream Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error as e:
                    print(f"Display not available: {e}")
                    print("Continuing in headless mode...")
                    display_available = False

            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  {frame_count} frames, {fps:.1f} FPS", end='\r')

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nReceived {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

        if frame_count > 0:
            print("✓ Stream is working!")
        else:
            print("✗ No frames received")

        cap.release()
        if display_available:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test camera stream")
    parser.add_argument("source", help="Camera ID (0, 1, ...) or stream URL (http://...)")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in seconds")
    parser.add_argument("--display", action="store_true", help="Show video display (requires X11)")
    args = parser.parse_args()

    test_stream(args.source, args.duration, args.display)
