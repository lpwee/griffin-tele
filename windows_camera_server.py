"""Camera streaming server for Windows.

Run this script on Windows to stream camera frames to WSL.
Install dependencies: pip install opencv-python flask

Usage:
    python windows_camera_server.py --camera 0 --port 5000
"""

import argparse
import cv2
from flask import Flask, Response
import time

app = Flask(__name__)

# Global camera object
camera = None
camera_lock = None


def init_camera(camera_id: int):
    """Initialize the camera."""
    global camera
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera initialized: {width}x{height} @ {fps} FPS")

    camera = cap
    return cap


def generate_frames():
    """Generate frames from camera."""
    global camera

    if camera is None:
        print("Error: Camera not initialized")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Error reading frame")
            break

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not ret:
            print("Error encoding frame")
            continue

        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

        # Print stats every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Streamed {frame_count} frames ({fps:.1f} FPS)")


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    """Health check endpoint."""
    return {'status': 'ok', 'camera': camera is not None}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera streaming server for Windows")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    print(f"Starting camera server on {args.host}:{args.port}")
    print(f"Camera ID: {args.camera}")
    print()

    # Initialize camera
    if init_camera(args.camera) is None:
        return

    print(f"\nServer ready!")
    print(f"Stream URL: http://localhost:{args.port}/video_feed")
    print(f"For WSL, use: http://$(hostname).local:{args.port}/video_feed")
    print(f"Or find your Windows IP with: ipconfig")
    print("\nPress Ctrl+C to stop")
    print()

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if camera is not None:
            camera.release()


if __name__ == "__main__":
    main()
