"""
Pose Landmarker Live Stream
Real-time pose detection using MediaPipe and OpenCV
"""

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Global variable to store the annotated frame
annotated_frame = None


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on the image."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback function for async pose detection results."""
    global annotated_frame
    # Ensure output_image is an RGB NumPy array for drawing
    rgb_frame = output_image.numpy_view()
    if result.pose_landmarks:
        annotated_frame = draw_landmarks_on_image(rgb_frame, result)
    else:
        annotated_frame = rgb_frame  # No landmarks, just show the original frame


def main():
    global annotated_frame

    # STEP 1: Create a PoseLandmarker object for live stream.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback)

    # Create the PoseLandmarker detector
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 2: Initialize webcam capture.
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10
    frame_delay = 1.0 / fps

    print(f"Capturing video at {frame_width}x{frame_height} @ {fps} FPS")

    # Initialize a window to display the video
    cv2.namedWindow('Pose Landmarker Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Landmarker Live', frame_width, frame_height)

    # Timestamp for asynchronous processing
    start_time = time.time()

    # STEP 3: Loop to read frames and perform detection.
    while cap.isOpened():
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("End of stream or error reading frame.")
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get current timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Perform pose detection asynchronously
        detector.detect_async(mp_image, timestamp_ms)

        # Display the annotated frame if available
        if annotated_frame is not None:
            # Convert back to BGR for OpenCV display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Pose Landmarker Live', display_frame)
        else:
            # If no results yet, display original frame
            cv2.imshow('Pose Landmarker Live', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Limit frame rate to target FPS
        elapsed = time.time() - frame_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    # STEP 4: Release resources.
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("Live stream ended.")


if __name__ == "__main__":
    main()
