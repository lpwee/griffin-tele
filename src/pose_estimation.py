"""Pose estimation module using MediaPipe Pose and Hands."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if TYPE_CHECKING:
    from .depth_processor import DepthProcessor
    from .camera import CameraIntrinsics


@dataclass
class ArmPose:
    """Extracted arm pose from MediaPipe landmarks."""

    # Wrist position relative to shoulder
    # For webcam: normalized coordinates (0-1 range)
    # For RGB-D: metric coordinates in meters (camera frame)
    wrist_position: np.ndarray  # shape (3,) - x, y, z

    # Elbow and shoulder positions
    # For webcam: normalized coordinates
    # For RGB-D: absolute 3D positions in meters (camera frame)
    elbow_position: np.ndarray  # shape (3,)
    shoulder_position: np.ndarray  # shape (3,)

    # Hand orientation (roll, pitch, yaw) in radians
    # Only available when hand is tracked, otherwise zeros
    wrist_orientation: np.ndarray  # shape (3,)

    # Confidence/visibility scores
    visibility: float

    # Whether tracking is valid this frame
    is_valid: bool

    # Whether hand landmarks were used (required for orientation)
    hand_tracked: bool = False

    # Whether positions are in metric coordinates (meters)
    # True for RGB-D mode, False for webcam mode
    is_metric: bool = False

    # Whether depth lookup succeeded (only relevant when is_metric=True)
    depth_valid: bool = False


class PoseEstimator:
    """Wrapper for MediaPipe pose estimation with arm extraction."""

    # MediaPipe Pose landmark indices (33 total)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22

    # MediaPipe Hand landmark indices (only those needed for orientation)
    HAND_WRIST = 0
    HAND_INDEX_MCP = 5
    HAND_MIDDLE_MCP = 9
    HAND_PINKY_MCP = 17

    def __init__(
        self,
        pose_model_path: str = "pose_landmarker.task",
        hand_model_path: Optional[str] = "hand_landmarker.task",
        use_right_arm: bool = True,
        min_visibility: float = 0.5,
        min_hand_confidence: float = 0.5,
        depth_processor: Optional["DepthProcessor"] = None,
    ):
        """Initialize pose estimator with optional hand tracking and depth.

        Args:
            pose_model_path: Path to the MediaPipe pose landmarker model.
            hand_model_path: Path to the MediaPipe hand landmarker model.
                           Set to None to disable hand tracking.
            use_right_arm: If True, track right arm; otherwise left arm.
            min_visibility: Minimum visibility score for pose landmarks.
            min_hand_confidence: Minimum confidence for hand detection.
            depth_processor: Optional depth processor for RGB-D mode.
        """
        self.use_right_arm = use_right_arm
        self.min_visibility = min_visibility
        self.min_hand_confidence = min_hand_confidence
        self._latest_pose_result: Optional[vision.PoseLandmarkerResult] = None
        self._latest_hand_result: Optional[vision.HandLandmarkerResult] = None
        self._use_hand_model = hand_model_path is not None

        # Depth processing for RGB-D mode
        self._depth_processor = depth_processor
        self._current_depth_frame: Optional[np.ndarray] = None
        self._current_intrinsics: Optional["CameraIntrinsics"] = None

        # Set landmark indices based on arm choice
        if use_right_arm:
            self.shoulder_idx = self.RIGHT_SHOULDER
            self.elbow_idx = self.RIGHT_ELBOW
            self.wrist_idx = self.RIGHT_WRIST
            self.pinky_idx = self.RIGHT_PINKY
            self.index_idx = self.RIGHT_INDEX
            self.thumb_idx = self.RIGHT_THUMB
            self._target_handedness = "Right"
        else:
            self.shoulder_idx = self.LEFT_SHOULDER
            self.elbow_idx = self.LEFT_ELBOW
            self.wrist_idx = self.LEFT_WRIST
            self.pinky_idx = self.LEFT_PINKY
            self.index_idx = self.LEFT_INDEX
            self.thumb_idx = self.LEFT_THUMB
            self._target_handedness = "Left"

        # Create pose detector
        pose_base = python.BaseOptions(model_asset_path=pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._pose_result_callback,
        )
        self._pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # Create hand detector if model provided
        self._hand_detector: Optional[vision.HandLandmarker] = None
        if self._use_hand_model:
            hand_base = python.BaseOptions(model_asset_path=hand_model_path)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=min_hand_confidence,
                min_hand_presence_confidence=min_hand_confidence,
                min_tracking_confidence=min_hand_confidence,
                result_callback=self._hand_result_callback,
            )
            self._hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    def _pose_result_callback(
        self,
        result: vision.PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Callback for async pose detection results."""
        self._latest_pose_result = result

    def _hand_result_callback(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Callback for async hand detection results."""
        self._latest_hand_result = result

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int,
        depth_frame: Optional[np.ndarray] = None,
        intrinsics: Optional["CameraIntrinsics"] = None,
    ) -> None:
        """Process a frame asynchronously with both pose and hand models.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3).
            timestamp_ms: Timestamp in milliseconds.
            depth_frame: Optional aligned depth image (H, W) in mm.
            intrinsics: Optional camera intrinsics for 3D deprojection.
        """
        # Store depth data for get_arm_pose
        self._current_depth_frame = depth_frame
        self._current_intrinsics = intrinsics

        # MediaPipe expects a contiguous uint8 RGB image with shape (H, W, 3).
        if rgb_frame is None:
            raise ValueError("PoseEstimator.process_frame received None for rgb_frame")

        if not isinstance(rgb_frame, np.ndarray):
            raise ValueError(
                f"PoseEstimator.process_frame expected np.ndarray, "
                f"got {type(rgb_frame)}"
            )

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(
                "PoseEstimator.process_frame expected image with shape "
                "(H, W, 3), got "
                f"{rgb_frame.shape}"
            )

        # Ensure correct dtype and memory layout
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)

        rgb_frame = np.ascontiguousarray(rgb_frame)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run pose detection
        self._pose_detector.detect_async(mp_image, timestamp_ms)

        # Run hand detection if enabled
        if self._hand_detector is not None:
            self._hand_detector.detect_async(mp_image, timestamp_ms)

    def get_arm_pose(self) -> ArmPose:
        """Extract arm pose from the latest detection results.

        Uses pose landmarks for arm position tracking and hand landmarks
        for wrist orientation. When depth data is available, provides
        metric 3D positions; otherwise uses normalized camera coordinates.

        Returns:
            ArmPose with current arm state, or invalid pose if no detection.
        """
        if self._latest_pose_result is None or not self._latest_pose_result.pose_landmarks:
            return self._invalid_pose()

        pose_landmarks = self._latest_pose_result.pose_landmarks[0]

        # Extract pose landmarks
        shoulder = pose_landmarks[self.shoulder_idx]
        elbow = pose_landmarks[self.elbow_idx]
        wrist = pose_landmarks[self.wrist_idx]

        # Check visibility
        min_vis = min(shoulder.visibility, elbow.visibility, wrist.visibility)
        if min_vis < self.min_visibility:
            return self._invalid_pose()

        # Check if we should use depth-based 3D reconstruction
        use_depth = (
            self._current_depth_frame is not None
            and self._current_intrinsics is not None
            and self._depth_processor is not None
        )

        if use_depth:
            # Try to get metric 3D positions using depth
            arm_pose = self._get_arm_pose_with_depth(
                shoulder, elbow, wrist, min_vis
            )
            if arm_pose is not None:
                return arm_pose
            # Fall through to normalized mode if depth fails

        # Fallback: normalized coordinates (webcam mode)
        return self._get_arm_pose_normalized(shoulder, elbow, wrist, min_vis)

    def _get_arm_pose_normalized(
        self,
        shoulder,
        elbow,
        wrist,
        visibility: float,
    ) -> ArmPose:
        """Get arm pose using normalized camera coordinates (webcam mode)."""
        # Extract arm positions from pose (normalized 0-1)
        shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
        elbow_pos = np.array([elbow.x, elbow.y, elbow.z])
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])

        # Try to get hand landmarks for orientation
        hand_landmarks = self.get_matching_hand()
        hand_tracked = hand_landmarks is not None

        if hand_tracked:
            orientation = self._calculate_hand_orientation(hand_landmarks)
        else:
            orientation = np.zeros(3)

        # Calculate wrist position relative to shoulder
        relative_wrist_pos = wrist_pos - shoulder_pos

        return ArmPose(
            wrist_position=relative_wrist_pos,
            elbow_position=elbow_pos,
            shoulder_position=shoulder_pos,
            wrist_orientation=orientation,
            visibility=visibility,
            is_valid=True,
            hand_tracked=hand_tracked,
            is_metric=False,
            depth_valid=False,
        )

    def _get_arm_pose_with_depth(
        self,
        shoulder,
        elbow,
        wrist,
        visibility: float,
    ) -> Optional[ArmPose]:
        """Get arm pose using depth data for metric 3D positions.

        Args:
            shoulder, elbow, wrist: MediaPipe landmark objects.
            visibility: Minimum visibility score.

        Returns:
            ArmPose with metric positions, or None if depth lookup fails.
        """
        intrinsics = self._current_intrinsics
        depth = self._current_depth_frame
        h, w = depth.shape

        # Convert normalized coords to pixels
        def to_pixel(landmark):
            return int(landmark.x * w), int(landmark.y * h)

        shoulder_px = to_pixel(shoulder)
        elbow_px = to_pixel(elbow)
        wrist_px = to_pixel(wrist)

        # Lookup depth for each joint
        shoulder_depth = self._depth_processor.lookup_depth(
            depth, shoulder_px[0], shoulder_px[1], "shoulder"
        )
        elbow_depth = self._depth_processor.lookup_depth(
            depth, elbow_px[0], elbow_px[1], "elbow"
        )
        wrist_depth = self._depth_processor.lookup_depth(
            depth, wrist_px[0], wrist_px[1], "wrist"
        )

        # All joints need valid depth
        if None in (shoulder_depth, elbow_depth, wrist_depth):
            return None

        # Deproject to 3D (meters, camera frame)
        shoulder_3d = self._depth_processor.deproject(
            shoulder_px[0], shoulder_px[1], shoulder_depth, intrinsics
        )
        elbow_3d = self._depth_processor.deproject(
            elbow_px[0], elbow_px[1], elbow_depth, intrinsics
        )
        wrist_3d = self._depth_processor.deproject(
            wrist_px[0], wrist_px[1], wrist_depth, intrinsics
        )

        # Compute relative wrist position (key output for IK)
        relative_wrist = wrist_3d - shoulder_3d

        # Get orientation from hand landmarks
        hand_landmarks = self.get_matching_hand()
        hand_tracked = hand_landmarks is not None

        if hand_tracked:
            orientation = self._calculate_hand_orientation(hand_landmarks)
        else:
            orientation = np.zeros(3)

        return ArmPose(
            wrist_position=relative_wrist,
            elbow_position=elbow_3d,
            shoulder_position=shoulder_3d,
            wrist_orientation=orientation,
            visibility=visibility,
            is_valid=True,
            hand_tracked=hand_tracked,
            is_metric=True,
            depth_valid=True,
        )

    def get_matching_hand(self):
        """Get hand landmarks matching the target handedness."""
        if (self._latest_hand_result is None or
            not self._latest_hand_result.hand_landmarks or
            not self._latest_hand_result.handedness):
            return None

        # Find hand matching target handedness (Right or Left)
        for i, handedness_list in enumerate(self._latest_hand_result.handedness):
            if handedness_list and len(handedness_list) > 0:
                # Note: MediaPipe returns mirrored handedness for selfie camera
                # "Right" in results = left hand in reality (and vice versa)
                # So we look for the opposite of what we want
                detected_hand = handedness_list[0].category_name
                # For a selfie/front camera, the labels are mirrored
                # If we want to track the user's right arm, we look for "Left" label
                target = "Left" if self._target_handedness == "Right" else "Right"
                if detected_hand == target:
                    return self._latest_hand_result.hand_landmarks[i]

        # If no matching hand, return first detected hand as fallback
        if self._latest_hand_result.hand_landmarks:
            return self._latest_hand_result.hand_landmarks[0]

        return None

    def _calculate_hand_orientation(self, hand_landmarks) -> np.ndarray:
        """Calculate wrist orientation from hand landmarks.

        Uses the hand's coordinate frame for accurate orientation.
        """
        wrist = hand_landmarks[self.HAND_WRIST]
        index_mcp = hand_landmarks[self.HAND_INDEX_MCP]
        pinky_mcp = hand_landmarks[self.HAND_PINKY_MCP]
        middle_mcp = hand_landmarks[self.HAND_MIDDLE_MCP]

        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
        index_pos = np.array([index_mcp.x, index_mcp.y, index_mcp.z])
        pinky_pos = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z])
        middle_pos = np.array([middle_mcp.x, middle_mcp.y, middle_mcp.z])

        # Hand forward direction (wrist to middle finger base)
        forward = middle_pos - wrist_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return np.zeros(3)
        forward = forward / forward_norm

        # Hand right direction (pinky to index)
        right = index_pos - pinky_pos
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1, 0, 0])
        else:
            right = right / right_norm

        # Hand up direction (cross product)
        up = np.cross(forward, right)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            up = np.array([0, 1, 0])
        else:
            up = up / up_norm

        # Calculate Euler angles from the hand's coordinate frame
        # Pitch: angle from horizontal
        pitch = np.arcsin(np.clip(-forward[1], -1, 1))

        # Yaw: rotation in horizontal plane
        yaw = np.arctan2(forward[0], -forward[2])

        # Roll: rotation around forward axis
        roll = np.arctan2(right[1], up[1])

        return np.array([roll, pitch, yaw])

    def _invalid_pose(self) -> ArmPose:
        """Return an invalid pose marker."""
        return ArmPose(
            wrist_position=np.zeros(3),
            elbow_position=np.zeros(3),
            shoulder_position=np.zeros(3),
            wrist_orientation=np.zeros(3),
            visibility=0.0,
            is_valid=False,
            hand_tracked=False,
            is_metric=False,
            depth_valid=False,
        )

    def get_latest_pose_result(self) -> Optional[vision.PoseLandmarkerResult]:
        """Get the latest raw pose detection result."""
        return self._latest_pose_result

    def get_latest_hand_result(self) -> Optional[vision.HandLandmarkerResult]:
        """Get the latest raw hand detection result."""
        return self._latest_hand_result

    # Pose landmark connections (subset focusing on upper body)
    POSE_CONNECTIONS = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 24),  # hips
        (15, 17), (15, 19), (15, 21),  # left hand
        (16, 18), (16, 20), (16, 22),  # right hand
        (17, 19), (18, 20),  # hand edges
    ]

    # Hand landmark connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),  # palm
    ]

    def draw_landmarks(self, rgb_image: np.ndarray) -> np.ndarray:
        """Draw all pose and hand landmarks on image.

        Args:
            rgb_image: RGB image to draw on.

        Returns:
            Image with landmarks drawn.
        """
        annotated = np.copy(rgb_image)
        h, w = annotated.shape[:2]

        # Draw pose landmarks
        if self._latest_pose_result and self._latest_pose_result.pose_landmarks:
            for pose_landmarks in self._latest_pose_result.pose_landmarks:
                # Draw connections
                for start_idx, end_idx in self.POSE_CONNECTIONS:
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        start = pose_landmarks[start_idx]
                        end = pose_landmarks[end_idx]
                        if start.visibility > 0.5 and end.visibility > 0.5:
                            pt1 = (int(start.x * w), int(start.y * h))
                            pt2 = (int(end.x * w), int(end.y * h))
                            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

                # Draw landmarks
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        pt = (int(landmark.x * w), int(landmark.y * h))
                        cv2.circle(annotated, pt, 4, (255, 0, 0), -1)

        # Draw hand landmarks
        if self._latest_hand_result and self._latest_hand_result.hand_landmarks:
            for hand_landmarks in self._latest_hand_result.hand_landmarks:
                # Draw connections
                for start_idx, end_idx in self.HAND_CONNECTIONS:
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        pt1 = (int(start.x * w), int(start.y * h))
                        pt2 = (int(end.x * w), int(end.y * h))
                        cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)

                # Draw landmarks
                for landmark in hand_landmarks:
                    pt = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(annotated, pt, 3, (255, 0, 255), -1)

        return annotated

    def close(self):
        """Release resources."""
        self._pose_detector.close()
        if self._hand_detector is not None:
            self._hand_detector.close()
