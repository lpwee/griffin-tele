"""Pose estimation module using MediaPipe Pose and Hands."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class ArmPose:
    """Extracted arm pose from MediaPipe landmarks."""

    # Wrist position in normalized coordinates (0-1)
    wrist_position: np.ndarray  # shape (3,) - x, y, z

    # Elbow and shoulder for orientation calculation
    elbow_position: np.ndarray  # shape (3,)
    shoulder_position: np.ndarray  # shape (3,)

    # Hand orientation (roll, pitch, yaw) in radians
    wrist_orientation: np.ndarray  # shape (3,)

    # Gripper state (0.0 = closed, 1.0 = open)
    gripper_openness: float

    # Confidence/visibility scores
    visibility: float

    # Whether tracking is valid this frame
    is_valid: bool


class PoseEstimator:
    """Wrapper for MediaPipe pose estimation with arm extraction."""

    # MediaPipe Pose landmark indices
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

    def __init__(
        self,
        model_path: str = "pose_landmarker.task",
        use_right_arm: bool = True,
        min_visibility: float = 0.5,
    ):
        """Initialize pose estimator.

        Args:
            model_path: Path to the MediaPipe pose landmarker model.
            use_right_arm: If True, track right arm; otherwise left arm.
            min_visibility: Minimum visibility score to consider landmarks valid.
        """
        self.use_right_arm = use_right_arm
        self.min_visibility = min_visibility
        self._latest_result: Optional[vision.PoseLandmarkerResult] = None

        # Set landmark indices based on arm choice
        if use_right_arm:
            self.shoulder_idx = self.RIGHT_SHOULDER
            self.elbow_idx = self.RIGHT_ELBOW
            self.wrist_idx = self.RIGHT_WRIST
            self.pinky_idx = self.RIGHT_PINKY
            self.index_idx = self.RIGHT_INDEX
            self.thumb_idx = self.RIGHT_THUMB
        else:
            self.shoulder_idx = self.LEFT_SHOULDER
            self.elbow_idx = self.LEFT_ELBOW
            self.wrist_idx = self.LEFT_WRIST
            self.pinky_idx = self.LEFT_PINKY
            self.index_idx = self.LEFT_INDEX
            self.thumb_idx = self.LEFT_THUMB

        # Create detector with live stream mode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def _result_callback(
        self,
        result: vision.PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        """Callback for async pose detection results."""
        self._latest_result = result

    def process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int) -> None:
        """Process a frame asynchronously.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3).
            timestamp_ms: Timestamp in milliseconds.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._detector.detect_async(mp_image, timestamp_ms)

    def get_arm_pose(self) -> ArmPose:
        """Extract arm pose from the latest detection result.

        Returns:
            ArmPose with current arm state, or invalid pose if no detection.
        """
        if self._latest_result is None or not self._latest_result.pose_landmarks:
            return self._invalid_pose()

        landmarks = self._latest_result.pose_landmarks[0]

        # Extract key landmarks
        shoulder = landmarks[self.shoulder_idx]
        elbow = landmarks[self.elbow_idx]
        wrist = landmarks[self.wrist_idx]
        pinky = landmarks[self.pinky_idx]
        index_finger = landmarks[self.index_idx]
        thumb = landmarks[self.thumb_idx]

        # Check visibility
        min_vis = min(shoulder.visibility, elbow.visibility, wrist.visibility)
        if min_vis < self.min_visibility:
            return self._invalid_pose()

        # Extract positions
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
        elbow_pos = np.array([elbow.x, elbow.y, elbow.z])
        shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])

        # Calculate wrist orientation from forearm direction and hand landmarks
        orientation = self._calculate_wrist_orientation(
            wrist_pos, elbow_pos,
            np.array([index_finger.x, index_finger.y, index_finger.z]),
            np.array([pinky.x, pinky.y, pinky.z]),
        )

        # Estimate gripper openness from thumb-index distance
        gripper_openness = self._calculate_gripper_openness(
            np.array([thumb.x, thumb.y, thumb.z]),
            np.array([index_finger.x, index_finger.y, index_finger.z]),
        )

        return ArmPose(
            wrist_position=wrist_pos,
            elbow_position=elbow_pos,
            shoulder_position=shoulder_pos,
            wrist_orientation=orientation,
            gripper_openness=gripper_openness,
            visibility=min_vis,
            is_valid=True,
        )

    def _calculate_wrist_orientation(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        index_finger: np.ndarray,
        pinky: np.ndarray,
    ) -> np.ndarray:
        """Calculate wrist orientation as roll, pitch, yaw.

        Uses forearm direction and hand plane to estimate orientation.
        """
        # Forearm direction (elbow to wrist)
        forearm = wrist - elbow
        forearm_norm = np.linalg.norm(forearm)
        if forearm_norm < 1e-6:
            return np.zeros(3)
        forearm = forearm / forearm_norm

        # Hand plane normal (cross product of index-wrist and pinky-wrist)
        to_index = index_finger - wrist
        to_pinky = pinky - wrist
        hand_normal = np.cross(to_index, to_pinky)
        hand_norm = np.linalg.norm(hand_normal)
        if hand_norm < 1e-6:
            hand_normal = np.array([0, 0, 1])
        else:
            hand_normal = hand_normal / hand_norm

        # Calculate Euler angles
        # Pitch: angle from horizontal (forearm y component)
        pitch = np.arcsin(np.clip(-forearm[1], -1, 1))

        # Yaw: angle in horizontal plane
        yaw = np.arctan2(forearm[0], -forearm[2])

        # Roll: rotation around forearm axis (from hand normal)
        # Project hand normal onto plane perpendicular to forearm
        roll = np.arctan2(hand_normal[0], hand_normal[1])

        return np.array([roll, pitch, yaw])

    def _calculate_gripper_openness(
        self,
        thumb: np.ndarray,
        index_finger: np.ndarray,
    ) -> float:
        """Calculate gripper openness from thumb-index distance.

        Returns value between 0 (closed) and 1 (open).
        """
        distance = np.linalg.norm(thumb - index_finger)
        # Normalize: ~0.02 closed, ~0.15 open (in normalized coords)
        openness = (distance - 0.02) / 0.13
        return float(np.clip(openness, 0.0, 1.0))

    def _invalid_pose(self) -> ArmPose:
        """Return an invalid pose marker."""
        return ArmPose(
            wrist_position=np.zeros(3),
            elbow_position=np.zeros(3),
            shoulder_position=np.zeros(3),
            wrist_orientation=np.zeros(3),
            gripper_openness=0.0,
            visibility=0.0,
            is_valid=False,
        )

    def close(self):
        """Release resources."""
        self._detector.close()
