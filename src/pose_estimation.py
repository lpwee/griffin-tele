"""Pose estimation module using MediaPipe Pose and Hands."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
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

    # Whether hand landmarks were used (vs just pose landmarks)
    hand_tracked: bool = False


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

    # MediaPipe Hand landmark indices (21 total)
    HAND_WRIST = 0
    HAND_THUMB_CMC = 1
    HAND_THUMB_MCP = 2
    HAND_THUMB_IP = 3
    HAND_THUMB_TIP = 4
    HAND_INDEX_MCP = 5
    HAND_INDEX_PIP = 6
    HAND_INDEX_DIP = 7
    HAND_INDEX_TIP = 8
    HAND_MIDDLE_MCP = 9
    HAND_MIDDLE_PIP = 10
    HAND_MIDDLE_DIP = 11
    HAND_MIDDLE_TIP = 12
    HAND_RING_MCP = 13
    HAND_RING_PIP = 14
    HAND_RING_DIP = 15
    HAND_RING_TIP = 16
    HAND_PINKY_MCP = 17
    HAND_PINKY_PIP = 18
    HAND_PINKY_DIP = 19
    HAND_PINKY_TIP = 20

    def __init__(
        self,
        pose_model_path: str = "pose_landmarker.task",
        hand_model_path: Optional[str] = "hand_landmarker.task",
        use_right_arm: bool = True,
        min_visibility: float = 0.5,
        min_hand_confidence: float = 0.5,
    ):
        """Initialize pose estimator with optional hand tracking.

        Args:
            pose_model_path: Path to the MediaPipe pose landmarker model.
            hand_model_path: Path to the MediaPipe hand landmarker model.
                           Set to None to disable hand tracking.
            use_right_arm: If True, track right arm; otherwise left arm.
            min_visibility: Minimum visibility score for pose landmarks.
            min_hand_confidence: Minimum confidence for hand detection.
        """
        self.use_right_arm = use_right_arm
        self.min_visibility = min_visibility
        self.min_hand_confidence = min_hand_confidence
        self._latest_pose_result: Optional[vision.PoseLandmarkerResult] = None
        self._latest_hand_result: Optional[vision.HandLandmarkerResult] = None
        self._use_hand_model = hand_model_path is not None

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

    def process_frame(self, rgb_frame: np.ndarray, timestamp_ms: int) -> None:
        """Process a frame asynchronously with both pose and hand models.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3).
            timestamp_ms: Timestamp in milliseconds.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run pose detection
        self._pose_detector.detect_async(mp_image, timestamp_ms)

        # Run hand detection if enabled
        if self._hand_detector is not None:
            self._hand_detector.detect_async(mp_image, timestamp_ms)

    def get_arm_pose(self) -> ArmPose:
        """Extract arm pose from the latest detection results.

        Uses pose landmarks for arm tracking and hand landmarks for
        precise gripper control and wrist orientation when available.

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

        # Extract arm positions from pose
        shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
        elbow_pos = np.array([elbow.x, elbow.y, elbow.z])
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])

        # Get pose landmarks for fallback calculations
        pinky = pose_landmarks[self.pinky_idx]
        index_finger = pose_landmarks[self.index_idx]
        thumb = pose_landmarks[self.thumb_idx]

        # Always use pose-based orientation (forearm direction) for pitch and yaw
        # This is more stable as it uses larger body landmarks
        pose_orientation = self._calculate_wrist_orientation(
            wrist_pos, elbow_pos,
            np.array([index_finger.x, index_finger.y, index_finger.z]),
            np.array([pinky.x, pinky.y, pinky.z]),
        )

        # Try to get hand landmarks for gripper and roll override
        hand_landmarks = self._get_matching_hand()
        hand_tracked = hand_landmarks is not None

        if hand_tracked:
            # Use hand landmarks for roll (gripper rotation) and gripper openness
            hand_roll = self._calculate_hand_roll(hand_landmarks)
            orientation = np.array([hand_roll, pose_orientation[1], pose_orientation[2]])
            gripper_openness = self._calculate_hand_gripper_openness(hand_landmarks)
        else:
            # Use full pose-based orientation and gripper
            orientation = pose_orientation
            gripper_openness = self._calculate_gripper_openness(
                np.array([thumb.x, thumb.y, thumb.z]),
                np.array([index_finger.x, index_finger.y, index_finger.z]),
            )

        # Calculate wrist position relative to shoulder (shoulder = base)
        relative_wrist_pos = wrist_pos - shoulder_pos

        return ArmPose(
            wrist_position=relative_wrist_pos,
            elbow_position=elbow_pos,
            shoulder_position=shoulder_pos,
            wrist_orientation=orientation,
            gripper_openness=gripper_openness,
            visibility=min_vis,
            is_valid=True,
            hand_tracked=hand_tracked,
        )

    def _get_matching_hand(self):
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

    def _calculate_hand_orientation(self, hand_landmarks, elbow_pos: np.ndarray) -> np.ndarray:
        """Calculate wrist orientation from detailed hand landmarks.

        Uses the hand's coordinate frame for more accurate orientation.
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

    def _calculate_hand_roll(self, hand_landmarks) -> float:
        """Calculate gripper roll (rotation around forward axis) from hand landmarks.

        Uses the hand's coordinate frame to extract just the roll component,
        which represents the gripper's twist orientation.
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
            return 0.0
        forward = forward / forward_norm

        # Hand right direction (pinky to index)
        right = index_pos - pinky_pos
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return 0.0
        right = right / right_norm

        # Hand up direction (cross product)
        up = np.cross(forward, right)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            return 0.0
        up = up / up_norm

        # Roll = rotation around forward axis
        roll = np.arctan2(right[1], up[1])
        return roll

    def _calculate_hand_gripper_openness(self, hand_landmarks) -> float:
        """Calculate gripper openness from detailed hand landmarks.

        Uses multiple finger positions for robust fist/open detection.
        """
        # Get fingertip and MCP (knuckle) positions
        thumb_tip = hand_landmarks[self.HAND_THUMB_TIP]
        index_tip = hand_landmarks[self.HAND_INDEX_TIP]
        middle_tip = hand_landmarks[self.HAND_MIDDLE_TIP]
        ring_tip = hand_landmarks[self.HAND_RING_TIP]
        pinky_tip = hand_landmarks[self.HAND_PINKY_TIP]

        index_mcp = hand_landmarks[self.HAND_INDEX_MCP]
        middle_mcp = hand_landmarks[self.HAND_MIDDLE_MCP]
        ring_mcp = hand_landmarks[self.HAND_RING_MCP]
        pinky_mcp = hand_landmarks[self.HAND_PINKY_MCP]

        wrist = hand_landmarks[self.HAND_WRIST]

        # Calculate finger extension ratios
        # (distance from wrist to tip) / (distance from wrist to mcp)
        def finger_extension(tip, mcp):
            wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
            tip_pos = np.array([tip.x, tip.y, tip.z])
            mcp_pos = np.array([mcp.x, mcp.y, mcp.z])

            tip_dist = np.linalg.norm(tip_pos - wrist_pos)
            mcp_dist = np.linalg.norm(mcp_pos - wrist_pos)

            if mcp_dist < 1e-6:
                return 1.0
            return tip_dist / mcp_dist

        # Average extension of 4 fingers (excluding thumb)
        extensions = [
            finger_extension(index_tip, index_mcp),
            finger_extension(middle_tip, middle_mcp),
            finger_extension(ring_tip, ring_mcp),
            finger_extension(pinky_tip, pinky_mcp),
        ]
        avg_extension = np.mean(extensions)

        # Also check thumb-index pinch distance for gripper
        thumb_pos = np.array([thumb_tip.x, thumb_tip.y, thumb_tip.z])
        index_pos = np.array([index_tip.x, index_tip.y, index_tip.z])
        pinch_dist = np.linalg.norm(thumb_pos - index_pos)

        # Combine: low extension OR small pinch = closed gripper
        # extension ~1.3 = closed fist, ~1.8 = open hand
        extension_openness = (avg_extension - 1.3) / 0.5
        # pinch ~0.02 = pinched, ~0.15 = open
        pinch_openness = (pinch_dist - 0.02) / 0.13

        # Use minimum (either fist or pinch closes gripper)
        openness = min(extension_openness, pinch_openness)
        return float(np.clip(openness, 0.0, 1.0))

    def _calculate_wrist_orientation(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        index_finger: np.ndarray,
        pinky: np.ndarray,
    ) -> np.ndarray:
        """Calculate wrist orientation from pose landmarks (fallback)."""
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
        pitch = np.arcsin(np.clip(-forearm[1], -1, 1))
        yaw = np.arctan2(forearm[0], -forearm[2])
        roll = np.arctan2(hand_normal[0], hand_normal[1])

        return np.array([roll, pitch, yaw])

    def _calculate_gripper_openness(
        self,
        thumb: np.ndarray,
        index_finger: np.ndarray,
    ) -> float:
        """Calculate gripper openness from pose landmarks (fallback)."""
        distance = np.linalg.norm(thumb - index_finger)
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
            hand_tracked=False,
        )

    def get_latest_pose_result(self) -> Optional[vision.PoseLandmarkerResult]:
        """Get the latest raw pose detection result."""
        return self._latest_pose_result

    def get_latest_hand_result(self) -> Optional[vision.HandLandmarkerResult]:
        """Get the latest raw hand detection result."""
        return self._latest_hand_result

    def draw_landmarks(self, rgb_image: np.ndarray) -> np.ndarray:
        """Draw all pose and hand landmarks on image.

        Args:
            rgb_image: RGB image to draw on.

        Returns:
            Image with landmarks drawn.
        """
        annotated = np.copy(rgb_image)

        # Draw pose landmarks
        if self._latest_pose_result and self._latest_pose_result.pose_landmarks:
            for pose_landmarks in self._latest_pose_result.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    ) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        # Draw hand landmarks
        if self._latest_hand_result and self._latest_hand_result.hand_landmarks:
            for hand_landmarks in self._latest_hand_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    ) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

        return annotated

    def close(self):
        """Release resources."""
        self._pose_detector.close()
        if self._hand_detector is not None:
            self._hand_detector.close()
