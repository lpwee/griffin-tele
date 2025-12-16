"""Gripper controller module for Piper robot arm.

Separates gripper control from the inverse kinematics chain,
since the gripper (prismatic joints 7 and 8) doesn't affect
end-effector pose and should be controlled independently.

Uses MediaPipe hand landmarks for gripper openness detection.
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np


@dataclass
class GripperConfig:
    """Configuration for gripper control."""

    # Physical gripper limits (meters)
    # 0.0 = fully closed, 0.08 = fully open
    min_position: float = 0.0
    max_position: float = 0.08

    # Smoothing factor (0 = no smoothing, 1 = infinite smoothing)
    smoothing_alpha: float = 0.3

    # Speed limit (meters per second) - prevents jerky movements
    max_speed: float = 0.1  # 10cm/s max gripper speed


@dataclass
class GripperState:
    """Current state of the gripper."""

    # Target position in meters (0.0 = closed, 0.08 = open)
    position: float

    # Raw openness value from pose estimation (0.0 to 1.0)
    raw_openness: float


class GripperController:
    """Controls gripper position independently from arm IK.

    The Piper arm has 6 revolute joints for positioning plus 2 prismatic
    gripper joints. This controller handles the gripper separately from
    the IK solver, which only deals with the 6 arm joints.

    Computes gripper openness from MediaPipe hand landmarks.
    """

    # MediaPipe Hand landmark indices
    HAND_WRIST = 0
    HAND_THUMB_TIP = 4
    HAND_INDEX_MCP = 5
    HAND_INDEX_TIP = 8
    HAND_MIDDLE_MCP = 9
    HAND_MIDDLE_TIP = 12
    HAND_RING_MCP = 13
    HAND_RING_TIP = 16
    HAND_PINKY_MCP = 17
    HAND_PINKY_TIP = 20

    def __init__(self, config: Optional[GripperConfig] = None):
        """Initialize gripper controller.

        Args:
            config: Gripper configuration. Uses defaults if None.
        """
        self.config = config or GripperConfig()
        self._prev_position: Optional[float] = None
        self._prev_time: Optional[float] = None

    def update(
        self,
        hand_landmarks: Optional[List[Any]],
        current_time: Optional[float] = None,
    ) -> Optional[GripperState]:
        """Update gripper state from hand landmarks.

        Args:
            hand_landmarks: MediaPipe hand landmarks list (21 landmarks).
                           If None, returns None (no hand detected).
            current_time: Current timestamp in seconds. Used for rate limiting.
                         If None, rate limiting is disabled.

        Returns:
            GripperState with the computed gripper position, or None if no hand.
        """
        if hand_landmarks is None:
            return None

        cfg = self.config

        # Compute gripper openness from hand landmarks
        gripper_openness = self._calculate_openness(hand_landmarks)

        # Map openness (0-1) to physical position (0-0.08m)
        target_position = gripper_openness * cfg.max_position

        # Clamp to physical limits
        target_position = np.clip(target_position, cfg.min_position, cfg.max_position)

        # Apply smoothing
        if self._prev_position is not None:
            alpha = cfg.smoothing_alpha
            smoothed = alpha * self._prev_position + (1 - alpha) * target_position

            # Apply rate limiting if time is available
            if current_time is not None and self._prev_time is not None:
                dt = current_time - self._prev_time
                if dt > 0:
                    max_delta = cfg.max_speed * dt
                    delta = smoothed - self._prev_position
                    if abs(delta) > max_delta:
                        smoothed = self._prev_position + np.sign(delta) * max_delta

            target_position = smoothed

        # Update state
        self._prev_position = target_position
        self._prev_time = current_time

        return GripperState(
            position=float(target_position),
            raw_openness=gripper_openness,
        )

    def reset(self):
        """Reset smoothing state."""
        self._prev_position = None
        self._prev_time = None

    @property
    def current_position(self) -> float:
        """Get the current gripper position.

        Returns:
            Current gripper position in meters, or 0.0 if not yet updated.
        """
        return self._prev_position if self._prev_position is not None else 0.0

    def _calculate_openness(self, hand_landmarks: List[Any]) -> float:
        """Calculate gripper openness from hand landmarks.

        Uses finger extension and thumb-index pinch distance.
        """
        # Get landmark positions
        wrist = hand_landmarks[self.HAND_WRIST]
        thumb_tip = hand_landmarks[self.HAND_THUMB_TIP]
        index_tip = hand_landmarks[self.HAND_INDEX_TIP]
        middle_tip = hand_landmarks[self.HAND_MIDDLE_TIP]
        ring_tip = hand_landmarks[self.HAND_RING_TIP]
        pinky_tip = hand_landmarks[self.HAND_PINKY_TIP]

        index_mcp = hand_landmarks[self.HAND_INDEX_MCP]
        middle_mcp = hand_landmarks[self.HAND_MIDDLE_MCP]
        ring_mcp = hand_landmarks[self.HAND_RING_MCP]
        pinky_mcp = hand_landmarks[self.HAND_PINKY_MCP]

        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])

        def finger_extension(tip, mcp) -> float:
            """Calculate finger extension ratio (tip dist / mcp dist from wrist)."""
            tip_pos = np.array([tip.x, tip.y, tip.z])
            mcp_pos = np.array([mcp.x, mcp.y, mcp.z])
            tip_dist = np.linalg.norm(tip_pos - wrist_pos)
            mcp_dist = np.linalg.norm(mcp_pos - wrist_pos)
            if mcp_dist < 1e-6:
                return 1.0
            return float(tip_dist / mcp_dist)

        # Average extension of 4 fingers (excluding thumb)
        extensions = [
            finger_extension(index_tip, index_mcp),
            finger_extension(middle_tip, middle_mcp),
            finger_extension(ring_tip, ring_mcp),
            finger_extension(pinky_tip, pinky_mcp),
        ]
        avg_extension = np.mean(extensions)

        # Thumb-index pinch distance
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
