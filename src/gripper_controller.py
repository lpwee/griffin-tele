"""Gripper controller module for Piper robot arm.

Separates gripper control from the inverse kinematics chain,
since the gripper (prismatic joints 7 and 8) doesn't affect
end-effector pose and should be controlled independently.
"""

from dataclasses import dataclass
from typing import Optional
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
    """

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
        gripper_openness: float,
        current_time: Optional[float] = None,
    ) -> GripperState:
        """Update gripper state from pose estimation.

        Args:
            gripper_openness: Normalized openness from pose estimation (0.0 to 1.0).
                             0.0 = fully closed, 1.0 = fully open.
            current_time: Current timestamp in seconds. Used for rate limiting.
                         If None, rate limiting is disabled.

        Returns:
            GripperState with the computed gripper position.
        """
        cfg = self.config

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
