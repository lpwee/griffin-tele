"""Workspace mapping from operator space to robot space."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .pose_estimation import ArmPose


@dataclass
class WorkspaceConfig:
    """Configuration for workspace mapping."""

    # Robot workspace bounds (meters)
    # Conservative bounds to stay within Piper's reachable workspace
    # Max reach ~0.5m, but corners of a box can exceed this
    robot_x_range: tuple[float, float] = (-0.15, 0.15)  # left/right (30cm total)
    robot_y_range: tuple[float, float] = (0.20, 0.35)   # forward (15cm depth)
    robot_z_range: tuple[float, float] = (0.15, 0.30)   # up/down (15cm height)

    # Operator workspace bounds (normalized 0-1 coordinates from camera)
    # These define the "active zone" in front of the camera
    operator_x_range: tuple[float, float] = (0.3, 0.7)   # left/right in frame
    operator_y_range: tuple[float, float] = (0.3, 0.8)   # top/bottom in frame
    operator_z_range: tuple[float, float] = (-0.3, 0.3)  # depth (from pose estimation)

    # Smoothing factor (0 = no smoothing, 1 = infinite smoothing)
    smoothing_alpha: float = 0.3

    # Dead zone threshold (movements smaller than this are ignored)
    dead_zone: float = 0.01


@dataclass
class RobotTarget:
    """Target pose for the robot end effector."""

    # Position in robot base frame (meters)
    position: np.ndarray  # shape (3,) - x, y, z

    # Orientation as roll, pitch, yaw (radians)
    orientation: np.ndarray  # shape (3,)

    # Gripper position (0 = closed, 0.08 = fully open, in meters)
    gripper: float

    # Whether this is a valid target
    is_valid: bool


class WorkspaceMapper:
    """Maps operator arm pose to robot target pose."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        """Initialize workspace mapper.

        Args:
            config: Workspace configuration. Uses defaults if None.
        """
        self.config = config or WorkspaceConfig()
        self._prev_position: Optional[np.ndarray] = None
        self._prev_orientation: Optional[np.ndarray] = None
        self._prev_gripper: Optional[float] = None

    def map_pose(self, arm_pose: ArmPose) -> RobotTarget:
        """Map operator arm pose to robot target.

        Args:
            arm_pose: Current arm pose from pose estimation.

        Returns:
            Robot target pose, or invalid target if pose is invalid.
        """
        if not arm_pose.is_valid:
            return RobotTarget(
                position=np.zeros(3),
                orientation=np.zeros(3),
                gripper=0.0,
                is_valid=False,
            )

        # Map position from operator space to robot space
        position = self._map_position(arm_pose.wrist_position)

        # Map orientation (direct mapping with possible offset)
        orientation = self._map_orientation(arm_pose.wrist_orientation)

        # Map gripper (0-1 to 0-0.08m)
        gripper = arm_pose.gripper_openness * 0.08

        # Apply smoothing
        position = self._smooth(position, self._prev_position)
        orientation = self._smooth_angle(orientation, self._prev_orientation)
        gripper = self._smooth_scalar(gripper, self._prev_gripper)

        # Update previous values
        self._prev_position = position.copy()
        self._prev_orientation = orientation.copy()
        self._prev_gripper = gripper

        return RobotTarget(
            position=position,
            orientation=orientation,
            gripper=gripper,
            is_valid=True,
        )

    def _map_position(self, operator_pos: np.ndarray) -> np.ndarray:
        """Map operator position to robot workspace.

        Note: Camera coordinates are:
        - x: left (0) to right (1)
        - y: top (0) to bottom (1)
        - z: depth (negative = closer)

        Robot coordinates (standard):
        - x: right (+) / left (-)
        - y: forward (+)
        - z: up (+)
        """
        cfg = self.config

        # Normalize operator position to 0-1 range within active zone
        op_x = self._normalize(operator_pos[0], cfg.operator_x_range)
        op_y = self._normalize(operator_pos[1], cfg.operator_y_range)
        op_z = self._normalize(operator_pos[2], cfg.operator_z_range)

        # Map to robot workspace
        # Flip x because camera left = robot right when facing operator
        robot_x = self._denormalize(1.0 - op_x, cfg.robot_x_range)
        # y (vertical in camera) maps to z (up/down in robot)
        robot_z = self._denormalize(1.0 - op_y, cfg.robot_z_range)
        # z (depth in camera) maps to y (forward/back in robot)
        robot_y = self._denormalize(op_z, cfg.robot_y_range)

        return np.array([robot_x, robot_y, robot_z])

    def _map_orientation(self, operator_orient: np.ndarray) -> np.ndarray:
        """Map operator wrist orientation to robot end-effector orientation.

        For now, apply a simple mapping. May need calibration offsets.
        """
        # Mirror roll for left/right symmetry
        roll = -operator_orient[0]
        pitch = operator_orient[1]
        yaw = -operator_orient[2]  # Mirror yaw as well

        return np.array([roll, pitch, yaw])

    def _normalize(self, value: float, range_: tuple[float, float]) -> float:
        """Normalize value from range to 0-1, clamped."""
        min_val, max_val = range_
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))

    def _denormalize(self, value: float, range_: tuple[float, float]) -> float:
        """Map 0-1 value to range."""
        min_val, max_val = range_
        return min_val + value * (max_val - min_val)

    def _smooth(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply exponential smoothing to position."""
        if previous is None:
            return current

        alpha = self.config.smoothing_alpha
        smoothed = alpha * previous + (1 - alpha) * current

        # Apply dead zone
        delta = np.linalg.norm(smoothed - previous)
        if delta < self.config.dead_zone:
            return previous

        return smoothed

    def _smooth_angle(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply smoothing to angles, handling wraparound."""
        if previous is None:
            return current

        alpha = self.config.smoothing_alpha

        # Handle angle wraparound for each component
        smoothed = np.zeros(3)
        for i in range(3):
            diff = current[i] - previous[i]
            # Wrap to [-pi, pi]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            smoothed[i] = previous[i] + (1 - alpha) * diff

        return smoothed

    def _smooth_scalar(
        self,
        current: float,
        previous: Optional[float],
    ) -> float:
        """Apply exponential smoothing to scalar."""
        if previous is None:
            return current

        alpha = self.config.smoothing_alpha
        return alpha * previous + (1 - alpha) * current

    def reset(self):
        """Reset smoothing state."""
        self._prev_position = None
        self._prev_orientation = None
        self._prev_gripper = None
