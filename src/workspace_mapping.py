"""Workspace mapping from operator space to robot space."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .pose_estimation import ArmPose


@dataclass
class WorkspaceConfig:
    """Configuration for workspace mapping."""

    # Robot workspace bounds (meters)
    # Based on IK reachability analysis and 626mm working radius spec
    # Robot X = forward reach, Y = left/right, Z = up/down
    robot_x_range: tuple[float, float] = (0.15, 0.40)   # forward reach
    robot_y_range: tuple[float, float] = (-0.25, 0.25)  # left/right
    robot_z_range: tuple[float, float] = (0.05, 0.35)   # up/down

    # Operator workspace bounds (wrist position relative to shoulder)
    # These are in normalized camera coordinates, with shoulder as origin
    # Estimated from typical arm proportions (~60cm arm, ~200cm camera view)
    operator_x_range: tuple[float, float] = (-0.25, 0.25)  # arm reach left/right of shoulder
    operator_y_range: tuple[float, float] = (-0.05, 0.30)  # wrist above(-) to below(+) shoulder
    operator_z_range: tuple[float, float] = (-0.15, 0.40)  # arm extension depth

    # Smoothing factor (0 = no smoothing, 1 = infinite smoothing)
    smoothing_alpha: float = 0.3

    # Dead zone threshold (movements smaller than this are ignored)
    dead_zone: float = 0.01

    # Orientation mapping configuration
    # The robot's home orientation (from FK analysis): roll=90°, pitch=85°, yaw=0°
    # We need to map operator orientation relative to this home orientation
    orientation_enabled: bool = True

    # Orientation offset (added to mapped orientation to match robot home frame)
    # These values calibrate the "neutral" operator pose to robot home
    orientation_offset: tuple[float, float, float] = (
        1.5708,  # roll offset: 90° (robot home roll)
        1.4835,  # pitch offset: 85° (robot home pitch)
        0.0,     # yaw offset: 0°
    )

    # Orientation scale factors (how much operator movement affects robot)
    # Values < 1 reduce sensitivity, > 1 increase it
    orientation_scale: tuple[float, float, float] = (
        0.5,  # roll scale (wrist twist)
        0.5,  # pitch scale (arm tilt)
        1.0,  # yaw scale (arm swing)
    )

    # Orientation limits (clamp mapped orientation to safe range, in radians)
    orientation_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-1.5, 1.5),   # roll limits (±86°)
        (-1.0, 1.5),   # pitch limits (-57° to +86°)
        (-1.5, 1.5),   # yaw limits (±86°)
    )


@dataclass
class RobotTarget:
    """Target pose for the robot end effector.

    Note: Gripper control is handled separately by GripperController.
    This class only contains the 6-DOF arm pose.
    """

    # Position in robot base frame (meters)
    position: np.ndarray  # shape (3,) - x, y, z

    # Orientation as roll, pitch, yaw (radians)
    orientation: np.ndarray  # shape (3,)

    # Whether this is a valid target
    is_valid: bool


class WorkspaceMapper:
    """Maps operator arm pose to robot target pose."""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        """Initialize workspace mapper.

        Args:
            config: Workspace configuration. Uses defaults if None.

        Note: Gripper control is now handled separately by GripperController.
        """
        self.config = config or WorkspaceConfig()
        self._prev_position: Optional[np.ndarray] = None
        self._prev_orientation: Optional[np.ndarray] = None

    def map_pose(self, arm_pose: ArmPose) -> RobotTarget:
        """Map operator arm pose to robot target.

        Args:
            arm_pose: Current arm pose from pose estimation.

        Returns:
            Robot target pose, or invalid target if pose is invalid.

        Note: Gripper control is handled separately by GripperController.
        """
        if not arm_pose.is_valid:
            return RobotTarget(
                position=np.zeros(3),
                orientation=np.zeros(3),
                is_valid=False,
            )

        # Map position from operator space to robot space
        position = self._map_position(arm_pose.wrist_position)

        # Map orientation (direct mapping with possible offset)
        orientation = self._map_orientation(arm_pose.wrist_orientation)

        # Apply smoothing
        position = self._smooth(position, self._prev_position)
        orientation = self._smooth_angle(orientation, self._prev_orientation)

        # Update previous values
        self._prev_position = position.copy()
        self._prev_orientation = orientation.copy()

        return RobotTarget(
            position=position,
            orientation=orientation,
            is_valid=True,
        )

    def _map_position(self, operator_pos: np.ndarray) -> np.ndarray:
        """Map operator position to robot workspace.

        Note: Operator position is wrist relative to shoulder in camera coords:
        - x: left (-) to right (+) of shoulder
        - y: above (-) to below (+) shoulder
        - z: depth relative to shoulder plane (negative = closer to camera)

        Robot coordinates (from FK analysis):
        - x: forward reach (arm extension)
        - y: left (-) / right (+)
        - z: up (+) / down (-)
        """
        cfg = self.config

        # Normalize operator position to 0-1 range within arm reach
        op_x = self._normalize(operator_pos[0], cfg.operator_x_range)
        op_y = self._normalize(operator_pos[1], cfg.operator_y_range)
        op_z = self._normalize(operator_pos[2], cfg.operator_z_range)

        # Map to robot workspace
        # Camera depth (z) → robot forward (x): closer to camera = less reach
        robot_x = self._denormalize(1.0 - op_z, cfg.robot_x_range)
        # Camera left/right (x) → robot left/right (y): flip for mirror effect
        robot_y = self._denormalize(1.0 - op_x, cfg.robot_y_range)
        # Camera up/down (y) → robot up/down (z): flip because camera y increases downward
        robot_z = self._denormalize(1.0 - op_y, cfg.robot_z_range)

        return np.array([robot_x, robot_y, robot_z])

    def _map_orientation(self, operator_orient: np.ndarray) -> np.ndarray:
        """Map operator wrist orientation to robot end-effector orientation.

        Coordinate frame transformation:
        - Camera/MediaPipe: X-right, Y-down, Z-into-screen
        - Robot (Piper): X-forward, Y-left, Z-up

        The robot's home orientation (all joints=0) has the end-effector at
        approximately roll=90°, pitch=85°, yaw=0° in world frame.

        We map operator orientation changes relative to a "neutral" pose
        to robot orientation changes relative to home.
        """
        cfg = self.config

        if not cfg.orientation_enabled:
            # Return robot home orientation when disabled
            return np.array(cfg.orientation_offset)

        # Extract operator angles
        op_roll = operator_orient[0]   # wrist twist
        op_pitch = operator_orient[1]  # forearm tilt (up/down)
        op_yaw = operator_orient[2]    # forearm swing (left/right)

        # Transform from camera frame to robot frame:
        # - Mirror roll for intuitive control (twist left = gripper rotates left)
        # - Negate pitch: camera Y-down means positive pitch is arm tilting down,
        #   but robot Z-up means we need to invert for intuitive mapping
        # - Mirror yaw for intuitive control (swing left = robot points left)
        robot_roll = -op_roll
        robot_pitch = -op_pitch
        robot_yaw = -op_yaw

        # Apply scale factors (reduce sensitivity for more controlled movement)
        robot_roll *= cfg.orientation_scale[0]
        robot_pitch *= cfg.orientation_scale[1]
        robot_yaw *= cfg.orientation_scale[2]

        # Apply orientation offset (robot home orientation)
        robot_roll += cfg.orientation_offset[0]
        robot_pitch += cfg.orientation_offset[1]
        robot_yaw += cfg.orientation_offset[2]

        # Clamp to safe limits
        robot_roll = np.clip(robot_roll, *cfg.orientation_limits[0])
        robot_pitch = np.clip(robot_pitch, *cfg.orientation_limits[1])
        robot_yaw = np.clip(robot_yaw, *cfg.orientation_limits[2])

        # Normalize angles to [-pi, pi] range
        result = np.array([robot_roll, robot_pitch, robot_yaw])
        result = np.mod(result + np.pi, 2 * np.pi) - np.pi

        return result

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
            # Wrap diff to [-pi, pi]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            smoothed[i] = previous[i] + (1 - alpha) * diff

        # Normalize output to [-pi, pi] to prevent drift
        smoothed = np.mod(smoothed + np.pi, 2 * np.pi) - np.pi

        return smoothed

    def reset(self):
        """Reset smoothing state."""
        self._prev_position = None
        self._prev_orientation = None
