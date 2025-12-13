"""Inverse kinematics solver for AgileX Piper 6-DOF arm."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# IKPy import - will be added as dependency
try:
    import ikpy.chain
    import ikpy.link
    IKPY_AVAILABLE = True
except ImportError:
    IKPY_AVAILABLE = False


@dataclass
class JointAngles:
    """Joint angles for the Piper arm."""

    # Joint angles in radians (6 joints)
    angles: np.ndarray  # shape (6,)

    # Whether the IK solution is valid
    is_valid: bool

    # IK solver residual error (lower is better)
    error: float


class PiperIK:
    """Inverse kinematics solver for AgileX Piper arm.

    Uses Modified DH parameters from firmware >= S-V1.6-3:
    Joint | alpha    | a        | d       | theta_offset
    ------|----------|----------|---------|-------------
    1     | 0        | 0        | 0.123   | 0
    2     | -π/2     | 0        | 0       | -172.22°
    3     | 0        | 0.28503  | 0       | -102.78°
    4     | π/2      | -0.021984| 0.25075 | 0
    5     | -π/2     | 0        | 0       | 0
    6     | π/2      | 0        | 0.091   | 0
    """

    # DH parameters for Piper (firmware >= S-V1.6-3)
    # Format: (alpha, a, d, theta_offset)
    DH_PARAMS = [
        (0.0, 0.0, 0.123, 0.0),
        (-np.pi / 2, 0.0, 0.0, np.radians(-172.22)),
        (0.0, 0.28503, 0.0, np.radians(-102.78)),
        (np.pi / 2, -0.021984, 0.25075, 0.0),
        (-np.pi / 2, 0.0, 0.0, 0.0),
        (np.pi / 2, 0.0, 0.091, 0.0),
    ]

    # Joint limits (radians) - approximate, check actual robot specs
    JOINT_LIMITS = [
        (-2.618, 2.618),   # Joint 1: ±150°
        (-1.571, 1.571),   # Joint 2: ±90°
        (-1.571, 1.571),   # Joint 3: ±90°
        (-2.618, 2.618),   # Joint 4: ±150°
        (-1.571, 1.571),   # Joint 5: ±90°
        (-2.618, 2.618),   # Joint 6: ±150°
    ]

    def __init__(self):
        """Initialize IK solver."""
        if not IKPY_AVAILABLE:
            raise ImportError(
                "ikpy is required for inverse kinematics. "
                "Install with: uv add ikpy"
            )

        # Build kinematic chain using IKPy
        self._chain = self._build_chain()

        # Store home position (all zeros after offset)
        self._home_angles = np.zeros(6)

    def _build_chain(self) -> "ikpy.chain.Chain":
        """Build IKPy kinematic chain from DH parameters."""
        links = []

        # Base link (fixed)
        links.append(ikpy.link.OriginLink())

        # Add each joint
        for i, (alpha, a, d, theta_offset) in enumerate(self.DH_PARAMS):
            bounds = self.JOINT_LIMITS[i]
            links.append(
                ikpy.link.URDFLink(
                    name=f"joint_{i + 1}",
                    origin_translation=[a, 0, d],
                    origin_orientation=[alpha, 0, 0],
                    rotation=[0, 0, 1],  # Rotation around Z axis
                    bounds=bounds,
                )
            )

        # End effector (fixed)
        links.append(
            ikpy.link.URDFLink(
                name="end_effector",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=None,
            )
        )

        return ikpy.chain.Chain(
            name="piper",
            links=links,
            active_links_mask=[False] + [True] * 6 + [False],
        )

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_angles: Optional[np.ndarray] = None,
    ) -> JointAngles:
        """Solve inverse kinematics for target end-effector pose.

        Args:
            target_position: Target position [x, y, z] in meters.
            target_orientation: Target orientation [roll, pitch, yaw] in radians.
                              If None, only position is considered.
            initial_angles: Initial guess for joint angles. Uses previous
                          solution or home position if None.

        Returns:
            JointAngles with solution or invalid result if no solution found.
        """
        # Build target transformation matrix
        target_matrix = np.eye(4)
        target_matrix[:3, 3] = target_position

        if target_orientation is not None:
            target_matrix[:3, :3] = self._euler_to_rotation(target_orientation)

        # Set initial guess
        if initial_angles is None:
            initial_angles = self._home_angles

        # Pad initial angles for IKPy (includes base and end effector)
        initial_full = np.zeros(8)
        initial_full[1:7] = initial_angles

        # Solve IK
        try:
            if target_orientation is not None:
                # Full pose IK
                result = self._chain.inverse_kinematics(
                    target_matrix,
                    initial_position=initial_full,
                    orientation_mode="all",
                )
            else:
                # Position-only IK
                result = self._chain.inverse_kinematics(
                    target_matrix,
                    initial_position=initial_full,
                    orientation_mode=None,
                )

            # Extract joint angles (skip base and end effector)
            angles = result[1:7]

            # Verify solution by forward kinematics
            fk_result = self._chain.forward_kinematics(result)
            position_error = np.linalg.norm(
                fk_result[:3, 3] - target_position
            )

            # Check if solution is within bounds
            in_bounds = all(
                self.JOINT_LIMITS[i][0] <= angles[i] <= self.JOINT_LIMITS[i][1]
                for i in range(6)
            )

            is_valid = position_error < 0.01 and in_bounds  # 1cm tolerance

            if is_valid:
                self._home_angles = angles.copy()

            return JointAngles(
                angles=angles,
                is_valid=is_valid,
                error=position_error,
            )

        except Exception:
            return JointAngles(
                angles=np.zeros(6),
                is_valid=False,
                error=float("inf"),
            )

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            angles: Joint angles in radians, shape (6,).

        Returns:
            End effector position [x, y, z] in meters.
        """
        full_angles = np.zeros(8)
        full_angles[1:7] = angles
        fk_result = self._chain.forward_kinematics(full_angles)
        return fk_result[:3, 3]

    def _euler_to_rotation(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix.

        Uses XYZ convention (roll around X, pitch around Y, yaw around Z).
        """
        roll, pitch, yaw = euler

        # Rotation matrices
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Combined rotation matrix (ZYX order)
        r = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

        return r
