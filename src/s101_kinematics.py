"""Inverse kinematics solver for SO-101 6-DOF arm.

Uses IKPy library with the s101.urdf kinematic model.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import os

import ikpy.chain


@dataclass
class S101JointAngles:
    """Joint angles for the SO-101 arm."""

    # Joint angles in radians (5 joints for arm, gripper separate)
    angles: np.ndarray  # shape (5,) or (6,) with gripper

    # Whether the IK solution is valid
    is_valid: bool

    # IK solver residual error (lower is better)
    error: float


class S101IK:
    """Inverse kinematics solver for SO-101 arm.

    Loads kinematic chain from s101.urdf file.

    Joint order (from URDF):
    0: shoulder_pan  - base rotation
    1: shoulder_lift - shoulder pitch
    2: elbow_flex    - elbow pitch
    3: wrist_flex    - wrist pitch
    4: wrist_roll    - wrist roll
    5: gripper       - gripper (not used for IK)
    """

    # Joint limits (radians) from URDF
    JOINT_LIMITS = [
        (-1.91986, 1.91986),   # shoulder_pan: ~±110°
        (-1.74533, 1.74533),   # shoulder_lift: ~±100°
        (-1.69, 1.69),         # elbow_flex: ~±97°
        (-1.65806, 1.65806),   # wrist_flex: ~±95°
        (-2.74385, 2.84121),   # wrist_roll: ~-157° to +163°
    ]

    # Joint names for reference
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]

    def __init__(
        self,
        urdf_path: str = "urdf/s101.urdf",
        verbose: bool = False,
    ):
        """Initialize IK solver.

        Args:
            urdf_path: Path to the URDF file.
            verbose: If True, print debug info for IK solutions.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        # Load kinematic chain from URDF
        # The SO-101 chain structure (as loaded by IKPy):
        # 0: Base link (OriginLink) - fixed
        # 1: shoulder_pan (URDFLink) - revolute
        # 2: shoulder_lift (URDFLink) - revolute
        # 3: elbow_flex (URDFLink) - revolute
        # 4: wrist_flex (URDFLink) - revolute
        # 5: wrist_roll (URDFLink) - revolute
        # 6: gripper_frame_joint (URDFLink) - fixed (end effector frame)
        self._chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[
                False,  # Base link (fixed origin)
                True,   # shoulder_pan (revolute)
                True,   # shoulder_lift (revolute)
                True,   # elbow_flex (revolute)
                True,   # wrist_flex (revolute)
                True,   # wrist_roll (revolute)
                False,  # gripper_frame_joint (fixed)
            ]
        )

        # Number of links in the chain (for padding angles)
        self._num_links = len(self._chain.links)

        # Home position (all joints at zero)
        self._home_angles = np.zeros(5)

        # Last valid solution (for warm starting)
        self._last_solution = self._home_angles.copy()

        # Debug settings
        self.verbose = verbose
        self._frame_count = 0
        self._consecutive_failures = 0

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_angles: Optional[np.ndarray] = None,
        max_retries: int = 3,
    ) -> S101JointAngles:
        """Solve inverse kinematics for target end-effector pose.

        Args:
            target_position: Target position [x, y, z] in meters.
            target_orientation: Target orientation [roll, pitch, yaw] in radians.
                              If None, only position is considered.
            initial_angles: Initial guess for joint angles (5 values).
                          Uses previous solution or home if None.
            max_retries: Number of random restarts to try if initial solve fails.

        Returns:
            S101JointAngles with solution or invalid result if no solution found.
        """
        self._frame_count += 1

        # Build target transformation matrix
        target_matrix = np.eye(4)
        target_matrix[:3, 3] = target_position

        if target_orientation is not None:
            target_matrix[:3, :3] = self._euler_to_rotation(target_orientation)

        # Prepare initial guesses
        initial_guesses = []

        # First: provided initial angles
        if initial_angles is not None:
            initial_guesses.append(initial_angles.copy())

        # Second: last valid solution (warm start)
        initial_guesses.append(self._last_solution.copy())

        # Third: home position
        initial_guesses.append(self._home_angles.copy())

        # Additional: random restarts within joint limits
        for _ in range(max_retries):
            random_angles = np.array([
                np.random.uniform(low, high)
                for low, high in self.JOINT_LIMITS
            ])
            initial_guesses.append(random_angles)

        best_angles = None
        best_error = float('inf')

        for init_angles in initial_guesses:
            # Pad angles for IKPy chain
            initial_full = self._pad_angles(init_angles)

            try:
                if target_orientation is not None:
                    # Full pose IK
                    result = self._chain.inverse_kinematics_frame(
                        target_matrix,
                        initial_position=initial_full,
                        orientation_mode="all",
                    )
                else:
                    # Position-only IK
                    result = self._chain.inverse_kinematics(
                        target_position,
                        initial_position=initial_full,
                    )

                # Extract joint angles
                angles = self._extract_angles(result)

                # Compute forward kinematics to get error
                fk_result = self._chain.forward_kinematics(result)
                position_error = np.linalg.norm(fk_result[:3, 3] - target_position)

                if position_error < best_error:
                    best_error = position_error
                    best_angles = angles.copy()

                    # Early exit if good enough
                    if position_error < 0.005:  # 5mm tolerance
                        break

            except Exception as e:
                if self.verbose:
                    print(f"[S101IK] IK attempt failed: {e}")
                continue

        # Check if we found a valid solution
        if best_angles is None:
            self._consecutive_failures += 1
            if self.verbose:
                print(f"[S101IK] Frame {self._frame_count}: ✗ No solution found")
            return S101JointAngles(
                angles=np.zeros(5),
                is_valid=False,
                error=float("inf"),
            )

        # Verify solution is within joint limits
        in_bounds = all(
            self.JOINT_LIMITS[i][0] <= best_angles[i] <= self.JOINT_LIMITS[i][1]
            for i in range(5)
        )

        is_valid = best_error < 0.025 and in_bounds  # 2.5cm tolerance

        if is_valid:
            self._last_solution = best_angles.copy()
            self._consecutive_failures = 0

            if self.verbose and self._frame_count % 30 == 0:
                angles_deg = np.degrees(best_angles)
                print(f"[S101IK] Frame {self._frame_count}: ✓ Valid | "
                      f"Error: {best_error:.4f}m | "
                      f"Angles: {angles_deg.round(1)}")
        else:
            self._consecutive_failures += 1
            if self.verbose and self._frame_count % 30 == 0:
                reason = "OOB" if not in_bounds else f"Error {best_error:.4f}m"
                print(f"[S101IK] Frame {self._frame_count}: ✗ Invalid ({reason})")

        return S101JointAngles(
            angles=best_angles,
            is_valid=bool(is_valid),
            error=float(best_error),
        )

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            angles: Joint angles in radians, shape (5,).

        Returns:
            End effector position [x, y, z] in meters.
        """
        full_angles = self._pad_angles(angles)
        fk_result = self._chain.forward_kinematics(full_angles)
        return fk_result[:3, 3]

    def get_end_effector_pose(self, angles: np.ndarray) -> tuple:
        """Get full end-effector pose (position and orientation).

        Args:
            angles: Joint angles in radians, shape (5,).

        Returns:
            Tuple of (position [x,y,z], rotation_matrix [3x3])
        """
        full_angles = self._pad_angles(angles)
        fk_result = self._chain.forward_kinematics(full_angles)
        return fk_result[:3, 3], fk_result[:3, :3]

    def _pad_angles(self, angles: np.ndarray) -> np.ndarray:
        """Pad joint angles for the full IKPy chain (7 links)."""
        full_angles = np.zeros(self._num_links)  # 7 links
        # Place the 5 active joint angles in positions 1-5
        # (indices 1-5 correspond to shoulder_pan through wrist_roll)
        full_angles[1:6] = angles[:5]
        return full_angles

    def _extract_angles(self, full_angles: np.ndarray) -> np.ndarray:
        """Extract the 5 arm joint angles from full chain angles."""
        return full_angles[1:6].copy()

    def _euler_to_rotation(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix.

        Uses XYZ convention (roll around X, pitch around Y, yaw around Z).
        """
        roll, pitch, yaw = euler

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

    def get_workspace_bounds(self) -> dict:
        """Estimate workspace bounds based on arm geometry.

        Returns approximate reachable workspace in meters.
        """
        # Estimate from forward kinematics at various configurations
        # These are approximate values based on the SO-101 dimensions
        return {
            "x_range": (0.05, 0.30),   # Forward reach
            "y_range": (-0.25, 0.25),  # Left/right
            "z_range": (0.0, 0.35),    # Height
        }


def create_s101_ik(
    urdf_path: str = "urdf/s101.urdf",
    verbose: bool = False,
) -> S101IK:
    """Factory function to create S101 IK solver.

    Args:
        urdf_path: Path to the URDF file.
        verbose: Enable verbose logging.

    Returns:
        S101IK solver instance.
    """
    return S101IK(urdf_path=urdf_path, verbose=verbose)
