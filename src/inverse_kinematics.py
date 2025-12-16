"""Inverse kinematics solver for AgileX Piper 6-DOF arm."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

import ikpy.chain
import ikpy.link


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

    Loads kinematic chain from URDF file (piper_description.urdf).
    """

    # Joint limits (radians) from robot specifications
    JOINT_LIMITS = [
        (-2.618, 2.618),   # Joint 1: [-150°, 150°]
        (0, 3.14),         # Joint 2: [0°, 180°]
        (-2.967, 0),       # Joint 3: [-170°, 0°]
        (-1.745, 1.745),   # Joint 4: [-100°, 100°]
        (-1.22, 1.22),     # Joint 5: [-70°, 70°]
        (-2.0944, 2.0944), # Joint 6: [-120°, 120°]
    ]

    def __init__(self, verbose: bool = False, urdf_path: str = "urdf/piper_description.urdf"):
        """Initialize IK solver.

        Args:
            verbose: If True, print debug info for IK solutions.
            urdf_path: Path to the URDF file describing the robot.
        """
        # Build kinematic chain from URDF file
        import os
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        # Load chain from URDF, excluding gripper joints (joint7, joint8)
        # The gripper has prismatic joints that don't affect end-effector pose
        # and should be controlled separately via GripperController
        self._chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[
                False,  # base_link (fixed)
                True,   # joint1 (revolute)
                True,   # joint2 (revolute)
                True,   # joint3 (revolute)
                True,   # joint4 (revolute)
                True,   # joint5 (revolute)
                True,   # joint6 (revolute)
                False,  # joint7 (prismatic - gripper, excluded)
                False,  # joint8 (prismatic - gripper, excluded)
            ]
        )

        # Store home position (all zeros after offset)
        self._home_angles = np.zeros(6)

        # Debug logging
        self.verbose = verbose
        self._frame_count = 0
        self._consecutive_failures = 0


    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_angles: Optional[np.ndarray] = None,
        max_retries: int = 3,
    ) -> JointAngles:
        """Solve inverse kinematics for target end-effector pose.

        Args:
            target_position: Target position [x, y, z] in meters.
            target_orientation: Target orientation [roll, pitch, yaw] in radians.
                              If None, only position is considered.
            initial_angles: Initial guess for joint angles. Uses previous
                          solution or home position if None.
            max_retries: Number of random restarts to try if initial solve fails.

        Returns:
            JointAngles with solution or invalid result if no solution found.
        """
        self._frame_count += 1

        # Build target transformation matrix
        target_matrix = np.eye(4)
        target_matrix[:3, 3] = target_position

        if target_orientation is not None:
            target_matrix[:3, :3] = self._euler_to_rotation(target_orientation)

        # Try with given initial guess first, then random restarts
        initial_guesses = []

        # First try: use provided initial angles or home
        if initial_angles is not None:
            initial_guesses.append(initial_angles.copy())
        initial_guesses.append(self._home_angles.copy())

        # Add random restarts within joint limits
        for _ in range(max_retries):
            random_angles = np.array([
                np.random.uniform(low, high)
                for low, high in self.JOINT_LIMITS
            ])
            initial_guesses.append(random_angles)

        best_angles = None
        best_error = float('inf')

        for init_angles in initial_guesses:
            # Pad initial angles for IKPy (chain has 9 links: base + 6 joints + 2 gripper links)
            initial_full = np.zeros(9)
            initial_full[1:7] = init_angles

            # Solve IK
            try:
                if target_orientation is not None:
                    # Full pose IK - use inverse_kinematics_frame for 4x4 matrix with orientation
                    result = self._chain.inverse_kinematics_frame(
                        target_matrix,
                        initial_position=initial_full,
                        orientation_mode="all",
                    )
                else:
                    # Position-only IK - just pass the position vector
                    result = self._chain.inverse_kinematics(
                        target_position,
                        initial_position=initial_full,
                    )

                # Extract joint angles and compute error
                angles = result[1:7]
                fk_result = self._chain.forward_kinematics(result)
                position_error = np.linalg.norm(fk_result[:3, 3] - target_position)

                if position_error < best_error:
                    best_error = position_error
                    best_angles = angles.copy()

                    # Early exit if good enough
                    if position_error < 0.005:
                        break

            except Exception:
                continue

        # Return failure if no solution found
        if best_angles is None:
            self._consecutive_failures += 1
            if self.verbose:
                print(f"[IK] Frame {self._frame_count}: ✗ No solution found")
                print(f"     Target pos: {target_position}, orient: {target_orientation}")
            return JointAngles(
                angles=np.zeros(6),
                is_valid=False,
                error=float("inf"),
            )

        # Use best result found
        angles = best_angles
        position_error = best_error

        # Check if solution is within bounds
        in_bounds = all(
            self.JOINT_LIMITS[i][0] <= angles[i] <= self.JOINT_LIMITS[i][1]
            for i in range(6)
        )

        is_valid = position_error < 0.025 and in_bounds  # 2.5cm tolerance

        if is_valid:
            self._home_angles = angles.copy()
            self._consecutive_failures = 0

            if self.verbose and self._frame_count % 30 == 0:
                angles_deg = np.degrees(angles)
                print(f"[IK] Frame {self._frame_count}: ✓ Valid | "
                      f"Error: {position_error:.4f}m | "
                      f"Angles: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°, "
                      f"{angles_deg[3]:.1f}°, {angles_deg[4]:.1f}°, {angles_deg[5]:.1f}°]")
        else:
            self._consecutive_failures += 1
            if self.verbose and self._frame_count % 30 == 0:
                angles_deg = np.degrees(angles)
                reason = "OOB" if not in_bounds else f"Error {position_error:.4f}m"
                print(f"[IK] Frame {self._frame_count}: ✗ Invalid ({reason}) | "
                      f"Angles: [{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°, {angles_deg[2]:.1f}°, "
                      f"{angles_deg[3]:.1f}°, {angles_deg[4]:.1f}°, {angles_deg[5]:.1f}°]")

        return JointAngles(
            angles=angles,
            is_valid=bool(is_valid),
            error=float(position_error),
        )

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            angles: Joint angles in radians, shape (6,).

        Returns:
            End effector position [x, y, z] in meters.
        """
        full_angles = np.zeros(9)  # Chain has 9 links
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
