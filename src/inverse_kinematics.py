"""Unified inverse kinematics solver for robot arms.

Automatically parses URDF files to extract joint configuration and limits.
Works with any robot arm URDF (Piper, S101, or custom).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import os
import xml.etree.ElementTree as ET

import ikpy.chain


@dataclass
class JointAngles:
    """Joint angles result from IK solver."""

    # Joint angles in radians (variable size based on robot)
    angles: np.ndarray

    # Whether the IK solution is valid
    is_valid: bool

    # IK solver residual error (lower is better)
    error: float


class RobotIK:
    """Unified inverse kinematics solver for robot arms.

    Automatically loads kinematic chain from URDF file and extracts:
    - Number of active joints
    - Joint limits from URDF
    - Active links mask

    Works with any robot arm URDF (Piper 6-DOF, S101 5-DOF, or custom).
    """

    def __init__(
        self,
        urdf_path: str,
        end_effector_link: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize IK solver from URDF.

        Args:
            urdf_path: Path to the URDF file describing the robot.
            end_effector_link: Name of the end-effector link. If None, will try to
                              auto-detect (looks for 'gripper_frame', 'link6', etc.)
            verbose: If True, print debug info for IK solutions.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        self.urdf_path = urdf_path
        self.verbose = verbose

        # Parse URDF to extract joint info
        self._joint_info = self._parse_urdf(urdf_path, end_effector_link)

        # Build kinematic chain
        self._chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=self._joint_info["active_mask"],
        )

        # Store configuration
        self._num_joints = self._joint_info["num_active_joints"]
        self._num_links = len(self._chain.links)
        self._joint_limits = self._joint_info["joint_limits"]
        self._joint_names = self._joint_info["joint_names"]

        # Home position (all joints at zero)
        self._home_angles = np.zeros(self._num_joints)

        # Last valid solution (for warm starting)
        self._last_solution = self._home_angles.copy()

        # Debug tracking
        self._frame_count = 0
        self._consecutive_failures = 0

        if verbose:
            print(f"[RobotIK] Loaded {urdf_path}")
            print(f"[RobotIK] Active joints: {self._num_joints}")
            print(f"[RobotIK] Joint names: {self._joint_names}")

    def _parse_urdf(
        self, urdf_path: str, end_effector_link: Optional[str]
    ) -> dict:
        """Parse URDF to extract joint configuration.

        Returns dict with:
        - num_active_joints: Number of active (non-fixed, non-gripper) joints
        - joint_limits: List of (lower, upper) tuples
        - joint_names: List of joint names
        - active_mask: Boolean mask for ikpy chain
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Build link-joint graph
        joints = []
        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            joint_type = joint.get("type")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")

            # Extract limits if present
            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower", 0))
                upper = float(limit_elem.get("upper", 0))
            else:
                lower, upper = -np.pi, np.pi

            joints.append({
                "name": joint_name,
                "type": joint_type,
                "parent": parent,
                "child": child,
                "limits": (lower, upper),
            })

        # Determine end-effector link
        if end_effector_link is None:
            # Auto-detect: look for common end-effector patterns
            link_names = [link.get("name") for link in root.findall("link")]
            # Priority order for end-effector detection
            ee_candidates = [
                "gripper_frame_link",  # S101
                "gripper_base",        # Piper
                "link6",               # Generic 6-DOF
                "link5",               # Generic 5-DOF
            ]
            for candidate in ee_candidates:
                if candidate in link_names:
                    end_effector_link = candidate
                    break

            if end_effector_link is None:
                # Fallback: use the last link in the chain
                end_effector_link = link_names[-1] if link_names else None

        # Find the kinematic chain from base to end-effector
        # We need to identify which joints are "arm" joints (revolute)
        # and which are gripper joints (to exclude)

        # Gripper-related joint patterns to exclude
        gripper_patterns = ["gripper", "jaw", "finger", "joint7", "joint8"]

        # Collect active joints (revolute joints not related to gripper)
        active_joints = []
        for j in joints:
            if j["type"] == "revolute":
                is_gripper = any(
                    pattern in j["name"].lower()
                    for pattern in gripper_patterns
                )
                if not is_gripper:
                    active_joints.append(j)

        # Sort by order in URDF (assumed to be kinematic chain order)
        # The chain goes: base -> joint1 -> link1 -> joint2 -> ... -> end_effector

        # Build active_links_mask for ikpy
        # ikpy chain includes: origin + links for each joint
        # We need to mark which links have active joints

        # Load the chain temporarily to get link order
        temp_chain = ikpy.chain.Chain.from_urdf_file(urdf_path)

        active_mask = []
        joint_limits = []
        joint_names = []
        active_joint_names = {j["name"] for j in active_joints}

        for i, link in enumerate(temp_chain.links):
            # Check if this link corresponds to an active joint
            if hasattr(link, 'name') and link.name in active_joint_names:
                active_mask.append(True)
                # Find the joint info
                for j in active_joints:
                    if j["name"] == link.name:
                        joint_limits.append(j["limits"])
                        joint_names.append(j["name"])
                        break
            else:
                active_mask.append(False)

        return {
            "num_active_joints": len(joint_names),
            "joint_limits": joint_limits,
            "joint_names": joint_names,
            "active_mask": active_mask,
            "end_effector_link": end_effector_link,
        }

    @property
    def num_joints(self) -> int:
        """Number of active joints in the arm."""
        return self._num_joints

    @property
    def joint_limits(self) -> list:
        """Joint limits as list of (lower, upper) tuples."""
        return self._joint_limits

    @property
    def joint_names(self) -> list:
        """Names of active joints."""
        return self._joint_names

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
            initial_angles: Initial guess for joint angles.
                          Uses previous solution or home if None.
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
                for low, high in self._joint_limits
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
                    print(f"[RobotIK] IK attempt failed: {e}")
                continue

        # Check if we found a valid solution
        if best_angles is None:
            self._consecutive_failures += 1
            if self.verbose:
                print(f"[RobotIK] Frame {self._frame_count}: ✗ No solution found")
            return JointAngles(
                angles=np.zeros(self._num_joints),
                is_valid=False,
                error=float("inf"),
            )

        # Verify solution is within joint limits
        in_bounds = all(
            self._joint_limits[i][0] <= best_angles[i] <= self._joint_limits[i][1]
            for i in range(self._num_joints)
        )

        is_valid = best_error < 0.025 and in_bounds  # 2.5cm tolerance

        if is_valid:
            self._last_solution = best_angles.copy()
            self._consecutive_failures = 0

            if self.verbose and self._frame_count % 30 == 0:
                angles_deg = np.degrees(best_angles)
                print(f"[RobotIK] Frame {self._frame_count}: ✓ Valid | "
                      f"Error: {best_error:.4f}m | "
                      f"Angles: {angles_deg.round(1)}")
        else:
            self._consecutive_failures += 1
            if self.verbose and self._frame_count % 30 == 0:
                reason = "OOB" if not in_bounds else f"Error {best_error:.4f}m"
                print(f"[RobotIK] Frame {self._frame_count}: ✗ Invalid ({reason})")

        return JointAngles(
            angles=best_angles,
            is_valid=bool(is_valid),
            error=float(best_error),
        )

    def forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics.

        Args:
            angles: Joint angles in radians.

        Returns:
            End effector position [x, y, z] in meters.
        """
        full_angles = self._pad_angles(angles)
        fk_result = self._chain.forward_kinematics(full_angles)
        return fk_result[:3, 3]

    def get_end_effector_pose(self, angles: np.ndarray) -> tuple:
        """Get full end-effector pose (position and orientation).

        Args:
            angles: Joint angles in radians.

        Returns:
            Tuple of (position [x,y,z], rotation_matrix [3x3])
        """
        full_angles = self._pad_angles(angles)
        fk_result = self._chain.forward_kinematics(full_angles)
        return fk_result[:3, 3], fk_result[:3, :3]

    def _pad_angles(self, angles: np.ndarray) -> np.ndarray:
        """Pad joint angles for the full IKPy chain."""
        full_angles = np.zeros(self._num_links)
        # Find active joint indices
        active_idx = 0
        for i, is_active in enumerate(self._joint_info["active_mask"]):
            if is_active and active_idx < len(angles):
                full_angles[i] = angles[active_idx]
                active_idx += 1
        return full_angles

    def _extract_angles(self, full_angles: np.ndarray) -> np.ndarray:
        """Extract active joint angles from full chain angles."""
        angles = []
        for i, is_active in enumerate(self._joint_info["active_mask"]):
            if is_active:
                angles.append(full_angles[i])
        return np.array(angles)

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

        Samples forward kinematics at various joint configurations to
        estimate the reachable workspace.

        Returns:
            Dictionary with x_range, y_range, z_range tuples.
        """
        # Sample positions at home and various joint limits
        positions = []

        # Home position
        home = np.zeros(self._num_joints)
        positions.append(self.forward_kinematics(home))

        # Sample at joint limit combinations
        for i in range(self._num_joints):
            for limit_idx in [0, 1]:  # lower and upper
                angles = np.zeros(self._num_joints)
                angles[i] = self._joint_limits[i][limit_idx]
                try:
                    pos = self.forward_kinematics(angles)
                    positions.append(pos)
                except Exception:
                    pass

        positions = np.array(positions)

        return {
            "x_range": (float(positions[:, 0].min()), float(positions[:, 0].max())),
            "y_range": (float(positions[:, 1].min()), float(positions[:, 1].max())),
            "z_range": (float(positions[:, 2].min()), float(positions[:, 2].max())),
        }


def create_robot_ik(
    urdf_path: str,
    verbose: bool = False,
) -> RobotIK:
    """Factory function to create robot IK solver.

    Args:
        urdf_path: Path to the URDF file.
        verbose: Enable verbose logging.

    Returns:
        RobotIK solver instance.
    """
    return RobotIK(urdf_path=urdf_path, verbose=verbose)
