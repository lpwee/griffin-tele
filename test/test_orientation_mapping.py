"""Diagnostic tool for orientation mapping between camera and robot frames.

This script helps visualize and debug the coordinate frame transformation
between MediaPipe camera coordinates and the Piper robot URDF coordinates.

Camera/MediaPipe frame (normalized):
  - X: left (-) to right (+) in image
  - Y: top (-) to bottom (+) in image
  - Z: depth, closer to camera is negative

Robot URDF frame (from joint analysis):
  - Joint 1 (base): rotates around Z (vertical), controls yaw
  - Joint 4: rotates around Z in its frame, controls wrist roll
  - Joint 5: rotates around Z in its frame, controls wrist pitch
  - Joint 6: rotates around Z in its frame, controls gripper roll

End-effector orientation at home position (all joints = 0):
  - Points forward (+Y in world frame after FK)
  - Gripper opens along X axis
"""

import numpy as np
from src.inverse_kinematics import RobotIK

# IKPy is always available (it's a dependency)
IKPY_AVAILABLE = True


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (XYZ convention) to rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # ZYX rotation order (yaw, pitch, roll applied in that order)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract Euler angles (roll, pitch, yaw) from rotation matrix."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0

    return roll, pitch, yaw


def analyze_robot_home_orientation():
    """Analyze the robot's end-effector orientation at home position."""
    if not IKPY_AVAILABLE:
        print("IKPy not available, skipping robot analysis")
        return

    ik = RobotIK(urdf_path="urdf/piper_description.urdf")

    print("=" * 60)
    print("ROBOT END-EFFECTOR ORIENTATION ANALYSIS")
    print("=" * 60)

    # Home position (all zeros)
    home_angles = np.zeros(6)
    full_angles = np.zeros(9)
    full_angles[1:7] = home_angles

    fk_result = ik._chain.forward_kinematics(full_angles)

    print("\n1. Home position (all joints = 0):")
    print(f"   Position: {fk_result[:3, 3]}")
    print(f"   Rotation matrix:\n{fk_result[:3, :3]}")

    # Extract orientation
    R = fk_result[:3, :3]
    roll, pitch, yaw = rotation_matrix_to_euler(R)
    print(f"   Euler (RPY): roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")

    # Test orientations
    print("\n2. Testing joint effects on end-effector orientation:")

    test_cases = [
        ("Joint 1 = +45° (base rotation)", [np.radians(45), 0, 0, 0, 0, 0]),
        ("Joint 1 = -45° (base rotation)", [np.radians(-45), 0, 0, 0, 0, 0]),
        ("Joint 4 = +45° (wrist roll)", [0, 0, 0, np.radians(45), 0, 0]),
        ("Joint 5 = +45° (wrist pitch)", [0, 0, 0, 0, np.radians(45), 0]),
        ("Joint 6 = +45° (gripper roll)", [0, 0, 0, 0, 0, np.radians(45)]),
    ]

    for name, angles in test_cases:
        full = np.zeros(9)
        full[1:7] = angles
        fk = ik._chain.forward_kinematics(full)
        R = fk[:3, :3]
        r, p, y = rotation_matrix_to_euler(R)
        print(f"\n   {name}:")
        print(f"      Position: [{fk[0,3]:.3f}, {fk[1,3]:.3f}, {fk[2,3]:.3f}]")
        print(f"      Euler: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")


def analyze_camera_to_robot_mapping():
    """Analyze the mapping between camera and robot orientation frames."""
    print("\n" + "=" * 60)
    print("CAMERA TO ROBOT ORIENTATION MAPPING")
    print("=" * 60)

    print("""
Camera frame (MediaPipe, looking at user):
  - User moves hand LEFT  → camera X decreases
  - User moves hand UP    → camera Y decreases
  - User moves hand FORWARD (away from camera) → camera Z increases

  Hand orientation from forearm:
  - Roll: rotation around forearm axis (wrist twist)
  - Pitch: forearm tilting up/down
  - Yaw: forearm pointing left/right

Robot frame (Piper, facing forward):
  - Robot reaches LEFT    → robot Y increases (or base rotates)
  - Robot reaches UP      → robot Z increases
  - Robot reaches FORWARD → robot X increases

  End-effector orientation:
  - Roll: gripper rotation (joint 6)
  - Pitch: wrist up/down (joint 5)
  - Yaw: base rotation (joint 1) for large motions, wrist for small

Mapping (current implementation):
  Camera          →  Robot
  -------            ------
  X (left/right)  →  -Y (mirrored for intuitive control)
  Y (up/down)     →  -Z (inverted, camera Y is down)
  Z (depth)       →  X (forward reach, inverted)

  roll            →  -roll (mirrored)
  pitch           →  pitch
  yaw             →  -yaw (mirrored)
""")


def test_ik_with_orientation():
    """Test IK solver with various orientations."""
    if not IKPY_AVAILABLE:
        print("IKPy not available")
        return

    ik = RobotIK(urdf_path="urdf/piper_description.urdf")

    print("\n" + "=" * 60)
    print("IK SOLVER ORIENTATION TESTS")
    print("=" * 60)

    # Test position (middle of workspace)
    test_pos = np.array([0.25, 0.0, 0.25])

    print(f"\nTest position: {test_pos}")

    # Test 1: Position only
    print("\n1. Position-only IK:")
    result = ik.solve(test_pos, None)
    print(f"   Valid: {result.is_valid}, Error: {result.error:.4f}")
    if result.is_valid:
        print(f"   Angles (deg): {np.degrees(result.angles)}")

    # Test 2: Various orientations
    test_orientations = [
        ("Neutral (0, 0, 0)", np.array([0, 0, 0])),
        ("Roll +30°", np.array([np.radians(30), 0, 0])),
        ("Roll -30°", np.array([np.radians(-30), 0, 0])),
        ("Pitch +30° (tilt down)", np.array([0, np.radians(30), 0])),
        ("Pitch -30° (tilt up)", np.array([0, np.radians(-30), 0])),
        ("Yaw +30° (point left)", np.array([0, 0, np.radians(30)])),
        ("Yaw -30° (point right)", np.array([0, 0, np.radians(-30)])),
    ]

    print("\n2. Position + Orientation IK:")
    for name, orient in test_orientations:
        result = ik.solve(test_pos, orient)
        status = "✓" if result.is_valid else "✗"
        print(f"   {status} {name}: valid={result.is_valid}, error={result.error:.4f}")
        if result.is_valid:
            # Verify resulting orientation
            full = np.zeros(9)
            full[1:7] = result.angles
            fk = ik._chain.forward_kinematics(full)
            R = fk[:3, :3]
            r, p, y = rotation_matrix_to_euler(R)
            print(f"      Achieved: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")


def suggest_orientation_mapping():
    """Suggest the correct orientation mapping based on analysis."""
    print("\n" + "=" * 60)
    print("SUGGESTED ORIENTATION MAPPING")
    print("=" * 60)

    print("""
Based on frame analysis, the correct mapping should be:

1. Camera coordinates to Robot coordinates:
   - Camera roll (wrist twist) → Robot roll (joint 6 / gripper rotation)
   - Camera pitch (arm tilt)   → Robot pitch (affects joints 2,3,5)
   - Camera yaw (arm swing)    → Robot yaw (joint 1 for large, 4 for small)

2. Coordinate transform (accounting for frame differences):

   # Camera: X-right, Y-down, Z-into-screen
   # Robot: X-forward, Y-left, Z-up

   # For a front-facing camera (selfie mode), mirroring is needed:
   robot_roll  = -camera_roll   # Mirror for intuitive control
   robot_pitch = -camera_pitch  # Invert: camera Y-down, robot Z-up
   robot_yaw   = -camera_yaw    # Mirror for intuitive control

3. Additional considerations:
   - Add configurable offsets for calibration
   - Consider limiting orientation range to avoid singularities
   - The home orientation of the robot end-effector should be accounted for
""")


def test_workspace_mapper_orientation():
    """Test the updated WorkspaceMapper orientation mapping."""
    from src.workspace_mapping import WorkspaceMapper, WorkspaceConfig

    print("\n" + "=" * 60)
    print("WORKSPACE MAPPER ORIENTATION TEST")
    print("=" * 60)

    mapper = WorkspaceMapper(WorkspaceConfig())
    cfg = mapper.config

    print(f"\nOrientation enabled: {cfg.orientation_enabled}")
    print(f"Orientation offset: {np.degrees(cfg.orientation_offset)} degrees")
    print(f"Orientation scale: {cfg.orientation_scale}")

    # Test mapping of neutral operator orientation
    test_orientations = [
        ("Neutral (0, 0, 0)", np.array([0, 0, 0])),
        ("Roll +30°", np.array([np.radians(30), 0, 0])),
        ("Roll -30°", np.array([np.radians(-30), 0, 0])),
        ("Pitch +30°", np.array([0, np.radians(30), 0])),
        ("Pitch -30°", np.array([0, np.radians(-30), 0])),
        ("Yaw +30°", np.array([0, 0, np.radians(30)])),
        ("Yaw -30°", np.array([0, 0, np.radians(-30)])),
    ]

    print("\nOperator orientation → Robot orientation mapping:")
    for name, op_orient in test_orientations:
        robot_orient = mapper._map_orientation(op_orient)
        print(f"  {name}:")
        print(f"    Operator: roll={np.degrees(op_orient[0]):.1f}°, pitch={np.degrees(op_orient[1]):.1f}°, yaw={np.degrees(op_orient[2]):.1f}°")
        print(f"    Robot:    roll={np.degrees(robot_orient[0]):.1f}°, pitch={np.degrees(robot_orient[1]):.1f}°, yaw={np.degrees(robot_orient[2]):.1f}°")


def test_ik_with_mapped_orientation():
    """Test IK solver with the new mapped orientations."""
    if not IKPY_AVAILABLE:
        print("IKPy not available")
        return

    from src.workspace_mapping import WorkspaceMapper, WorkspaceConfig

    ik = RobotIK(urdf_path="urdf/piper_description.urdf")
    mapper = WorkspaceMapper(WorkspaceConfig())

    print("\n" + "=" * 60)
    print("IK WITH MAPPED ORIENTATION TEST")
    print("=" * 60)

    test_pos = np.array([0.25, 0.0, 0.25])
    print(f"\nTest position: {test_pos}")

    test_orientations = [
        ("Neutral (0, 0, 0)", np.array([0, 0, 0])),
        ("Roll +30°", np.array([np.radians(30), 0, 0])),
        ("Roll -30°", np.array([np.radians(-30), 0, 0])),
        ("Pitch +30°", np.array([0, np.radians(30), 0])),
        ("Pitch -30°", np.array([0, np.radians(-30), 0])),
        ("Yaw +30°", np.array([0, 0, np.radians(30)])),
        ("Yaw -30°", np.array([0, 0, np.radians(-30)])),
    ]

    print("\nIK results with mapped orientations:")
    for name, op_orient in test_orientations:
        robot_orient = mapper._map_orientation(op_orient)
        result = ik.solve(test_pos, robot_orient)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {name}: valid={result.is_valid}, error={result.error:.4f}")
        if result.is_valid:
            print(f"      Angles (deg): [{', '.join(f'{np.degrees(a):.1f}' for a in result.angles)}]")


if __name__ == "__main__":
    analyze_robot_home_orientation()
    analyze_camera_to_robot_mapping()
    test_ik_with_orientation()
    suggest_orientation_mapping()
    test_workspace_mapper_orientation()
    test_ik_with_mapped_orientation()
