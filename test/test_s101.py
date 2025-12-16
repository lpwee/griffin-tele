#!/usr/bin/env python3
"""Test script for SO-101 arm interface.

This script tests the S101 interface in mock mode and optionally with real hardware.

Usage:
    # Test with mock robot (no hardware needed)
    python test/test_s101.py

    # Test with real hardware
    python test/test_s101.py --real --port /dev/ttyACM0
"""

import argparse
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s101_interface import S101Robot, MockS101Robot, S101Config, create_s101_robot
from src.s101_kinematics import S101IK, create_s101_ik


def test_mock_robot():
    """Test the mock S101 robot."""
    print("\n" + "=" * 60)
    print("Testing MockS101Robot")
    print("=" * 60)

    robot = create_s101_robot(use_mock=True)

    # Test connection
    print("\n1. Testing connection...")
    assert robot.connect(), "Connection failed"
    print("   ✓ Connected")

    # Test enable
    print("\n2. Testing enable...")
    assert robot.enable(), "Enable failed"
    print("   ✓ Motors enabled")

    # Test get_state
    print("\n3. Testing get_state...")
    state = robot.get_state()
    print(f"   Joint positions: {np.degrees(state.joint_positions).round(1)}")
    print(f"   Gripper: {state.gripper_position:.3f}m")
    print(f"   Enabled: {state.is_enabled}")
    print("   ✓ State retrieved")

    # Test set_joint_positions
    print("\n4. Testing set_joint_positions...")
    target = np.array([0.5, -0.3, 0.2, -0.1, 0.4])  # radians
    assert robot.set_joint_positions(target, gripper=0.05), "Set positions failed"
    print(f"   Target: {np.degrees(target).round(1)} deg")

    # Wait for simulated movement
    time.sleep(0.5)
    state = robot.get_state()
    print(f"   Current: {np.degrees(state.joint_positions).round(1)} deg")
    print("   ✓ Positions commanded")

    # Test emergency stop
    print("\n5. Testing emergency_stop...")
    robot.emergency_stop()
    state = robot.get_state()
    assert not state.is_enabled, "Robot should be disabled after e-stop"
    print("   ✓ Emergency stop worked")

    # Test disconnect
    print("\n6. Testing disconnect...")
    robot.disconnect()
    print("   ✓ Disconnected")

    print("\n✓ All MockS101Robot tests passed!")


def test_ik_solver():
    """Test the S101 IK solver."""
    print("\n" + "=" * 60)
    print("Testing S101IK")
    print("=" * 60)

    ik = create_s101_ik(urdf_path="urdf/s101.urdf", verbose=True)

    # Test forward kinematics at home position
    print("\n1. Testing forward kinematics at home...")
    home_angles = np.zeros(5)
    home_pos = ik.forward_kinematics(home_angles)
    print(f"   Home position: {home_pos.round(4)}")
    print("   ✓ FK computed")

    # Test IK for reachable position
    print("\n2. Testing IK for reachable target...")
    target_pos = np.array([0.15, 0.0, 0.20])  # 15cm forward, 20cm up
    result = ik.solve(target_pos)
    print(f"   Target: {target_pos}")
    print(f"   Valid: {result.is_valid}")
    print(f"   Error: {result.error:.4f}m")
    print(f"   Angles: {np.degrees(result.angles).round(1)} deg")

    if result.is_valid:
        # Verify with FK
        achieved_pos = ik.forward_kinematics(result.angles)
        print(f"   Achieved: {achieved_pos.round(4)}")
        print("   ✓ IK solution found")
    else:
        print("   ⚠ No valid IK solution (may be out of workspace)")

    # Test IK with orientation
    print("\n3. Testing IK with orientation...")
    target_orient = np.array([0, np.pi/4, 0])  # 45° pitch
    result = ik.solve(target_pos, target_orientation=target_orient)
    print(f"   Target orient: {np.degrees(target_orient).round(1)} deg")
    print(f"   Valid: {result.is_valid}")
    print(f"   Error: {result.error:.4f}m")
    if result.is_valid:
        print("   ✓ IK with orientation found")

    # Test workspace bounds
    print("\n4. Testing workspace bounds...")
    bounds = ik.get_workspace_bounds()
    print(f"   X range: {bounds['x_range']}")
    print(f"   Y range: {bounds['y_range']}")
    print(f"   Z range: {bounds['z_range']}")
    print("   ✓ Workspace bounds retrieved")

    print("\n✓ All S101IK tests passed!")


def test_real_robot(port: str):
    """Test with real S101 hardware."""
    print("\n" + "=" * 60)
    print(f"Testing Real S101 Robot on {port}")
    print("=" * 60)
    print("\n⚠ WARNING: This will move the real robot!")
    print("  Ensure the robot is in a safe position and press Enter to continue...")
    input()

    config = S101Config(port=port)
    robot = S101Robot(config)

    try:
        # Connect
        print("\n1. Connecting...")
        if not robot.connect():
            print("   ✗ Connection failed - check port and power")
            return False

        # Read current state (without enabling)
        print("\n2. Reading current state...")
        state = robot.get_state()
        print(f"   Joint positions: {np.degrees(state.joint_positions).round(1)} deg")
        print(f"   Gripper: {state.gripper_position:.3f}m")

        # Ask before enabling
        print("\n3. Enable motors? This will hold current position.")
        print("   Press Enter to enable, or Ctrl+C to abort...")
        input()

        if not robot.enable():
            print("   ✗ Enable failed")
            return False

        print("   ✓ Motors enabled")

        # Small movement test
        print("\n4. Testing small movement...")
        print("   Moving joint 0 (shoulder_pan) by 10 degrees...")
        current = state.joint_positions.copy()
        target = current.copy()
        target[0] += np.radians(10)

        robot.set_joint_positions(target)
        time.sleep(1.0)

        new_state = robot.get_state()
        print(f"   New position: {np.degrees(new_state.joint_positions).round(1)} deg")

        # Return to original
        print("\n5. Returning to original position...")
        robot.set_joint_positions(current)
        time.sleep(1.0)

        print("\n6. Disabling motors...")
        robot.disable()

        print("\n✓ Real robot test completed!")
        return True

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        robot.disable()
        return False
    finally:
        robot.disconnect()


def test_integration():
    """Test integrated teleoperation flow with mock robot."""
    print("\n" + "=" * 60)
    print("Testing Integration (Mock Robot + IK)")
    print("=" * 60)

    # Create components
    robot = create_s101_robot(use_mock=True)
    ik = create_s101_ik(verbose=False)

    # Connect and enable
    robot.connect()
    robot.enable()

    print("\n1. Simulating teleoperation loop...")
    targets = [
        np.array([0.15, 0.0, 0.15]),
        np.array([0.15, 0.1, 0.20]),
        np.array([0.20, -0.1, 0.15]),
        np.array([0.15, 0.0, 0.25]),
    ]

    for i, target in enumerate(targets):
        print(f"\n   Target {i+1}: {target}")

        # Solve IK
        result = ik.solve(target)
        if result.is_valid:
            # Command robot
            robot.set_joint_positions(result.angles, gripper=0.04)
            time.sleep(0.1)

            # Read state
            state = robot.get_state()
            print(f"   Joints: {np.degrees(state.joint_positions).round(1)} deg")
            print(f"   ✓ Valid (error: {result.error:.4f}m)")
        else:
            print(f"   ✗ No valid IK solution")

    robot.disable()
    robot.disconnect()

    print("\n✓ Integration test completed!")


def main():
    parser = argparse.ArgumentParser(description="Test S101 arm interface")
    parser.add_argument("--real", action="store_true", help="Test with real hardware")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for real robot")
    parser.add_argument("--ik-only", action="store_true", help="Only test IK solver")
    args = parser.parse_args()

    print("=" * 60)
    print("S101 (LeRobot/SO-101) Interface Test Suite")
    print("=" * 60)

    try:
        if args.ik_only:
            test_ik_solver()
        elif args.real:
            test_real_robot(args.port)
        else:
            test_mock_robot()
            test_ik_solver()
            test_integration()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
