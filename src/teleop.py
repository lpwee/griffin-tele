"""Main teleoperation loop."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .pose_estimation import PoseEstimator, ArmPose
from .workspace_mapping import WorkspaceMapper, WorkspaceConfig, RobotTarget
from .inverse_kinematics import PiperIK, JointAngles, IKPY_AVAILABLE
from .robot_interface import RobotInterface, create_robot, RobotState


class TeleoperationController:
    """Main controller for camera-based teleoperation."""

    def __init__(
        self,
        robot: RobotInterface,
        pose_estimator: PoseEstimator,
        workspace_mapper: WorkspaceMapper,
        ik_solver: Optional[PiperIK] = None,
        target_fps: float = 30.0,
        show_arm_viz: bool = False,
        output_file: Optional[str] = None,
    ):
        """Initialize teleoperation controller.

        Args:
            robot: Robot interface (mock or real).
            pose_estimator: MediaPipe pose estimator.
            workspace_mapper: Workspace mapping configuration.
            ik_solver: Inverse kinematics solver. If None, IK is disabled.
            target_fps: Target control loop frequency.
            show_arm_viz: If True, show 3D arm visualization (only for mock mode).
            output_file: Path to CSV file for recording joint angles.
        """
        self.robot = robot
        self.pose_estimator = pose_estimator
        self.workspace_mapper = workspace_mapper
        self.ik_solver = ik_solver
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.show_arm_viz = show_arm_viz
        self.output_file = output_file

        # State
        self._running = False
        self._enabled = False  # Dead-man switch state
        self._last_valid_joints: Optional[np.ndarray] = None

        # Arm visualizer (lazy init)
        self._arm_viz = None

        # Output file handle
        self._output_handle = None

    def run(self, camera_id: int = 0, show_video: bool = True):
        """Run the teleoperation loop.

        Args:
            camera_id: Camera device ID.
            show_video: If True, display video with overlay.
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {frame_width}x{frame_height}")

        # Connect and enable robot
        if not self.robot.connect():
            print("Error: Could not connect to robot")
            cap.release()
            return

        if not self.robot.enable():
            print("Error: Could not enable robot")
            self.robot.disconnect()
            cap.release()
            return

        # Setup display window
        if show_video:
            cv2.namedWindow("Teleoperation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Teleoperation", frame_width, frame_height)

        # Setup arm visualizer for mock mode
        if self.show_arm_viz:
            from .arm_visualizer import PiperArmVisualizer
            self._arm_viz = PiperArmVisualizer(width=400, height=400)
            cv2.namedWindow("Piper Arm", cv2.WINDOW_NORMAL)
            print("3D arm visualization: enabled")

        # Setup output file for recording joint angles
        if self.output_file:
            self._output_handle = open(self.output_file, 'w')
            # Write CSV header
            self._output_handle.write("timestamp,joint1,joint2,joint3,joint4,joint5,joint6,gripper\n")
            print(f"Recording joint angles to: {self.output_file}")

        print("\n=== Teleoperation Started ===")
        print("Controls:")
        print("  - Make a FIST to enable robot control (dead-man switch)")
        print("  - Open hand to disable (robot holds position)")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset workspace mapping")
        print("  - Press 'e' for emergency stop")
        print()

        self._running = True
        start_time = time.time()
        frame_count = 0

        try:
            while self._running:
                loop_start = time.time()

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get timestamp
                timestamp_ms = int((time.time() - start_time) * 1000)

                # Process pose
                self.pose_estimator.process_frame(rgb_frame, timestamp_ms)
                arm_pose = self.pose_estimator.get_arm_pose()

                # Check dead-man switch (closed fist = enabled)
                # Gripper openness < 0.3 means fist is closed
                dead_man_active = arm_pose.is_valid and arm_pose.gripper_openness < 0.3

                if dead_man_active and not self._enabled:
                    print("[Teleop] Control ENABLED (fist detected)")
                    self._enabled = True
                elif not dead_man_active and self._enabled:
                    print("[Teleop] Control DISABLED (open hand)")
                    self._enabled = False

                # Process control
                robot_target: Optional[RobotTarget] = None
                joint_angles: Optional[JointAngles] = None

                # Always compute IK when pose is valid (for visualization)
                if arm_pose.is_valid:
                    # Map to robot workspace
                    robot_target = self.workspace_mapper.map_pose(arm_pose)

                    if robot_target.is_valid and self.ik_solver is not None:
                        # Solve IK
                        joint_angles = self.ik_solver.solve(
                            robot_target.position,
                            robot_target.orientation,
                            self._last_valid_joints,
                        )

                        if joint_angles.is_valid:
                            # Always update visualization joints
                            self._last_valid_joints = joint_angles.angles.copy()

                            # Only send to robot if control is enabled (dead-man switch)
                            if self._enabled:
                                self.robot.set_joint_positions(
                                    joint_angles.angles,
                                    robot_target.gripper,
                                )

                # Write joint angles to output file (whenever we have valid IK)
                if self._output_handle is not None and joint_angles is not None and joint_angles.is_valid:
                    t = time.time() - start_time
                    angles = joint_angles.angles
                    gripper = robot_target.gripper if robot_target else 0.0
                    self._output_handle.write(
                        f"{t:.4f},{angles[0]:.6f},{angles[1]:.6f},{angles[2]:.6f},"
                        f"{angles[3]:.6f},{angles[4]:.6f},{angles[5]:.6f},{gripper:.6f}\n"
                    )
                    self._output_handle.flush()

                # Display
                if show_video:
                    display_frame = self._draw_overlay(
                        frame, rgb_frame, arm_pose, robot_target, joint_angles
                    )
                    cv2.imshow("Teleoperation", display_frame)

                # Update arm visualization (mock mode only)
                if self._arm_viz is not None:
                    target_pos = robot_target.position if robot_target and robot_target.is_valid else None
                    gripper = robot_target.gripper / 0.08 if robot_target and robot_target.is_valid else 0.0
                    arm_img = self._arm_viz.update(
                        joint_angles=self._last_valid_joints,
                        target_position=target_pos,
                        gripper_openness=gripper,
                        is_valid=joint_angles.is_valid if joint_angles else False,
                    )
                    cv2.imshow("Piper Arm", arm_img)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("Resetting workspace mapping")
                    self.workspace_mapper.reset()
                elif key == ord('e'):
                    print("EMERGENCY STOP")
                    self.robot.emergency_stop()
                    self._enabled = False

                # Maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted")

        finally:
            # Cleanup
            self._running = False
            self.robot.disable()
            self.robot.disconnect()
            cap.release()
            if self._arm_viz is not None:
                self._arm_viz.close()
            if self._output_handle is not None:
                self._output_handle.close()
                print(f"Joint angles saved to: {self.output_file}")
            cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} FPS)")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        arm_pose: ArmPose,
        robot_target: Optional[RobotTarget],
        joint_angles: Optional[JointAngles],
    ) -> np.ndarray:
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]

        # Draw skeleton on RGB frame, then convert to BGR
        annotated_rgb = self.pose_estimator.draw_landmarks(rgb_frame)
        display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Status background
        cv2.rectangle(display, (10, 10), (350, 220), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (350, 220), (255, 255, 255), 1)

        # Status text
        y = 35
        line_height = 22

        # Model info
        cv2.putText(display, "Pose: MediaPipe Pose Landmarker (33 pts)",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += line_height

        # Hand model status
        if arm_pose.hand_tracked:
            cv2.putText(display, "Hand: MediaPipe Hand Landmarker (21 pts)",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Hand: Not detected (using pose fallback)",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        y += line_height

        # Tracking status
        if arm_pose.is_valid:
            cv2.putText(display, f"Tracking: OK (vis={arm_pose.visibility:.2f})",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Tracking: LOST",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += line_height

        # Dead-man switch status
        if self._enabled:
            cv2.putText(display, "Control: ENABLED (fist)",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Control: DISABLED",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y += line_height

        # Gripper openness
        if arm_pose.is_valid:
            cv2.putText(display, f"Hand: {'FIST' if arm_pose.gripper_openness < 0.3 else 'OPEN'} ({arm_pose.gripper_openness:.2f})",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height

        # Robot target
        if robot_target is not None and robot_target.is_valid:
            pos = robot_target.position
            cv2.putText(display, f"Target: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += line_height

        # IK status
        if joint_angles is not None:
            if joint_angles.is_valid:
                cv2.putText(display, f"IK: OK (err={joint_angles.error:.4f})",
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display, f"IK: FAILED (err={joint_angles.error:.4f})",
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += line_height

        # Draw wrist marker if tracking
        if arm_pose.is_valid:
            wrist_x = int(arm_pose.wrist_position[0] * w)
            wrist_y = int(arm_pose.wrist_position[1] * h)
            color = (0, 255, 0) if self._enabled else (128, 128, 128)
            cv2.circle(display, (wrist_x, wrist_y), 15, color, 3)
            cv2.circle(display, (wrist_x, wrist_y), 5, color, -1)

        return display


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera-based teleoperation for Piper arm")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--mock", action="store_true", help="Use mock robot (no hardware)")
    parser.add_argument("--can", type=str, default="can0", help="CAN interface for real robot")
    parser.add_argument("--pose-model", type=str, default="pose_landmarker.task",
                       help="Path to MediaPipe pose model")
    parser.add_argument("--hand-model", type=str, default="hand_landmarker.task",
                       help="Path to MediaPipe hand model (use 'none' to disable)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--no-video", action="store_true", help="Disable video display")
    parser.add_argument("--left-arm", action="store_true", help="Track left arm instead of right")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output CSV file for recording joint angles (auto-generated if not specified)")
    parser.add_argument("--no-record", action="store_true", help="Disable joint angle recording")
    args = parser.parse_args()

    # Create components
    print("Initializing...")

    # Generate output filename if recording is enabled
    output_file = args.output
    if not args.no_record and output_file is None:
        # Create outputs directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(outputs_dir / f"joints_{timestamp}.csv")
    elif args.no_record:
        output_file = None

    hand_model = args.hand_model if args.hand_model.lower() != "none" else None

    pose_estimator = PoseEstimator(
        pose_model_path=args.pose_model,
        hand_model_path=hand_model,
        use_right_arm=not args.left_arm,
    )

    print(f"Pose model: {args.pose_model}")
    print(f"Hand model: {hand_model or 'disabled'}")

    workspace_mapper = WorkspaceMapper(WorkspaceConfig())

    ik_solver: Optional[PiperIK] = None
    if IKPY_AVAILABLE:
        try:
            ik_solver = PiperIK()
            print("IK solver: enabled")
        except Exception as e:
            print(f"IK solver: disabled ({e})")
    else:
        print("IK solver: disabled (ikpy not installed)")

    robot = create_robot(use_mock=args.mock, can_name=args.can)

    controller = TeleoperationController(
        robot=robot,
        pose_estimator=pose_estimator,
        workspace_mapper=workspace_mapper,
        ik_solver=ik_solver,
        target_fps=args.fps,
        show_arm_viz=args.mock,  # Show 3D arm visualization in mock mode
        output_file=output_file,
    )

    # Run
    controller.run(camera_id=args.camera, show_video=not args.no_video)

    # Cleanup
    pose_estimator.close()


if __name__ == "__main__":
    main()
