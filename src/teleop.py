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
from .inverse_kinematics import PiperIK, JointAngles
from .robot_interface import RobotInterface, create_robot, RobotState
from .gripper_controller import GripperController, GripperConfig, GripperState


class TeleoperationController:
    """Main controller for camera-based teleoperation."""

    def __init__(
        self,
        robot: RobotInterface,
        pose_estimator: PoseEstimator,
        workspace_mapper: WorkspaceMapper,
        ik_solver: Optional[PiperIK] = None,
        gripper_controller: Optional[GripperController] = None,
        target_fps: float = 30.0,
        mock: bool = False,
        output_file: Optional[str] = None,
    ):
        """Initialize teleoperation controller.

        Args:
            robot: Robot interface (mock or real).
            pose_estimator: MediaPipe pose estimator.
            workspace_mapper: Workspace mapping configuration.
            ik_solver: Inverse kinematics solver. If None, IK is disabled.
            gripper_controller: Gripper controller. If None, Gripper is disabled.
            target_fps: Target control loop frequency.
            mock: If True, enable mock mode with 3D arm visualization.
            output_file: Path to CSV file for recording joint angles.
        """
        self.robot = robot
        self.pose_estimator = pose_estimator
        self.workspace_mapper = workspace_mapper
        self.ik_solver = ik_solver
        self.gripper_controller = gripper_controller
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.mock = mock
        self.output_file = output_file

        # State
        self._running = False
        self._last_valid_joints: Optional[np.ndarray] = None
        self._last_gripper_state: Optional[GripperState] = None

        # Arm visualizer (lazy init)
        self._arm_viz = None

        # Output file handle
        self._output_handle = None

    def run(self, camera_id: int = 0):
        """Run the teleoperation loop.

        Args:
            camera_id: Camera device ID.
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
        cv2.namedWindow("Teleoperation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Teleoperation", frame_width, frame_height)

        # Setup arm visualizer for mock mode
        if self.mock:
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

                # Process gripper through dedicated controller (uses hand landmarks)
                current_time = time.time() - start_time
                gripper_state: Optional[GripperState] = None
                if self.gripper_controller is not None:
                    hand_landmarks = self.pose_estimator.get_matching_hand()
                    gripper_state = self.gripper_controller.update(
                        hand_landmarks,
                        current_time,
                    )
                    if gripper_state is not None:
                        self._last_gripper_state = gripper_state

                # Process control
                robot_target: Optional[RobotTarget] = None
                joint_angles: Optional[JointAngles] = None

                # Compute IK when pose is valid
                if arm_pose.is_valid:
                    # Map to robot workspace (arm pose only, gripper handled separately)
                    robot_target = self.workspace_mapper.map_pose(arm_pose)

                    if robot_target.is_valid and self.ik_solver is not None:
                        # Solve IK with position and orientation
                        # Orientation is enabled/disabled via WorkspaceConfig.orientation_enabled
                        target_orientation = robot_target.orientation if self.workspace_mapper.config.orientation_enabled else None
                        joint_angles = self.ik_solver.solve(
                            robot_target.position,
                            target_orientation,
                            self._last_valid_joints,
                        )

                        if joint_angles.is_valid:
                            self._last_valid_joints = joint_angles.angles.copy()

                            # Send to robot
                            gripper_pos = gripper_state.position if gripper_state else 0.0
                            self.robot.set_joint_positions(
                                joint_angles.angles,
                                gripper_pos,
                            )

                # Write joint angles to output file (whenever we have valid IK)
                if self._output_handle is not None and joint_angles is not None and joint_angles.is_valid:
                    t = time.time() - start_time
                    angles = joint_angles.angles
                    gripper = gripper_state.position if gripper_state else 0.0
                    self._output_handle.write(
                        f"{t:.4f},{angles[0]:.6f},{angles[1]:.6f},{angles[2]:.6f},"
                        f"{angles[3]:.6f},{angles[4]:.6f},{angles[5]:.6f},{gripper:.6f}\n"
                    )
                    self._output_handle.flush()

                # Display
                display_frame = self._draw_overlay(
                    frame, rgb_frame, arm_pose, robot_target, joint_angles, gripper_state
                )
                cv2.imshow("Teleoperation", display_frame)

                # Update arm visualization (mock mode only)
                # Uses IK results directly from current frame
                if self._arm_viz is not None:
                    target_pos = robot_target.position if robot_target and robot_target.is_valid else None
                    # Normalize gripper position (0-0.08m) to openness (0-1) for visualization
                    gripper_openness_viz = gripper_state.position / 0.08 if gripper_state else 0.0
                    # Use current frame's IK joint angles if valid, otherwise fall back to last valid
                    viz_joints = joint_angles.angles if (joint_angles and joint_angles.is_valid) else self._last_valid_joints
                    ik_error = joint_angles.error if joint_angles else 0.0
                    arm_img = self._arm_viz.update(
                        joint_angles=viz_joints,
                        target_position=target_pos,
                        gripper_openness=gripper_openness_viz,
                        is_valid=joint_angles.is_valid if joint_angles else False,
                        ik_error=ik_error,
                    )
                    cv2.imshow("Piper Arm", arm_img)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    print("Resetting workspace mapping and gripper")
                    self.workspace_mapper.reset()
                    if self.gripper_controller is not None:
                        self.gripper_controller.reset()
                elif key == ord('e'):
                    print("EMERGENCY STOP")
                    self.robot.emergency_stop()

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
        gripper_state: Optional[GripperState],
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

        # Gripper state
        if gripper_state is not None:
            cv2.putText(display, f"Gripper: {gripper_state.raw_openness:.2f}",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display, "Gripper: No hand",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
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

        # Draw workspace limits box and wrist marker if tracking
        if arm_pose.is_valid:
            # Get workspace config for drawing limits
            cfg = self.workspace_mapper.config

            # Shoulder position (absolute in camera coords)
            shoulder_x = arm_pose.shoulder_position[0]
            shoulder_y = arm_pose.shoulder_position[1]

            # Draw workspace limits box (relative to shoulder)
            # Box corners in pixel coordinates
            box_left = int((shoulder_x + cfg.operator_x_range[0]) * w)
            box_right = int((shoulder_x + cfg.operator_x_range[1]) * w)
            box_top = int((shoulder_y + cfg.operator_y_range[0]) * h)
            box_bottom = int((shoulder_y + cfg.operator_y_range[1]) * h)

            # Draw workspace box (cyan dashed-style with corners)
            box_color = (255, 255, 0)  # Cyan in BGR
            # Draw corner markers instead of full rectangle for less clutter
            corner_len = 20
            # Top-left corner
            cv2.line(display, (box_left, box_top), (box_left + corner_len, box_top), box_color, 2)
            cv2.line(display, (box_left, box_top), (box_left, box_top + corner_len), box_color, 2)
            # Top-right corner
            cv2.line(display, (box_right, box_top), (box_right - corner_len, box_top), box_color, 2)
            cv2.line(display, (box_right, box_top), (box_right, box_top + corner_len), box_color, 2)
            # Bottom-left corner
            cv2.line(display, (box_left, box_bottom), (box_left + corner_len, box_bottom), box_color, 2)
            cv2.line(display, (box_left, box_bottom), (box_left, box_bottom - corner_len), box_color, 2)
            # Bottom-right corner
            cv2.line(display, (box_right, box_bottom), (box_right - corner_len, box_bottom), box_color, 2)
            cv2.line(display, (box_right, box_bottom), (box_right, box_bottom - corner_len), box_color, 2)

            # Draw shoulder marker (base reference)
            shoulder_px = int(shoulder_x * w)
            shoulder_py = int(shoulder_y * h)
            cv2.drawMarker(display, (shoulder_px, shoulder_py), box_color,
                          cv2.MARKER_CROSS, 15, 2)

            # Calculate absolute wrist position (shoulder + relative wrist)
            abs_wrist_x = shoulder_x + arm_pose.wrist_position[0]
            abs_wrist_y = shoulder_y + arm_pose.wrist_position[1]
            wrist_x = int(abs_wrist_x * w)
            wrist_y = int(abs_wrist_y * h)

            # Draw wrist marker
            cv2.circle(display, (wrist_x, wrist_y), 15, (0, 255, 0), 3)
            cv2.circle(display, (wrist_x, wrist_y), 5, (0, 255, 0), -1)

        return display


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Camera-based teleoperation for Piper arm")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--mock", action="store_true", help="Use mock robot (no hardware)")
    parser.add_argument("--can", type=str, default="can0", help="CAN interface for real robot")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--left-arm", action="store_true", help="Track left arm instead of right")
    parser.add_argument("--no-record", action="store_true", help="Disable joint angle recording")
    parser.add_argument("--verbose-ik", action="store_true", help="Print IK solver debug info")
    args = parser.parse_args()

    # Create components
    print("Initializing...")

    # Generate output filename if recording is enabled
    output_file = None
    if not args.no_record:
        # Create outputs directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = str(outputs_dir / f"joints_{timestamp}.csv")

    pose_estimator = PoseEstimator(
        pose_model_path="pose_landmarker.task",
        hand_model_path="hand_landmarker.task",
        use_right_arm=not args.left_arm,
    )

    workspace_mapper = WorkspaceMapper(WorkspaceConfig())

    ik_solver = PiperIK(verbose=args.verbose_ik)

    robot = create_robot(use_mock=args.mock, can_name=args.can)

    controller = TeleoperationController(
        robot=robot,
        pose_estimator=pose_estimator,
        workspace_mapper=workspace_mapper,
        ik_solver=ik_solver,
        target_fps=args.fps,
        mock=args.mock,
        output_file=output_file,
    )

    # Run
    controller.run(camera_id=args.camera)

    # Cleanup
    pose_estimator.close()


if __name__ == "__main__":
    main()
