"""Robot POV visualization showing operator arm and robot arm side-by-side."""

from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .pose_estimation import ArmPose


@dataclass
class RobotPOVConfig:
    """Configuration for the Robot POV visualization."""

    # Scene geometry (meters)
    desk_width: float = 1.0
    desk_depth: float = 0.6
    desk_height: float = 0.0  # Ground level in viz

    # Robot mounting (two arms on either side)
    arm_base_separation: float = 0.6  # Distance between arm bases (left-right)
    arm_base_forward: float = 0.1  # How far forward from viewer

    # Scaling
    robot_reach: float = 0.5
    operator_arm_length: float = 0.6

    # Virtual camera view angle (top-down POV like the reference image)
    view_elev: float = 70  # High elevation for top-down view
    view_azim: float = 90  # Looking forward into workspace (X=left-right, Y=forward-back)


class RobotPOVVisualizer:
    """
    Visualizes operator arm movements from the robot's perspective.

    Shows both:
    - Operator arm (cyan): 3 joints (shoulder, elbow, wrist) transformed to robot frame
    - Robot FK arm (orange): 6 joints computed via forward kinematics from IK solution
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

    def __init__(
        self,
        config: Optional[RobotPOVConfig] = None,
        width: int = 500,
        height: int = 500,
    ):
        """Initialize the Robot POV visualizer.

        Args:
            config: Configuration for the visualization. Uses defaults if None.
            width: Width of the visualization image.
            height: Height of the visualization image.
        """
        self.config = config or RobotPOVConfig()
        self.width = width
        self.height = height
        self.dpi = 100

        # Create figure with dark background
        self.fig = plt.figure(
            figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi
        )
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Style settings (dark theme)
        self.fig.patch.set_facecolor("#1a1a1a")
        self.ax.set_facecolor("#1a1a1a")

        # Set consistent axis colors
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#333333")
        self.ax.yaxis.pane.set_edgecolor("#333333")
        self.ax.zaxis.pane.set_edgecolor("#333333")

        # Grid and labels
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)", color="white", fontsize=8)
        self.ax.set_ylabel("Y (m)", color="white", fontsize=8)
        self.ax.set_zlabel("Z (m)", color="white", fontsize=8)
        self.ax.tick_params(colors="white", labelsize=6)

        # Set axis limits
        limit = 0.6
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-0.1, limit)

        # Set view angle
        self.ax.view_init(elev=self.config.view_elev, azim=self.config.view_azim)

        # Canvas for rendering to numpy array
        self.canvas = FigureCanvasAgg(self.fig)

        # Colors
        self.operator_color = "#00d4aa"  # Cyan/teal for operator arm
        self.operator_joint_color = "#00ffcc"
        self.robot_color = "#ff6b35"  # Orange for robot arm
        self.robot_joint_colors = [
            "#ff6b6b",
            "#ffa502",
            "#ffd93d",
            "#6bcb77",
            "#4d96ff",
            "#845ef7",
        ]
        self.gripper_color = "#ff6b9d"

        # Pre-compute transformation matrices
        self._compute_transforms()

    def _compute_transforms(self):
        """Pre-compute the transformation matrices for operator to robot frame."""
        # Frame rotation (camera → robot POV)
        # Camera: X=right, Y=down, Z=forward (toward camera)
        # Robot POV: X=right, Y=forward (into workspace), Z=up
        #
        # Mapping for top-down view:
        # - Operator moves hand right → arm moves right (X → X)
        # - Operator moves hand down (in camera) → arm moves forward into workspace (Y → Y)
        # - Operator moves hand forward (depth) → arm moves up (Z → Z)
        self.R_frame = np.array(
            [
                [1, 0, 0],   # X stays X (right)
                [0, 1, 0],   # Y stays Y (down in camera → forward in workspace)
                [0, 0, 1],   # Z stays Z (depth → up)
            ]
        )

    def _operator_to_robot_frame(self, point_operator: np.ndarray) -> np.ndarray:
        """Transform a point from operator/camera frame to robot frame.

        Args:
            point_operator: 3D point in operator/camera frame (meters).

        Returns:
            3D point in robot frame (meters).
        """
        # Scale operator arm to robot workspace
        scale = self.config.robot_reach / self.config.operator_arm_length

        # Apply rotation and scaling
        point_robot = self.R_frame @ (point_operator * scale)

        return point_robot

    def dh_transform(
        self, alpha: float, a: float, d: float, theta: float
    ) -> np.ndarray:
        """Compute DH transformation matrix.

        Args:
            alpha: Link twist angle.
            a: Link length.
            d: Link offset.
            theta: Joint angle.

        Returns:
            4x4 transformation matrix.
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array(
            [
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0, 0, 0, 1],
            ]
        )

    def forward_kinematics(
        self, joint_angles: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Compute forward kinematics for all joints.

        Args:
            joint_angles: Array of 6 joint angles in radians.

        Returns:
            Tuple of (positions, end_effector_transform):
            - positions: List of 3D positions for base and each joint.
            - end_effector_transform: 4x4 transformation matrix of end effector.
        """
        positions = [np.array([0, 0, 0])]  # Base position
        T = np.eye(4)

        for i, (alpha, a, d, theta_offset) in enumerate(self.DH_PARAMS):
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(alpha, a, d, theta)
            positions.append(T[:3, 3].copy())

        return positions, T.copy()

    def _draw_scene(self):
        """Draw static scene elements (workspace, sink, arm bases, legend)."""
        # Draw workspace surface (counter/desk)
        hw = self.config.desk_width / 2
        hd = self.config.desk_depth / 2
        workspace_z = -0.05  # Slightly below arm level

        # Workspace rectangle
        desk_x = [-hw, hw, hw, -hw, -hw]
        desk_y = [-hd, -hd, hd, hd, -hd]
        desk_z = [workspace_z] * 5
        self.ax.plot3D(desk_x, desk_y, desk_z, color="#666666", linewidth=2)

        # Draw sink (ellipse in center of workspace)
        theta = np.linspace(0, 2 * np.pi, 30)
        sink_rx, sink_ry = 0.15, 0.10
        sink_x = sink_rx * np.cos(theta)
        sink_y = sink_ry * np.sin(theta)
        sink_z = np.full_like(theta, workspace_z)
        self.ax.plot3D(sink_x, sink_y, sink_z, color="#888888", linewidth=1.5)

        # Draw robot arm base markers (left and right)
        # Arms are at the near edge (positive Y with azim=90 view)
        base_sep = self.config.arm_base_separation / 2
        base_y = hd - 0.05  # Near edge (positive Y)

        # Left arm base
        self._draw_arm_base(-base_sep, base_y, 0, "L")
        # Right arm base
        self._draw_arm_base(base_sep, base_y, 0, "R")

        # Draw coordinate axes at center
        self._draw_coordinate_frame(np.eye(4), scale=0.06, show_labels=True)

        # Draw legend
        self.ax.text2D(
            0.02,
            0.98,
            "Operator (cyan)",
            transform=self.ax.transAxes,
            color=self.operator_color,
            fontsize=8,
            verticalalignment="top",
        )
        self.ax.text2D(
            0.02,
            0.93,
            "Robot FK (orange)",
            transform=self.ax.transAxes,
            color=self.robot_color,
            fontsize=8,
            verticalalignment="top",
        )

    def _draw_arm_base(self, x: float, y: float, z: float, label: str):
        """Draw a robot arm base marker.

        Args:
            x, y, z: Position of the base.
            label: Label for the arm (e.g., "L" or "R").
        """
        # Draw base circle
        theta = np.linspace(0, 2 * np.pi, 20)
        r = 0.03
        bx = x + r * np.cos(theta)
        by = y + r * np.sin(theta)
        bz = np.full_like(theta, z)
        self.ax.plot3D(bx, by, bz, color="#444444", linewidth=2)

        # Draw label
        self.ax.text(x, y, z + 0.02, label, color="#888888", fontsize=7, ha="center")

    def _draw_operator_arm(self, arm_pose: ArmPose):
        """Draw the operator's arm transformed to robot frame.

        Args:
            arm_pose: The operator's arm pose from pose estimation.
        """
        if not arm_pose.is_valid:
            return

        # Get joint positions in camera frame
        # wrist_position is relative to shoulder, so we need to compute absolute positions
        shoulder = arm_pose.shoulder_position
        elbow = arm_pose.elbow_position
        # wrist_position is relative to shoulder
        wrist_rel = arm_pose.wrist_position

        # Transform to robot frame
        # For the operator arm, we use the relative positions centered at the right arm base
        elbow_rel = elbow - shoulder
        wrist_abs = wrist_rel  # Already relative to shoulder

        # Position at right arm base (positive Y with azim=90 view)
        base_sep = self.config.arm_base_separation / 2
        hd = self.config.desk_depth / 2
        base_offset = np.array([base_sep, hd - 0.05, 0])

        shoulder_robot = base_offset.copy()
        elbow_robot = base_offset + self._operator_to_robot_frame(elbow_rel)
        wrist_robot = base_offset + self._operator_to_robot_frame(wrist_abs)

        positions = [shoulder_robot, elbow_robot, wrist_robot]

        # Draw links
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            self.ax.plot3D(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=self.operator_color,
                linewidth=4,
                alpha=0.8,
            )

        # Draw joints
        for pos in positions:
            self.ax.scatter(
                *pos,
                color=self.operator_joint_color,
                s=80,
                edgecolors="white",
                linewidths=1,
                zorder=5,
            )

        # Draw wrist marker (larger)
        self.ax.scatter(
            *wrist_robot,
            color=self.operator_joint_color,
            s=120,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=6,
        )

    def _draw_robot_arm(
        self,
        joint_angles: np.ndarray,
        gripper_openness: float = 0.0,
        is_valid: bool = True,
    ):
        """Draw the robot arm using forward kinematics.

        Args:
            joint_angles: Array of 6 joint angles in radians.
            gripper_openness: Gripper openness (0-1).
            is_valid: Whether the IK solution is valid.
        """
        # Compute joint positions
        positions_local, ee_transform = self.forward_kinematics(joint_angles)

        # Position at right arm base (same as operator arm)
        base_sep = self.config.arm_base_separation / 2
        hd = self.config.desk_depth / 2
        base_offset = np.array([base_sep, hd - 0.05, 0])

        # Offset all positions to the arm base
        positions = [pos + base_offset for pos in positions_local]

        # Draw links
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            self.ax.plot3D(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=self.robot_color,
                linewidth=3,
                alpha=0.7,
            )

        # Draw joints
        for i, pos in enumerate(positions[:-1]):
            color = (
                self.robot_joint_colors[i]
                if i < len(self.robot_joint_colors)
                else "#ffffff"
            )
            self.ax.scatter(
                *pos, color=color, s=60, edgecolors="white", linewidths=1, zorder=5
            )

        # Draw end effector
        ee_pos = positions[-1]
        ee_color = "#00ff00" if is_valid else "#ff0000"
        self.ax.scatter(
            *ee_pos,
            color=ee_color,
            s=100,
            marker="o",
            edgecolors="white",
            linewidths=2,
            zorder=6,
        )

        # Draw gripper
        self._draw_gripper(positions[-2], positions[-1], gripper_openness)

    def _draw_gripper(
        self, wrist_pos: np.ndarray, ee_pos: np.ndarray, openness: float
    ):
        """Draw a simple gripper representation.

        Args:
            wrist_pos: Wrist joint position.
            ee_pos: End effector position.
            openness: Gripper openness (0-1).
        """
        direction = ee_pos - wrist_pos
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return

        direction = direction / length

        # Perpendicular direction for gripper fingers
        if abs(direction[2]) < 0.9:
            perp = np.cross(direction, [0, 0, 1])
        else:
            perp = np.cross(direction, [1, 0, 0])
        perp = perp / np.linalg.norm(perp)

        # Gripper width based on openness
        width = 0.02 + openness * 0.04  # 2cm to 6cm

        # Finger positions
        finger1 = ee_pos + perp * width / 2
        finger2 = ee_pos - perp * width / 2

        # Draw gripper fingers
        self.ax.plot3D(
            [ee_pos[0], finger1[0]],
            [ee_pos[1], finger1[1]],
            [ee_pos[2], finger1[2]],
            color=self.gripper_color,
            linewidth=2,
        )
        self.ax.plot3D(
            [ee_pos[0], finger2[0]],
            [ee_pos[1], finger2[1]],
            [ee_pos[2], finger2[2]],
            color=self.gripper_color,
            linewidth=2,
        )

    def _draw_coordinate_frame(
        self, transform: np.ndarray, scale: float = 0.1, show_labels: bool = True
    ):
        """Draw XYZ coordinate frame.

        Args:
            transform: 4x4 transformation matrix defining the frame.
            scale: Length of the axis arrows.
            show_labels: If True, add X/Y/Z labels at axis tips.
        """
        origin = transform[:3, 3]

        # X axis (red)
        x_end = origin + transform[:3, 0] * scale
        self.ax.plot3D(
            [origin[0], x_end[0]],
            [origin[1], x_end[1]],
            [origin[2], x_end[2]],
            color="red",
            linewidth=2,
            alpha=0.9,
        )
        if show_labels:
            self.ax.text(
                x_end[0], x_end[1], x_end[2], "X", color="red", fontsize=7
            )

        # Y axis (green)
        y_end = origin + transform[:3, 1] * scale
        self.ax.plot3D(
            [origin[0], y_end[0]],
            [origin[1], y_end[1]],
            [origin[2], y_end[2]],
            color="lime",
            linewidth=2,
            alpha=0.9,
        )
        if show_labels:
            self.ax.text(
                y_end[0], y_end[1], y_end[2], "Y", color="lime", fontsize=7
            )

        # Z axis (blue)
        z_end = origin + transform[:3, 2] * scale
        self.ax.plot3D(
            [origin[0], z_end[0]],
            [origin[1], z_end[1]],
            [origin[2], z_end[2]],
            color="cyan",
            linewidth=2,
            alpha=0.9,
        )
        if show_labels:
            self.ax.text(
                z_end[0], z_end[1], z_end[2], "Z", color="cyan", fontsize=7
            )

    def update(
        self,
        arm_pose: Optional[ArmPose] = None,
        joint_angles: Optional[np.ndarray] = None,
        gripper_openness: float = 0.0,
        is_valid: bool = True,
    ) -> np.ndarray:
        """Update visualization and return as BGR image.

        Args:
            arm_pose: Operator's arm pose from pose estimation.
            joint_angles: IK solution joint angles in radians (6,).
            gripper_openness: Gripper openness (0-1).
            is_valid: Whether the IK solution is valid.

        Returns:
            BGR image as numpy array.
        """
        # Clear previous plot
        self.ax.cla()

        # Restore axis settings after clear
        self.ax.set_facecolor("#1a1a1a")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X", color="white", fontsize=8)
        self.ax.set_ylabel("Y", color="white", fontsize=8)
        self.ax.set_zlabel("Z", color="white", fontsize=8)
        self.ax.tick_params(colors="white", labelsize=6)

        limit = 0.6
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-0.1, limit)

        self.ax.view_init(elev=self.config.view_elev, azim=self.config.view_azim)

        # Draw static scene elements
        self._draw_scene()

        # Draw operator arm (cyan) if pose is available
        if arm_pose is not None and arm_pose.is_valid:
            self._draw_operator_arm(arm_pose)

        # Draw robot FK arm (orange) if joint angles are available
        if joint_angles is not None:
            self._draw_robot_arm(joint_angles, gripper_openness, is_valid)

        # Title
        ik_status = "Valid" if is_valid else "Invalid"
        title = f"Robot POV | IK: {ik_status}"
        if joint_angles is not None:
            angles_deg = np.degrees(joint_angles)
            title += f"\nJoints: [{angles_deg[0]:.0f}, {angles_deg[1]:.0f}, {angles_deg[2]:.0f}, {angles_deg[3]:.0f}, {angles_deg[4]:.0f}, {angles_deg[5]:.0f}]"
        self.ax.set_title(title, color="white", fontsize=8, pad=10)

        # Render to image
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        img = np.asarray(buf)

        # Convert RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img_bgr

    def close(self):
        """Clean up matplotlib resources."""
        plt.close(self.fig)


def test_visualizer():
    """Test the visualizer with animated motion.

    Shows a top-down view of the workspace with:
    - Operator arm (cyan) moving in response to simulated pose
    - Robot FK arm (orange) moving with animated joint angles
    - Workspace with sink and dual arm base markers
    """
    viz = RobotPOVVisualizer(width=600, height=600)

    cv2.namedWindow("Robot POV", cv2.WINDOW_NORMAL)
    print("Robot POV Visualizer Test")
    print("Press 'q' to quit")

    t = 0
    while True:
        # Create a mock ArmPose
        # Simulate operator arm moving in camera frame
        shoulder = np.array([0.0, 0.0, 0.5])  # Camera frame
        elbow_offset = np.array([0.0, 0.15 + 0.05 * np.sin(t), 0.0])
        wrist_offset = np.array([
            0.1 * np.sin(t * 0.7),      # Left-right motion
            0.25 + 0.1 * np.sin(t * 0.5),  # Forward-back motion
            0.1 * np.cos(t * 0.7),      # Up-down motion
        ])

        mock_pose = ArmPose(
            wrist_position=wrist_offset,
            elbow_position=shoulder + elbow_offset,
            shoulder_position=shoulder,
            wrist_orientation=np.zeros(3),
            visibility=1.0,
            is_valid=True,
            hand_tracked=False,
            is_metric=True,
            depth_valid=True,
        )

        # Animate robot joint angles (simulating IK solution)
        angles = np.array([
            np.sin(t * 0.5) * 0.3,      # Base rotation
            np.sin(t * 0.3) * 0.2 - 0.3,  # Shoulder
            np.sin(t * 0.4) * 0.3,      # Elbow
            np.sin(t * 0.6) * 0.5,      # Wrist 1
            np.sin(t * 0.7) * 0.3,      # Wrist 2
            t * 0.3,                     # Wrist 3 (continuous rotation)
        ])

        gripper = (np.sin(t) + 1) / 2  # 0 to 1

        img = viz.update(
            arm_pose=mock_pose,
            joint_angles=angles,
            gripper_openness=gripper,
            is_valid=True,
        )

        cv2.imshow("Robot POV", img)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        t += 0.05

    viz.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_visualizer()
