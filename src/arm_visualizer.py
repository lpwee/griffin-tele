"""3D visualization of the Piper arm using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import cv2


class PiperArmVisualizer:
    """Real-time 3D visualization of the Piper arm."""

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

    def __init__(self, width: int = 400, height: int = 400):
        """Initialize the visualizer.

        Args:
            width: Width of the visualization image.
            height: Height of the visualization image.
        """
        self.width = width
        self.height = height
        self.dpi = 100

        # Create figure with dark background
        self.fig = plt.figure(figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Style settings
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#1a1a1a')

        # Set consistent axis colors
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#333333')
        self.ax.yaxis.pane.set_edgecolor('#333333')
        self.ax.zaxis.pane.set_edgecolor('#333333')

        # Grid and labels
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)', color='white', fontsize=8)
        self.ax.set_ylabel('Y (m)', color='white', fontsize=8)
        self.ax.set_zlabel('Z (m)', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=6)

        # Set axis limits based on robot reach
        limit = 0.6
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-0.1, limit)

        # Set view angle
        self.ax.view_init(elev=25, azim=45)

        # Canvas for rendering to numpy array
        self.canvas = FigureCanvasAgg(self.fig)

        # Joint colors
        self.joint_colors = ['#ff6b6b', '#ffa502', '#ffd93d', '#6bcb77', '#4d96ff', '#845ef7']
        self.link_color = '#4dabf7'
        self.gripper_color = '#ff6b9d'

        # Store current joint angles
        self._current_angles = np.zeros(6)
        self._target_angles: Optional[np.ndarray] = None
        self._gripper_open = 0.0

    def dh_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
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

        return np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles: np.ndarray) -> list[np.ndarray]:
        """Compute forward kinematics for all joints.

        Args:
            joint_angles: Array of 6 joint angles in radians.

        Returns:
            List of 3D positions for base and each joint.
        """
        positions = [np.array([0, 0, 0])]  # Base position
        T = np.eye(4)

        for i, (alpha, a, d, theta_offset) in enumerate(self.DH_PARAMS):
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(alpha, a, d, theta)
            positions.append(T[:3, 3].copy())

        return positions

    def update(
        self,
        joint_angles: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        gripper_openness: float = 0.0,
        is_valid: bool = True,
    ) -> np.ndarray:
        """Update visualization and return as BGR image.

        Args:
            joint_angles: Current joint angles in radians (6,).
            target_position: Target end-effector position for visualization.
            gripper_openness: Gripper openness (0-1).
            is_valid: Whether the IK solution is valid.

        Returns:
            BGR image as numpy array.
        """
        if joint_angles is not None:
            self._current_angles = joint_angles
        self._gripper_open = gripper_openness

        # Clear previous plot
        self.ax.cla()

        # Restore axis settings after clear
        self.ax.set_facecolor('#1a1a1a')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X', color='white', fontsize=8)
        self.ax.set_ylabel('Y', color='white', fontsize=8)
        self.ax.set_zlabel('Z', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=6)

        limit = 0.6
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-0.1, limit)

        # Compute joint positions
        positions = self.forward_kinematics(self._current_angles)

        # Draw base platform
        self._draw_base()

        # Draw links
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            self.ax.plot3D(
                [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=self.link_color, linewidth=4, alpha=0.8
            )

        # Draw joints
        for i, pos in enumerate(positions[:-1]):
            color = self.joint_colors[i] if i < len(self.joint_colors) else '#ffffff'
            self.ax.scatter(*pos, color=color, s=100, edgecolors='white', linewidths=1, zorder=5)

        # Draw end effector
        ee_pos = positions[-1]
        ee_color = '#00ff00' if is_valid else '#ff0000'
        self.ax.scatter(*ee_pos, color=ee_color, s=150, marker='o', edgecolors='white', linewidths=2, zorder=6)

        # Draw gripper
        self._draw_gripper(positions[-2], positions[-1], gripper_openness)

        # Draw target position if provided
        if target_position is not None:
            self.ax.scatter(
                *target_position, color='#ffff00', s=80, marker='x',
                linewidths=2, alpha=0.7, zorder=4
            )
            # Draw line to target
            self.ax.plot3D(
                [ee_pos[0], target_position[0]],
                [ee_pos[1], target_position[1]],
                [ee_pos[2], target_position[2]],
                color='#ffff00', linestyle='--', alpha=0.5, linewidth=1
            )

        # Draw coordinate frame at base
        self._draw_coordinate_frame(np.eye(4), scale=0.1)

        # Title
        angles_deg = np.degrees(self._current_angles)
        self.ax.set_title(
            f'Piper Arm - Joints: [{angles_deg[0]:.0f}°, {angles_deg[1]:.0f}°, {angles_deg[2]:.0f}°, '
            f'{angles_deg[3]:.0f}°, {angles_deg[4]:.0f}°, {angles_deg[5]:.0f}°]',
            color='white', fontsize=8, pad=10
        )

        # Render to image
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        img = np.asarray(buf)

        # Convert RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img_bgr

    def _draw_base(self):
        """Draw the robot base platform."""
        # Draw a cylinder-like base
        theta = np.linspace(0, 2 * np.pi, 20)
        r = 0.05
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)

        self.ax.plot3D(x, y, z, color='#666666', linewidth=2)
        self.ax.plot_surface(
            np.outer(np.cos(theta), [0, r]),
            np.outer(np.sin(theta), [0, r]),
            np.zeros((20, 2)),
            color='#444444', alpha=0.5
        )

    def _draw_gripper(self, wrist_pos: np.ndarray, ee_pos: np.ndarray, openness: float):
        """Draw a simple gripper representation."""
        # Direction from wrist to end effector
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
            [ee_pos[0], finger1[0]], [ee_pos[1], finger1[1]], [ee_pos[2], finger1[2]],
            color=self.gripper_color, linewidth=3
        )
        self.ax.plot3D(
            [ee_pos[0], finger2[0]], [ee_pos[1], finger2[1]], [ee_pos[2], finger2[2]],
            color=self.gripper_color, linewidth=3
        )

    def _draw_coordinate_frame(self, transform: np.ndarray, scale: float = 0.1):
        """Draw XYZ coordinate frame."""
        origin = transform[:3, 3]

        # X axis (red)
        x_end = origin + transform[:3, 0] * scale
        self.ax.plot3D([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]],
                      color='red', linewidth=2, alpha=0.7)

        # Y axis (green)
        y_end = origin + transform[:3, 1] * scale
        self.ax.plot3D([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]],
                      color='green', linewidth=2, alpha=0.7)

        # Z axis (blue)
        z_end = origin + transform[:3, 2] * scale
        self.ax.plot3D([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]],
                      color='blue', linewidth=2, alpha=0.7)

    def close(self):
        """Clean up matplotlib resources."""
        plt.close(self.fig)


def test_visualizer():
    """Test the visualizer with animated joint motion."""
    viz = PiperArmVisualizer(width=600, height=600)

    cv2.namedWindow('Piper Arm', cv2.WINDOW_NORMAL)

    t = 0
    while True:
        # Animate joints
        angles = np.array([
            np.sin(t * 0.5) * 0.5,
            np.sin(t * 0.3) * 0.3 - 0.3,
            np.sin(t * 0.4) * 0.4,
            np.sin(t * 0.6) * 0.8,
            np.sin(t * 0.7) * 0.5,
            t * 0.5,  # Continuous rotation
        ])

        gripper = (np.sin(t) + 1) / 2  # 0 to 1

        img = viz.update(
            joint_angles=angles,
            target_position=np.array([0.3, 0.2, 0.3]),
            gripper_openness=gripper,
            is_valid=True,
        )

        cv2.imshow('Piper Arm', img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        t += 0.05

    viz.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_visualizer()
