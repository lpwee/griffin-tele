"""Griffin Teleoperation - Camera-based teleoperation for AgileX Piper arm."""

__version__ = "0.1.0"

from .gripper_controller import GripperController, GripperConfig, GripperState
from .inverse_kinematics import PiperIK, JointAngles
from .workspace_mapping import WorkspaceMapper, WorkspaceConfig, RobotTarget
from .pose_estimation import PoseEstimator, ArmPose
from .robot_interface import RobotInterface, create_robot, RobotState
from .teleop import TeleoperationController
from .camera import CameraInterface, WebcamCamera, OrbbecCamera, create_camera, CameraFrame, CameraIntrinsics
from .depth_processor import DepthProcessor, DepthConfig
from .robot_pov_visualizer import RobotPOVVisualizer, RobotPOVConfig

__all__ = [
    # Gripper control (separate from IK)
    "GripperController",
    "GripperConfig",
    "GripperState",
    # Inverse kinematics (6-DOF arm only)
    "PiperIK",
    "JointAngles",
    # Workspace mapping
    "WorkspaceMapper",
    "WorkspaceConfig",
    "RobotTarget",
    # Pose estimation
    "PoseEstimator",
    "ArmPose",
    # Robot interface
    "RobotInterface",
    "create_robot",
    "RobotState",
    # Main controller
    "TeleoperationController",
    # Camera abstraction
    "CameraInterface",
    "WebcamCamera",
    "OrbbecCamera",
    "create_camera",
    "CameraFrame",
    "CameraIntrinsics",
    # Depth processing
    "DepthProcessor",
    "DepthConfig",
    # Robot POV visualization
    "RobotPOVVisualizer",
    "RobotPOVConfig",
]
