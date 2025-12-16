"""Griffin Teleoperation - Camera-based teleoperation for AgileX Piper arm."""

__version__ = "0.1.0"

from .gripper_controller import GripperController, GripperConfig, GripperState
from .inverse_kinematics import PiperIK, JointAngles, IKPY_AVAILABLE
from .workspace_mapping import WorkspaceMapper, WorkspaceConfig, RobotTarget
from .pose_estimation import PoseEstimator, ArmPose
from .robot_interface import RobotInterface, create_robot, RobotState
from .teleop import TeleoperationController

__all__ = [
    # Gripper control (separate from IK)
    "GripperController",
    "GripperConfig",
    "GripperState",
    # Inverse kinematics (6-DOF arm only)
    "PiperIK",
    "JointAngles",
    "IKPY_AVAILABLE",
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
]
