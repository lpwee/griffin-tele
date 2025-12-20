"""Griffin Teleoperation - Camera-based teleoperation for AgileX Piper arm."""

__version__ = "0.1.0"

from .gripper_controller import GripperController, GripperConfig, GripperState
from .inverse_kinematics import RobotIK, JointAngles, create_robot_ik
from .workspace_mapping import WorkspaceMapper, WorkspaceConfig, RobotTarget
from .pose_estimation import PoseEstimator, ArmPose
from .robot_interface import RobotInterface, create_robot, RobotState
from .teleop import TeleoperationController

__all__ = [
    # Gripper control (separate from IK)
    "GripperController",
    "GripperConfig",
    "GripperState",
    # Inverse kinematics - unified
    "RobotIK",
    "JointAngles",
    "create_robot_ik",
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
