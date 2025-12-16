"""Griffin Teleoperation - Camera-based teleoperation for robot arms.

Supports:
- AgileX Piper 6-DOF arm (CAN interface)
- SO-101 / LeRobot arm (Feetech STS3215 servos, serial interface)
"""

__version__ = "0.1.0"

from .gripper_controller import GripperController, GripperConfig, GripperState
from .inverse_kinematics import PiperIK, JointAngles
from .workspace_mapping import WorkspaceMapper, WorkspaceConfig, RobotTarget
from .pose_estimation import PoseEstimator, ArmPose
from .robot_interface import RobotInterface, create_robot, RobotState
from .teleop import TeleoperationController

# S101 / LeRobot arm support
from .s101_interface import (
    S101Robot,
    MockS101Robot,
    S101Config,
    create_s101_robot,
    FeetechServoController,
)
from .s101_kinematics import S101IK, S101JointAngles, create_s101_ik

__all__ = [
    # Gripper control (separate from IK)
    "GripperController",
    "GripperConfig",
    "GripperState",
    # Inverse kinematics - Piper
    "PiperIK",
    "JointAngles",
    # Inverse kinematics - S101
    "S101IK",
    "S101JointAngles",
    "create_s101_ik",
    # Workspace mapping
    "WorkspaceMapper",
    "WorkspaceConfig",
    "RobotTarget",
    # Pose estimation
    "PoseEstimator",
    "ArmPose",
    # Robot interface - Generic
    "RobotInterface",
    "create_robot",
    "RobotState",
    # Robot interface - S101/LeRobot
    "S101Robot",
    "MockS101Robot",
    "S101Config",
    "create_s101_robot",
    "FeetechServoController",
    # Main controller
    "TeleoperationController",
]
