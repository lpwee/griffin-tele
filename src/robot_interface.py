"""Robot interface for AgileX Piper arm."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import time


@dataclass
class RobotState:
    """Current state of the robot."""

    # Joint positions in radians
    joint_positions: np.ndarray  # shape (6,)

    # Gripper position in meters (0 = closed, 0.08 = open)
    gripper_position: float

    # Whether robot is enabled and ready
    is_enabled: bool

    # Timestamp of state reading
    timestamp: float


class RobotInterface(ABC):
    """Abstract interface for robot control."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        pass

    @abstractmethod
    def enable(self) -> bool:
        """Enable robot motors.

        Returns:
            True if enable successful.
        """
        pass

    @abstractmethod
    def disable(self) -> None:
        """Disable robot motors (safe stop)."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state.

        Returns:
            Current joint positions and gripper state.
        """
        pass

    @abstractmethod
    def set_joint_positions(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> bool:
        """Command robot to move to joint positions.

        Args:
            positions: Target joint positions in radians, shape (6,).
            gripper: Target gripper position in meters (0-0.08).

        Returns:
            True if command was accepted.
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Emergency stop - immediately halt all motion."""
        pass


class MockRobot(RobotInterface):
    """Mock robot for testing without hardware."""

    def __init__(self, simulate_latency: bool = True):
        """Initialize mock robot.

        Args:
            simulate_latency: If True, simulate realistic command latency.
        """
        self._simulate_latency = simulate_latency
        self._connected = False
        self._enabled = False
        self._joint_positions = np.zeros(6)
        self._gripper_position = 0.0
        self._target_positions = np.zeros(6)
        self._target_gripper = 0.0

    def connect(self) -> bool:
        print("[MockRobot] Connected")
        self._connected = True
        return True

    def disconnect(self) -> None:
        print("[MockRobot] Disconnected")
        self._connected = False
        self._enabled = False

    def enable(self) -> bool:
        if not self._connected:
            print("[MockRobot] Cannot enable: not connected")
            return False
        print("[MockRobot] Motors enabled")
        self._enabled = True
        return True

    def disable(self) -> None:
        print("[MockRobot] Motors disabled")
        self._enabled = False

    def get_state(self) -> RobotState:
        # Simulate movement towards target
        if self._enabled:
            alpha = 0.1  # Movement speed
            self._joint_positions += alpha * (
                self._target_positions - self._joint_positions
            )
            self._gripper_position += alpha * (
                self._target_gripper - self._gripper_position
            )

        return RobotState(
            joint_positions=self._joint_positions.copy(),
            gripper_position=self._gripper_position,
            is_enabled=self._enabled,
            timestamp=time.time(),
        )

    def set_joint_positions(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> bool:
        if not self._enabled:
            return False

        self._target_positions = positions.copy()
        if gripper is not None:
            self._target_gripper = gripper

        if self._simulate_latency:
            time.sleep(0.001)  # 1ms simulated latency

        return True

    def emergency_stop(self) -> None:
        print("[MockRobot] EMERGENCY STOP")
        self._enabled = False
        self._target_positions = self._joint_positions.copy()


class PiperRobot(RobotInterface):
    """Interface for real AgileX Piper arm via piper_sdk.

    Requires piper_sdk to be installed and CAN interface configured.
    """

    def __init__(self, can_name: str = "can0"):
        """Initialize Piper robot interface.

        Args:
            can_name: CAN interface name (e.g., "can0").
        """
        self._can_name = can_name
        self._piper = None
        self._connected = False
        self._enabled = False

    def connect(self) -> bool:
        try:
            from piper_sdk import C_PiperInterface

            self._piper = C_PiperInterface(self._can_name)
            self._piper.ConnectPort()
            self._connected = True
            print(f"[PiperRobot] Connected via {self._can_name}")
            return True
        except ImportError:
            print("[PiperRobot] piper_sdk not installed")
            return False
        except Exception as e:
            print(f"[PiperRobot] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._piper is not None:
            try:
                self.disable()
            except Exception:
                pass
            self._piper = None
        self._connected = False
        print("[PiperRobot] Disconnected")

    def enable(self) -> bool:
        if not self._connected or self._piper is None:
            return False

        try:
            self._piper.EnableArm(7)  # Enable all joints
            self._enabled = True
            print("[PiperRobot] Motors enabled")
            return True
        except Exception as e:
            print(f"[PiperRobot] Enable failed: {e}")
            return False

    def disable(self) -> None:
        if self._piper is not None:
            try:
                self._piper.DisableArm(7)
            except Exception:
                pass
        self._enabled = False
        print("[PiperRobot] Motors disabled")

    def get_state(self) -> RobotState:
        if not self._connected or self._piper is None:
            return RobotState(
                joint_positions=np.zeros(6),
                gripper_position=0.0,
                is_enabled=False,
                timestamp=time.time(),
            )

        try:
            # Get joint feedback from SDK
            arm_status = self._piper.GetArmStatus()
            joint_positions = np.array([
                arm_status.joint_state.joint_1.joint_angle,
                arm_status.joint_state.joint_2.joint_angle,
                arm_status.joint_state.joint_3.joint_angle,
                arm_status.joint_state.joint_4.joint_angle,
                arm_status.joint_state.joint_5.joint_angle,
                arm_status.joint_state.joint_6.joint_angle,
            ])
            # Convert from degrees to radians if needed
            joint_positions = np.radians(joint_positions)

            gripper_position = arm_status.gripper_state.grippers_angle / 1000.0

            return RobotState(
                joint_positions=joint_positions,
                gripper_position=gripper_position,
                is_enabled=self._enabled,
                timestamp=time.time(),
            )
        except Exception as e:
            print(f"[PiperRobot] State read error: {e}")
            return RobotState(
                joint_positions=np.zeros(6),
                gripper_position=0.0,
                is_enabled=False,
                timestamp=time.time(),
            )

    def set_joint_positions(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> bool:
        if not self._enabled or self._piper is None:
            return False

        try:
            # Convert radians to degrees (or motor units as needed)
            positions_deg = np.degrees(positions)

            # Send joint command
            # Note: Actual piper_sdk API may differ - adjust as needed
            self._piper.MotionCtrl_1(
                int(positions_deg[0] * 1000),
                int(positions_deg[1] * 1000),
                int(positions_deg[2] * 1000),
            )
            self._piper.MotionCtrl_2(
                int(positions_deg[3] * 1000),
                int(positions_deg[4] * 1000),
                int(positions_deg[5] * 1000),
            )

            if gripper is not None:
                # Gripper in mm
                self._piper.GripperCtrl(int(gripper * 1000), 500)

            return True
        except Exception as e:
            print(f"[PiperRobot] Command error: {e}")
            return False

    def emergency_stop(self) -> None:
        print("[PiperRobot] EMERGENCY STOP")
        if self._piper is not None:
            try:
                self._piper.EmergencyStop()
            except Exception:
                pass
        self._enabled = False


def create_robot(
    robot_type: str = "mock",
    can_name: str = "can0",
    port: str = "/dev/ttyACM0",
    use_mock: bool = None,  # Deprecated, use robot_type instead
) -> RobotInterface:
    """Factory function to create robot interface.

    Args:
        robot_type: Robot type - "mock", "piper", "s101", "mock_s101"
        can_name: CAN interface name for Piper robot.
        port: Serial port for S101 robot.
        use_mock: Deprecated. Use robot_type="mock" instead.

    Returns:
        Robot interface instance.
    """
    # Handle legacy use_mock parameter
    if use_mock is not None:
        import warnings
        warnings.warn(
            "use_mock parameter is deprecated. Use robot_type='mock' or 'piper' instead.",
            DeprecationWarning
        )
        if use_mock:
            return MockRobot()
        else:
            return PiperRobot(can_name)

    robot_type = robot_type.lower()

    if robot_type == "mock":
        return MockRobot()
    elif robot_type == "piper":
        return PiperRobot(can_name)
    elif robot_type == "s101":
        from src.s101_interface import create_s101_robot
        return create_s101_robot(use_mock=False, port=port)
    elif robot_type == "mock_s101":
        from src.s101_interface import create_s101_robot
        return create_s101_robot(use_mock=True, port=port)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. "
                        f"Valid types: mock, piper, s101, mock_s101")
