"""Robot interface for SO-101 (LeRobot) 6-DOF arm with Feetech STS3215 servos.

The SO-101 is a low-cost 6-DOF robot arm designed by Hugging Face/RobotStudio
for use with the LeRobot library. It uses Feetech STS3215 serial bus servos.

References:
- https://huggingface.co/docs/lerobot/so101
- https://github.com/TheRobotStudio/SO-ARM100
- https://github.com/ftservo/FTServo_Python
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import time
import serial
import struct


@dataclass
class S101Config:
    """Configuration for SO-101 arm."""

    # Serial port settings
    port: str = "/dev/ttyACM0"
    baudrate: int = 1_000_000
    timeout: float = 0.1

    # Motor IDs (1-indexed as per Feetech convention)
    motor_ids: tuple = (1, 2, 3, 4, 5, 6)

    # Joint names corresponding to motor IDs
    joint_names: tuple = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )

    # Position range for STS3215 servos
    # 0-4096 steps, center at 2048 (180 degrees each direction)
    position_min: int = 0
    position_max: int = 4096
    position_center: int = 2048

    # Steps per radian (4096 steps / 2*pi radians)
    steps_per_radian: float = 4096 / (2 * np.pi)

    # Joint limits in radians (from URDF)
    joint_limits: tuple = (
        (-1.91986, 1.91986),   # shoulder_pan
        (-1.74533, 1.74533),   # shoulder_lift
        (-1.69, 1.69),         # elbow_flex
        (-1.65806, 1.65806),   # wrist_flex
        (-2.74385, 2.84121),   # wrist_roll
        (-0.174533, 1.74533),  # gripper
    )

    # Home position in radians (all joints centered)
    home_position: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Gripper range in meters (mapped from gripper joint angle)
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.08


# Feetech STS3215 Protocol Constants
class STS3215Protocol:
    """Protocol constants for Feetech STS3215 servos."""

    # Instruction codes
    INST_PING = 0x01
    INST_READ = 0x02
    INST_WRITE = 0x03
    INST_REG_WRITE = 0x04
    INST_ACTION = 0x05
    INST_SYNC_READ = 0x82
    INST_SYNC_WRITE = 0x83

    # Control table addresses
    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58
    ADDR_PRESENT_LOAD = 60
    ADDR_PRESENT_VOLTAGE = 62
    ADDR_PRESENT_TEMPERATURE = 63
    ADDR_LOCK = 55
    ADDR_MODE = 33

    # Packet markers
    HEADER = 0xFF

    # Broadcast ID
    BROADCAST_ID = 0xFE


class FeetechServoController:
    """Low-level controller for Feetech STS3215 servos.

    Implements the Feetech serial protocol for reading/writing servo positions.
    """

    def __init__(self, config: S101Config):
        self.config = config
        self._serial: Optional[serial.Serial] = None
        self._connected = False

    def connect(self) -> bool:
        """Open serial connection to servo bus."""
        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._connected = True
            return True
        except Exception as e:
            print(f"[FeetechController] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Close serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False

    def _checksum(self, data: bytes) -> int:
        """Calculate checksum for packet."""
        return (~sum(data)) & 0xFF

    def _build_packet(self, servo_id: int, instruction: int, params: bytes = b'') -> bytes:
        """Build a command packet."""
        length = len(params) + 2  # instruction + checksum
        packet = bytes([
            STS3215Protocol.HEADER,
            STS3215Protocol.HEADER,
            servo_id,
            length,
            instruction,
        ]) + params
        checksum = self._checksum(packet[2:])
        return packet + bytes([checksum])

    def _send_packet(self, packet: bytes) -> bool:
        """Send packet to servo bus."""
        if self._serial is None:
            return False
        try:
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()
            return True
        except Exception as e:
            print(f"[FeetechController] Send error: {e}")
            return False

    def _receive_packet(self, expected_length: int = 6) -> Optional[bytes]:
        """Receive response packet from servo."""
        if self._serial is None:
            return None
        try:
            # Read header
            header = self._serial.read(4)
            if len(header) < 4 or header[0:2] != b'\xff\xff':
                return None

            servo_id = header[2]
            length = header[3]

            # Read remaining data
            data = self._serial.read(length)
            if len(data) < length:
                return None

            # Verify checksum
            packet_data = bytes([servo_id, length]) + data[:-1]
            expected_checksum = self._checksum(packet_data)
            if data[-1] != expected_checksum:
                return None

            return header + data
        except Exception as e:
            print(f"[FeetechController] Receive error: {e}")
            return None

    def ping(self, servo_id: int) -> bool:
        """Ping a servo to check if it's connected."""
        packet = self._build_packet(servo_id, STS3215Protocol.INST_PING)
        if not self._send_packet(packet):
            return False
        response = self._receive_packet()
        return response is not None

    def enable_torque(self, servo_id: int, enable: bool = True) -> bool:
        """Enable or disable servo torque."""
        params = bytes([STS3215Protocol.ADDR_TORQUE_ENABLE, 1 if enable else 0])
        packet = self._build_packet(servo_id, STS3215Protocol.INST_WRITE, params)
        return self._send_packet(packet)

    def set_torque_all(self, enable: bool = True) -> bool:
        """Enable or disable torque on all servos."""
        success = True
        for motor_id in self.config.motor_ids:
            if not self.enable_torque(motor_id, enable):
                success = False
            time.sleep(0.001)  # Small delay between commands
        return success

    def read_position(self, servo_id: int) -> Optional[int]:
        """Read current position of a servo (0-4096)."""
        params = bytes([STS3215Protocol.ADDR_PRESENT_POSITION, 2])
        packet = self._build_packet(servo_id, STS3215Protocol.INST_READ, params)
        if not self._send_packet(packet):
            return None

        response = self._receive_packet(8)
        if response is None or len(response) < 8:
            return None

        # Extract position (2 bytes, little-endian)
        position = struct.unpack('<H', response[5:7])[0]
        return position

    def read_positions_all(self) -> Optional[np.ndarray]:
        """Read positions of all servos."""
        positions = []
        for motor_id in self.config.motor_ids:
            pos = self.read_position(motor_id)
            if pos is None:
                return None
            positions.append(pos)
            time.sleep(0.0005)  # Small delay between reads
        return np.array(positions)

    def write_position(self, servo_id: int, position: int, speed: int = 0) -> bool:
        """Write goal position to a servo.

        Args:
            servo_id: Motor ID (1-6)
            position: Goal position (0-4096)
            speed: Goal speed (0 = max speed)
        """
        position = max(0, min(4096, position))
        params = struct.pack('<BHH', STS3215Protocol.ADDR_GOAL_POSITION, position, speed)
        packet = self._build_packet(servo_id, STS3215Protocol.INST_WRITE, params)
        return self._send_packet(packet)

    def sync_write_positions(self, positions: Dict[int, int], speed: int = 0) -> bool:
        """Write positions to multiple servos simultaneously.

        Args:
            positions: Dict mapping motor_id to position
            speed: Goal speed for all servos
        """
        if not positions:
            return True

        # Build sync write packet
        # Format: start_addr, data_length, [id1, data1..., id2, data2..., ...]
        start_addr = STS3215Protocol.ADDR_GOAL_POSITION
        data_length = 4  # 2 bytes position + 2 bytes speed

        params = bytes([start_addr, data_length])
        for motor_id, pos in positions.items():
            pos = max(0, min(4096, pos))
            params += bytes([motor_id]) + struct.pack('<HH', pos, speed)

        packet = self._build_packet(
            STS3215Protocol.BROADCAST_ID,
            STS3215Protocol.INST_SYNC_WRITE,
            params
        )
        return self._send_packet(packet)


class S101Robot:
    """Robot interface for SO-101 arm.

    Implements the same interface as PiperRobot for compatibility
    with the teleoperation system.
    """

    def __init__(self, config: Optional[S101Config] = None):
        """Initialize S101 robot interface.

        Args:
            config: Robot configuration. Uses defaults if None.
        """
        self.config = config or S101Config()
        self._controller = FeetechServoController(self.config)
        self._connected = False
        self._enabled = False
        self._last_positions = np.zeros(6)
        self._last_gripper = 0.0

    def connect(self) -> bool:
        """Connect to the robot."""
        if self._controller.connect():
            # Verify all motors are responding
            print(f"[S101Robot] Checking motors on {self.config.port}...")
            all_found = True
            for motor_id in self.config.motor_ids:
                if self._controller.ping(motor_id):
                    print(f"  Motor {motor_id}: OK")
                else:
                    print(f"  Motor {motor_id}: NOT FOUND")
                    all_found = False
                time.sleep(0.01)

            if all_found:
                self._connected = True
                print(f"[S101Robot] Connected - all {len(self.config.motor_ids)} motors found")
                return True
            else:
                print("[S101Robot] Some motors not responding")
                self._controller.disconnect()
                return False
        return False

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._enabled:
            self.disable()
        self._controller.disconnect()
        self._connected = False
        print("[S101Robot] Disconnected")

    def enable(self) -> bool:
        """Enable robot motors (torque on)."""
        if not self._connected:
            print("[S101Robot] Cannot enable: not connected")
            return False

        if self._controller.set_torque_all(True):
            self._enabled = True
            print("[S101Robot] Motors enabled")
            return True
        return False

    def disable(self) -> None:
        """Disable robot motors (torque off)."""
        self._controller.set_torque_all(False)
        self._enabled = False
        print("[S101Robot] Motors disabled")

    def get_state(self):
        """Get current robot state.

        Returns:
            RobotState with current joint positions and gripper state.
        """
        # Import here to avoid circular dependency
        from src.robot_interface import RobotState

        if not self._connected:
            return RobotState(
                joint_positions=np.zeros(6),
                gripper_position=0.0,
                is_enabled=False,
                timestamp=time.time(),
            )

        # Read motor positions
        raw_positions = self._controller.read_positions_all()
        if raw_positions is None:
            return RobotState(
                joint_positions=self._last_positions.copy(),
                gripper_position=self._last_gripper,
                is_enabled=self._enabled,
                timestamp=time.time(),
            )

        # Convert from steps to radians
        # Center position (2048) = 0 radians
        joint_positions = (raw_positions - self.config.position_center) / self.config.steps_per_radian

        # Extract gripper position and convert to meters
        gripper_angle = joint_positions[5]  # Last joint is gripper
        gripper_position = self._gripper_angle_to_meters(gripper_angle)

        # Store for fallback
        self._last_positions = joint_positions[:5].copy()
        self._last_gripper = gripper_position

        return RobotState(
            joint_positions=joint_positions[:5],  # First 5 joints for arm
            gripper_position=gripper_position,
            is_enabled=self._enabled,
            timestamp=time.time(),
        )

    def set_joint_positions(
        self,
        positions: np.ndarray,
        gripper: Optional[float] = None,
    ) -> bool:
        """Command robot to move to joint positions.

        Args:
            positions: Target joint positions in radians, shape (5,) or (6,).
            gripper: Target gripper position in meters (0-0.08).

        Returns:
            True if command was accepted.
        """
        if not self._enabled:
            return False

        # Build full position array
        if len(positions) == 5:
            # Add gripper position
            if gripper is not None:
                gripper_angle = self._gripper_meters_to_angle(gripper)
            else:
                gripper_angle = 0.0
            full_positions = np.append(positions, gripper_angle)
        else:
            full_positions = positions.copy()
            if gripper is not None:
                full_positions[5] = self._gripper_meters_to_angle(gripper)

        # Convert from radians to steps
        # 0 radians = center position (2048)
        raw_positions = (full_positions * self.config.steps_per_radian +
                        self.config.position_center).astype(int)

        # Clamp to valid range
        raw_positions = np.clip(raw_positions, 0, 4096)

        # Build position dict for sync write
        positions_dict = {
            motor_id: int(raw_positions[i])
            for i, motor_id in enumerate(self.config.motor_ids)
        }

        return self._controller.sync_write_positions(positions_dict)

    def emergency_stop(self) -> None:
        """Emergency stop - immediately disable all motors."""
        print("[S101Robot] EMERGENCY STOP")
        self.disable()

    def go_home(self) -> bool:
        """Move robot to home position."""
        if not self._enabled:
            return False
        return self.set_joint_positions(
            np.array(self.config.home_position[:5]),
            gripper=self.config.gripper_max_m / 2  # Half open
        )

    def _gripper_angle_to_meters(self, angle: float) -> float:
        """Convert gripper joint angle (radians) to opening distance (meters)."""
        # Map gripper angle range to meters range
        angle_min, angle_max = self.config.joint_limits[5]
        angle_normalized = (angle - angle_min) / (angle_max - angle_min)
        return angle_normalized * self.config.gripper_max_m

    def _gripper_meters_to_angle(self, meters: float) -> float:
        """Convert gripper opening distance (meters) to joint angle (radians)."""
        meters = max(0, min(self.config.gripper_max_m, meters))
        angle_min, angle_max = self.config.joint_limits[5]
        angle_normalized = meters / self.config.gripper_max_m
        return angle_min + angle_normalized * (angle_max - angle_min)


class MockS101Robot:
    """Mock S101 robot for testing without hardware."""

    def __init__(self, config: Optional[S101Config] = None):
        self.config = config or S101Config()
        self._connected = False
        self._enabled = False
        self._joint_positions = np.zeros(5)
        self._gripper_position = 0.04  # Half open
        self._target_positions = np.zeros(5)
        self._target_gripper = 0.04

    def connect(self) -> bool:
        print("[MockS101Robot] Connected")
        self._connected = True
        return True

    def disconnect(self) -> None:
        print("[MockS101Robot] Disconnected")
        self._connected = False
        self._enabled = False

    def enable(self) -> bool:
        if not self._connected:
            return False
        print("[MockS101Robot] Motors enabled")
        self._enabled = True
        return True

    def disable(self) -> None:
        print("[MockS101Robot] Motors disabled")
        self._enabled = False

    def get_state(self):
        from src.robot_interface import RobotState

        # Simulate movement towards target
        if self._enabled:
            alpha = 0.1
            self._joint_positions += alpha * (self._target_positions - self._joint_positions)
            self._gripper_position += alpha * (self._target_gripper - self._gripper_position)

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

        self._target_positions = positions[:5].copy()
        if gripper is not None:
            self._target_gripper = gripper

        time.sleep(0.001)  # Simulate latency
        return True

    def emergency_stop(self) -> None:
        print("[MockS101Robot] EMERGENCY STOP")
        self._enabled = False

    def go_home(self) -> bool:
        if not self._enabled:
            return False
        return self.set_joint_positions(np.zeros(5), gripper=0.04)


def create_s101_robot(
    use_mock: bool = True,
    port: str = "/dev/ttyACM0",
    config: Optional[S101Config] = None,
):
    """Factory function to create S101 robot interface.

    Args:
        use_mock: If True, create mock robot for testing.
        port: Serial port for real robot.
        config: Optional configuration override.

    Returns:
        S101Robot or MockS101Robot instance.
    """
    if config is None:
        config = S101Config(port=port)
    else:
        config.port = port

    if use_mock:
        return MockS101Robot(config)
    else:
        return S101Robot(config)
