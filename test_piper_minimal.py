import time
import math
from piper_sdk import C_PiperInterface

def main():
    # Configuration
    CAN_PORT = 'piper_can'  # Change if your CAN port is different
    
    print("Creating Piper interface...")
    piper = C_PiperInterface(can_name=CAN_PORT)
    
    print(f"Connecting to CAN port: {CAN_PORT}")
    piper.ConnectPort()
    
    if not piper.isOk():
        print("ERROR: Failed to connect to CAN port")
        return
    
    print("Connection successful!")
    
    # Enable the arm
    print("Enabling arm...")
    piper.EnableArm(7)
    time.sleep(1.0)
    
    # Check enable status
    print("Checking enable status...")
    enable_status = (
        piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
        piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
        piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
        piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
        piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
        piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
    )
    print(f"Arm enabled: {enable_status}")
    
    if not enable_status:
        print("WARNING: Arm may not be fully enabled")
    
    # Initialize gripper
    print("Initializing gripper...")
    piper.GripperCtrl(0, 1000, 0x02, 0)  # Reset/initialize mode
    piper.GripperCtrl(0, 1000, 0x01, 0)  # Normal control mode
    
    # Set motion control mode (velocity control, speed 100)
    print("Setting motion control mode...")
    piper.MotionCtrl_2(0x01, 0x01, 100)
    
    # Convert radians to degrees * 1000 (format expected by JointCtrl)
    # Factor: 1000 * 180 / π ≈ 57324.840764
    def rad_to_joint_value(rad):
        return int(round(rad * 1000 * 180 / math.pi))
    
    # Test joint positions (in radians)
    # Set all joints to 0 degrees (home position)
    print("\nSetting joints to home position (0 degrees)...")
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(2.0)
    
    # Move to a test position (small angles in radians)
    print("Moving to test position...")
    test_joints = [
        math.radians(10),   # joint1: 10 degrees
        math.radians(-10),  # joint2: -10 degrees
        math.radians(5),    # joint3: 5 degrees
        math.radians(0),    # joint4: 0 degrees
        math.radians(0),    # joint5: 0 degrees
        math.radians(0),    # joint6: 0 degrees
    ]
    
    joint_values = [rad_to_joint_value(angle) for angle in test_joints]
    print(f"Joint values (degrees*1000): {joint_values}")
    piper.JointCtrl(*joint_values)
    time.sleep(2.0)
    
    # Return to home
    print("Returning to home position...")
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(2.0)
    
    # Test gripper open/close
    print("\n=== Gripper Test ===")
    
    # Close gripper (position 0)
    print("Closing gripper...")
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(2.0)
    
    # Open gripper (position 80000 = max open)
    print("Opening gripper (max)...")
    piper.GripperCtrl(80000, 1000, 0x01, 0)
    time.sleep(2.0)
    
    # Partially open (50% = 40000)
    print("Partially opening gripper (50%)...")
    piper.GripperCtrl(40000, 1000, 0x01, 0)
    time.sleep(2.0)
    
    # Close again
    print("Closing gripper again...")
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(2.0)
    
    print("Gripper test complete!")
    
    print("\nTest complete!")
    print("Press Ctrl+C to exit, or the script will continue running...")
    
    # Keep running to maintain control
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Disable gripper
        piper.GripperCtrl(0, 1000, 0x02, 0)
        # Disable arm
        piper.DisableArm(7)
        print("Arm and gripper disabled. Exiting.")

if __name__ == "__main__":
    main()