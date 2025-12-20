"""Debug IK issues."""

from src.inverse_kinematics import RobotIK
import numpy as np

ik = RobotIK(urdf_path="urdf/piper_description.urdf", verbose=True)

# These simple positions should definitely be reachable
test_cases = [
    [0.0, 0.25, 0.20],   # dead center
    [0.0, 0.20, 0.15],   # close and low
    [-0.20, 0.15, 0.10], # failing corner
    [0.10, 0.20, 0.15],  # slight offset
]

print("Debugging IK solutions...")
print("=" * 70)

for pos in test_cases:
    target = np.array(pos)
    print(f"\nTarget: {pos}")

    # Try position-only IK
    result = ik.solve(target, None)
    print(f"  Position-only: valid={result.is_valid}, err={result.error:.4f}m")
    print(f"  Angles (deg): {np.degrees(result.angles).round(1)}")

    # Verify with FK
    fk_pos = ik.forward_kinematics(result.angles)
    print(f"  FK result: {fk_pos.round(4)}")
    print(f"  Actual error: {np.linalg.norm(fk_pos - target):.4f}m")

# Check the chain's home position
print("\n" + "=" * 70)
print("Home position (all zeros):")
home_pos = ik.forward_kinematics(np.zeros(6))
print(f"  FK at zeros: {home_pos}")

# Check extended arm position
print("\nExtended arm (joint2=-90deg):")
extended = np.array([0, -np.pi/2, 0, 0, 0, 0])
ext_pos = ik.forward_kinematics(extended)
print(f"  FK: {ext_pos}")
