"""Verify the new workspace bounds are all reachable."""

from src.inverse_kinematics import RobotIK
import numpy as np

ik = RobotIK(urdf_path="urdf/piper_description.urdf")

# New workspace bounds
x_range = (-0.15, 0.15)
y_range = (0.20, 0.35)
z_range = (0.15, 0.30)

print("Testing all corners of new workspace...")
print("=" * 60)

corners = []
for x in x_range:
    for y in y_range:
        for z in z_range:
            corners.append([x, y, z])

all_valid = True
for corner in corners:
    target = np.array(corner)
    result = ik.solve(target, None)
    status = "✓" if result.is_valid else "✗"
    print(f"{status} [{corner[0]:+.2f}, {corner[1]:.2f}, {corner[2]:.2f}] err={result.error:.4f}m")
    if not result.is_valid:
        all_valid = False

print("=" * 60)
if all_valid:
    print("All corners are reachable!")
else:
    print("WARNING: Some corners are NOT reachable!")

# Also test center and some intermediate points
print("\nTesting grid of points...")
valid_count = 0
total_count = 0
for x in np.linspace(x_range[0], x_range[1], 5):
    for y in np.linspace(y_range[0], y_range[1], 5):
        for z in np.linspace(z_range[0], z_range[1], 5):
            target = np.array([x, y, z])
            result = ik.solve(target, None)
            total_count += 1
            if result.is_valid:
                valid_count += 1

print(f"Grid test: {valid_count}/{total_count} points reachable ({100*valid_count/total_count:.1f}%)")
