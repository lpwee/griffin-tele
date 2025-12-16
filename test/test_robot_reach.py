"""Find the actual reachable workspace of the Piper arm."""

from src.inverse_kinematics import PiperIK
import numpy as np

ik = PiperIK()

# Calculate theoretical max reach from DH params
# Link lengths from DH: d1=0.123, a3=0.28503, d4=0.25075, d6=0.091
d1 = 0.123
a3 = 0.28503
d4 = 0.25075
d6 = 0.091
theoretical_reach = a3 + d4 + d6
print(f"Theoretical max reach (extended): {theoretical_reach:.3f}m")
print(f"Base height: {d1:.3f}m")

# Sample the workspace to find actual reachable region
print("\nSampling workspace to find reachable region...")

reachable_points = []
unreachable_points = []

# Grid search
for x in np.linspace(-0.5, 0.5, 21):
    for y in np.linspace(0.0, 0.6, 13):
        for z in np.linspace(0.0, 0.5, 11):
            target = np.array([x, y, z])
            result = ik.solve(target, None)
            if result.is_valid:
                reachable_points.append(target)
            else:
                unreachable_points.append(target)

reachable_points = np.array(reachable_points)
print(f"\nReachable points: {len(reachable_points)}")
print(f"Unreachable points: {len(unreachable_points)}")

if len(reachable_points) > 0:
    print(f"\nReachable bounds:")
    print(f"  X: [{reachable_points[:, 0].min():.3f}, {reachable_points[:, 0].max():.3f}]")
    print(f"  Y: [{reachable_points[:, 1].min():.3f}, {reachable_points[:, 1].max():.3f}]")
    print(f"  Z: [{reachable_points[:, 2].min():.3f}, {reachable_points[:, 2].max():.3f}]")

    # Find a safe inner box (positions that are definitely reachable)
    # Check what range works for y=0.25 (middle of forward reach)
    mid_y_points = reachable_points[np.abs(reachable_points[:, 1] - 0.25) < 0.05]
    if len(mid_y_points) > 0:
        print(f"\nAt y≈0.25 (mid reach):")
        print(f"  X: [{mid_y_points[:, 0].min():.3f}, {mid_y_points[:, 0].max():.3f}]")
        print(f"  Z: [{mid_y_points[:, 2].min():.3f}, {mid_y_points[:, 2].max():.3f}]")

# Recommend safe workspace bounds
print("\n" + "=" * 50)
print("RECOMMENDED WORKSPACE CONFIG:")
print("=" * 50)
print("""
robot_x_range: tuple[float, float] = (-0.25, 0.25)  # left/right
robot_y_range: tuple[float, float] = (0.15, 0.40)   # forward
robot_z_range: tuple[float, float] = (0.05, 0.35)   # up/down
""")
