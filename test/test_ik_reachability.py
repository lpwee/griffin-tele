"""Test IK reachability for various positions."""

from src.inverse_kinematics import RobotIK
import numpy as np

ik = RobotIK(urdf_path="urdf/piper_description.urdf", verbose=True)

# Test various target positions to see which ones fail
test_positions = [
    ([0.2, 0.3, 0.2], 'typical front'),
    ([0.0, 0.3, 0.3], 'center front high'),
    ([0.3, 0.1, 0.1], 'right side low'),
    ([0.0, 0.5, 0.2], 'far front'),
    ([0.0, 0.2, 0.5], 'high up'),
    ([0.4, 0.4, 0.1], 'far corner'),
    ([-0.2, 0.3, 0.2], 'left side'),
]

print('Testing IK reachability...')
print('=' * 70)

for pos, desc in test_positions:
    target = np.array(pos)
    # Test position-only first
    result = ik.solve(target, None)
    print(f'{desc:20s} pos={pos} -> valid={result.is_valid}, err={result.error:.4f}m')

print()
print('=' * 70)
print('Testing with orientation...')

# Test with orientation
for pos, desc in test_positions[:3]:
    target = np.array(pos)
    orient = np.array([0.0, 0.0, 0.0])
    result = ik.solve(target, orient)
    print(f'{desc:20s} -> valid={result.is_valid}, err={result.error:.4f}m')
