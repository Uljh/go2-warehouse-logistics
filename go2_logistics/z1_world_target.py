"""World-frame Z1 targets compensated with the measured floating mount pose.

This adapter only returns bounded motor position requests. It neither fixes
the robot base nor changes a payload or rigid-body state.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation


def link_transform(position, quaternion_xyzw):
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(np.r_[position, quaternion]).all()
            or abs(np.linalg.norm(quaternion)-1.) > 1e-3):
        raise ValueError('Measured link pose must have a finite position and unit XYZW quaternion')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def validate_transform(transform):
    value = np.asarray(transform, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-8)
            or not np.allclose(value[:3, :3].T@value[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Arm target and measured mount must be rigid transforms')
    return value


def measured_world_arm_target(kinematics, world_target, mount_world, previous, dt):
    """Apply inverse measured mount transform, original-limit IK and slew.

    Residuals describe the rate-limited request, not the physical tracking
    error. The caller must measure the actual gripper link separately.
    """
    goal = validate_transform(world_target)
    mount = validate_transform(mount_world)
    previous = np.asarray(previous, dtype=float)
    if (previous.shape != (6,) or not np.isfinite(previous).all()
            or not math.isfinite(dt) or not 0 < dt <= .02):
        raise ValueError('Invalid previous Z1 command or control period')
    local_goal = np.linalg.solve(mount, goal)
    # Resolve the 10 um contact-admittance increments instead of accumulating
    # them into 100 um command jumps at the former numerical deadband.
    candidate, info = kinematics.inverse_kinematics(local_goal, previous, max_iters=80,
        position_tolerance=1e-5, orientation_tolerance=5e-4)
    if not info['success']:
        raise RuntimeError(f'Measured-base Z1 IK rejected unreachable target: {info}')
    lower = np.maximum(kinematics.limits[:, 0], previous-math.pi*dt)
    upper = np.minimum(kinematics.limits[:, 1], previous+math.pi*dt)
    if np.any(lower > upper):
        raise RuntimeError('Previous arm command is outside original position/rate feasibility')
    command = np.clip(candidate, lower, upper)
    requested_world = mount@kinematics.fk(command)
    return command, dict(
        maximum_target_step_rad=float(np.abs(command-previous).max()),
        request_position_residual_m=float(np.linalg.norm(requested_world[:3, 3]-goal[:3, 3])),
        request_orientation_residual_rad=float(Rotation.from_matrix(
            requested_world[:3, :3]@goal[:3, :3].T).magnitude()),
        rate_limited=bool(np.max(np.abs(command-candidate)) > 1e-9),
        local_target=local_goal.tolist())
