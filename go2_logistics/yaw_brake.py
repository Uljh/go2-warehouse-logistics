"""Opt-in pre-support yaw damping through measured foot contacts and motors.

No base wrench or state writes. The policy retains its position targets.
Contact estimates and Jacobians currently come from the simulator.
"""
import numpy as np


def allocate_yaw_brake(feet, measured_forces, jacobians, yaw_rate, gain=3., blend=1.):
    feet, measured, jac = (np.asarray(x, dtype=float) for x in (feet, measured_forces, jacobians))
    if feet.shape != (4, 3) or measured.shape != (4, 3) or jac.shape != (4, 3, 3):
        raise ValueError('Expected four feet, contact forces and leg Jacobians')
    if not all(np.isfinite(x).all() for x in (feet, measured, jac, np.array([yaw_rate, gain, blend]))):
        raise ValueError('Non-finite brake input')
    if gain < 0 or not 0 <= blend <= 1:
        raise ValueError('Invalid brake gain or blend')
    requested = float(np.clip(-gain*yaw_rate, -.8, .8))*blend
    # Fit Fx/Fy/Mx/My/Mz simultaneously, using only horizontal contact forces.
    matrix = np.zeros((5, 8))
    for i, (x, y, z) in enumerate(feet):
        matrix[:, 2*i:2*i+2] = [[1, 0], [0, 1], [0, -z], [z, 0], [-y, x]]
    target = np.array([0., 0., 0., 0., requested])
    value = np.linalg.lstsq(matrix, target, rcond=1e-6)[0]
    forces = np.column_stack((value.reshape(4, 2), np.zeros(4)))
    if np.linalg.norm(matrix@value-target) > 1e-6 or np.any(measured[:, 2] <= 5.):
        forces[:] = 0.
    # Budget only unused friction, and at most 15% of measured normal force.
    # Uniform scaling preserves the zero-force/zero-roll/zero-pitch couple.
    budget = np.maximum(0., np.minimum(.15*measured[:, 2],
        .45*measured[:, 2]-np.linalg.norm(measured[:, :2], axis=1)))
    scale = min(1., float(np.min(budget/np.maximum(np.linalg.norm(forces, axis=1), 1e-12))))
    effort = -np.einsum('lij,li->lj', jac, forces)
    scale = min(scale, 1./max(float(np.abs(effort).max()), 1e-12))
    return forces*scale, effort*scale, requested


class PreSupportYawBrake:
    def __init__(self):
        self.blend = 0.

    def apply(self, controller, requested, dt):
        from .balance_recovery import _tensor
        data, torch = controller.robot.data, controller.torch
        linear = _tensor(data.root_lin_vel_w)[0]
        angular = _tensor(data.root_ang_vel_b)[0]
        gravity = float(_tensor(data.projected_gravity_b)[0, 2])
        contacts = _tensor(controller.sensor.data.net_forces_w)[0, controller.contact_ids]
        eligible = bool(requested and controller.weight == 0. and gravity < -.99
            and float(linear.norm()) < .06 and float(angular[:2].norm()) < .15
            and float(angular.norm()) < .5 and bool((contacts[:, 2] > 5.).all()))
        self.blend = min(1., self.blend+dt/.1) if eligible else 0.
        zeros = torch.zeros(12, device=controller.robot.device)
        telemetry = dict(eligible=eligible, blend=self.blend, active=False)
        if not eligible:
            return zeros, telemetry
        feet = (_tensor(data.body_link_pose_w)[0, controller.feet, :3]-_tensor(data.root_pos_w)[0]).cpu().numpy()
        all_jac = _tensor(data.body_link_jacobian_w)[0]
        blocks = np.stack([all_jac[foot-int(controller.robot.is_fixed_base), :3][:, controller.columns[leg]].cpu().numpy()
                           for leg, foot in enumerate(controller.feet)])
        yaw_rate = float(controller.quat_apply(_tensor(data.root_quat_w)[0], angular)[2])
        forces, effort, moment = allocate_yaw_brake(feet, contacts.cpu().numpy(), blocks, yaw_rate, blend=self.blend)
        telemetry.update(active=bool(np.any(effort)), feet_relative_w=feet.tolist(),
            measured_contact_force_w=contacts.cpu().tolist(), jacobians=blocks.tolist(),
            yaw_rate_world=yaw_rate, requested_moment_nm=moment, allocated_force_w=forces.tolist(),
            effort_per_leg_nm=effort.tolist(), actual_moment_nm=np.cross(feet, forces).sum(axis=0).tolist())
        return torch.as_tensor(effort.T.reshape(-1), device=zeros.device, dtype=zeros.dtype), telemetry
