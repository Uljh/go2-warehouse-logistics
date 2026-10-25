"""Experimental contact-force support during bounded stance recovery.

The floating base is never written or externally forced. Ground reaction
requests are converted to leg motor feedforward torques through measured
Jacobians. This uses simulator state/contact estimates, not a trained policy.
"""
from dataclasses import dataclass
import math

import numpy as np


class SupportAllocationError(RuntimeError):
    """Keep the rejected optimization problem; never send an unchecked effort."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.support_allocation_diagnostics = diagnostics


@dataclass(frozen=True)
class BalanceConfig:
    height_m: float = .33
    height_kp: float = 450.
    height_kd: float = 60.
    tilt_kp: float = 35.
    tilt_kd: float = 4.
    friction: float = .45
    torque_limit_nm: float = 18.
    joint_rate_radps: float = 1.2
    enter_time_s: float = .20
    exit_time_s: float = .25
    allocator: str = 'sequential'
    stationary_payload_qp: bool = False
    loaded_recovery_qp: bool = False
    total_motor_qp: bool = False
    joint_motor_qp: bool = False
    handoff_mode: str = 'legacy'
    exit_joint_rate_radps: float = 4.
    handoff_release_seconds: float = 0.
    recenter_foot: bool = False
    offload_foot: bool = False
    measured_com: bool = False
    leg_ik: str = 'clipped'
    stance_fit: bool = False
    target_update: str = 'legacy'
    touchdown_ik: bool = False
    airborne_touchdown: bool = False
    workspace_reference: bool = False
    workspace_settled_entry: bool = False
    workspace_entry_max_height_m: float | None = None


def support_scope_requested(recovering, stationary_balance, scope='all', settled_navigation=False):
    """Select support without bypassing the controller's effort fade-out."""
    if scope not in ('all','stationary','settled','settledall'):
        raise ValueError('Unknown balance support scope')
    return bool(stationary_balance or recovering and (scope=='all' or scope in ('settled','settledall') and settled_navigation))


def workspace_scope_requested(recovering, payload_held, phase):
    """Limit the candidate to the empty return phase actually diagnosed."""
    return bool(recovering and not payload_held and phase=='returning')


def workspace_entry_settled(height,gravity_z,speed,angular_speed,contacts):
    """Select the low, planted stance being diagnosed, not braking at full height."""
    values=(height,gravity_z,speed,angular_speed)
    return bool(all(math.isfinite(value) for value in values)
        and height<.305 and gravity_z<-.985 and speed<.12 and angular_speed<.5
        and len(contacts)==4 and all(contacts))


class SettledPayloadSupport:
    """Latch support after 100 ms of measured stationary, level recovery.

    Once lifting, velocity can legitimately rise: do not chatter at the entry
    velocity threshold. Losing three-foot support or level attitude releases
    the latch through the existing controller fade. No recovery time is added.
    """
    def __init__(self):
        self.active=False
        self.stable_time_s=0.
        self.eligible=False

    def update(self,recovering_payload,linear,angular,gravity_z,normals,dt):
        geometry_ok=(recovering_payload and gravity_z<-.99 and sum(value>1. for value in normals)>=3)
        self.eligible=bool(geometry_ok and math.sqrt(sum(value*value for value in linear))<.06
                           and math.sqrt(sum(value*value for value in angular))<.15)
        if not geometry_ok:
            self.active=False
            self.stable_time_s=0.
        elif not self.active:
            self.stable_time_s=self.stable_time_s+dt if self.eligible else 0.
            self.active=self.stable_time_s>=.10-1e-9
        return self.active


def recenter_trajectory(start, target, support_shift, elapsed):
    """Bounded shift/lift/transfer/lower/settle trajectory for one swing foot.

    Position references only. Real contact forces and actuator limits still
    determine whether the physical robot can execute this candidate step.
    """
    start, target, shift = (np.asarray(value, dtype=float) for value in (start, target, support_shift))
    if start.shape != (3,) or target.shape != (3,) or shift.shape != (2,):
        raise ValueError('Invalid recenter geometry')
    if not all(np.isfinite(v).all() for v in (start,target,shift)) or not math.isfinite(elapsed):
        raise ValueError('Non-finite recenter trajectory')
    def smooth(value):
        value=np.clip(value,0.,1.)
        return value*value*(3.-2.*value)
    foot=start.copy()
    if elapsed<.30:
        phase='shift'
        body=shift*smooth(elapsed/.30)
    elif elapsed<.48:
        phase='lift'
        body=shift
        foot[2]+=.045*smooth((elapsed-.30)/.18)
    elif elapsed<.76:
        phase='transfer'
        body=shift
        foot=start+(target-start)*smooth((elapsed-.48)/.28)
        foot[2]=max(start[2],target[2])+.045
    elif elapsed<.96:
        phase='lower'
        body=shift
        foot=target.copy()
        foot[2]+=.045*(1.-smooth((elapsed-.76)/.20))
    else:
        phase='settle' if elapsed<1.12 else 'complete'
        foot=target.copy()
        # Keep the physically shifted stance. Pulling the root back in 280ms
        # excited >.3m/s motion in the first trial and exhausted its timeout.
        body=shift.copy()
    return foot,body,phase


def support_triangle_contains(point, vertices, margin=.005):
    """Check measured root projection against the three remaining feet."""
    point=np.asarray(point,dtype=float)
    vertices=np.asarray(vertices,dtype=float)
    if point.shape!=(2,) or vertices.shape!=(3,2) or not np.isfinite(vertices).all() or not np.isfinite(point).all():
        return False
    edges=np.roll(vertices,-1,axis=0)-vertices
    signed=edges[:,0]*(point[1]-vertices[:,1])-edges[:,1]*(point[0]-vertices[:,0])
    lengths=np.linalg.norm(edges,axis=1)
    if min(lengths)<1e-6:
        return False
    distances=signed/lengths
    return bool(np.all(distances>=margin) or np.all(distances<=-margin))


def mass_weighted_center(positions, masses, payload=None):
    positions,masses=np.asarray(positions,dtype=float),np.asarray(masses,dtype=float)
    if positions.shape!=(len(masses),3) or not np.isfinite(positions).all() or not np.isfinite(masses).all() or np.any(masses<0) or masses.sum()<=0:
        raise ValueError('Invalid measured body masses or centers')
    total=float(masses.sum());weighted=(positions*masses[:,None]).sum(axis=0)
    if payload is not None:
        point,mass=payload
        point=np.asarray(point,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all() or not math.isfinite(mass) or mass<0:
            raise ValueError('Invalid held payload center or mass')
        weighted+=point*mass;total+=mass
    return weighted/total,total


def minimal_support_shift(point, vertices, margin=.015, max_shift=.12):
    """Move only as far toward the triangle interior as support requires.

    Binary search along the root-to-centroid line; this is a conservative
    feasible shift, not a global distance optimum or full CoM estimator.
    """
    point,vertices=np.asarray(point,dtype=float),np.asarray(vertices,dtype=float)
    if support_triangle_contains(point,vertices,margin):
        return np.zeros(2)
    center=vertices.mean(axis=0)
    if not support_triangle_contains(center,vertices,margin):
        return None
    lo,hi=0.,1.
    for _ in range(30):
        mid=(lo+hi)/2
        if support_triangle_contains(point+mid*(center-point),vertices,margin):
            hi=mid
        else:
            lo=mid
    shift=hi*(center-point)
    return shift if np.linalg.norm(shift)<=max_shift else None


def allocate_support_qp(feet_relative, contacts, vertical_force, moment_xy, horizontal_force,
                        *, friction=.45, per_foot_limit=150., jacobians=None, torque_limit=18.,
                        motor_bias=None, motor_lower=None, motor_upper=None, effort_scale=1.):
    """Jointly fit support wrench within contact, friction and motor bounds.

    Attitude and vertical support take priority over planar braking. Unlike
    sequential allocation, planar forces may change when their tipping moment
    makes the desired attitude wrench infeasible. A conservative L1 friction
    pyramid implies the circular Coulomb bound. This predicts requested motor
    feedforward only: actual PD + motor saturation remains measured separately.
    """
    from scipy.optimize import LinearConstraint, minimize
    feet = np.asarray(feet_relative, dtype=float)
    mask = np.asarray(contacts, dtype=bool)
    target = np.asarray([*horizontal_force, vertical_force, *moment_xy], dtype=float)
    if feet.shape != (4, 3) or mask.shape != (4,) or target.shape != (5,):
        raise ValueError('Expected four feet and a finite support wrench')
    if not np.isfinite(feet).all() or not np.isfinite(target).all() or not 0 < friction <= 1 or per_foot_limit <= 0:
        raise ValueError('Invalid support input')
    active = np.flatnonzero(mask)
    forces = np.zeros((4, 3))
    if not len(active):
        return forces, target[2:]
    count = len(active)
    wrench = np.zeros((5, count*3))
    inequalities = []
    lower, upper = [], []
    jac = None if jacobians is None else np.asarray(jacobians, dtype=float)
    if jac is not None and (jac.shape != (4, 3, 3) or not np.isfinite(jac).all() or torque_limit <= 0):
        raise ValueError('Invalid leg dynamics')
    motor_bounded=motor_bias is not None
    if motor_bounded:
        bias,lo,hi=(np.asarray(v,dtype=float) for v in (motor_bias,motor_lower,motor_upper))
        if (jac is None or any(v.shape!=(4,3) or not np.isfinite(v).all() for v in (bias,lo,hi))
                or np.any(lo>hi) or not math.isfinite(effort_scale) or not 0<effort_scale<=1.):
            raise ValueError('Invalid total motor envelope')
    for slot, foot in enumerate(active):
        xyz = slice(slot*3, slot*3+3)
        x, y, z = feet[foot]
        wrench[:3, xyz] = np.eye(3)
        wrench[3:, xyz] = [[0., -z, y], [z, 0., -x]]
        for sx, sy in ((1.,1.),(1.,-1.),(-1.,1.),(-1.,-1.)):
            row = np.zeros(count*3)
            row[xyz] = [sx, sy, -friction]
            inequalities.append(row)
            lower.append(-np.inf)
            upper.append(0.)
        if jac is not None:
            for axis,motor in enumerate(-jac[foot].T):
                row = np.zeros(count*3)
                row[xyz] = motor
                inequalities.append(row)
                lower.append(-torque_limit)
                upper.append(torque_limit)
                if motor_bounded:
                    inequalities.append(row*effort_scale)
                    lower.append(lo[foot,axis]-bias[foot,axis])
                    upper.append(hi[foot,axis]-bias[foot,axis])
    weights = np.array([.2, .2, 1., 8., 8.])
    matrix = weights[:, None] * wrench
    desired = weights * target
    regularization = 1e-4
    hessian = matrix.T @ matrix + regularization * np.eye(count*3)
    linear = matrix.T @ desired
    constraints = LinearConstraint(np.asarray(inequalities), lower, upper)
    bounds = [(-per_foot_limit, per_foot_limit), (-per_foot_limit, per_foot_limit), (0., per_foot_limit)] * count
    result = minimize(lambda value: .5*value@hessian@value-linear@value,
        np.zeros(count*3), jac=lambda value:hessian@value-linear, method='SLSQP',
        bounds=bounds, constraints=constraints, options=dict(maxiter=80, ftol=1e-7))
    value = result.x
    feasible = (np.isfinite(value).all()
        and np.all(np.asarray(inequalities)@value <= np.asarray(upper)+1e-5)
        and np.all(np.asarray(inequalities)@value >= np.asarray(lower)-1e-5))
    if not result.success or not feasible:
        # A separate feasibility-only LP distinguishes an empty constraint set
        # from an optimizer/conditioning failure. It is diagnostic only: even
        # a feasible LP never substitutes a force command after a QP failure.
        from scipy.optimize import linprog
        rows = np.asarray(inequalities)
        lo_rows, hi_rows = np.asarray(lower), np.asarray(upper)
        finite_lo, finite_hi = np.isfinite(lo_rows), np.isfinite(hi_rows)
        feasibility = linprog(np.zeros(count*3),
            A_ub=np.r_[rows[finite_hi], -rows[finite_lo]],
            b_ub=np.r_[hi_rows[finite_hi], -lo_rows[finite_lo]],
            bounds=bounds, method='highs')
        diagnostics = dict(
            feet_relative=feet.tolist(), contacts=mask.tolist(),
            vertical_force_n=float(vertical_force), moment_xy_nm=list(map(float, moment_xy)),
            horizontal_force_n=list(map(float, horizontal_force)), friction=float(friction),
            per_foot_limit_n=float(per_foot_limit), jacobians=None if jac is None else jac.tolist(),
            feedforward_limit_nm=float(torque_limit), effort_scale=float(effort_scale),
            motor_bias_nm=bias.tolist() if motor_bounded else None,
            motor_lower_nm=lo.tolist() if motor_bounded else None,
            motor_upper_nm=hi.tolist() if motor_bounded else None,
            qp_status=int(result.status), qp_message=str(result.message),
            qp_solution=[float(x) if math.isfinite(x) else None for x in value],
            qp_solution_feasible=bool(feasible),
            feasibility_lp_status=int(feasibility.status),
            feasibility_lp_message=str(feasibility.message),
            constraint_set_feasible=True if feasibility.status == 0 else False if feasibility.status == 2 else None,
            command_sent=False,
            scope='Rejected pre-step force request only; diagnostic LP never sends motor commands')
        raise SupportAllocationError(f'Constrained support allocation failed: {result.message}', diagnostics)
    forces[active] = value.reshape(count, 3)
    return forces, (target-wrench@value)[2:]


def pd_motor_envelope(stiffness,damping,effort_limit,velocity_limit,saturation_effort,
                      position,velocity,position_target,velocity_target):
    """Predict the native explicit PD/DC motor equation without changing its limits."""
    arrays=[np.asarray(v,dtype=float) for v in (stiffness,damping,effort_limit,velocity_limit,
        saturation_effort,position,velocity,position_target,velocity_target)]
    if any(v.shape!=(12,) or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Motor envelope requires twelve finite measured leg channels')
    kp,kd,cap,speed,stall,q,qd,goal,qdgoal=arrays
    if np.any(kp<0) or np.any(kd<0) or np.any(cap<=0) or np.any(speed<=0) or np.any(stall<=0):
        raise ValueError('Invalid original motor parameters')
    # This is the same speed clipping used by Isaac Lab DCMotor._clip_effort.
    measured=np.clip(qd,-speed*(1.+cap/stall),speed*(1.+cap/stall))
    lower=np.maximum(stall*(-1.-measured/speed),-cap)
    upper=np.minimum(stall*(1.-measured/speed),cap)
    bias=kp*(goal-q)+kd*(qdgoal-qd)
    return bias,lower,upper


def measured_motor_envelope(robot,leg_ids,position_target):
    """Read explicit DC actuator gains/caps in policy leg order, never PhysX zero PD gains."""
    slots={int(j):i for i,j in enumerate(leg_ids)}
    names=('stiffness','damping','effort_limit','velocity_limit')
    parameters={key:np.full(12,np.nan) for key in (*names,'saturation_effort')}
    for actuator in robot.actuators.values():
        indices=actuator.joint_indices
        if isinstance(indices,slice):indices=list(range(robot.num_joints))[indices]
        ids=[int(j) for j in indices]
        selected=[(i,slots[j]) for i,j in enumerate(ids) if j in slots]
        if not selected:continue
        if actuator.is_implicit_model or getattr(actuator.cfg,'saturation_effort',None) is None:
            raise ValueError('Total motor QP currently requires the original explicit DC leg actuators')
        for key in names:
            values=_tensor(getattr(actuator,key))[0].detach().cpu().numpy()
            for index,slot in selected:parameters[key][slot]=values[index]
        for _,slot in selected:parameters['saturation_effort'][slot]=float(actuator.cfg.saturation_effort)
    data=robot.data
    q=_tensor(data.joint_pos)[0,leg_ids].detach().cpu().numpy()
    qd=_tensor(data.joint_vel)[0,leg_ids].detach().cpu().numpy()
    qdgoal=_tensor(data.joint_vel_target)[0,leg_ids].detach().cpu().numpy()
    goal=position_target.detach().cpu().numpy()
    bias,lower,upper=pd_motor_envelope(*(parameters[key] for key in (*names,'saturation_effort')),
        q,qd,goal,qdgoal)
    return bias,lower,upper,dict(**{k:v.tolist() for k,v in parameters.items()},
        position_rad=q.tolist(),velocity_radps=qd.tolist(),position_target_rad=goal.tolist(),
        velocity_target_radps=qdgoal.tolist(),pd_bias_nm=bias.tolist(),
        lower_nm=lower.tolist(),upper_nm=upper.tolist())


def allocate_support(feet_relative, contacts, vertical_force, moment_xy, horizontal_force,
                     *, friction=.45, per_foot_limit=150.):
    """Nonnegative vertical-force fit, then friction-limited planar braking.

    Moment rows are scaled by a characteristic body length. Only measured
    contacts receive force. An infeasible wrench remains a residual; it is
    never satisfied by negative contact forces or a hidden base wrench.
    """
    from scipy.optimize import lsq_linear
    feet = np.asarray(feet_relative, dtype=float)
    mask = np.asarray(contacts, dtype=bool)
    target = np.asarray([vertical_force, *moment_xy], dtype=float)
    planar = np.asarray(horizontal_force, dtype=float)
    if feet.shape != (4, 3) or mask.shape != (4,) or target.shape != (3,) or planar.shape != (2,):
        raise ValueError('Expected four feet and a finite support wrench')
    if not all(np.isfinite(x).all() for x in (feet, target, planar)) or not 0 < friction <= 1 or per_foot_limit <= 0:
        raise ValueError('Invalid support input')
    forces = np.zeros((4, 3))
    active = np.flatnonzero(mask)
    if not len(active):
        return forces, target
    matrix = np.array([np.ones(len(active)), feet[active, 1], -feet[active, 0]])
    weights = np.array([1., 1. / .25, 1. / .25])
    planar_moment = np.zeros(2)
    for _ in range(3):
        adjusted = target - np.array([0., *planar_moment])
        solution = lsq_linear(weights[:, None] * matrix, weights * adjusted,
                              bounds=(0., per_foot_limit), method='bvls')
        if not solution.success or not np.isfinite(solution.x).all():
            raise RuntimeError('Contact-force allocation failed')
        normal = solution.x
        forces[active, 2] = normal
        total = normal.sum()
        if total > 1e-9:
            requested = planar[None, :] * normal[:, None] / total
            lengths = np.linalg.norm(requested, axis=1)
            factors = np.minimum(1., friction * normal / np.maximum(lengths, 1e-9))
            forces[active, :2] = requested * factors[:, None]
        horizontal_only = forces.copy()
        horizontal_only[:, 2] = 0.
        planar_moment = np.cross(feet, horizontal_only).sum(axis=0)[:2]
    actual = np.array([forces[:, 2].sum(), *np.cross(feet, forces).sum(axis=0)[:2]])
    return forces, target - actual


def leg_feedforward(jacobians, forces, torque_limit):
    """Oppose the upward ground reaction with actuator torque: -J.T @ f."""
    jac = np.asarray(jacobians, dtype=float)
    force = np.asarray(forces, dtype=float)
    if jac.shape != (4, 3, 3) or force.shape != (4, 3) or torque_limit <= 0:
        raise ValueError('Invalid leg force/Jacobian dimensions')
    if not np.isfinite(jac).all() or not np.isfinite(force).all():
        raise ValueError('Non-finite leg dynamics')
    return np.clip(-np.einsum('lij,li->lj', jac, force), -torque_limit, torque_limit)


def _tensor(value):
    return getattr(value, 'torch', value)


class ContactBalanceRecovery:
    def __init__(self, robot, sensor, leg_ids, action_scale, *, payload_mass=0., payload_state=None, config=None, stance_geometry=None, stance_clearance_check=None):
        import torch
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        self.torch, self.quat_apply, self.quat_inverse = torch, quat_apply, quat_apply_inverse
        self.robot, self.sensor = robot, sensor
        self.leg_ids, self.scale = leg_ids, action_scale
        self.config = config or BalanceConfig()
        if self.config.workspace_entry_max_height_m is not None and (
                not self.config.workspace_reference or
                not .22 < self.config.workspace_entry_max_height_m <= .305):
            raise ValueError('Low-stance workspace entry requires a height in (.22,.305] and workspace_reference')
        if self.config.allocator not in ('sequential', 'wrench_qp'):
            raise ValueError('Unknown balance allocator')
        if not isinstance(self.config.stationary_payload_qp,bool):
            raise ValueError('Stationary payload QP must be an explicit boolean')
        if not isinstance(self.config.loaded_recovery_qp,bool):
            raise ValueError('Loaded recovery QP must be an explicit boolean')
        if (not isinstance(self.config.total_motor_qp,bool)
                or self.config.total_motor_qp and not self.config.loaded_recovery_qp):
            raise ValueError('Total motor QP requires explicit loaded recovery QP')
        if not isinstance(self.config.joint_motor_qp,bool) or self.config.joint_motor_qp and not self.config.total_motor_qp:
            raise ValueError('Joint motor QP requires explicit total motor QP')
        if (not isinstance(self.config.airborne_touchdown,bool)
                or self.config.airborne_touchdown and not self.config.touchdown_ik):
            raise ValueError('Airborne touchdown requires an explicit boolean and touchdown IK')
        if self.config.handoff_mode not in ('legacy', 'settled_slew'):
            raise ValueError('Unknown recovery handoff mode')
        if (not math.isfinite(self.config.handoff_release_seconds)
                or self.config.handoff_release_seconds != 0. and not .25 <= self.config.handoff_release_seconds <= 2.
                or self.config.handoff_release_seconds and self.config.handoff_mode != 'settled_slew'):
            raise ValueError('Finite handoff release requires settled_slew and zero or .25..2 seconds')
        if self.config.leg_ik not in ('clipped','bounded'):
            raise ValueError('Unknown recovery leg IK mode')
        if self.config.target_update not in ('legacy','project_then_slew'):
            raise ValueError('Unknown recovery target update mode')
        if self.config.offload_foot and not self.config.recenter_foot:
            raise ValueError('Active foot unloading requires recenter_foot')
        if self.config.touchdown_ik and (self.config.recenter_foot or self.config.stance_fit):
            raise ValueError('Touchdown IK is a separate experiment from recentering and stance fitting')
        if self.config.stance_fit and (self.config.recenter_foot or stance_geometry is None):
            raise ValueError('Stance fitting requires leg geometry and cannot be combined with foot recentering')
        if self.config.workspace_reference and (stance_geometry is None or stance_clearance_check is None
                or self.config.stance_fit or self.config.recenter_foot or self.config.touchdown_ik):
            raise ValueError('Workspace reference requires geometry, sensed clearance and a separate experiment')
        self.stance_geometry=stance_geometry
        self.stance_clearance_check=stance_clearance_check
        self.stance_fit_plan=None
        self.stance_fit_active=False
        self.stance_fit_attempted=False
        self.stance_fit_started=0.
        self.stance_fit_anchors=None
        self.workspace_plan=None
        self.workspace_anchors=None
        self.workspace_active=False
        self.workspace_attempted=False
        names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.feet, _ = robot.find_bodies(names, preserve_order=True)
        self.contact_ids, _ = sensor.find_bodies(names, preserve_order=True)
        if len(self.feet) != 4 or len(self.contact_ids) != 4:
            raise RuntimeError('Force balance requires all four measured foot contacts')
        self.columns = [[leg_ids[leg + 4 * joint] + robot.num_base_dofs for joint in range(3)] for leg in range(4)]
        masses = _tensor(robot.data.default_mass)[0]
        self.mass = float(masses.sum()) + payload_mass
        self.payload_state=payload_state
        self.active = False
        self.weight = 0.
        self.target = self.origin = None
        self.yaw_quat = None
        self.ground_z = None
        self.activations = 0
        self.telemetry = {}
        if self.config.total_motor_qp:
            # Fail at initialization if this backend cannot expose the actual
            # DC motor model, before spending a full pickup on the candidate.
            measured_motor_envelope(robot,leg_ids,_tensor(robot.data.joint_pos)[0,leg_ids])
        self.handoff_active = False
        self.nominal_feet_local = None
        self.recenter = None
        self.recenter_active = False
        self.recenter_attempted = False
        self.recovery_elapsed = 0.
        self.recenter_count = 0
        self.recenter_completed = 0
        self.recenter_aborted = 0
        self.recenter_guarded = 0

    def check_stance_geometry(self):
        """Verify authored leg frames against the live articulation before use."""
        from scipy.spatial.transform import Rotation
        data=self.robot.data
        q=_tensor(data.joint_pos)[0,self.leg_ids].cpu().numpy()
        rotation=Rotation.from_quat(_tensor(data.root_quat_w)[0].cpu().numpy()).as_matrix()
        root=_tensor(data.root_pos_w)[0].cpu().numpy()
        measured=_tensor(data.body_link_pose_w)[0,self.feet,:3].cpu().numpy()
        predicted=self.stance_geometry.feet(q)@rotation.T+root
        position_error=float(np.linalg.norm(predicted-measured,axis=1).max())
        all_jac=_tensor(data.body_link_jacobian_w)[0]
        jacobian_error=0.
        for leg,chain in enumerate(self.stance_geometry.legs):
            _,jac=chain.forward(q[[leg,leg+4,leg+8]],with_jacobian=True)
            measured_jac=all_jac[self.feet[leg]-int(self.robot.is_fixed_base),:3][:,self.columns[leg]].cpu().numpy()
            jacobian_error=max(jacobian_error,float(np.abs(rotation@jac[:3]-measured_jac).max()))
        if position_error>.002 or jacobian_error>.005:
            raise RuntimeError(f'Stance FK/live mismatch: position={position_error}, jacobian={jacobian_error}')
        return dict(position_m=position_error,jacobian_max_error=jacobian_error)

    def apply(self, recovering, fallback, last_action, reference_action, dt, *, workspace_eligible=True,
              stationary_payload=False, loaded_recovery=False):
        torch, cfg, data = self.torch, self.config, self.robot.data
        zeros = torch.zeros(12, device=self.robot.device)
        if reference_action is None:
            self.telemetry = dict(active=False, reason='reference_not_ready')
            return fallback, zeros
        if self.ground_z is None:
            self.ground_z = float(_tensor(data.body_link_pose_w)[0, self.feet, 2].median())
        if cfg.recenter_foot and self.nominal_feet_local is None:
            self.nominal_feet_local = self.quat_inverse(
                _tensor(data.root_quat_w)[0].expand(4,4),
                _tensor(data.body_link_pose_w)[0,self.feet,:3]-_tensor(data.root_pos_w)[0]).clone()
        if recovering and not self.active:
            self.target = (_tensor(data.default_joint_pos)[0, self.leg_ids] + self.scale * last_action).clone()
            self.origin = _tensor(data.root_pos_w)[0].clone()
            x, y, z, w = _tensor(data.root_quat_w)[0].tolist()
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            self.yaw_quat = torch.tensor([0., 0., math.sin(yaw/2), math.cos(yaw/2)], device=self.robot.device)
            self.activations += 1
            self.handoff_active = True
            self.handoff_release_elapsed_s = 0.
            self.recenter = None
            self.recenter_active = False
            self.recenter_attempted = False
            self.recovery_elapsed = 0.
            self.stance_fit_plan=None
            self.stance_fit_active=False
            self.stance_fit_attempted=False
            self.stance_fit_anchors=None
            self.workspace_plan=None
            self.workspace_anchors=None
            self.workspace_active=False
            self.workspace_attempted=False
        self.active = recovering
        if recovering:
            self.recovery_elapsed += dt
        self.weight = min(1., self.weight + dt / cfg.enter_time_s) if recovering else max(0., self.weight - dt / cfg.exit_time_s)
        if self.weight <= 0:
            if cfg.handoff_mode == 'settled_slew' and self.handoff_active:
                step = cfg.exit_joint_rate_radps * dt / self.scale
                limited = last_action + (fallback - last_action).clamp(-step, step)
                fraction = 0.
                if cfg.handoff_release_seconds:
                    self.handoff_release_elapsed_s = getattr(self, 'handoff_release_elapsed_s', 0.) + dt
                    fraction = min(1., self.handoff_release_elapsed_s / cfg.handoff_release_seconds)
                    fraction = fraction*fraction*(3.-2.*fraction)
                action = (1.-fraction)*limited + fraction*fallback
                self.handoff_active = bool((fallback - action).abs().max() > 1e-6)
                self.telemetry = dict(active=False, blend=0., handoff_active=self.handoff_active,
                    handoff_mode=cfg.handoff_mode,
                    handoff_release_fraction=fraction,
                    handoff_release_elapsed_s=getattr(self, 'handoff_release_elapsed_s', 0.),
                    handoff_release_seconds=cfg.handoff_release_seconds,
                    max_target_step_rad=float((action-last_action).abs().max())*self.scale)
                return action, zeros
            self.telemetry = dict(active=False, blend=0.)
            return fallback, zeros
        default = _tensor(data.default_joint_pos)[0, self.leg_ids]
        root = _tensor(data.root_pos_w)[0]
        quat = _tensor(data.root_quat_w)[0]
        feet_world = _tensor(data.body_link_pose_w)[0, self.feet, :3]
        all_jac = _tensor(data.body_link_jacobian_w)[0]
        jac_world = [all_jac[foot - int(self.robot.is_fixed_base), :3][:, self.columns[leg]] for leg, foot in enumerate(self.feet)]
        root_goal = self.origin.clone()
        root_goal[2] = cfg.height_m
        desired_world = feet_world.clone()
        desired_world[:, 2] = self.ground_z
        contact_forces = _tensor(self.sensor.data.net_forces_w)[0, self.contact_ids]
        contacts = contact_forces[:, 2] > 1.
        velocity = _tensor(data.root_lin_vel_w)[0]
        angular = _tensor(data.root_ang_vel_b)[0]
        gravity = _tensor(data.projected_gravity_b)[0]
        center=root
        mass=self.mass
        if cfg.measured_com:
            payload=self.payload_state() if self.payload_state else None
            measured,mass=mass_weighted_center(_tensor(data.body_com_pose_w)[0,:,:3].cpu().numpy(),
                _tensor(data.body_mass)[0].cpu().numpy(),payload)
            center=torch.tensor(measured,device=root.device,dtype=root.dtype)
        if cfg.workspace_reference and not workspace_eligible and self.workspace_active:
            self.workspace_active=False
            self.workspace_anchors=None
            self.workspace_plan['outcome']='scope_exit'
        if cfg.workspace_reference and recovering and workspace_eligible:
            # Recovery already requests touchdown and a level 33 cm body.
            # Check whether that destination is reachable before continuing
            # to push an extended knee. This changes the body reference,
            # never the measured contact mask or the joint/motor bounds.
            q=_tensor(data.joint_pos)[0,self.leg_ids]
            limits=_tensor(data.joint_pos_limits)[0,self.leg_ids]
            margin=float(torch.minimum(q-limits[:,0],limits[:,1]-q).min())
            near_ground=bool(((feet_world[:,2]-self.ground_z).abs()<.03).all())
            entry_allowed=(not cfg.workspace_settled_entry or workspace_entry_settled(float(root[2]),
                float(gravity[2]),float(velocity.norm()),float(angular.norm()),contacts.cpu().tolist()))
            if cfg.workspace_entry_max_height_m is not None:
                entry_allowed = entry_allowed and float(root[2]) <= cfg.workspace_entry_max_height_m
            if (not self.workspace_attempted and int(contacts.sum())>=2
                    and near_ground and margin<.12 and entry_allowed):
                self.workspace_attempted=True
                grounded=feet_world.clone();grounded[:,2]=self.ground_z
                yaw=2.*math.atan2(float(self.yaw_quat[2]),float(self.yaw_quat[3]))
                self.workspace_plan=self.stance_geometry.fit(grounded.cpu().numpy(),root.cpu().numpy(),
                    yaw,q.cpu().numpy(),limits.cpu().numpy(),height=cfg.height_m)
                self.workspace_plan.update(measured_start_joint_margin_rad=margin,
                    geometry_check=self.check_stance_geometry(),
                    outcome='infeasible',measured_contacts_at_plan=contacts.cpu().tolist())
                if self.workspace_plan['success']:
                    clear=bool(self.stance_clearance_check(self.workspace_plan['root_target_w']))
                    self.workspace_plan.update(fresh_sensed_clearance=clear,outcome='clearance_rejected')
                    if clear:
                        self.workspace_anchors=grounded.clone()
                        self.workspace_planted_at_plan=contacts.clone()
                        self.workspace_active=True
                        self.workspace_scan_time=self.recovery_elapsed
                        self.workspace_plan['outcome']='executing'
            if self.workspace_anchors is not None:
                if self.recovery_elapsed-self.workspace_scan_time>=.10-1e-6:
                    self.workspace_plan['fresh_sensed_clearance']=bool(self.stance_clearance_check(self.workspace_plan['root_target_w']))
                    self.workspace_scan_time=self.recovery_elapsed
                # An airborne foot still has gait momentum; its motion is
                # not slip at a planted contact. Never use it to certify
                # support or to trigger this planted-anchor guard.
                planted=contacts & self.workspace_planted_at_plan
                slip=float((feet_world[planted,:2]-self.workspace_anchors[planted,:2]).norm(dim=1).max()) if bool(planted.any()) else 0.
                self.workspace_plan['maximum_anchor_slip_m']=max(self.workspace_plan.get('maximum_anchor_slip_m',0.),slip)
                if slip>.04 or not self.workspace_plan['fresh_sensed_clearance']:
                    self.workspace_anchors=None
                    self.workspace_active=False
                    self.origin[:2]=root[:2]
                    self.workspace_plan['outcome']='guarded_stop'
                else:
                    target=torch.tensor(self.workspace_plan['root_target_w'],device=root.device,dtype=root.dtype)
                    delta=target[:2]-self.origin[:2]
                    self.origin[:2]+=delta*min(1.,.10*dt/max(float(delta.norm()),1e-9))
                    self.workspace_active=bool((target[:2]-self.origin[:2]).norm()>.001)
                    if not self.workspace_active:
                        self.workspace_plan['outcome']='reference_completed'
                    desired_world=self.workspace_anchors.clone()
                root_goal[:2]=self.origin[:2]
        if cfg.stance_fit and recovering:
            # Keep the feet planted. A small body shift may restore knee
            # workspace without unloading a foot or changing the height goal.
            if (not self.stance_fit_attempted and self.recovery_elapsed<1.3
                    and bool(contacts.all()) and float(root[2])>.29
                    and float(gravity[2])<-.985 and float(velocity.norm())<.12
                    and float(angular.norm())<.40):
                q=_tensor(data.joint_pos)[0,self.leg_ids]
                limits=_tensor(data.joint_pos_limits)[0,self.leg_ids]
                margin=float(torch.minimum(q-limits[:,0],limits[:,1]-q).min())
                if margin<.12:
                    self.stance_fit_attempted=True
                    grounded=feet_world.clone();grounded[:,2]=self.ground_z
                    yaw=2.*math.atan2(float(self.yaw_quat[2]),float(self.yaw_quat[3]))
                    self.stance_fit_plan=self.stance_geometry.fit(grounded.cpu().numpy(),root.cpu().numpy(),
                        yaw,q.cpu().numpy(),limits.cpu().numpy(),height=cfg.height_m)
                    self.stance_fit_plan['measured_start_joint_margin_rad']=margin
                    self.stance_fit_plan['outcome']='no_shift_needed' if self.stance_fit_plan['success'] else 'infeasible'
                    if self.stance_fit_plan['success'] and self.stance_fit_plan['shift_m']>.005:
                        clear=bool(self.stance_clearance_check and self.stance_clearance_check(self.stance_fit_plan['root_target_w']))
                        self.stance_fit_plan['fresh_sensed_clearance']=clear
                        self.stance_fit_plan['outcome']='clearance_rejected'
                        if clear:
                            self.stance_fit_active=True
                            self.stance_fit_started=self.recovery_elapsed
                            self.stance_fit_scan_time=self.recovery_elapsed
                            # Plan and execute from the same measured root and
                            # four foot anchors. Re-reading feet every frame
                            # would silently move the reference with foot slip.
                            self.origin=root.clone()
                            self.stance_fit_start=root.clone()
                            self.stance_fit_anchors=grounded.clone()
                            self.stance_fit_plan['outcome']='executing'
            if self.stance_fit_active:
                elapsed=self.recovery_elapsed-self.stance_fit_started
                if self.recovery_elapsed-self.stance_fit_scan_time>=.10-1e-6:
                    self.stance_fit_plan['fresh_sensed_clearance']=bool(self.stance_clearance_check(self.stance_fit_plan['root_target_w']))
                    self.stance_fit_scan_time=self.recovery_elapsed
                if (int(contacts.sum())<3 or float(gravity[2])>-.990 or float(angular.norm())>.5
                        or not self.stance_fit_plan['fresh_sensed_clearance']):
                    self.stance_fit_active=False
                    self.stance_fit_anchors=None
                    self.origin[:2]=root[:2]
                    self.stance_fit_plan['outcome']='guarded_stop'
                else:
                    fraction=min(1.,elapsed/.50)
                    smooth=fraction*fraction*(3.-2.*fraction)
                    target=torch.tensor(self.stance_fit_plan['root_target_w'],device=root.device,dtype=root.dtype)
                    self.origin[:2]=self.stance_fit_start[:2]+smooth*(target[:2]-self.stance_fit_start[:2])
                    if fraction>=1.:
                        self.stance_fit_active=False
                        self.stance_fit_plan['outcome']='reference_completed'
                root_goal[:2]=self.origin[:2]
            if self.stance_fit_anchors is not None and recovering:
                desired_world=self.stance_fit_anchors.clone()
        recenter_phase = None
        swing_leg = None
        offload_leg = None
        if cfg.recenter_foot and recovering:
            if (not self.recenter_attempted and self.recovery_elapsed<1.5
                    and bool(contacts.all()) and float(root[2])>.305
                    and float(gravity[2])<-.990 and float(velocity.norm())<.10
                    and float(angular.norm())<.30):
                # Use the verified initial standing geometry, rotated by the
                # recovery yaw. Only one bounded step is attempted per event.
                nominal=self.quat_apply(self.yaw_quat.expand(4,4),self.nominal_feet_local)+root
                error=(nominal[:,:2]-feet_world[:,:2]).norm(dim=1)
                leg=int(error.argmax())
                if float(error[leg])>.085:
                    delta=nominal[leg,:2]-feet_world[leg,:2]
                    delta*=min(1.,.12/float(delta.norm()))
                    target=feet_world[leg].clone()
                    target[:2]+=delta
                    target[2]=self.ground_z
                    others=[i for i in range(4) if i!=leg]
                    shift=minimal_support_shift(center[:2].cpu().numpy(),feet_world[others,:2].cpu().numpy())
                    if shift is not None:
                        self.recenter=dict(leg=leg,start=feet_world[leg].cpu().numpy().copy(),
                            target=target.cpu().numpy().copy(),shift=shift.copy(),
                            anchors=feet_world.clone(),origin=root.clone(),elapsed=0.,offload_wait=0.)
                        self.recenter_active=True
                        self.recenter_count+=1
                self.recenter_attempted=True
            if self.recenter_active:
                step=self.recenter
                if cfg.measured_com and step.get('lift_started') and not step.get('guarded'):
                    others=[i for i in range(4) if i!=step['leg']]
                    safe=(bool(contacts[others].all()) and support_triangle_contains(
                        center[:2].cpu().numpy(),feet_world[others,:2].cpu().numpy(),margin=0.)
                        and float(gravity[2])<-math.cos(math.radians(6.)) and float(angular.norm())<.5)
                    if not safe:
                        step['guarded']=True
                        self.recenter_guarded+=1
                if .30<=step['elapsed']<.30+dt*1.01:
                    others=[i for i in range(4) if i!=step['leg']]
                    geometry_supported=(bool(contacts[others].all())
                        and support_triangle_contains(center[:2].cpu().numpy(),feet_world[others,:2].cpu().numpy()))
                    supported=geometry_supported and float(contact_forces[step['leg'],2])<15.
                    if not supported:
                        # Never lift a loaded foot before the physical body
                        # has shifted. Abandon the candidate before swinging
                        # if it cannot unload within the original timeout.
                        step['elapsed']=.30-dt
                        step['offload_wait']+=dt
                        if step['offload_wait']>.36:
                            self.recenter_active=False
                            self.recenter_aborted+=1
                            self.origin=root.clone()
                        elif cfg.offload_foot and geometry_supported:
                            # A minimum-norm four-contact fit keeps loading the
                            # candidate even inside the other support triangle.
                            # Deliberately remove its feedforward support, and
                            # remove joint-position preload below. Do not lift
                            # until the unchanged measured-force gate passes.
                            offload_leg=step['leg']
                foot,shift,recenter_phase=recenter_trajectory(step['start'],step['target'],step['shift'],step['elapsed'])
                desired_world=step['anchors'].clone()
                desired_world[:,2]=self.ground_z
                desired_world[step['leg']]=torch.as_tensor(foot,device=root.device,dtype=root.dtype)
                root_goal=step['origin'].clone()
                root_goal[:2]+=torch.as_tensor(shift,device=root.device,dtype=root.dtype)
                root_goal[2]=cfg.height_m
                if self.recenter_active and recenter_phase in ('lift','transfer','lower'):
                    swing_leg=step['leg']
                    step['lift_started']=True
                    contacts=contacts.clone()
                    contacts[swing_leg]=False
                step['elapsed']+=dt
                if step.get('guarded'):
                    # Keep the original joint-rate bound, but stop lateral
                    # swing immediately and request touchdown where the foot
                    # actually is. Original recovery timeout remains active.
                    desired_world=feet_world.clone()
                    desired_world[:,2]=self.ground_z
                    contacts=contact_forces[:,2]>1.
                    swing_leg=offload_leg=None
                    recenter_phase='guard_lower'
                    if float(contact_forces[step['leg'],2])>5.:
                        self.recenter_active=False
                        self.origin=root.clone()
                        root_goal=self.origin.clone();root_goal[2]=cfg.height_m
                        recenter_phase='guard_landed'
                elif recenter_phase=='complete' and self.recenter_active:
                    self.recenter_active=False
                    self.recenter_completed+=1
                    self.origin=root_goal.clone()
                elif not self.recenter_active:
                    recenter_phase='aborted_before_lift'
                    root_goal=self.origin.clone()
                    root_goal[2]=cfg.height_m
                    desired_world=feet_world.clone()
                    desired_world[:,2]=self.ground_z
        if offload_leg is not None:
            contacts=contacts.clone()
            contacts[offload_leg]=False
            recenter_phase='offloading'
        desired_local = self.quat_inverse(self.yaw_quat.expand(4, 4), desired_world - root_goal)
        measured_local = self.quat_inverse(quat.expand(4, 4), feet_world - root)
        errors = (desired_local - measured_local).clamp(-.06, .06)
        desired = _tensor(data.joint_pos)[0, self.leg_ids].clone()
        limits = _tensor(data.joint_pos_limits)[0, self.leg_ids]
        eye = torch.eye(3, device=self.robot.device)
        for leg, jac in enumerate(jac_world):
            local_jac = self.quat_inverse(quat.expand(3, 4), jac.T).T
            indices = [leg+4*joint for joint in range(3)]
            if cfg.leg_ik=='bounded':
                from .physical_manipulation import bounded_damped_increment
                current=desired[indices]
                lower=(limits[indices,0]+.015-current).clamp(min=-.25)
                upper=(limits[indices,1]-.015-current).clamp(max=.25)
                delta=bounded_damped_increment(local_jac,errors[leg],lower,upper,.03)
                desired[indices]+=delta
            else:
                delta = local_jac.T @ torch.linalg.solve(local_jac @ local_jac.T + .03**2 * eye, errors[leg])
                desired[indices] += delta.clamp(-.25, .25)
        if offload_leg is not None:
            indices=[offload_leg+4*joint for joint in range(3)]
            desired[indices]=_tensor(data.joint_pos)[0,self.leg_ids][indices]
        lower, upper = limits[:, 0] + .015, limits[:, 1] - .015
        previous_command = default + self.scale * last_action
        if cfg.target_update == 'project_then_slew':
            from .recovery_targets import continuous_recovery_target
            # Inherited RL requests can already lie outside the position box.
            # Project the destination, not the slewed command: snapping the
            # command afterward would silently bypass the original rate cap.
            self.target = continuous_recovery_target(self.target, desired, lower, upper,
                                                       cfg.joint_rate_radps * dt)
        else:
            self.target += (desired - self.target).clamp(-cfg.joint_rate_radps * dt, cfg.joint_rate_radps * dt)
            self.target = torch.maximum(torch.minimum(self.target, upper), lower)
        touchdown = []
        if cfg.touchdown_ik and recovering:
            from .touchdown_ik import touchdown_joint_target
            measured = _tensor(data.joint_pos)[0, self.leg_ids]
            for leg, jac in enumerate(jac_world):
                indices = [leg+4*joint for joint in range(3)]
                command, diagnostic = touchdown_joint_target(
                    measured[indices].cpu().numpy(), previous_command[indices].cpu().numpy(),
                    lower[indices].cpu().numpy(), upper[indices].cpu().numpy(), jac.cpu().numpy(),
                    self.quat_apply(quat, errors[leg]).cpu().numpy(), cfg.joint_rate_radps * dt,
                    contact_normal_n=float(contact_forces[leg, 2]),
                    near_limit_only=not (cfg.airborne_touchdown and loaded_recovery))
                touchdown.append(dict(leg=leg, **diagnostic))
                if command is not None:
                    # Do not independently clip/slew this final constrained
                    # solution, which would change its foot-motion direction.
                    self.target[indices] = torch.as_tensor(command, device=self.robot.device, dtype=self.target.dtype)
        torque_body = torch.stack((cfg.tilt_kp * gravity[1] - cfg.tilt_kd * angular[0],
                                   -cfg.tilt_kp * gravity[0] - cfg.tilt_kd * angular[1],
                                   angular[2] * 0.))
        moment_world = self.quat_apply(_tensor(data.root_quat_w)[0], torque_body)
        vertical = mass * 9.81 + cfg.height_kp * (cfg.height_m - float(root[2])) - cfg.height_kd * float(velocity[2])
        vertical = min(mass * 9.81 * 1.5, max(0., vertical))
        horizontal = -mass * (3. * (root[:2] - root_goal[:2]) + 3. * velocity[:2])
        feet = _tensor(data.body_link_pose_w)[0, self.feet, :3] - center
        blocks = np.stack([jac.cpu().numpy() for jac in jac_world])
        use_qp = (cfg.allocator == 'wrench_qp' or cfg.stationary_payload_qp and stationary_payload
                  or cfg.loaded_recovery_qp and loaded_recovery)
        support_action = (self.target - default) / self.scale
        action = support_action if recovering else self.weight * support_action + (1. - self.weight) * fallback
        if cfg.handoff_mode == 'settled_slew' and not recovering:
            step = cfg.exit_joint_rate_radps * dt / self.scale
            action = last_action + (action-last_action).clamp(-step, step)
        allocation = allocate_support_qp if use_qp else allocate_support
        extra = dict(jacobians=blocks, torque_limit=cfg.torque_limit_nm) if use_qp else {}
        motor_report=None
        joint_target_report=None
        if cfg.total_motor_qp and loaded_recovery and use_qp:
            bias,motor_lo,motor_hi,motor_report=measured_motor_envelope(
                self.robot,self.leg_ids,default+self.scale*action)
            # Leave a small numerical margin inside the measured native limits;
            # neither actuator gains nor native effort/speed caps are modified.
            extra.update(motor_bias=bias.reshape(3,4).T,
                motor_lower=(motor_lo+.1).reshape(3,4).T,
                motor_upper=(motor_hi-.1).reshape(3,4).T,effort_scale=self.weight)
        if cfg.joint_motor_qp and motor_report is not None:
            from .motor_target_support import solve
            forces,residual,selected,joint_target_report=solve(feet.cpu().numpy(),contacts.cpu().numpy(),vertical,
                moment_world[:2].cpu().numpy(),horizontal.cpu().numpy(),jacobians=blocks,
                friction=cfg.friction,per_foot_limit=mass*9.81*.8,feedforward_limit=cfg.torque_limit_nm,
                effort_scale=self.weight,motor=motor_report,previous_target=previous_command.cpu().numpy(),
                position_lower=lower.cpu().numpy(),position_upper=upper.cpu().numpy(),
                max_step=(cfg.joint_rate_radps if recovering else cfg.exit_joint_rate_radps)*dt)
            selected=torch.as_tensor(selected,device=self.robot.device,dtype=action.dtype)
            action=(selected-default)/self.scale
            if recovering:self.target=selected.clone()
            bias,motor_lo,motor_hi,motor_report=measured_motor_envelope(self.robot,self.leg_ids,selected)
        else:
            forces, residual = allocation(feet.cpu().numpy(), contacts.cpu().numpy(), vertical,
                moment_world[:2].cpu().numpy(), horizontal.cpu().numpy(), friction=cfg.friction,
                per_foot_limit=mass * 9.81 * .8, **extra)
        torque_per_leg = leg_feedforward(blocks, forces, cfg.torque_limit_nm)
        raw_torque = -np.einsum('lij,li->lj', blocks, forces)
        effort = torch.tensor(torque_per_leg.T.reshape(-1), device=self.robot.device, dtype=fallback.dtype)
        # While recovering, hold the continuously slewed support target. After
        # recovery, hand back both position targets and torque over 250 ms.
        if motor_report is not None:
            motor_report.update(active=True,margin_nm=.1,effort_scale=self.weight,
                feedforward_nm=(self.weight*effort).cpu().tolist(),
                predicted_total_nm=(bias+self.weight*effort.detach().cpu().numpy()).tolist())
        from .recovery_targets import bound_violation
        self.telemetry = dict(active=bool(recovering), blend=self.weight, contacts=contacts.cpu().tolist(),
            allocator='wrench_qp' if use_qp else 'sequential',
            stationary_payload_qp_scope=bool(cfg.stationary_payload_qp and stationary_payload),
            loaded_recovery_qp_scope=bool(cfg.loaded_recovery_qp and loaded_recovery),
            total_motor_qp=motor_report,
            joint_motor_target=joint_target_report,
            raw_feedforward_max_nm=float(np.abs(raw_torque).max()),
            feedforward_clipping_max_nm=float(np.abs(raw_torque-torque_per_leg).max()),
            recovery_event=self.activations,
            target_update=cfg.target_update,
            touchdown_ik=touchdown,
            airborne_touchdown_scope=bool(cfg.airborne_touchdown and loaded_recovery),
            inherited_bound_violation_rad=float(bound_violation(default+self.scale*last_action, lower, upper)),
            target_bound_violation_rad=float(bound_violation(default+self.scale*action, lower, upper)),
            base_allocator=cfg.allocator,
            leg_ik=cfg.leg_ik,
            handoff_mode=cfg.handoff_mode,
            recenter_active=self.recenter_active,recenter_phase=recenter_phase,
            recenter_swing_leg=swing_leg,recenter_count=self.recenter_count,
            recenter_offload_leg=offload_leg,
            recenter_guarded=self.recenter_guarded,
            stance_fit_active=self.stance_fit_active if cfg.stance_fit else False,
            stance_fit_plan=self.stance_fit_plan if cfg.stance_fit else None,
            workspace_reference_active=self.workspace_active if cfg.workspace_reference else False,
            workspace_reference_plan=self.workspace_plan if cfg.workspace_reference else None,
            workspace_reference_eligible=bool(workspace_eligible) if cfg.workspace_reference else False,
            support_point_mode='measured_center_of_mass' if cfg.measured_com else 'root_projection',
            support_reference_w=center.cpu().tolist(),
            recenter_completed=self.recenter_completed,recenter_aborted=self.recenter_aborted,
            commanded_joint_position_rad=(default+self.scale*action).cpu().tolist(),
            ik_joint_position_rad=desired.cpu().tolist(),
            foot_error_local_m=errors.cpu().tolist(),
            max_target_step_rad=float((action-last_action).abs().max())*self.scale,
            foot_positions_w=feet_world.cpu().tolist(),
            root_velocity_w=velocity.cpu().tolist(), projected_gravity_b=gravity.cpu().tolist(),
            root_position_w=root.cpu().tolist(),
            measured_joint_margin_rad=float(torch.minimum(
                _tensor(data.joint_pos)[0,self.leg_ids]-limits[:,0],
                limits[:,1]-_tensor(data.joint_pos)[0,self.leg_ids]).min()),
            root_angular_velocity_b=angular.cpu().tolist(),
            support_mass_estimate_kg=mass,
            requested_vertical_n=vertical, measured_normal_n=contact_forces[:, 2].cpu().tolist(),
            allocated_force_w=forces.tolist(), wrench_fit_residual=residual.tolist(),
            max_feedforward_nm=float(effort.abs().max()) * self.weight,
            target_height_m=cfg.height_m)
        return action, self.weight * effort
