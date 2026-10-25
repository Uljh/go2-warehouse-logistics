"""Shared-world Z1 manipulation backend for the warehouse mission contract.

This is an opt-in contact-only backend. It consumes camera-estimated parcel
center/yaw, measured Z1 mount pose, the original six-axis URDF kinematics and
pair-filtered jaw contact. Every command is a bounded joint target; the parcel
remains an independent dynamic rigid body throughout pickup, carry and release.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .z1_asset import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME, prepare_z1_urdf
from .z1_kinematics import Z1Kinematics
from .z1_world_target import link_transform, measured_world_arm_target
from .z1_contact_control import (JawNormalAdmittance, GripperReleaseHandoff,
    PairedGripReadiness, unload_moving_jaw_preload)


_PROBE_ORIENTATION = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])


@dataclass(frozen=True)
class Z1FleetConfig:
    dropoff_position: tuple[float, float, float]
    grasp_mode: str = 'contact'
    parcel_size_xyz_m: tuple[float, float, float] = (.035, .10, .08)
    gripper_open_rad: float = -1.45
    jaw_sample_x_m: float = .15
    grasp_heading_reference_rad: float | None = None
    lift_height: float = .40
    approach_height: float = .10
    close_seconds: float = 24.
    close_on_stable_contact: bool = False
    approach_seconds: float = 8.
    lift_seconds: float = 12.
    place_seconds: float = 5.
    release_seconds: float = 2.
    settle_seconds: float = 2.
    tracking_settle_seconds: float = 2.
    force_target_n: float = 6.
    precontact_limit_n: float | None = None
    hard_contact_limit_n: float = 20.
    position_tolerance_m: float = .012
    orientation_tolerance_rad: float = .20
    placement_planar_tolerance: float = .04
    placement_height_tolerance: float = .015
    placement_surface_z: float | None = None
    placement_approach_height: float = .08
    release_clearance_backoff_m: float = 0.
    release_clearance_rise_m: float = 0.
    release_clearance_seconds: float = 2.
    placement_deck_size_xy: tuple[float, float] | None = None
    reset_before_grasp: bool = True
    stow_after_release: bool = True
    constraint_path: str = '/World/unused_Z1_assisted_grasp'

    def __post_init__(self):
        if self.grasp_mode != 'contact':
            raise ValueError('The Z1 warehouse backend supports contact-only grasping')
        if not isinstance(self.close_on_stable_contact,bool):
            raise ValueError('Measured early close must be an explicit boolean')
        if not (-math.pi/2 <= self.gripper_open_rad < 0):
            raise ValueError('Z1 gripper opening is outside its original range')
        if not math.isfinite(self.jaw_sample_x_m) or not .14 <= self.jaw_sample_x_m <= .16:
            raise ValueError('Fleet Z1 jaw insertion reference must remain within [.14,.16] metres')
        if self.grasp_heading_reference_rad is not None and not math.isfinite(self.grasp_heading_reference_rad):
            raise ValueError('Grasp heading reference must be finite')
        if not (len(self.parcel_size_xyz_m) == 3 and all(math.isfinite(v) and v > 0 for v in self.parcel_size_xyz_m)):
            raise ValueError('Invalid Z1 parcel dimensions')
        if not (0 < self.force_target_n < self.hard_contact_limit_n == 20.):
            raise ValueError('Z1 contact force must remain below the unchanged 20 N guard')
        if self.precontact_limit_n is not None and not .05 < self.precontact_limit_n <= self.force_target_n:
            raise ValueError('Single-jaw preload relief must be below the paired force target')
        if (self.placement_surface_z is None) != (self.placement_deck_size_xy is None):
            raise ValueError('Shelf verification requires both surface height and deck dimensions')
        for name in ('lift_height', 'approach_height', 'close_seconds', 'approach_seconds',
                     'lift_seconds', 'place_seconds', 'release_seconds', 'settle_seconds','tracking_settle_seconds',
                     'position_tolerance_m', 'orientation_tolerance_rad', 'release_clearance_seconds',
                     'placement_planar_tolerance', 'placement_height_tolerance','placement_approach_height'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('release_clearance_backoff_m', 'release_clearance_rise_m'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0. <= value <= .04:
                raise ValueError(f'{name} must be finite in [0,.04] metres')


def parcel_grasp_target(center_world, yaw_rad, tcp_in_stator_m, *, surface_normal=None, long_axis=None,
                        heading_reference_rad=None):
    """Build the validated side-pinch pose, aligned to an observed box yaw.

    The imported fleet parcel is 100 x 35 x 80 mm. Rotating the tested probe
    grasp by yaw+90 degrees aligns its local pinch axis with the parcel's short
    side. The returned transform locates the stator, not a virtual TCP.
    An optional fixed reference selects the nearest equivalent grasp side;
    it prevents the legacy yaw-zero branch cut from swapping the two jaws.
    The reference must be chosen before motion, never switched while holding.
    """
    center = np.asarray(center_world, dtype=float)
    tcp = np.asarray(tcp_in_stator_m, dtype=float)
    if (center.shape != (3,) or tcp.shape != (3,) or not np.isfinite(np.r_[center, tcp, yaw_rad]).all()):
        raise ValueError('Invalid observed box center, yaw or calibrated gripper reference')
    heading = (float(yaw_rad)+math.pi) % math.pi-math.pi/2
    if heading_reference_rad is not None:
        reference = float(heading_reference_rad)
        if not math.isfinite(reference):
            raise ValueError('Grasp heading reference must be finite')
        heading = reference+(heading-reference+math.pi/2) % math.pi-math.pi/2
    rotation = Rotation.from_euler('z', heading).as_matrix() @ _PROBE_ORIENTATION
    if surface_normal is not None:
        normal=np.asarray(surface_normal,dtype=float)
        along=np.asarray(long_axis if long_axis is not None else rotation[:,1],dtype=float)
        if (normal.shape!=(3,) or along.shape!=(3,) or not np.isfinite(np.r_[normal,along]).all()
                or abs(np.linalg.norm(normal)-1.)>.001 or normal[2]<math.cos(math.radians(15.))):
            raise ValueError('Invalid measured upward parcel face')
        along=along-normal*np.dot(normal,along)
        if np.linalg.norm(along)<.9:
            raise ValueError('Invalid measured long axis of the parcel face')
        along/=np.linalg.norm(along)
        if np.dot(along,rotation[:,1])<0:along=-along
        rotation=np.column_stack((-normal,along,-np.cross(normal,along)))
    target = np.eye(4)
    target[:3, :3] = rotation
    target[:3, 3] = center - rotation @ tcp
    return target


def interpolate_pose(start, end, fraction):
    """Rigid interpolation used for visibly continuous world-frame targets."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if (a.shape != (4, 4) or b.shape != (4, 4) or not np.isfinite(np.r_[a.ravel(), b.ravel(), fraction]).all()
            or not 0 <= fraction <= 1):
        raise ValueError('Invalid Z1 world-frame interpolation')
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + (b[:3, 3]-a[:3, 3])*fraction
    out[:3, :3] = Slerp([0., 1.], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))([fraction]).as_matrix()[0]
    return out


class PhysicalZ1Arm:
    """Z1 implementation of the existing fleet arm's small public contract."""

    def __init__(self, env: Any, stage: Any, box: Any, *, hand_path: str, box_path: str,
                 config: Z1FleetConfig, step_fn: Callable[[], None], robot_key='robot',
                 contact_sensor_prefix=''):
        import torch
        from .z1_contact_probe import ProbeConfig, geometry_report

        self.torch = torch
        self.env = env
        self.robot = env.unwrapped.scene[robot_key]
        self.stage = stage
        self.box = box
        self.box_path = box_path
        self.hand_path = hand_path
        self.robot_key = robot_key
        self.config = config
        self.step_fn = step_fn
        self.contact_sensor_prefix = contact_sensor_prefix
        self.contact_wrench_enabled = os.environ.get('GO2_Z1_CONTACT_WRENCH') == '1'
        self.device = self.robot.device
        self.arm_ids, arm_names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        self.gripper_ids, gripper_names = self.robot.find_joints([GRIPPER_JOINT_NAME], preserve_order=True)
        self.mount_id = self.robot.find_bodies(['link00'], preserve_order=True)[0][0]
        self.stator_id = self.robot.find_bodies(['gripperStator'], preserve_order=True)[0][0]
        self.mover_id = self.robot.find_bodies(['gripperMover'], preserve_order=True)[0][0]
        if len(self.arm_ids) != 6 or len(self.gripper_ids) != 1:
            raise RuntimeError(f'Expected official Z1 6+1 DOFs; got {arm_names}, {gripper_names}')
        self.sensors = [env.unwrapped.scene.sensors[contact_sensor_prefix + name]
                        for name in ('left_grip_contact', 'right_grip_contact')]
        urdf, spec = prepare_z1_urdf(overwrite=False)
        self.kinematics = Z1Kinematics.from_urdf(urdf)
        self.asset_spec = spec
        self.motor_efforts = torch.tensor([spec.joint_efforts_nm[name] for name in ARM_JOINT_NAMES],
                                          device=self.device, dtype=torch.float32)
        self.dt = float(env.unwrapped.step_dt)
        self._arm_target = self._joint_position(self.arm_ids).clone()
        self._previous_arm_target = self._arm_target.clone()
        self._home_target = _tensor(self.robot.data.default_joint_pos)[0, self.arm_ids].clone()
        self._gripper_target = float(self._joint_position(self.gripper_ids)[0])
        self._previous_gripper_target = self._gripper_target
        self._last_contact = self._read_contact()
        self._contact_enabled = False
        self._force_latched = False
        self._grip_reference = self._gripper_target
        self._normal_controller = JawNormalAdmittance(config.force_target_n, precontact_feedback=True,
                                                     precontact_limit_n=config.precontact_limit_n)
        self._release_handoff = GripperReleaseHandoff(config.gripper_open_rad, duration_s=config.release_seconds)
        self._grasp_tcp = None
        self._grasp_yaw = None
        self._grasp_box_to_stator = None
        self._carry_reference_local = None
        self._nominal_world_reference = None
        self._lost_pair_ticks = 0
        self._releasing = False
        self._released_once = False
        self.held = False
        self.state = 'waiting'
        self.last_error = ''
        self.detected_position = None
        self.detected_geometry = None
        self.contact_loss_detection = True
        self.loss_events = []
        self.joint = None
        self.verified_lift = 0.
        self.max_hold_error = 0.
        self.placement_metrics = None
        self._release_retreat_target = None
        self._release_retreat_complete = False
        self._release_clearance_target = None
        self._release_complete = False
        self.kinematic_plan = None
        self.kinematic_plan_history = []
        self.kinematic_plan_failures = []
        self.close_completion = None
        probe = ProbeConfig(parcel_size_xyz_m=(.035, .10, config.parcel_size_xyz_m[2]),
                            jaw_sample_x_m=config.jaw_sample_x_m, collision_mode='convexDecomposition')
        self.geometry = geometry_report(probe)
        if not self.geometry.get('matches_fleet_parcel_dimensions'):
            raise RuntimeError('Z1 calibrated jaw geometry does not match the warehouse parcel')
        self.tcp_in_stator = np.asarray(self.geometry['tcp_in_stator_m'], dtype=float)
        self._verify_live_kinematics()
        self.runtime_integrity = self._verify_physical_asset()
        print(f'PHYSICAL_Z1 ready arm_joints={arm_names} gripper={gripper_names} '
              f'parcel_size_m={config.parcel_size_xyz_m} grasp_model=measured_contact '
              f'constraint_created=False control_dt_s={self.dt:.4f}', flush=True)

    def _joint_position(self, ids):
        return _tensor(self.robot.data.joint_pos)[0, ids]

    def _mount_pose(self):
        row = _tensor(self.robot.data.body_link_pose_w)[0, self.mount_id]
        return link_transform(row[:3].detach().cpu().numpy(), row[3:7].detach().cpu().numpy())

    def _tip_pose_matrix(self):
        row = _tensor(self.robot.data.body_link_pose_w)[0, self.stator_id]
        return link_transform(row[:3].detach().cpu().numpy(), row[3:7].detach().cpu().numpy())

    def _box_pose_matrix(self):
        pos = _tensor(self.box.data.root_pos_w)[0].detach().cpu().numpy()
        quat = _tensor(self.box.data.root_quat_w)[0].detach().cpu().numpy()
        return link_transform(pos, quat)

    def _verify_live_kinematics(self):
        predicted = self._mount_pose() @ self.kinematics.fk(self._joint_position(self.arm_ids).detach().cpu().numpy())
        measured = self._tip_pose_matrix()
        position_error = float(np.linalg.norm(predicted[:3, 3]-measured[:3, 3]))
        angle_error = float(Rotation.from_matrix(predicted[:3, :3]@measured[:3, :3].T).magnitude())
        report = dict(position_error_m=position_error, orientation_error_rad=angle_error)
        if position_error > .002 or angle_error > .01:
            raise RuntimeError(f'Z1 URDF FK disagrees with the live articulation: {report}')
        self.kinematic_plan = dict(operation='live_model_check', live_model_verification=report)

    def _read_contact(self):
        normals, frictions, totals = [], [], []
        for sensor in self.sensors:
            raw = sensor.data.force_matrix_w
            if raw is None:
                raise RuntimeError('Z1 pair-filtered normal force matrix is unavailable')
            normal = self._active_pair_vector(sensor, raw)
            friction_raw = getattr(sensor.data, 'friction_forces_w', None)
            if friction_raw is None:
                raise RuntimeError('Z1 pair-filtered friction force matrix is unavailable')
            friction = self._active_pair_vector(sensor, friction_raw)
            total = normal+friction
            normals.append(normal)
            frictions.append(friction)
            totals.append(float(self.torch.linalg.vector_norm(total).item()))
        normal_magnitudes = [float(self.torch.linalg.vector_norm(value).item()) for value in normals]
        return dict(normal_vectors=normals, friction_vectors=frictions,
                    normal_magnitudes=normal_magnitudes, total_magnitudes=totals)

    def _active_pair_vector(self, sensor, matrix):
        """Keep another queued parcel's contact out of this order's force loop."""
        paths = [value.replace('{ENV_REGEX_NS}', '/World/envs/env_0').replace('/env_.*', '/env_0')
                 for value in sensor.cfg.filter_prim_paths_expr]
        if self.box_path not in paths:
            raise RuntimeError(f'Z1 jaw sensor has no explicit filter for active parcel {self.box_path}: {paths}')
        index = paths.index(self.box_path)
        tensor = _tensor(matrix)
        if tensor.ndim != 4 or tensor.shape[-1] != 3 or tensor.shape[2] != len(paths):
            raise RuntimeError(f'Z1 pair-filtered force matrix has unexpected shape: {tuple(tensor.shape)}')
        return tensor[0, :, index, :].sum(0)

    def contact_wrench_diagnostic(self, contact):
        """Read raw active-order points without changing contacts or controller inputs."""
        if not self.contact_wrench_enabled or self.state not in ('closing_fingers','lift','failed'):
            return None
        from .z1_contact_wrench import read_pair_wrench
        origin=_tensor(self.box.data.root_pos_w)[0].detach().cpu().numpy()
        result=[]
        for jaw,sensor,normal,friction in zip(('fixed','moving'),self.sensors,
                contact['normal_vectors'],contact['friction_vectors']):
            paths=[value.replace('{ENV_REGEX_NS}', '/World/envs/env_0').replace('/env_.*','/env_0')
                   for value in sensor.cfg.filter_prim_paths_expr]
            if self.box_path not in paths:
                raise RuntimeError('Raw contact diagnostic has no active parcel filter')
            view=sensor.contact_view
            if view.sensor_count!=1 or view.filter_count!=len(paths):
                raise RuntimeError('Raw contact diagnostic requires a single jaw and explicit parcel filters')
            entry=read_pair_wrench(view,sensor._sim_physics_dt,origin,
                normal.detach().cpu().numpy(),friction.detach().cpu().numpy(),
                pair=(0,paths.index(self.box_path)))
            entry.update(jaw=jaw,active_payload_path=self.box_path,
                         physics_dt_s=float(sensor._sim_physics_dt))
            result.append(entry)
        return result

    def payload_sleep_diagnostic(self):
        """Read the native sleep flag; never wake or alter a payload."""
        if not self.contact_wrench_enabled or self.state not in ('closing_fingers','lift','failed'):
            return None
        if not hasattr(self,'_sleep_query'):
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools, Sdf, UsdUtils
            stage_id=UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
            if stage_id<0:
                raise RuntimeError('Sleep diagnostic requires the active cached USD stage')
            self._sleep_query=(get_physx_simulation_interface(),stage_id,PhysicsSchemaTools,Sdf)
        api,stage_id,schema,sdf=self._sleep_query
        return dict(active_payload_path=self.box_path,
            sleeping=bool(api.is_sleeping(stage_id,schema.sdfPathToInt(sdf.Path(self.box_path)))),
            source='PhysX simulation interface is_sleeping; read only')

    def _verify_physical_asset(self):
        from pxr import UsdPhysics
        from .z1_asset import ARM_LINK_NAMES, GRIPPER_LINK_NAMES
        bodies = self.robot.find_bodies(list(ARM_LINK_NAMES)+list(GRIPPER_LINK_NAMES), preserve_order=True)[0]
        mass = float(_tensor(self.robot.data.default_mass)[0, bodies].sum())
        limits = _tensor(self.robot.data.joint_pos_limits)[0, self.arm_ids+self.gripper_ids].detach().cpu().numpy()
        expected = np.asarray([self.asset_spec.joint_limits[name]
                               for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT_NAME)])
        joints = []
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
                if any(str(path) == self.box_path or str(path).startswith(self.box_path+'/') for path in targets):
                    joints.append(str(prim.GetPath()))
        kinematic = bool(UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(self.box_path)).GetKinematicEnabledAttr().Get())
        passed = (not self.robot.is_fixed_base and abs(mass-self.asset_spec.total_mass_kg) < 1e-4
                  and len(bodies) == 9 and np.allclose(limits, expected, atol=1e-4) and not joints and not kinematic)
        report = dict(passed=passed, arm_mass_kg=mass, expected_arm_mass_kg=self.asset_spec.total_mass_kg,
                      fixed_base=bool(self.robot.is_fixed_base), actual_joint_limits_rad=limits.tolist(),
                      payload_joints=joints, payload_kinematic=kinematic, runtime_state_writes=0)
        if not passed:
            raise RuntimeError(f'Z1 runtime original-asset or contact-only integrity failed: {report}')
        return report

    def _hard_contact_guard(self, contact=None):
        sample = self._last_contact if contact is None else contact
        values = sample['normal_magnitudes'] + sample['total_magnitudes']
        if not np.isfinite(values).all() or max(values, default=0.) > self.config.hard_contact_limit_n:
            raise RuntimeError(f'Z1 jaw force exceeded the unchanged 20 N limit: {values} '
                               f'robot={self.robot_key} parcel={self.box_path} phase={self.state}')

    def _set_phase(self, phase):
        self.state = phase
        prim = self.stage.GetPrimAtPath(self.box_path)
        if prim.IsValid():
            prim.SetCustomDataByKey('payload_state', phase)

    def _fail(self, reason):
        self.last_error = reason
        self._set_phase('failed')
        print(f'PHYSICAL_Z1 failed reason={reason}', flush=True)
        return False

    def apply_targets(self):
        """Write arm position/velocity/effort requests; never write any body state."""
        if self.held and self.state == 'carried' and self._carry_reference_local is not None:
            self._nominal_world_reference = self._mount_pose()@self._carry_reference_local
            goal = self._update_force_targets(self._nominal_world_reference, 'carry')
            requested, _ = measured_world_arm_target(self.kinematics, goal, self._mount_pose(),
                self._arm_target.detach().cpu().numpy(), self.dt)
            self._arm_target = self.torch.as_tensor(requested, device=self.device, dtype=self._arm_target.dtype)
        self.robot.set_joint_position_target_index(target=self._arm_target.unsqueeze(0), joint_ids=self.arm_ids)
        grip = self.torch.tensor([[self._gripper_target]], device=self.device, dtype=self._arm_target.dtype)
        self.robot.set_joint_position_target_index(target=grip, joint_ids=self.gripper_ids)
        dt = self.dt
        if not 0 < dt <= .01+1e-9:
            raise RuntimeError(f'Z1 contact loop requires a control period <=10 ms; got dt={dt}')
        velocity = ((self._arm_target-self._previous_arm_target)/dt).clamp(-math.pi, math.pi)
        self.robot.set_joint_velocity_target_index(target=velocity.unsqueeze(0), joint_ids=self.arm_ids)
        gravity = _tensor(self.robot.data.gravity_compensation_forces)[:, self.robot.num_base_dofs:]
        effort_caps = self.motor_efforts.to(dtype=gravity.dtype)
        ff = gravity[:, self.arm_ids]
        ff = self.torch.maximum(self.torch.minimum(ff, effort_caps), -effort_caps)
        self.robot.set_joint_effort_target_index(target=ff, joint_ids=self.arm_ids)
        self._previous_arm_target = self._arm_target.clone()
        self._previous_gripper_target = self._gripper_target

    def _step(self):
        # The shared scheduler writes each arm exactly once before stepping.
        # A second write here would erase the velocity feedforward difference.
        self.step_fn()
        self._last_contact = self._read_contact()
        self._hard_contact_guard()

    def prepare(self):
        if self.held:
            raise RuntimeError('Cannot clear Z1 grasp state while a payload is held')
        self._gripper_target = self.config.gripper_open_rad
        self._previous_gripper_target = self._gripper_target
        self._force_latched = False
        self._carry_reference_local = None
        self._nominal_world_reference = None
        self._contact_enabled = False
        self._releasing = False
        self._released_once = False
        self.detected_position = None
        self.detected_geometry = None
        self.last_error = ''
        self.verified_lift = 0.
        self.max_hold_error = 0.
        self.placement_metrics = None
        self._release_retreat_target = None
        self._release_retreat_complete = False
        self._release_clearance_target = None
        self._release_complete = False
        self.loss_events.clear()
        self.close_completion = None
        self._set_phase('waiting')
        self.apply_targets()

    def disable_constraint(self):
        # Contact-only by construction: there is no authored parcel joint.
        return None

    def _preflight(self, goals, operation):
        mount = self._mount_pose()
        q = self._joint_position(self.arm_ids).detach().cpu().numpy().copy()
        sampled = 0
        minimum_margin = float('inf')
        try:
            for a, b in zip(goals, goals[1:]):
                distance = float(np.linalg.norm(b[:3, 3]-a[:3, 3]))
                pieces = max(1, math.ceil(distance/.01))
                for index in range(1, pieces+1):
                    target = interpolate_pose(a, b, index/pieces)
                    local = np.linalg.solve(mount, target)
                    q, info = self.kinematics.inverse_kinematics(local, q, max_iters=120,
                        position_tolerance=1e-4, orientation_tolerance=2e-3)
                    if not info['success']:
                        raise RuntimeError(f'Z1 path IK failed at sample {sampled}: {info}')
                    margin = np.minimum(q-self.kinematics.limits[:, 0], self.kinematics.limits[:, 1]-q)
                    minimum_margin = min(minimum_margin, float(margin.min()))
                    sampled += 1
        except Exception as error:
            report = dict(operation=operation, passed=False, sampled_waypoints=sampled,
                          minimum_joint_margin_rad=None, error=str(error))
            self.kinematic_plan_failures.append(report)
            raise RuntimeError(f'Z1 {operation} preflight rejected: {error}') from error
        report = dict(operation=operation, passed=True, sampled_waypoints=sampled,
                      minimum_joint_margin_rad=minimum_margin, collision_check=False,
                      measured_mount_pose=mount.tolist())
        self.kinematic_plan_history.append(report)
        self.kinematic_plan = report
        return report

    def _update_force_targets(self, goal, phase, close_fraction=None):
        dt = self.dt
        contact = self._last_contact
        paired = min(contact['normal_magnitudes']) > .05
        if phase == 'close':
            scheduled = self.config.gripper_open_rad*(1.-float(close_fraction))
            measured = float(self._joint_position(self.gripper_ids)[0])
            observed_moving = contact['normal_magnitudes'][1]
            if self._force_latched or observed_moving > .05:
                if not self._force_latched:
                    self._grip_reference = measured
                self._force_latched = True
                self._grip_reference += float(np.clip((self.config.force_target_n-observed_moving)*.01*dt,
                                                      -.0005, .0005))
            else:
                preload = .008 if measured > -1.10 else .035
                self._grip_reference = min(scheduled, measured+preload)
            self._gripper_target = unload_moving_jaw_preload(
                self._grip_reference, measured, observed_moving, self.config.force_target_n,
                self._previous_gripper_target, dt, self.config.gripper_open_rad)
            self._grip_reference = self._gripper_target
        elif self._contact_enabled and phase in ('lift', 'carry', 'carry_settle', 'delivery_align',
                                                  'place_preapproach', 'lowering'):
            observed_moving = contact['normal_magnitudes'][1]
            if self._force_latched or observed_moving > .05:
                self._force_latched = True
                self._grip_reference += float(np.clip((self.config.force_target_n-observed_moving)*.01*dt,
                                                      -.0005, .0005))
            self._gripper_target = unload_moving_jaw_preload(
                self._grip_reference, float(self._joint_position(self.gripper_ids)[0]), observed_moving,
                self.config.force_target_n, self._previous_gripper_target, dt, self.config.gripper_open_rad)
            self._grip_reference = self._gripper_target
        self._gripper_target = float(np.clip(self._gripper_target, self.config.gripper_open_rad, 0.))
        if self._contact_enabled:
            normal = goal[:3, 2]
            fixed_normal = max(0., -float(np.dot(contact['normal_vectors'][0].detach().cpu().numpy(), normal)))
            admittance_phase = {'lowering':'lower', 'place_preapproach':'delivery_align'}.get(phase, phase)
            offset = self._normal_controller.update(fixed_normal, paired, admittance_phase, dt)
            adjusted = goal.copy()
            adjusted[:3, 3] += normal*offset
            return adjusted
        return goal

    def _move_world(self, target, phase, duration_s, *, tolerance=None, force_loop=True):
        if self.step_fn is None:
            raise RuntimeError('PhysicalZ1Arm requires a shared-world policy step callback')
        self._set_phase(phase)
        # The force controller adds its correction to a nominal reference.
        # Starting the next segment at the measured, already-corrected wrist
        # would apply that offset twice at every phase boundary.
        start = (self._nominal_world_reference.copy() if self._nominal_world_reference is not None
                 else self._tip_pose_matrix())
        dt = self.dt
        count = max(1, math.ceil(duration_s/dt))
        target = np.asarray(target, dtype=float)
        for index in range(count):
            fraction = (index+1)/count
            smooth = fraction*fraction*(3.-2.*fraction)
            nominal = interpolate_pose(start, target, smooth)
            self._nominal_world_reference = nominal.copy()
            if force_loop:
                goal = self._update_force_targets(nominal, phase)
            else:
                goal = nominal
            try:
                requested, telemetry = measured_world_arm_target(self.kinematics, goal, self._mount_pose(),
                    self._arm_target.detach().cpu().numpy(), dt)
            except Exception as error:
                self._fail(f'{phase} world-frame IK: {error}')
                return False
            self._arm_target = self.torch.as_tensor(requested, device=self.device, dtype=self._arm_target.dtype)
            self._step()
            self._previous_gripper_target = self._gripper_target
            if index == 0 or (index+1) % max(1, round(1./dt)) == 0:
                print(f'PHYSICAL_Z1 progress phase={phase} step={index+1}/{count} '
                      f'ik_step={telemetry["maximum_target_step_rad"]:.4f} '
                      f'contacts={self._last_contact["normal_magnitudes"]}', flush=True)
        error = self._pose_error(target)
        allowed = self.config.position_tolerance_m if tolerance is None else tolerance
        # A completed interpolation is only a request. Let the unchanged
        # feedback controller settle for a bounded period before accepting the
        # measured pose; the position/orientation tolerances stay unchanged.
        if error[0]>allowed or error[1]>self.config.orientation_tolerance_rad:
            print(f'PHYSICAL_Z1 tracking_settle phase={phase} initial_error={error}',flush=True)
            for _ in range(max(1,math.ceil(self.config.tracking_settle_seconds/self.dt))):
                self._nominal_world_reference=target.copy()
                goal=self._update_force_targets(target,phase) if force_loop else target
                try:
                    requested,_=measured_world_arm_target(self.kinematics,goal,self._mount_pose(),
                        self._arm_target.detach().cpu().numpy(),self.dt)
                except Exception as failure:
                    return self._fail(f'{phase} settling IK: {failure}')
                self._arm_target=self.torch.as_tensor(requested,device=self.device,dtype=self._arm_target.dtype)
                self._step()
                error=self._pose_error(target)
                if error[0]<=allowed and error[1]<=self.config.orientation_tolerance_rad:
                    break
        if error[0] > allowed or error[1] > self.config.orientation_tolerance_rad:
            return self._fail(f'{phase} tracking outside tolerance: position={error[0]:.4f}m angle={error[1]:.3f}rad')
        return True

    def _pose_error(self, target):
        measured = self._tip_pose_matrix()
        position = float(np.linalg.norm(measured[:3, 3]-target[:3, 3]))
        angle = float(Rotation.from_matrix(measured[:3, :3]@target[:3, :3].T).magnitude())
        return position, angle

    def _tool_pose(self):
        pose = self._tip_pose_matrix()
        return (self.torch.as_tensor(pose[:3, 3], device=self.device, dtype=self._arm_target.dtype),
                self.torch.as_tensor(Rotation.from_matrix(pose[:3, :3]).as_quat(), device=self.device,
                                     dtype=self._arm_target.dtype))

    def stow_empty(self, steps=None, phase='stow_empty_arm'):
        if self.held:
            raise RuntimeError('Cannot stow a Z1 arm while a parcel is held')
        dt = self.dt
        # Existing fleet callers pass 150 for a three-second empty-arm fold.
        # Preserve that physical minimum in the 200 Hz Z1 loop.
        count = max(steps or 0, math.ceil(3./dt))
        start = self._joint_position(self.arm_ids).clone()
        self._set_phase(phase)
        self._nominal_world_reference = None
        for index in range(count):
            f = (index+1)/count
            blend = f*f*(3.-2.*f)
            self._arm_target = start+blend*(self._home_target-start)
            self._step()
        self._gripper_target = self.config.gripper_open_rad

    def grasp(self, object_id: str) -> bool:
        if self.held:
            return self.verified_lift > .07
        if not object_id:
            return self._fail('empty parcel identity')
        if self.detected_position is None or self.detected_geometry is None:
            return self._fail('Z1 grasp requires camera RGB-D center and visible-box yaw geometry')
        if not self.detected_geometry.get('source', '').startswith('rendered_depth_visible_rectangle'):
            return self._fail('Z1 grasp geometry did not come from the rendered depth image')
        if self.config.reset_before_grasp:
            self.stow_empty(phase='prepare_grasp_posture')
        for _ in range(max(1, round(.6/self.dt))):
            self._step()
        self._set_phase('planning_grasp')
        observed = np.asarray(self.detected_geometry['center_world'], dtype=float)
        yaw = float(self.detected_geometry['yaw_rad'])
        grasp = parcel_grasp_target(observed, yaw, self.tcp_in_stator,
            surface_normal=self.detected_geometry.get('surface_normal_world'),
            long_axis=self.detected_geometry.get('long_axis_world'),
            heading_reference_rad=self.config.grasp_heading_reference_rad)
        pregrasp = grasp.copy()
        pregrasp[2, 3] += self.config.approach_height
        inward_normal = grasp[:3, 2]
        pregrasp[:3, 3] -= inward_normal*.04
        side = grasp.copy()
        side[:3, 3] -= inward_normal*.04
        closed = grasp.copy()
        closed[:3, 3] += inward_normal*.004
        lifted = closed.copy()
        lifted[2, 3] += self.config.lift_height
        try:
            preflight = self._preflight([self._tip_pose_matrix(), pregrasp, side, grasp, closed, lifted], 'grasp_lift')
        except RuntimeError as error:
            return self._fail(str(error))
        self._grasp_tcp = self.tcp_in_stator.copy()
        self._grasp_yaw = yaw
        initial_box = self._box_pose_matrix()
        if not self._move_world(pregrasp, 'pregrasp', self.config.approach_seconds):
            return False
        if not self._move_world(side, 'approach', self.config.approach_seconds):
            return False
        if not self._move_world(grasp, 'approach', self.config.approach_seconds):
            return False
        self._contact_enabled = True
        self._force_latched = False
        self._grip_reference = float(self._joint_position(self.gripper_ids)[0])
        self._normal_controller = JawNormalAdmittance(self.config.force_target_n, precontact_feedback=True,
                                                     precontact_limit_n=self.config.precontact_limit_n)
        self._set_phase('closing_fingers')
        count = max(1, math.ceil(self.config.close_seconds/self.dt))
        readiness = PairedGripReadiness(self.config.force_target_n)
        close_reason = 'trajectory_duration'
        for index in range(count):
            smooth = ((index+1)/count)**2*(3.-2.*((index+1)/count))
            nominal = grasp.copy()
            nominal[:3, 3] += inward_normal*.004*smooth
            self._nominal_world_reference = nominal.copy()
            goal = self._update_force_targets(nominal, 'close', smooth)
            try:
                requested, _ = measured_world_arm_target(self.kinematics, goal, self._mount_pose(),
                    self._arm_target.detach().cpu().numpy(), self.dt)
            except Exception as error:
                return self._fail(f'close world-frame IK: {error}')
            self._arm_target = self.torch.as_tensor(requested, device=self.device, dtype=self._arm_target.dtype)
            self._step()
            self._previous_gripper_target = self._gripper_target
            if min(self._last_contact['normal_magnitudes']) > .05:
                self._lost_pair_ticks += 1
            else:
                self._lost_pair_ticks = 0
            if index == 0 or (index+1) % max(1, round(2./self.dt)) == 0:
                print(f'PHYSICAL_Z1 closing={index+1}/{count} contact_n={self._last_contact["normal_magnitudes"]} '
                      f'gripper_target_rad={self._gripper_target:.4f}', flush=True)
            if self.config.close_on_stable_contact and readiness.update(
                    self._last_contact['normal_magnitudes'],self._last_contact['total_magnitudes'],self.dt):
                close_reason = 'measured_paired_force_stable_0.5s'
                break
        self.close_completion = dict(reason=close_reason,steps=index+1,elapsed_s=(index+1)*self.dt,
            budget_s=self.config.close_seconds,stable_force_s=readiness.stable_s,
            final_normal_n=list(self._last_contact['normal_magnitudes']),
            final_total_n=list(self._last_contact['total_magnitudes']))
        print('PHYSICAL_Z1 close_completed '+str(self.close_completion),flush=True)
        paired_ticks = self._lost_pair_ticks
        if paired_ticks < math.ceil(.1/self.dt):
            self._contact_enabled = False
            self._gripper_target = self.config.gripper_open_rad
            return self._fail('Refusing to lift without 0.1 s of measured opposing jaw contact')
        self.held = True
        self._lost_pair_ticks = 0
        self._grasp_box_to_stator = np.linalg.solve(self._tip_pose_matrix(), self._box_pose_matrix())
        if not self._move_world(lifted, 'lift', self.config.lift_seconds):
            return False
        final_box = self._box_pose_matrix()
        rise = float(final_box[2, 3]-initial_box[2, 3])
        self.verified_lift = rise
        if rise < .07:
            self.held = False
            return self._fail(f'Z1 did not physically lift the parcel: {rise:.3f}m')
        self._set_phase('carried')
        self._carry_reference_local = np.linalg.solve(self._mount_pose(), lifted)
        self.kinematic_plan = dict(preflight, object_id=object_id,
                                   verified_lift_m=rise, contact_only=True)
        self.kinematic_plan_history[-1] = self.kinematic_plan
        return True

    def sync(self, pose: Any = None):
        del pose
        if not self.held or self._releasing:
            return
        contact = self._read_contact()
        self._last_contact = contact
        self._hard_contact_guard()
        paired = min(contact['normal_magnitudes']) > .05
        self._lost_pair_ticks = 0 if paired else self._lost_pair_ticks+1
        stator = self._tip_pose_matrix()
        box = self._box_pose_matrix()
        expected = stator@self._grasp_box_to_stator
        drift = float(np.linalg.norm(expected[:3, 3]-box[:3, 3]))
        self.max_hold_error = max(self.max_hold_error, drift)
        if self._lost_pair_ticks >= math.ceil(.25/self.dt):
            self.held = False
            row = dict(reason=f'opposing jaw contact absent for 0.25s; normal_n={contact["normal_magnitudes"]}',
                       tool_position=stator[:3, 3].tolist(), source='paired_gripper_contact',
                       contact_force_n=contact['normal_magnitudes'], debounce_steps=self._lost_pair_ticks,
                       evaluation_displacement_m=drift)
            self.loss_events.append(row)
            self._fail(row['reason'])
            from .mission import PayloadLost
            raise PayloadLost(self.last_error)

    def release(self) -> bool:
        if not self.held:
            if self._released_once:
                return self._verify_placement() if self._release_complete else self._finish_release()
            return self._fail('no physical parcel held')
        actual_box = self._box_pose_matrix()
        yaw = Rotation.from_matrix(actual_box[:3, :3]).as_euler('xyz')[2]
        desired_box = np.eye(4)
        desired_box[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        desired_box[:3, 3] = np.asarray(self.config.dropoff_position)
        hand_from_box = self._grasp_box_to_stator
        desired_hand = desired_box@np.linalg.inv(hand_from_box)
        above_box = desired_box.copy()
        above_box[2, 3] += self.config.placement_approach_height
        above_hand = above_box@np.linalg.inv(hand_from_box)
        clearance_hand = None
        if self.config.release_clearance_backoff_m or self.config.release_clearance_rise_m:
            # Move away from the shelf, then gain extra height before the base
            # starts. A pure upward move can hit this short arm's joint limits.
            away = self._mount_pose()[:2, 3] - desired_box[:2, 3]
            distance = float(np.linalg.norm(away))
            if distance < 1e-6:
                return self._fail('Cannot determine release clearance direction from shelf and mount')
            clearance_hand = above_hand.copy()
            clearance_hand[:2, 3] += away/distance*self.config.release_clearance_backoff_m
            clearance_hand[2, 3] += self.config.release_clearance_rise_m
        release_path = [self._tip_pose_matrix(), above_hand, desired_hand, above_hand]
        if clearance_hand is not None:
            release_path.append(clearance_hand)
        try:
            self._preflight(release_path, 'shelf_release')
        except RuntimeError as error:
            return self._fail(str(error))
        if not self._move_world(above_hand, 'place_preapproach', self.config.place_seconds):
            return False
        if not self._move_world(desired_hand, 'lowering', self.config.place_seconds):
            return False
        # Retrace the preflighted approach after opening. An arbitrary extra
        # 10 cm rise can exceed original joint limits at the short-arm dock.
        self._release_retreat_target = above_hand.copy()
        self._release_retreat_complete = False
        self._release_clearance_target = clearance_hand
        self._set_phase('opening_fingers')
        self._releasing = True
        previous = float(self._gripper_target)
        self._release_handoff = GripperReleaseHandoff(self.config.gripper_open_rad,
                                                      duration_s=self.config.release_seconds)
        count = max(1, math.ceil(self.config.release_seconds/self.dt))
        for index in range(count):
            scheduled = self.config.gripper_open_rad
            self._gripper_target = self._release_handoff.update('release', previous, scheduled, self.dt)
            requested, _ = measured_world_arm_target(self.kinematics, desired_hand, self._mount_pose(),
                self._arm_target.detach().cpu().numpy(), self.dt)
            self._arm_target = self.torch.as_tensor(requested, device=self.device, dtype=self._arm_target.dtype)
            self._step()
            previous = self._gripper_target
        self._gripper_target = self.config.gripper_open_rad
        self.held = False
        self._contact_enabled = False
        self._released_once = True
        self._releasing = False
        return self._finish_release()

    def _finish_release(self):
        """Resume an interrupted retreat; a settled parcel alone is not done."""
        if self._release_retreat_target is None:
            return self._fail('Released parcel has no validated arm retreat target')
        clearance = getattr(self, '_release_clearance_target', None)
        if (self.config.release_clearance_backoff_m or self.config.release_clearance_rise_m) and clearance is None:
            return self._fail('Released parcel has no validated departure clearance target')
        if not getattr(self, '_release_retreat_complete', False):
            if not self._move_world(self._release_retreat_target, 'retreat', self.config.place_seconds,
                                    force_loop=False):
                return False
            self._release_retreat_complete = True
        if clearance is not None:
            if not self._move_world(clearance, 'retreat_clearance', self.config.release_clearance_seconds,
                                    force_loop=False):
                return False
        self._set_phase('settling')
        for _ in range(max(1, math.ceil(self.config.settle_seconds/self.dt))):
            self._step()
        if not self._verify_placement():
            return False
        self._set_phase('released')
        self._release_complete = True
        self.last_error = ''
        return True

    def _verify_placement(self):
        box = self._box_pose_matrix()
        desired = np.asarray(self.config.dropoff_position)
        planar = float(np.linalg.norm(box[:2, 3]-desired[:2]))
        center_error = abs(float(box[2, 3]-desired[2]))
        velocity = _tensor(self.box.data.root_lin_vel_w)[0]
        speed = float(self.torch.linalg.vector_norm(velocity).item())
        surface_error = center_error
        fits = None
        if self.config.placement_surface_z is not None:
            dimensions = np.asarray(self.box.cfg.spawn.size, dtype=float)
            axes = Rotation.from_matrix(box[:3, :3]).as_matrix()
            extents = np.abs(axes)@ (dimensions*.5)
            surface_error = abs(float(box[2, 3]-extents[2]-self.config.placement_surface_z))
            dx, dy = self.config.placement_deck_size_xy
            fits = bool(abs(box[0, 3]-desired[0])+extents[0] <= dx/2
                        and abs(box[1, 3]-desired[1])+extents[1] <= dy/2)
        success = bool(planar < self.config.placement_planar_tolerance
                   and surface_error < self.config.placement_height_tolerance
                   and speed < .10 and fits is not False)
        self.placement_metrics = dict(success=success, planar_error_m=planar,
            height_error_m=surface_error, nominal_center_height_error_m=center_error,
            speed_mps=speed, footprint_within_deck=fits)
        if not success:
            return self._fail(f'Z1 parcel did not settle on target shelf: {self.placement_metrics}')
        return True


def _tensor(value):
    return getattr(value, 'torch', value)
