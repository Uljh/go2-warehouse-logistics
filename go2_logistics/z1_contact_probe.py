"""Real-mass Z1 contact grasp probe (fixed support or free Go2, never fleet acceptance).

Preparation, trajectory planning and evidence analysis do not import Isaac.
Run ``--prepare-only`` first; run physics through IsaacLab's Python launcher.
Every run owns its USD adapter, URDF, source snapshots, trace and result files.
The imported official asset is referenced read-only. Fixed-support mode anchors
link00 to the world; --mounted-go2 instead uses a free Go2 articulation.
The payload has no authored joint or pose writer in either mode.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from go2_logistics.z1_asset import (  # noqa: E402
    ARM_JOINT_NAMES, DESCRIPTION, GRIPPER_JOINT_NAME, prepare_z1_urdf,
)
from go2_logistics.z1_kinematics import Z1Kinematics  # noqa: E402


@dataclass(frozen=True)
class ProbeConfig:
    # The fleet parcel is 100 x 35 x 80 mm. This probe rotates its long side
    # into world y, so the downward-facing jaws close across world x = 35 mm.
    parcel_size_xyz_m: tuple[float, float, float] = (.035, .10, .08)
    parcel_mass_kg: float = .08
    parcel_xy_m: tuple[float, float] = (.32, 0.)
    parcel_spawn_offset_xy_m: tuple[float, float] = (0., 0.)
    table_top_m: float = .18
    base_height_m: float = .20
    jaw_sample_x_m: float = .13
    fixed_jaw_clearance_m: float = .002
    gripper_open_rad: float = -1.45
    gripper_closed_target_rad: float = 0.
    approach_m: float = .10
    lift_m: float = .12
    dt_s: float = .005
    minimum_lift_m: float = .09
    contact_threshold_n: float = .05
    hold_duration_s: float = 3.
    hold_translation_drift_m: float = .02
    hold_contact_fraction: float = .90
    static_friction: float = 1.2
    dynamic_friction: float = 1.0
    # These are explicit controller choices, not manufacturer gain claims.
    arm_stiffness: float = 160.
    arm_damping: float = 12.
    gripper_stiffness: float = 8.
    gripper_damping: float = .4
    gravity_compensation: bool = False
    lateral_approach_m: float = 0.
    contact_feedback: bool = False
    grip_force_target_n: float = 3.
    hard_contact_limit_n: float = 20.
    close_advance_m: float = 0.
    collision_mode: str = 'convexHull'
    close_seconds: float = 0.
    arm_velocity_feedforward: bool = False
    grip_force_gain: float = .05
    motion_seconds: float = 3.
    wrist_contact_feedback: bool = False
    precontact_normal_feedback: bool = False
    contact_wrench: bool = False
    grip_tracking_antiwindup: bool = False
    approach_seconds: float = 0.
    retreat_seconds: float = 3.
    support_enter_seconds: float = .20
    loaded_carry_seconds: float = 0.
    loaded_carry_distance_m: float = .50
    loaded_delivery: bool = False
    carry_control: str = 'proportional'
    carry_route: str = 'direct'
    carry_arrival_stop: bool = False
    carry_brake_seconds: float = 0.
    carry_arrival_offset_xy_m: tuple[float,float] = (0.,0.)
    carry_recovery: bool = False
    strict_physics_clock: bool = False
    audit_render_clock: bool = False
    audit_leg_contacts: bool = False
    carry_reference: str = 'mount'
    carry_stop_reference: str = 'world'
    stop_support_enter_seconds: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.carry_brake_seconds) or not 0<=self.carry_brake_seconds<=1.5:
            raise ValueError('Arrival gait braking must be within 0..1.5 seconds')
        if self.carry_brake_seconds and not self.carry_arrival_stop:
            raise ValueError('Arrival gait braking requires measured arrival stopping')
        if (len(self.carry_arrival_offset_xy_m)!=2
                or not np.isfinite(self.carry_arrival_offset_xy_m).all()
                or np.linalg.norm(self.carry_arrival_offset_xy_m)>.10):
            raise ValueError('Carry stance-transition compensation must be finite and within 0.10 metres')
        if self.carry_reference not in ('mount','level'):
            raise ValueError('Carry reference must be mount or level')
        if self.carry_stop_reference not in ('world','mount'):
            raise ValueError('Carry stop reference must be world or mount')
        if self.carry_stop_reference=='mount' and self.carry_reference!='mount':
            raise ValueError('Mount stop reference requires a mount-frame carry reference')
        if self.loaded_delivery and not self.loaded_carry_seconds:
            raise ValueError('Destination delivery requires the loaded walking probe')
        if self.carry_control not in ('proportional','velocity_pi'):
            raise ValueError('Unknown carry command controller')
        if self.carry_route not in ('direct','retreat_aisle','offset_aisle'):
            raise ValueError('Unknown loaded carry route')
        if self.stop_support_enter_seconds is not None and not (
                math.isfinite(self.stop_support_enter_seconds) and .02<=self.stop_support_enter_seconds<=.50):
            raise ValueError('Stop support entry must remain between 0.02 and 0.50 seconds')
        if not (math.isfinite(self.arm_stiffness) and self.arm_stiffness>0
                and math.isfinite(self.arm_damping) and self.arm_damping>=0):
            raise ValueError('Arm motor gains must be finite, with positive stiffness and nonnegative damping')
        if not (math.isfinite(self.support_enter_seconds) and .02<=self.support_enter_seconds<=.50):
            raise ValueError('Support force entry must remain between 0.02 and 0.50 seconds')
        if not (math.isfinite(self.loaded_carry_seconds)
                and (self.loaded_carry_seconds==0 or 10<=self.loaded_carry_seconds<=20)
                and math.isfinite(self.loaded_carry_distance_m) and .3<=self.loaded_carry_distance_m<=.8):
            raise ValueError('Loaded carry requires 10..20 seconds and .3...8 metres')
        if any(not math.isfinite(x) or x <= 0 for x in self.parcel_size_xyz_m):
            raise ValueError('Parcel dimensions must be finite and positive')
        if (len(self.parcel_spawn_offset_xy_m)!=2 or
                any(not math.isfinite(value) for value in self.parcel_spawn_offset_xy_m) or
                math.hypot(*self.parcel_spawn_offset_xy_m)>.004):
            raise ValueError('Initial-only parcel offset must remain within a 4 mm radius')
        if not (-math.pi / 2 <= self.gripper_open_rad < 0):
            raise ValueError('Open angle must remain inside the original gripper limits')
        if not (.10 <= self.jaw_sample_x_m <= .17):
            raise ValueError('Experimental jaw insertion reference must remain within [.10,.17] m')
        if not (0 < self.dt_s <= .01 and 0 < self.minimum_lift_m < self.lift_m):
            raise ValueError('Invalid physics step or lift acceptance threshold')
        if not (0 <= self.lateral_approach_m <= .08):
            raise ValueError('Lateral approach must be bounded in [0,.08] m')
        if not (0 <= self.close_advance_m <= .01):
            raise ValueError('Closing advance must remain within 10 mm')
        if self.collision_mode not in ('convexHull','convexDecomposition'):
            raise ValueError('Unsupported jaw collision approximation')
        if not (0 <= self.close_seconds <= 30):
            raise ValueError('Closing phase must remain bounded within 30 seconds')
        if not (0 < self.grip_force_target_n < self.hard_contact_limit_n == 20
                and 0 < self.grip_force_gain <= .1 and 3 <= self.motion_seconds <= 10):
            raise ValueError('Invalid bounded contact controller or motion duration')
        if self.wrist_contact_feedback and not self.contact_feedback:
            raise ValueError('Wrist contact feedback requires the original grip loop and contact guards')
        if self.precontact_normal_feedback and not self.wrist_contact_feedback:
            raise ValueError('Precontact normal feedback requires bounded wrist contact feedback')
        if self.grip_tracking_antiwindup and not self.contact_feedback:
            raise ValueError('Moving-jaw tracking feedback requires measured contact feedback')
        if not (math.isfinite(self.approach_seconds) and (self.approach_seconds==0 or 1<=self.approach_seconds<=10)
                and math.isfinite(self.retreat_seconds) and 1<=self.retreat_seconds<=10):
            raise ValueError('Approach and retreat durations must be bounded within 1 to 10 seconds')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _mesh(path: Path):
    # trimesh reads STL coordinates without scaling them. scipy/trimesh are
    # already dependencies of the project's Isaac Python environment.
    import trimesh
    return trimesh.load(path, force='mesh', process=False)


def hull_slice_bounds(equations: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Intersect a scipy convex hull with a z-parallel line, in metres."""
    lower, upper = -math.inf, math.inf
    for a, b, c, d in equations:
        residual = a*x + b*y + d
        if abs(c) < 1e-10:
            if residual > 1e-8:
                return None
        elif c < 0:
            lower = max(lower, -residual/c)
        else:
            upper = min(upper, -residual/c)
    return (float(lower), float(upper)) if lower <= upper + 1e-8 else None


def triangle_slice_heights(triangles: np.ndarray, x: float, y: float) -> list[float]:
    """Actual STL surface intersections, independently of its convex hull."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b-a, c-a
    denominator = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    valid = abs(denominator) > 1e-14
    safe = np.where(valid, denominator, 1.)
    px, py = x-a[:, 0], y-a[:, 1]
    s = (px*v[:, 1] - py*v[:, 0])/safe
    t = (u[:, 0]*py - u[:, 1]*px)/safe
    valid &= (s >= -1e-8) & (t >= -1e-8) & (s+t <= 1+1e-8)
    return sorted(float(value) for value in (a[:, 2]+s*u[:, 2]+t*v[:, 2])[valid])


def geometry_report(config: ProbeConfig) -> dict:
    """Calibrate against the selected approximation and preserve source geometry.

    The fixed jaw's motor housing and finger are one collision mesh. Its
    convex hull fills a concavity. Using the visible finger tip as TCP would
    therefore begin with payload interpenetration in the current asset.
    """
    from scipy.spatial import ConvexHull
    files = {name: DESCRIPTION/'meshes/collision'/f'z1_Gripper{name}.STL' for name in ('Stator', 'Mover')}
    meshes = {name: _mesh(path) for name, path in files.items()}
    stator = meshes['Stator']
    hull = ConvexHull(stator.vertices)
    half_depth = config.parcel_size_xyz_m[2]/2
    x_min, x_max = config.jaw_sample_x_m-half_depth, config.jaw_sample_x_m+half_depth
    samples = []
    for x in np.linspace(x_min, x_max, 81):
        for y in np.linspace(-.04, .04, 17):
            bounds = hull_slice_bounds(hull.equations, float(x), float(y))
            if bounds is not None:
                heights = triangle_slice_heights(stator.triangles, float(x), float(y)) if config.collision_mode=='convexDecomposition' else []
                if config.collision_mode=='convexHull' or heights:
                    samples.append((float(x), float(y), max(heights) if heights else bounds[1]))
    if not samples:
        raise ValueError('Parcel depth does not overlap the fixed jaw collision hull')
    # Maximizing a piecewise concave hull surface over a rectangle can peak
    # between samples. With these data it peaks at x_min,y=0; retain 2 mm
    # clearance and record both method and spacing instead of claiming exact TCP.
    contact_x, contact_y, stator_surface = max(samples, key=lambda row: row[2])
    tcp_z = stator_surface + config.parcel_size_xyz_m[0]/2 + config.fixed_jaw_clearance_m
    q = config.gripper_open_rad
    R = np.array([[math.cos(q), 0, math.sin(q)], [0, 1, 0], [-math.sin(q), 0, math.cos(q)]])
    mover_open = meshes['Mover'].vertices@R.T + np.array([.049, 0., 0.])
    sections = []
    for x in (.09, .10, .12, .13, .14):
        original = triangle_slice_heights(stator.triangles, x, 0.)
        bounds = hull_slice_bounds(hull.equations, x, 0.)
        sections.append(dict(x_m=x, y_m=0., mesh_intersections_z_m=original,
            mesh_upper_surface_z_m=max(original) if original else None,
            hull_bounds_z_m=list(bounds) if bounds else None,
            filled_concavity_m=bounds[1]-max(original) if bounds and original else None))
    return dict(
        source_meshes={name: dict(path=str(path), sha256=_sha(path),
            bounds_m=meshes[name].bounds.tolist()) for name, path in files.items()},
        collision_approximation=config.collision_mode+'; verified again against USD before physics',
        stator_sections=sections,
        gripper_joint=dict(type='revolute', axis_in_stator=[0., 1., 0.], origin_m=[.049, 0., 0.],
            original_limits_rad=[-math.pi/2, 0.], open_rad=q, closed_target_rad=config.gripper_closed_target_rad),
        tcp_in_stator_m=[config.jaw_sample_x_m, 0., float(tcp_z)],
        tcp_status='geometric candidate from '+config.collision_mode+', not a physically validated grasp TCP',
        sampled_fixed_jaw_maximum=dict(x_m=contact_x, y_m=contact_y, z_m=stator_surface),
        sample_spacing_x_m=(x_max-x_min)/80, sample_spacing_y_m=.005,
        parcel_extent_in_stator_x_m=[x_min, x_max],
        mover_open_bounds_in_stator_m=[mover_open.min(0).tolist(), mover_open.max(0).tolist()],
        open_mover_depth_clearance_m=float(x_min-mover_open[:, 0].max()),
        parcel_size_world_xyz_m=list(config.parcel_size_xyz_m),
        matches_fleet_parcel_dimensions=True,
        limitation=('Single convex hull fills jaw concavity.' if config.collision_mode=='convexHull' else
            'Convex decomposition approximates the original jaw STL; cooked hull contacts require physical validation.')+
            ' Original mesh vertices, mass, inertia and joint limits retained.',
    )


def make_plan(kin: Z1Kinematics, geometry: dict, config: ProbeConfig) -> tuple[list[dict], dict]:
    """Six-axis bounded IK; the calibrated TCP and orientation remain explicit."""
    R = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    tcp = np.asarray(geometry['tcp_in_stator_m'])
    parcel = np.array([*config.parcel_xy_m, config.table_top_m+config.parcel_size_xyz_m[2]/2])
    base = np.array([0., 0., config.base_height_m])
    grasp = np.eye(4)
    grasp[:3, :3] = R
    grasp[:3, 3] = parcel-base-R@tcp
    pre = grasp.copy()
    pre[2, 3] += config.approach_m
    pre[0, 3] -= config.lateral_approach_m
    side = grasp.copy()
    side[0, 3] -= config.lateral_approach_m
    closed = grasp.copy()
    closed[0, 3] += config.close_advance_m
    lifted = closed.copy()
    lifted[2, 3] += config.lift_m
    seed = np.array([0., 1.8, -1.4, 1.2, 0., 0.])
    q, info = kin.inverse_kinematics(pre, seed)
    if not info['success']:
        raise RuntimeError(f'Pregrasp IK failed: {info}')
    initial = q.copy()
    commands, diagnostics = [], []
    # Cartesian paths are solved at <= 5 mm spacing, then time interpolated.
    approach = ([('approach', side, config.approach_seconds or 3., config.gripper_open_rad),
        ('approach', grasp, config.approach_seconds or 3., config.gripper_open_rad)] if config.lateral_approach_m else
        [('approach', grasp, config.approach_seconds or 2.5, config.gripper_open_rad)])
    schedule = [('initial_settle', pre, 1.5, config.gripper_open_rad)]+approach+[
        ('close', closed, config.close_seconds or (8. if config.contact_feedback else 2.), config.gripper_closed_target_rad),
        ('lift', lifted, config.motion_seconds, config.gripper_closed_target_rad),
        ('hold', lifted, config.hold_duration_s, config.gripper_closed_target_rad),
        ('lower', closed, config.motion_seconds, config.gripper_closed_target_rad),
        ('release', closed, 2., config.gripper_open_rad),
        ('retreat', pre, config.retreat_seconds, config.gripper_open_rad),
        ('settle', pre, 1.5, config.gripper_open_rad)]
    previous_pose, previous_grip = pre, config.gripper_open_rad
    for phase, goal, duration, grip in schedule:
        segments = max(1, math.ceil(np.linalg.norm(goal[:3, 3]-previous_pose[:3, 3])/.005))
        knots = [q.copy()]
        for k in range(1, segments+1):
            pose = goal.copy()
            pose[:3, 3] = previous_pose[:3, 3]+(goal[:3, 3]-previous_pose[:3, 3])*(k/segments)
            q, row = kin.inverse_kinematics(pose, q)
            diagnostics.append(dict(phase=phase, fraction=k/segments, **row))
            if not row['success']:
                raise RuntimeError(f'{phase} IK failed at {k}/{segments}: {row}')
            knots.append(q.copy())
        count = round(duration/config.dt_s)
        for step in range(1, count+1):
            f = step/count
            smooth = f*f*(3-2*f)
            cursor = smooth*segments
            segment = min(int(cursor), segments-1)
            blend = min(1., cursor-segment)
            command = (1-blend)*knots[segment] + blend*knots[segment+1]
            commands.append(dict(phase=phase, arm_rad=command.tolist(),
                gripper_rad=float(previous_grip+(grip-previous_grip)*smooth)))
        previous_pose, previous_grip = goal, grip
    return commands, dict(initial_arm_rad=initial.tolist(), initial_gripper_rad=config.gripper_open_rad,
        parcel_spawn_world_m=(parcel+np.r_[config.parcel_spawn_offset_xy_m,0.]).tolist(),
        nominal_parcel_world_m=parcel.tolist(),base_world_m=base.tolist(),
        grasp_stator_pose_in_base=grasp.tolist(), initial_stator_pose_in_base=pre.tolist(),
        lift_stator_pose_in_base=lifted.tolist(), ik_diagnostics=diagnostics,
        ticks=len(commands), duration_s=len(commands)*config.dt_s)


def build_fixed_asset(source: Path, destination: Path, base_height: float, *, collision_mode='convexHull') -> dict:
    """Read-only reference plus one world anchor; retain all original inertials."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Arm')
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(source.as_posix())
    root.GetPrim().GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    root.ClearXformOpOrder()
    root.AddTranslateOp().Set(Gf.Vec3d(0., 0., base_height))
    bodies, disabled = {}, []
    for prim in list(stage.Traverse()):
        schemas = prim.GetMetadata('apiSchemas')
        applied = schemas.GetAppliedItems() if schemas else []
        kept = [name for name in applied if not name.endswith('ArticulationRootAPI')]
        if kept != applied:
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(kept))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetName()] = prim
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                disabled.append(str(prim.GetPath()))
                prim.SetActive(False)
    base = bodies['link00']
    position = UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation()
    anchor = UsdPhysics.FixedJoint.Define(stage, '/Arm/ProbeBaseAnchor')
    anchor.CreateBody1Rel().SetTargets([base.GetPath()])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(position))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0.))
    anchor.CreateLocalRot0Attr(Gf.Quatf(1.))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1.))
    # A fixed articulation roots at the world joint. Rooting at the nested
    # body leaves its sibling world joint outside the articulation subtree.
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
    collision_paths = [prim.GetPath() for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and 'gripper' in str(prim.GetPath()).lower()]
    collisions = []
    for path in collision_paths:
        prim = stage.GetPrimAtPath(path)
        if collision_mode=='convexDecomposition':
            parent=prim
            while parent and parent.IsInstanceProxy():
                parent=parent.GetParent()
            if parent and parent.IsInstance():
                parent.SetInstanceable(False)
            prim=stage.GetPrimAtPath(path)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(collision_mode)
        if prim:
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            collisions.append(dict(path=str(prim.GetPath()), approximation=approximation))
    if len(collisions) != 2 or any(row['approximation'] != collision_mode for row in collisions):
        raise RuntimeError(f'TCP calibration requires the two audited convex hulls: {collisions}')
    stage.GetRootLayer().Export(str(destination))
    return dict(asset=str(destination), source=str(source), source_sha256=_sha(source),
        anchor_path=str(anchor.GetPath()), disabled_original_world_joints=disabled,
        body_paths={name: str(prim.GetPath()) for name, prim in bodies.items()},
        gripper_collisions=collisions, added_payload_joints=[])


def evaluate_trace(samples: list[dict], config: ProbeConfig, integrity: dict) -> dict:
    """Independent checks over measured motion, pair-filtered forces and audit.

    A commanded lift or dispatcher flag is never sufficient. Missing phases,
    missing contact arrays or a runtime exception cannot pass this evaluator.
    """
    checks, metrics = {}, {}
    expected = ('initial_settle', 'approach', 'close', 'lift', 'hold', 'lower', 'release', 'retreat', 'settle')
    phases = {name: [s for s in samples if s['phase'] == name] for name in expected}
    checks['all_phases_measured'] = all(phases.values())
    sequence=[]
    for sample in samples:
        if not sequence or sample['phase']!=sequence[-1]:
            sequence.append(sample['phase'])
    checks['phase_order'] = sequence==list(expected)
    checks['complete_control_timing'] = bool(samples) and all(
        sample.get('tick')==i+1 and isinstance(sample.get('time_s'),(int,float))
        and math.isfinite(sample['time_s']) and abs(sample['time_s']-(i+1)*config.dt_s)<1e-8
        for i,sample in enumerate(samples))
    checks['contact_force_guard'] = all(
        len(sample.get('jaw_contact_n',[]))==2 and all(math.isfinite(f) and 0<=f<=config.hard_contact_limit_n
            for f in sample['jaw_contact_n']) for sample in samples)
    checks['total_contact_force_recorded_and_guarded'] = bool(samples) and all(
        len(sample.get('jaw_total_contact_n',[]))==2
        and all(math.isfinite(force) and 0<=force<=config.hard_contact_limit_n
                for force in sample['jaw_total_contact_n']) for sample in samples)
    checks['asset_and_runtime_integrity'] = bool(integrity.get('passed'))
    checks['no_payload_constraints_or_pose_writes'] = (
        integrity.get('payload_joints') == [] and integrity.get('payload_pose_writes') == 0
        and integrity.get('resets_during_probe') == 0 and integrity.get('payload_kinematic') is False)
    if not samples:
        return dict(passed=False, checks=checks, metrics=metrics, scope='fixed-support Z1 contact probe only')
    baseline_samples = phases['initial_settle'][-max(1, round(.5/config.dt_s)):]
    baseline = float(np.median([s['parcel_pos_w'][2] for s in baseline_samples])) if baseline_samples else samples[0]['parcel_pos_w'][2]
    metrics['baseline_payload_z_m'] = baseline
    def paired(sample):
        forces = sample.get('jaw_contact_n', [])
        return len(forces) == 2 and all(f > config.contact_threshold_n for f in forces)
    run, longest = 0, 0
    for s in phases['close']:
        run = run+1 if paired(s) else 0
        longest = max(longest, run)
    metrics['close_paired_contact_longest_s'] = longest*config.dt_s
    checks['two_jaws_contact_before_lift'] = longest*config.dt_s >= .1
    # Exclude the first 0.5 s of the already stationary hold, preserving at
    # least 2.5 s of continuous evidence at the default duration.
    hold = phases['hold'][round(.5/config.dt_s):]
    metrics['hold_measured_s'] = len(hold)*config.dt_s
    if hold:
        lifts = [s['parcel_pos_w'][2]-baseline for s in hold]
        offsets = np.asarray([s['parcel_in_stator_m'] for s in hold])
        drift = float(np.max(np.linalg.norm(offsets-offsets[0], axis=1)))
        fraction = sum(paired(s) for s in hold)/len(hold)
        metrics.update(minimum_hold_lift_m=min(lifts), maximum_hold_lift_m=max(lifts),
            hold_translation_drift_m=drift, hold_paired_contact_fraction=fraction)
        checks['actual_lift_and_sustained_hold'] = bool(len(hold)*config.dt_s >= 2.4
            and min(lifts) >= config.minimum_lift_m and drift <= config.hold_translation_drift_m
            and fraction >= config.hold_contact_fraction)
    else:
        checks['actual_lift_and_sustained_hold'] = False
    settled = phases['settle'][-round(1./config.dt_s):]
    checks['released_and_stable_on_table'] = bool(len(settled)*config.dt_s >= .99 and all(
        abs(s['parcel_pos_w'][2]-(config.table_top_m+config.parcel_size_xyz_m[2]/2)) <= .012
        and np.linalg.norm(np.asarray(s['parcel_pos_w'][:2])-config.parcel_xy_m) <= .04
        and np.linalg.norm(s['parcel_lin_vel_w']) <= .025
        and np.linalg.norm(s['parcel_ang_vel_w']) <= .15
        and s.get('table_contact_n', 0.) > .1
        and max(s.get('jaw_contact_n', [math.inf])) < config.contact_threshold_n
        and s.get('gripper_rad', 0.) < -1.20
        and s.get('tcp_payload_distance_m', 0.) >= .075
        for s in settled))
    checks['finite_trace'] = all(np.isfinite(np.asarray(s['parcel_pos_w'])).all() for s in samples)
    return dict(passed=all(checks.values()), checks=checks, metrics=metrics,
        scope='fixed-support real-mass Z1 contact grasp; no Go2, navigation or fleet acceptance')


def prepare_run(output: Path, import_report: Path, config: ProbeConfig, *, directory_reserved=False,
                mounted_go2=False) -> dict:
    if config.loaded_carry_seconds and not mounted_go2:
        raise ValueError('Loaded walking requires the free Go2, never a fixed support')
    output.mkdir(parents=True, exist_ok=directory_reserved)
    imported = json.loads(import_report.read_text(encoding='utf-8'))
    if not imported.get('passed'):
        raise RuntimeError('An independently validated original-mass Z1 import is required')
    source = Path(imported['usd_path']).resolve()
    if _sha(source) != imported['usd_root_sha256']:
        raise RuntimeError('Imported USD root changed since its validation report')
    urdf, spec = prepare_z1_urdf(output/'z1_with_gripper.urdf', overwrite=True)
    if spec.generated_sha256 != imported['urdf_sha256']:
        raise RuntimeError('Generated official URDF differs from the imported model')
    kin = Z1Kinematics.from_urdf(urdf)
    geometry = geometry_report(config)
    if geometry['open_mover_depth_clearance_m'] <= .002:
        raise RuntimeError('Open gripper intersects the parcel depth range')
    commands, plan = make_plan(kin, geometry, config)
    if config.loaded_carry_seconds:
        from go2_logistics.z1_carry_control import extend_with_carry
        commands,plan=extend_with_carry(commands,plan,config.dt_s,
            config.loaded_carry_seconds,config.loaded_carry_distance_m,delivery=config.loaded_delivery,
            route=config.carry_route,arrival_offset_xy=config.carry_arrival_offset_xy_m)
    if mounted_go2:
        if imported.get('fixed_base'):
            raise RuntimeError('Mounted probe requires the independently validated floating Z1 source')
        from go2_logistics.mounted_z1_asset import build_mounted_z1_asset
        asset=build_mounted_z1_asset(ROOT/'artifacts/go2_asset/go2.usd',source,output/'go2_z1.usda',
            gripper_collision_mode=config.collision_mode)
    else:
        asset = build_fixed_asset(source, output/'fixed_z1.usda', config.base_height_m, collision_mode=config.collision_mode)
    from go2_logistics.import_z1_isaac import audit_import
    audited = audit_import(asset['asset'], spec)
    if not audited['passed']:
        raise RuntimeError('Fixed-support adapter changed original mass, inertia or limits')
    _json(output/'geometry.json', geometry)
    _json(output/'plan.json', plan)
    _json(output/'asset_audit.json', dict(**audited, adapter=asset))
    _json(output/'config.json', asdict(config))
    _json(output/'import_validation_source.json', imported)
    sources = {}
    for name in ('z1_contact_probe.py', 'z1_contact_control.py', 'z1_contact_wrench.py', 'z1_asset.py',
                 'z1_kinematics.py', 'import_z1_isaac.py','z1_mounted_contact.py','z1_world_target.py',
                 'mounted_z1_asset.py','balance_recovery.py','recovery_targets.py','z1_carry_control.py',
                 'fleet_runtime.py'):
        path = Path(__file__).with_name(name)
        (output/'source').mkdir(exist_ok=True)
        (output/'source'/name).write_bytes(path.read_bytes())
        sources[name] = _sha(path)
    _json(output/'manifest.json', dict(created_at=datetime.now().astimezone().isoformat(),
        source_sha256=sources, import_report=str(import_report.resolve()),
        scope='preflight only; no physics success implied', mounted_go2=mounted_go2,config=asdict(config)))
    return dict(output=output, config=config, spec=spec, geometry=geometry, plan=plan,
        commands=commands, asset=asset, audit=audited,mounted_go2=mounted_go2)


def _tensor(value):
    return getattr(value, 'torch', value)


def run_physics(prepared: dict, args, app) -> dict:
    """One bounded experiment; no environment auto-reset and no pose recovery."""
    os.environ['WARP_CACHE_PATH'] = str(prepared['output']/'warp_cache')
    output, cfg = prepared['output'], prepared['config']
    mounted=prepared.get('mounted_go2',False)
    samples, writer = [], None
    integrity = dict(passed=False, payload_pose_writes=0, resets_during_probe=0,
        payload_joints=None, payload_kinematic=None)
    failure = ''
    started = time.monotonic()
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from isaaclab.sim import SimulationContext
        from isaaclab_physx.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        from pxr import Usd, UsdPhysics
        sim = SimulationContext(sim_utils.SimulationCfg(device='cpu', dt=cfg.dt_s,
            render_interval=20, log_dir=str(output/'sim_logs')))
        sim.set_camera_view([1.45, -1.5, 1.05] if mounted else [1.2, -.9, .8], [.15, 0., .40] if mounted else [.2, 0., .3])
        ground = sim_utils.GroundPlaneCfg()
        ground.func('/World/Ground', ground)
        light = sim_utils.DomeLightCfg(intensity=1800.)
        light.func('/World/Light', light)
        material = sim_utils.RigidBodyMaterialCfg(static_friction=cfg.static_friction,
            dynamic_friction=cfg.dynamic_friction, friction_combine_mode='max')
        table = sim_utils.CuboidCfg(size=(.20, .22, cfg.table_top_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1, .3, .4)))
        table.func('/World/Table', table, translation=(*cfg.parcel_xy_m, cfg.table_top_m/2))
        if cfg.loaded_delivery:
            table.func('/World/DestinationTable',table,translation=(cfg.parcel_xy_m[0],
                cfg.parcel_xy_m[1]+cfg.loaded_carry_distance_m,cfg.table_top_m/2))
        if not mounted:
            support = sim_utils.CuboidCfg(size=(.12, .12, cfg.base_height_m),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .25, .25)))
            support.func('/World/Support', support, translation=(0., 0., cfg.base_height_m/2))
        all_names = list(ARM_JOINT_NAMES)+[GRIPPER_JOINT_NAME]
        spec = prepared['spec']
        robot_cfg=ArticulationCfg(prim_path='/World/Arm',
            spawn=sim_utils.UsdFileCfg(usd_path=str(output/'fixed_z1.usda'),
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., cfg.base_height_m), joint_pos=dict(zip(all_names,
                prepared['plan']['initial_arm_rad']+[cfg.gripper_open_rad]))),
            actuators=dict(arm=ImplicitActuatorCfg(joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit_sim={name: spec.joint_efforts_nm[name] for name in ARM_JOINT_NAMES},
                velocity_limit_sim=math.pi, stiffness=cfg.arm_stiffness, damping=cfg.arm_damping),
                gripper=ImplicitActuatorCfg(joint_names_expr=[GRIPPER_JOINT_NAME],
                    effort_limit_sim=spec.joint_efforts_nm[GRIPPER_JOINT_NAME], velocity_limit_sim=3.1415,
                    stiffness=cfg.gripper_stiffness, damping=cfg.gripper_damping)))
        if mounted:
            from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
            mounted_cfg=UNITREE_GO2_CFG.replace(prim_path='/World/Robot')
            mounted_cfg.spawn.usd_path=prepared['asset']['asset']
            mounted_cfg.spawn.variants={'Physics':'physx'}
            mounted_cfg.spawn.articulation_props.solver_position_iteration_count=16
            mounted_cfg.spawn.articulation_props.solver_velocity_iteration_count=4
            mounted_cfg.init_state.pos=(0.,0.,cfg.base_height_m-.10)
            mounted_cfg.init_state.joint_pos.update(robot_cfg.init_state.joint_pos)
            mounted_cfg.actuators.update(robot_cfg.actuators)
            robot_cfg=mounted_cfg
        robot=Articulation(robot_cfg)
        parcel = RigidObject(RigidObjectCfg(prim_path='/World/Parcel',
            spawn=sim_utils.CuboidCfg(size=cfg.parcel_size_xyz_m,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=cfg.parcel_mass_kg),
                activate_contact_sensors=True, physics_material=material,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9, .08, .03))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(prepared['plan']['parcel_spawn_world_m']))))
        paths = {name: path.replace('/Robot/' if mounted else '/Arm/',
            '/World/Robot/' if mounted else '/World/Arm/', 1) for name, path in prepared['asset']['body_paths'].items()}
        # The URDF keeps links nested with resetXformStack. The generic
        # activator stops below its first rigid body, so activate each jaw
        # explicitly rather than assuming spawn reached every nested link.
        for name in ('gripperStator', 'gripperMover'):
            sim_utils.activate_contact_sensors(paths[name], stage=sim.stage)
        contacts = [ContactSensor(ContactSensorCfg(prim_path=paths[name], update_period=0.,track_friction_forces=True,
            track_contact_points=cfg.contact_wrench,
            max_contact_data_count_per_prim=128,
            filter_prim_paths_expr=['/World/Parcel'])) for name in ('gripperStator', 'gripperMover')]
        # A payload sensor filtered to the table is separate from jaw sensors.
        # Static collision bodies are valid PhysX filter targets.
        table_contact = ContactSensor(ContactSensorCfg(prim_path='/World/Parcel', update_period=0.,track_friction_forces=True,
            track_contact_points=cfg.contact_wrench,
            max_contact_data_count_per_prim=128,
            filter_prim_paths_expr=['/World/Table']))
        destination_contact=(ContactSensor(ContactSensorCfg(prim_path='/World/Parcel',update_period=0.,
            track_friction_forces=True,track_contact_points=cfg.contact_wrench,
            max_contact_data_count_per_prim=128,filter_prim_paths_expr=['/World/DestinationTable']))
            if cfg.loaded_delivery else None)
        contact_sensors=contacts+[table_contact]+([destination_contact] if destination_contact else [])
        feet_contact=ContactSensor(ContactSensorCfg(prim_path='/World/Robot/.*_foot',update_period=0.)) if mounted else None
        leg_table_contacts={}
        if mounted and cfg.audit_leg_contacts:
            filters=['/World/Table']+(['/World/DestinationTable'] if cfg.loaded_delivery else [])
            for leg in ('FL','FR','RL','RR'):
                for part in ('calf','foot'):
                    name=f'{leg}_{part}'
                    leg_table_contacts[name]=ContactSensor(ContactSensorCfg(
                        prim_path=f'/World/Robot/{name}',update_period=0.,
                        track_friction_forces=True,max_contact_data_count_per_prim=64,
                        filter_prim_paths_expr=filters))
        camera = None
        if args.video:
            from isaaclab.sensors import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg
            from go2_logistics.video_recording import get_mission_writer
            camera = Camera(CameraCfg(prim_path='/World/ProbeCamera', update_period=.1,
                height=480, width=640, data_types=['rgb'], renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=20.955),
                offset=CameraCfg.OffsetCfg(pos=(1.1, -.9, .8), convention='world')))
            writer = get_mission_writer(output/'contact_probe.mp4', fps=10)
        sim.reset()  # The one initial physics initialization; never called again.
        if cfg.strict_physics_clock:
            # Kit app pumping must not introduce uncounted physical steps.
            # Explicit PhysicsManager.step remains the sole simulation clock.
            sim.set_setting('/app/player/playSimulations',False)
        arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        grip_ids, _ = robot.find_joints([GRIPPER_JOINT_NAME], preserve_order=True)
        body_ids, _ = robot.find_bodies(['gripperStator', 'gripperMover'], preserve_order=True)
        if len(arm_ids) != 6 or len(grip_ids) != 1 or len(body_ids) != 2 or bool(robot.is_fixed_base)==mounted:
            raise RuntimeError(f'Expected fixed-support Z1 six-axis arm and original rotating gripper; '
                f'arm_ids={arm_ids}, gripper_ids={grip_ids}, body_ids={body_ids}, fixed={robot.is_fixed_base}')
        for sensor, name in zip(contacts, ('gripperStator', 'gripperMover')):
            if sensor.body_names != [name]:
                raise RuntimeError(f'Jaw sensor resolved unexpected rigid bodies: {sensor.body_names}')
        initial = _tensor(robot.data.default_joint_pos).clone()
        if mounted:
            # SimulationContext.reset performs initialization/warm-up before
            # user control. Restore only the initial robot state, once, before
            # measurements; never reset or write state inside the control loop.
            robot.write_root_pose_to_sim_index(root_pose=torch.tensor([[0.,0.,cfg.base_height_m-.10,0.,0.,0.,1.]]))
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6)))
        robot.write_joint_position_to_sim_index(position=initial)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial))
        robot.reset()
        if mounted:
            sim.forward()
            robot.update(0.)
        ids = arm_ids+grip_ids
        from go2_logistics.z1_asset import ARM_LINK_NAMES,GRIPPER_LINK_NAMES
        arm_bodies=robot.find_bodies(list(ARM_LINK_NAMES)+list(GRIPPER_LINK_NAMES),preserve_order=True)[0]
        actual_mass = float(_tensor(robot.data.default_mass)[0,arm_bodies].sum())
        actual_limits = _tensor(robot.data.joint_pos_limits)[0, ids].cpu().numpy()
        expected_limits = np.asarray([spec.joint_limits[n] for n in all_names])
        payload_joints = []
        robot_world_joints=[]
        for prim in sim.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
                if mounted and str(prim.GetPath()).startswith('/World/Robot/') and (
                        not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets()):
                    robot_world_joints.append(str(prim.GetPath()))
                if any(str(path) == '/World/Parcel' or str(path).startswith('/World/Parcel/') for path in targets):
                    payload_joints.append(str(prim.GetPath()))
        payload_kinematic = bool(UsdPhysics.RigidBodyAPI(sim.stage.GetPrimAtPath('/World/Parcel')).GetKinematicEnabledAttr().Get())
        integrity.update(actual_arm_mass_kg=actual_mass, expected_arm_mass_kg=spec.total_mass_kg,
            actual_joint_limits_rad=actual_limits.tolist(), expected_joint_limits_rad=expected_limits.tolist(),
            payload_joints=payload_joints, payload_kinematic=payload_kinematic,
            fixed_base=bool(robot.is_fixed_base), initial_joint_state_writes=1,
            initial_root_state_writes=int(mounted),initial_root_pos_w=_tensor(robot.data.root_pos_w)[0].cpu().tolist(),
            robot_world_joints=robot_world_joints,
            passed=bool(abs(actual_mass-spec.total_mass_kg) < 1e-4
                and np.allclose(actual_limits, expected_limits, atol=1e-4) and not payload_joints and not payload_kinematic))
        _json(output/'runtime_integrity.json', integrity)
        if not integrity['passed']:
            raise RuntimeError('Runtime mass, limits or payload-integrity audit failed')
        offset = torch.tensor(prepared['geometry']['tcp_in_stator_m'], dtype=torch.float32)
        initial_physics_step=int(sim.get_physics_step_count())
        _json(output/'sensor_timing.json',dict(requested_dt_s=cfg.dt_s,sim_dt_s=float(sim.get_physics_dt()),
            sensor_dt_s=[float(sensor._sim_physics_dt) for sensor in contact_sensors],
            raw_contact_capacity=128))
        previous_phase = ''
        render_clock_checks=0
        render_clock_max_errors={}
        grip_reference = cfg.gripper_open_rad
        grip_force_latched = False
        paired_close_ticks = 0
        longest_paired_close_ticks = 0
        lost_contact_ticks = 0
        previous_arm_target = torch.tensor(prepared['plan']['initial_arm_rad'])
        from go2_logistics.z1_contact_control import (GripperReleaseHandoff, JawNormalAdmittance,
            wrist_offset_target,unload_moving_jaw_preload)
        normal_controller=JawNormalAdmittance(cfg.grip_force_target_n,
            precontact_feedback=cfg.precontact_normal_feedback,contact_threshold_n=cfg.contact_threshold_n)
        release_controller=GripperReleaseHandoff(cfg.gripper_open_rad)
        previous_grip_target=cfg.gripper_open_rad
        wrist_kinematics=Z1Kinematics.from_urdf(output/'z1_with_gripper.urdf')
        balance=None
        if mounted:
            from go2_logistics.balance_recovery import ContactBalanceRecovery,BalanceConfig
            from go2_logistics.mounted_robot_asset import LEG_JOINT_NAMES
            from go2_logistics.z1_world_target import link_transform,measured_world_arm_target
            from scipy.spatial.transform import Rotation
            leg_ids=robot.find_joints(list(LEG_JOINT_NAMES),preserve_order=True)[0]
            mount_body=robot.find_bodies(['link00'],preserve_order=True)[0][0]
            balance=ContactBalanceRecovery(robot,feet_contact,leg_ids,.25,
                config=BalanceConfig(measured_com=True,target_update='project_then_slew',handoff_mode='settled_slew',
                    enter_time_s=cfg.support_enter_seconds))
            balance.ground_z=.02
            last_leg_action=torch.zeros(12)
            nominal_mount=np.eye(4);nominal_mount[:3,3]=prepared['plan']['base_world_m']
            pose=_tensor(robot.data.body_link_pose_w)[0]
            mount_world=link_transform(pose[mount_body,:3].cpu().numpy(),pose[mount_body,3:7].cpu().numpy())
            actual_tip=link_transform(pose[body_ids[0],:3].cpu().numpy(),pose[body_ids[0],3:7].cpu().numpy())
            if np.linalg.norm((mount_world@wrist_kinematics.fk(initial[0,arm_ids].cpu().numpy()))[:3,3]-actual_tip[:3,3])>.002:
                raise RuntimeError('Free-base URDF FK disagrees with the measured gripper link')
        _json(output/'wrist_controller.json',dict(enabled=cfg.wrist_contact_feedback,**asdict(normal_controller)))
        carry_relative_goal=carry_stop_goal=None
        gait_actor=None
        gait_command=np.zeros(3)
        gait_action=torch.zeros(12)
        previous_world_reference=None
        actor_observation=None
        delivery_align_start=None
        delivery_align_ticks=0
        if cfg.loaded_carry_seconds:
            from go2_logistics.z1_carry_control import (carry_velocity,load_gait_actor,planar_mount_transform,
                interpolate_world_goal,CarryVelocityFeedback,CarryStanceGuard,CarryWaypointFollower,CarryArrivalStop)
            carry_feedback=CarryVelocityFeedback()
            carry_stance_guard=CarryStanceGuard()
            carry_waypoints=CarryWaypointFollower(prepared['plan']['carry_waypoints_root_xy_m'])
            carry_arrival=CarryArrivalStop(brake_seconds=cfg.carry_brake_seconds)
            from go2_logistics.fleet_runtime import gait_observation
            from tensordict import TensorDict
            checkpoint=Path(args.gait_checkpoint).resolve()
            agent_path=checkpoint.with_name('agent.yaml')
            gait_actor=load_gait_actor(checkpoint,agent_path)
            (output/'gait_checkpoint.pt').write_bytes(checkpoint.read_bytes())
            (output/'gait_agent.yaml').write_bytes(agent_path.read_bytes())
            training_manifest=checkpoint.with_name('manifest.json')
            training_scope='Unknown training scope; see checkpoint provenance'
            if training_manifest.exists():
                (output/'gait_training_manifest.json').write_bytes(training_manifest.read_bytes())
                training_scope=json.loads(training_manifest.read_text(encoding='utf-8')).get('training_scope',training_scope)
            _json(output/'carry_policy.json',dict(checkpoint=str(checkpoint),sha256=_sha(checkpoint),
                agent_sha256=_sha(agent_path),observation_dimension=48,action_dimension=12,
                inference_hz=50.,training_scope=training_scope,
                route=cfg.carry_route,route_scope='Known-layout measured-position waypoint feedback; no sensed obstacle avoidance'))
        with (output/'trace.jsonl').open('w', encoding='utf-8') as trace, torch.inference_mode():
            for tick, command in enumerate(prepared['commands']):
                if not app.is_running():
                    raise RuntimeError('Simulator closed before the probe completed')
                phase = command['phase']
                phase_changed=phase!=previous_phase
                if phase_changed:
                    if phase == 'lift' and cfg.contact_feedback and longest_paired_close_ticks*cfg.dt_s < .1:
                        raise RuntimeError('Refusing to lift without 0.1 s of measured opposing jaw contact')
                    print('Z1_CONTACT_PHASE '+phase, flush=True)
                    if phase=='delivery_align':
                        root=_tensor(robot.data.root_pos_w)[0].cpu().numpy()
                        if np.linalg.norm(root[:2]-prepared['plan']['carry_destination_root_xy_m'])>.05:
                            raise RuntimeError('Refusing destination release: loaded walk has not reached the original 5 cm stop tolerance')
                        delivery_align_start=previous_world_reference.copy()
                    if phase=='carry_settle' and cfg.stop_support_enter_seconds is not None:
                        from dataclasses import replace
                        balance.config=replace(balance.config,enter_time_s=cfg.stop_support_enter_seconds)
                    previous_phase = phase
                target = torch.tensor([command['arm_rad']+[command['gripper_rad']]], dtype=torch.float32)
                grip_tracking_unloaded=False
                if cfg.contact_feedback and phase in ('close', 'lift', 'hold', 'carry', 'carry_settle', 'delivery_align', 'lower'):
                    # Only the moving finger is driven by this joint. A
                    # stator contact from the arm approach must not freeze an
                    # otherwise fully open moving finger. Either jaw still
                    # independently triggers the hard force guard below.
                    observed = samples[-1]['jaw_contact_n'][1] if samples else 0.
                    if grip_force_latched or observed > cfg.contact_threshold_n:
                        if not grip_force_latched:
                            grip_reference = float(_tensor(robot.data.joint_pos)[0, grip_ids[0]])
                        grip_force_latched = True
                        grip_reference += float(np.clip((cfg.grip_force_target_n-observed)*cfg.grip_force_gain*cfg.dt_s, -.0005, .0005))
                    else:
                        # Limit closing preload while still following the
                        # free-space opening schedule; the motor remains PD.
                        measured_grip = float(_tensor(robot.data.joint_pos)[0, grip_ids[0]])
                        preload = .008 if measured_grip > -1.10 else .035
                        grip_reference = min(command['gripper_rad'], measured_grip+preload)
                    if cfg.grip_tracking_antiwindup:
                        prior_reference=grip_reference
                        grip_reference=unload_moving_jaw_preload(grip_reference,
                            float(_tensor(robot.data.joint_pos)[0,grip_ids[0]]),observed,
                            cfg.grip_force_target_n,previous_grip_target,cfg.dt_s,cfg.gripper_open_rad)
                        grip_tracking_unloaded=grip_reference<prior_reference-1e-10
                    target[0, 6] = float(np.clip(grip_reference, cfg.gripper_open_rad, 0.))
                target[0,6]=release_controller.update(phase,previous_grip_target,float(target[0,6]),cfg.dt_s)
                previous_grip_target=float(target[0,6])
                wrist_diagnostic=None
                world_goal=None
                if mounted:
                    pose=_tensor(robot.data.body_link_pose_w)[0,mount_body]
                    mount_world=link_transform(pose[:3].cpu().numpy(),pose[3:7].cpu().numpy())
                    world_goal=nominal_mount@wrist_kinematics.fk(command['arm_rad'])
                    if phase=='carry':
                        carry_frame=(planar_mount_transform(_tensor(robot.data.root_pos_w)[0].cpu().numpy(),
                            _tensor(robot.data.root_quat_w)[0].cpu().numpy())
                            if cfg.carry_reference=='level' else mount_world)
                        if carry_relative_goal is None:
                            carry_relative_goal=np.linalg.solve(carry_frame,world_goal)
                        world_goal=carry_frame@carry_relative_goal
                    elif phase=='carry_settle':
                        if cfg.carry_stop_reference=='mount':
                            world_goal=mount_world@carry_relative_goal
                        else:
                            if carry_stop_goal is None:
                                carry_stop_goal=previous_world_reference.copy()
                            world_goal=carry_stop_goal.copy()
                    elif cfg.loaded_delivery and phase in ('delivery_align','lower','release','retreat','settle'):
                        world_goal[:3,3]+=prepared['plan']['delivery_translation_w_m']
                        if phase=='delivery_align':
                            delivery_align_ticks+=1
                            world_goal=interpolate_world_goal(delivery_align_start,world_goal,
                                min(1.,delivery_align_ticks*cfg.dt_s/8.))
                    previous_world_reference=world_goal.copy()
                if cfg.wrist_contact_feedback:
                    # The force on the stator points opposite its inward
                    # normal. Magnitude alone would include tangential load.
                    normal=(world_goal if mounted else wrist_kinematics.fk(command['arm_rad']))[:3,2]
                    measured_normal=max(0.,-float(np.dot(samples[-1]['jaw_contact_vectors_w_n'][0],normal))) if samples else 0.
                    paired=bool(samples and min(samples[-1]['jaw_contact_n'])>cfg.contact_threshold_n)
                    correction=normal_controller.update(measured_normal,paired,phase,cfg.dt_s)
                    if mounted:
                        world_goal[:3,3]+=normal*correction
                        wrist_diagnostic=dict(offset_m=correction,inward_normal_w=normal.tolist())
                    else:
                        wrist_target,wrist_diagnostic=wrist_offset_target(wrist_kinematics,command['arm_rad'],
                            previous_arm_target.cpu().numpy(),correction,cfg.dt_s)
                        target[0,:6]=torch.as_tensor(wrist_target,dtype=target.dtype)
                    wrist_diagnostic.update(measured_stator_normal_n=measured_normal,
                        filtered_stator_normal_n=normal_controller.filtered_force_n,
                        latched=normal_controller.latched,precontact_active=normal_controller.precontact_active)
                world_request=None
                if mounted:
                    arm_target,world_request=measured_world_arm_target(wrist_kinematics,world_goal,mount_world,
                        previous_arm_target.cpu().numpy(),cfg.dt_s)
                    target[0,:6]=torch.as_tensor(arm_target,dtype=target.dtype)
                    if phase=='carry':
                        root_xy=_tensor(robot.data.root_pos_w)[0,:2].cpu().numpy()
                        carry_destination,waypoint_changed=carry_waypoints.update(root_xy)
                        if waypoint_changed:
                            carry_feedback.bias_w[:]=0.
                        x,y,z,w=_tensor(robot.data.root_quat_w)[0].cpu().tolist()
                        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        was_arrived=carry_arrival.arrived
                        recovering=cfg.carry_arrival_stop and carry_arrival.update(root_xy,
                            prepared['plan']['carry_arrival_target_root_xy_m'],
                            _tensor(robot.data.root_lin_vel_w)[0,:2].cpu().numpy(),
                            carry_waypoints.index==len(carry_waypoints.waypoints)-1,
                            dt=cfg.dt_s,stance_ready=(
                                float(_tensor(robot.data.root_pos_w)[0,2])>=.305
                                and float(_tensor(robot.data.projected_gravity_b)[0,2])<=-.995
                                and float(_tensor(robot.data.root_ang_vel_b)[0].norm())<=.12
                                and int((_tensor(feet_contact.data.net_forces_w)[0,balance.contact_ids,2]>1.).sum())>=3))
                        if recovering and not was_arrived:
                            from dataclasses import replace
                            balance.config=replace(balance.config,enter_time_s=cfg.stop_support_enter_seconds or .35)
                            print('Z1_CARRY_ARRIVED '+json.dumps(dict(tick=tick+1,root_xy_m=root_xy.tolist())),flush=True)
                        if cfg.carry_recovery and not recovering:
                            recovering=carry_stance_guard.update(float(_tensor(robot.data.root_pos_w)[0,2]),
                                float(_tensor(robot.data.projected_gravity_b)[0,2]),
                                float(_tensor(robot.data.root_lin_vel_w)[0].norm()),
                                float(_tensor(robot.data.root_ang_vel_b)[0].norm()),
                                (_tensor(feet_contact.data.net_forces_w)[0,balance.contact_ids,2]>1.).cpu().tolist(),cfg.dt_s)
                        if recovering or carry_arrival.braking:
                            gait_command+=np.clip(-gait_command,-.4*cfg.dt_s,.4*cfg.dt_s)
                            carry_feedback.bias_w*=max(0.,1.-cfg.dt_s/.2)
                        elif cfg.carry_control=='velocity_pi':
                            gait_command=carry_feedback.update(root_xy,yaw,carry_destination,
                                _tensor(robot.data.root_lin_vel_w)[0,:2].cpu().numpy(),gait_command,cfg.dt_s)
                        else:
                            gait_command=carry_velocity(root_xy,yaw,carry_destination,
                                gait_command,cfg.dt_s)
                        if tick%4==0 or phase_changed:
                            obs=gait_observation(robot,leg_ids,torch.as_tensor(gait_command,dtype=torch.float32)[None],
                                last_leg_action[None])
                            actor_observation=obs[0].cpu().tolist()
                            gait_action=gait_actor(TensorDict({'policy':obs},batch_size=[1]))[0]
                            if gait_action.shape!=(12,) or not torch.isfinite(gait_action).all():
                                raise RuntimeError('Loaded gait returned invalid leg actions')
                        leg_action,leg_effort=balance.apply(recovering,gait_action,last_leg_action,torch.zeros(12),cfg.dt_s)
                    else:
                        leg_action,leg_effort=balance.apply(True,last_leg_action,last_leg_action,torch.zeros(12),cfg.dt_s)
                    robot.set_joint_position_target_index(target=(_tensor(robot.data.default_joint_pos)[0,leg_ids]+.25*leg_action)[None],joint_ids=leg_ids)
                    robot.set_joint_effort_target_index(target=leg_effort[None],joint_ids=leg_ids)
                    last_leg_action=leg_action.clone()
                robot.set_joint_position_target_index(target=target, joint_ids=ids)
                arm_velocity_target = torch.zeros((1,len(arm_ids)),dtype=torch.float32)
                if cfg.arm_velocity_feedforward:
                    arm_velocity_target = ((target[:,:6]-previous_arm_target)/cfg.dt_s).clamp(-math.pi,math.pi)
                robot.set_joint_velocity_target_index(target=arm_velocity_target,joint_ids=arm_ids)
                previous_arm_target = target[0,:6].clone()
                gravity_feedforward = torch.zeros((1, len(arm_ids)), dtype=torch.float32)
                if cfg.gravity_compensation:
                    gravity = _tensor(robot.data.gravity_compensation_forces)[:, robot.num_base_dofs:]
                    motor_limits = torch.tensor([spec.joint_efforts_nm[name] for name in ARM_JOINT_NAMES])
                    gravity_feedforward = torch.maximum(torch.minimum(gravity[:, arm_ids], motor_limits), -motor_limits)
                robot.set_joint_effort_target_index(target=gravity_feedforward, joint_ids=arm_ids)
                robot.write_data_to_sim()
                parcel.write_data_to_sim()
                sim.step(render=False if cfg.strict_physics_clock else args.visual or args.video)
                robot.update(cfg.dt_s)
                parcel.update(cfg.dt_s)
                for sensor in contact_sensors:
                    sensor.update(cfg.dt_s, force_recompute=True)
                if feet_contact:
                    feet_contact.update(cfg.dt_s,force_recompute=True)
                for sensor in leg_table_contacts.values():
                    sensor.update(cfg.dt_s,force_recompute=True)
                p = _tensor(parcel.data.root_pos_w)[0]
                stator_pos = _tensor(robot.data.body_pos_w)[0, body_ids[0]]
                stator_quat = _tensor(robot.data.body_quat_w)[0, body_ids[0]]
                tcp = stator_pos+quat_apply(stator_quat, offset)
                forces = []
                friction_forces=[]
                for sensor in contact_sensors:
                    value = sensor.data.force_matrix_w
                    if value is None:
                        raise RuntimeError('Pair-filtered contact force matrix is unavailable')
                    vector = _tensor(value)[0].reshape(-1, 3).sum(0)
                    forces.append(vector)
                    friction=sensor.data.friction_forces_w
                    if friction is None:
                        raise RuntimeError('Pair-filtered friction force matrix is unavailable')
                    friction_forces.append(_tensor(friction)[0].reshape(-1,3).sum(0))
                total_forces=[normal+friction for normal,friction in zip(forces,friction_forces)]
                contact_wrenches=None
                if cfg.contact_wrench:
                    from go2_logistics.z1_contact_wrench import read_pair_wrench
                    contact_wrenches=[read_pair_wrench(
                        sensor.contact_view,sensor._sim_physics_dt,p.cpu().numpy(),
                        normal.cpu().numpy(),friction.cpu().numpy())
                        for sensor,normal,friction in zip(contact_sensors,forces,friction_forces)]
                q = _tensor(robot.data.joint_pos)[0, ids]
                sample = dict(tick=tick+1, time_s=(tick+1)*cfg.dt_s, phase=phase,
                    physics_step_count=int(sim.get_physics_step_count())-initial_physics_step,
                    physics_dt_s=float(sim.get_physics_dt()),
                    contact_wrenches=contact_wrenches,
                    arm_rad=q[:6].cpu().tolist(), gripper_rad=float(q[6]),
                    arm_velocity_radps=_tensor(robot.data.joint_vel)[0,arm_ids].cpu().tolist(),
                    target_rad=target[0].cpu().tolist(),
                    planned_arm_rad=command['arm_rad'], wrist_contact_feedback=wrist_diagnostic,
                    applied_torque_nm=_tensor(robot.data.applied_torque)[0, ids].cpu().tolist(),
                    gravity_feedforward_nm=gravity_feedforward[0].cpu().tolist(),
                    arm_velocity_target_radps=arm_velocity_target[0].cpu().tolist(),
                    contact_feedback_latched=grip_force_latched,
                    grip_tracking_unloaded=grip_tracking_unloaded,
                    release_start_target_rad=release_controller.start_rad,
                    parcel_pos_w=p.cpu().tolist(), parcel_quat_xyzw=_tensor(parcel.data.root_quat_w)[0].cpu().tolist(),
                    parcel_lin_vel_w=_tensor(parcel.data.root_lin_vel_w)[0].cpu().tolist(),
                    parcel_ang_vel_w=_tensor(parcel.data.root_ang_vel_w)[0].cpu().tolist(),
                    stator_pos_w=stator_pos.cpu().tolist(), stator_quat_xyzw=stator_quat.cpu().tolist(),
                    parcel_in_stator_m=quat_apply_inverse(stator_quat, p-stator_pos).cpu().tolist(),
                    tcp_pos_w=tcp.cpu().tolist(), tcp_payload_distance_m=float(torch.linalg.norm(tcp-p)),
                    jaw_contact_n=[float(torch.linalg.norm(v)) for v in forces[:2]],
                    jaw_contact_vectors_w_n=[v.cpu().tolist() for v in forces[:2]],
                    jaw_friction_vectors_w_n=[v.cpu().tolist() for v in friction_forces[:2]],
                    jaw_total_contact_vectors_w_n=[v.cpu().tolist() for v in total_forces[:2]],
                    jaw_total_contact_n=[float(torch.linalg.norm(v)) for v in total_forces[:2]],
                    contact_force_semantics='jaw_contact_n is normal-only; total includes pair-filtered friction',
                    table_contact_n=float(torch.linalg.norm(forces[2])))
                if destination_contact:
                    sample['destination_table_contact_n']=float(torch.linalg.norm(forces[3]))
                if mounted:
                    root=_tensor(robot.data.root_pos_w)[0]
                    gravity=_tensor(robot.data.projected_gravity_b)[0]
                    pose=_tensor(robot.data.body_link_pose_w)[0,body_ids[0]]
                    actual_tip=link_transform(pose[:3].cpu().numpy(),pose[3:7].cpu().numpy())
                    tracking=dict(position_error_m=float(np.linalg.norm(actual_tip[:3,3]-world_goal[:3,3])),
                        orientation_error_rad=float(Rotation.from_matrix(actual_tip[:3,:3]@world_goal[:3,:3].T).magnitude()))
                    sample.update(root_pos_w=root.cpu().tolist(),projected_gravity_b=gravity.cpu().tolist(),
                        root_quat_xyzw=_tensor(robot.data.root_quat_w)[0].cpu().tolist(),
                        root_lin_vel_w=_tensor(robot.data.root_lin_vel_w)[0].cpu().tolist(),
                        foot_normal_n=_tensor(feet_contact.data.net_forces_w)[0,balance.contact_ids,2].cpu().tolist(),
                        world_arm_tracking=tracking,world_arm_request=world_request,world_arm_goal=world_goal.tolist(),
                        stationary_support=balance.telemetry,
                        leg_applied_torque_nm=_tensor(robot.data.applied_torque)[0,leg_ids].cpu().tolist())
                    if gait_actor is not None:
                        sample['actor_observation']=actor_observation
                        sample['leg_joint_position_rad']=_tensor(robot.data.joint_pos)[0,leg_ids].cpu().tolist()
                        sample['leg_joint_velocity_radps']=_tensor(robot.data.joint_vel)[0,leg_ids].cpu().tolist()
                        sample['gait_action']=gait_action.cpu().tolist()
                        sample['applied_leg_action']=last_leg_action.cpu().tolist()
                        sample['carry_velocity_bias_w_mps']=carry_feedback.bias_w.tolist()
                        sample['carry_waypoint_index']=carry_waypoints.index
                        sample['carry_arrival_stop']=carry_arrival.arrived
                        sample['carry_arrival_brake']=dict(active=carry_arrival.braking and not carry_arrival.arrived,
                            elapsed_s=carry_arrival.elapsed_s,settled_s=carry_arrival.settled_s)
                        sample['carry_recovery']=dict(active=carry_stance_guard.active,count=carry_stance_guard.count,
                            elapsed_s=carry_stance_guard.elapsed_s,settled_s=carry_stance_guard.settled_s)
                        if tick%20==0:
                            import warp as wp
                            backend_velocity=wp.to_torch(robot.root_view.get_dof_velocities())
                            sample['backend_leg_joint_velocity_radps']=backend_velocity[0,leg_ids].cpu().tolist()
                            sample['backend_arm_joint_velocity_radps']=backend_velocity[0,arm_ids].cpu().tolist()
                        sample['carry_command_b']=gait_command.tolist() if phase=='carry' else [0.,0.,0.]
                        sample['carry_endpoint_error_m']=float(np.linalg.norm(root[:2].cpu().numpy()
                            -prepared['plan']['carry_destination_root_xy_m']))
                    if leg_table_contacts:
                        sample['leg_table_contacts']={name:dict(
                            normal_vectors_w_n=_tensor(sensor.data.force_matrix_w)[0,0].cpu().tolist(),
                            friction_vectors_w_n=_tensor(sensor.data.friction_forces_w)[0,0].cpu().tolist())
                            for name,sensor in leg_table_contacts.items()}
                        sample['leg_contact_table_order']=['source']+(['destination'] if cfg.loaded_delivery else [])
                        sample['foot_positions_w']=_tensor(robot.data.body_link_pose_w)[0,balance.feet,:3].cpu().tolist()
                trace.write(json.dumps(sample, allow_nan=False)+'\n')
                samples.append(sample)
                if mounted and (float(root[2])<.25 or float(gravity[2])>-math.cos(math.radians(20.))):
                    raise RuntimeError(f'Free-base stance guard at tick {tick+1}: height={float(root[2])}, gravity_z={float(gravity[2])}')
                if phase == 'close':
                    paired_close_ticks = paired_close_ticks+1 if min(sample['jaw_contact_n']) > cfg.contact_threshold_n else 0
                    longest_paired_close_ticks = max(longest_paired_close_ticks, paired_close_ticks)
                if phase in ('lift','hold','carry','carry_settle','delivery_align') and cfg.contact_feedback:
                    lost_contact_ticks = lost_contact_ticks+1 if min(sample['jaw_contact_n']) <= cfg.contact_threshold_n else 0
                    if lost_contact_ticks*cfg.dt_s > .25:
                        raise RuntimeError('Opposing jaw contact lost for more than 0.25 s during lift/hold')
                if max(sample['jaw_contact_n']+sample['jaw_total_contact_n']) > cfg.hard_contact_limit_n:
                    raise RuntimeError(f'Jaw contact exceeded unchanged 20 N guard at tick {tick+1}: '
                        f'normal={sample["jaw_contact_n"]}, total={sample["jaw_total_contact_n"]}')
                if tick % 20 == 0:
                    trace.flush()
                    if camera:
                        camera.set_world_poses_from_view(torch.tensor([[1.45, -1.5, 1.05] if mounted else [1.1, -.9, .8]]), torch.tensor([[.15, 0., .40] if mounted else [.22, 0., .28]]))
                        def capture_frame():
                            sim.render()
                            camera.update(.1, force_recompute=True)
                            return camera.data.output['rgb'][0,...,:3].cpu().numpy().copy()
                        if cfg.audit_render_clock:
                            from go2_logistics.fleet_gui_check import backend_snapshot,check_frozen
                            frames=[]
                            try:
                                clock_check=check_frozen(lambda:backend_snapshot(sim,{'robot':robot},{'parcel':parcel}),
                                    lambda:frames.append(capture_frame()),pumps=1)
                            except RuntimeError as exc:
                                _json(output/'render_clock_audit.json',dict(passed=False,tick=tick+1,
                                    error=str(exc),strict_physics_clock=cfg.strict_physics_clock))
                                if frames:writer.append_data(frames[0])
                                raise
                            render_clock_checks+=1
                            for key,value in clock_check['max_state_errors'].items():
                                render_clock_max_errors[key]=max(render_clock_max_errors.get(key,0.),value)
                            _json(output/'render_clock_audit.json',dict(passed=True,tick=tick+1,
                                strict_physics_clock=cfg.strict_physics_clock,checks=render_clock_checks,
                                max_state_errors=render_clock_max_errors))
                            writer.append_data(frames[0])
                        else:
                            writer.append_data(capture_frame())
                if float(p[2]) < max(.01,cfg.table_top_m-.05) or not torch.isfinite(p).all():
                    raise RuntimeError(f'Payload fell outside the table scenario at tick {tick+1}')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if writer:
            writer.close()
        if cfg.loaded_carry_seconds:
            from go2_logistics.z1_carry_control import evaluate_loaded_carry_trace
            evidence=evaluate_loaded_carry_trace(samples,cfg,integrity,prepared['plan'])
        elif mounted:
            from go2_logistics.z1_mounted_contact import evaluate_mounted_trace
            evidence=evaluate_mounted_trace(samples,cfg,integrity)
        else:
            evidence = evaluate_trace(samples, cfg, integrity)
        evidence.update(failure=failure, measured_ticks=len(samples),
            measured_seconds=len(samples)*cfg.dt_s, requested_ticks=prepared['plan']['ticks'],
            wall_seconds=time.monotonic()-started, initial_physics_resets=1,
            runtime_integrity=integrity, output=str(output))
        evidence['passed'] = evidence['passed'] and not failure and len(samples) == prepared['plan']['ticks']
        _json(output/'result.json', evidence)
        print('Z1_CONTACT_RESULT '+json.dumps(evidence), flush=True)
        app.close()
    return evidence


def analyze_run(output: Path) -> dict:
    config_data = json.loads((output/'config.json').read_text(encoding='utf-8'))
    config = ProbeConfig(**config_data)
    samples = [json.loads(line) for line in (output/'trace.jsonl').read_text(encoding='utf-8').splitlines() if line]
    integrity = json.loads((output/'runtime_integrity.json').read_text(encoding='utf-8'))
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if config.loaded_carry_seconds:
        from go2_logistics.z1_carry_control import evaluate_loaded_carry_trace
        evidence=evaluate_loaded_carry_trace(samples,config,integrity,
            json.loads((output/'plan.json').read_text(encoding='utf-8')))
    elif manifest.get('mounted_go2'):
        from go2_logistics.z1_mounted_contact import evaluate_mounted_trace
        evidence=evaluate_mounted_trace(samples,config,integrity)
    else:
        evidence = evaluate_trace(samples, config, integrity)
    planned = json.loads((output/'plan.json').read_text(encoding='utf-8'))
    evidence['checks']['all_requested_ticks_measured'] = len(samples) == planned['ticks']
    evidence['passed'] = evidence['passed'] and evidence['checks']['all_requested_ticks_measured']
    _json(output/'independent_evidence.json', evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh run directory; existing paths are refused')
    parser.add_argument('--import-report', type=Path, default=ROOT/'artifacts/real_arm_z1/usd/import_validation.json')
    parser.add_argument('--prepare-only', action='store_true', help='Geometry, original physical-data audit and IK only; no Isaac process')
    parser.add_argument('--analyze', type=Path, help='Independently re-evaluate a specific completed run without Isaac')
    parser.add_argument('--video', action='store_true', help='Record a 640x480, 10 Hz RTX camera; default has no camera')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--mounted-go2',action='store_true',help='Free Go2 with original-mass Z1, static contact support and measured-mount compensation')
    parser.add_argument('--table-height',type=float,help='Support surface height; mounted default 0.12 m, fixed default 0.18 m')
    parser.add_argument('--parcel-x',type=float,help='Nominal parcel x; mounted default 0.42 m, fixed default 0.32 m')
    parser.add_argument('--arm-base-height',type=float,help='World height of mounted Z1 link00; default 0.43 m')
    parser.add_argument('--gravity-compensation', action='store_true', help='Model-based arm gravity feedforward inside original motor limits')
    parser.add_argument('--lateral-approach',type=float,default=0.,help='Descend with bounded lateral clearance, then approach horizontally (metres)')
    parser.add_argument('--contact-feedback',action='store_true',help='Target 3 N opposing grip contact with a 20 N hard guard')
    parser.add_argument('--arm-stiffness',type=float,default=160.)
    parser.add_argument('--arm-damping',type=float,default=12.)
    parser.add_argument('--close-advance',type=float,default=0.,help='Bounded horizontal hand advance during closing, in metres')
    parser.add_argument('--collision-mode',choices=('convexHull','convexDecomposition'),default='convexHull')
    parser.add_argument('--close-seconds',type=float,default=0.,help='Explicit bounded closing duration, at most 30 seconds')
    parser.add_argument('--arm-velocity-feedforward',action='store_true',help='Use planned joint velocities in the original bounded PD motor drives')
    parser.add_argument('--grip-force',type=float,default=3.,help='Moving-finger contact target below the 20 N hard guard')
    parser.add_argument('--grip-gain',type=float,default=.05)
    parser.add_argument('--motion-seconds',type=float,default=3.,help='Lift and lower duration, between 3 and 10 seconds')
    parser.add_argument('--wrist-contact-feedback',action='store_true',help='Experimental bounded fixed-jaw normal-force wrist correction')
    parser.add_argument('--precontact-normal-feedback',action='store_true',help='Experimental unload-only sensed fixed-jaw preload control during closing before pairing')
    parser.add_argument('--contact-wrench',action='store_true',help='Read-only contact-point and friction-patch wrench diagnostic')
    parser.add_argument('--grip-tracking-antiwindup',action='store_true',help='Unload stored closing position error on measured excess force')
    parser.add_argument('--approach-seconds',type=float,default=0.,help='Per-segment approach time; zero preserves the existing schedule')
    parser.add_argument('--retreat-seconds',type=float,default=3.)
    parser.add_argument('--support-enter-seconds',type=float,default=.20,
        help='Measured-contact support torque ramp; bounded within 0.02 to 0.50 seconds')
    parser.add_argument('--loaded-carry',action='store_true',
        help='Experimental pick, loaded lateral walk and stop; separate acceptance, no table release')
    parser.add_argument('--loaded-delivery',action='store_true',help='Continue loaded walk to a separate known low table and contact release')
    parser.add_argument('--carry-seconds',type=float,default=16.)
    parser.add_argument('--carry-distance',type=float,default=.50)
    parser.add_argument('--carry-control',choices=('proportional','velocity_pi'),default='proportional')
    parser.add_argument('--carry-route',choices=('direct','retreat_aisle','offset_aisle'),default='direct')
    parser.add_argument('--carry-arrival-stop',action='store_true',help='Use measured final-dock arrival to enter motor stance for the remaining scheduled walk')
    parser.add_argument('--carry-brake-seconds',type=float,default=0.,
        help='Experimental zero-command gait braking before measured settled stance; zero preserves immediate arrival stopping')
    parser.add_argument('--carry-arrival-offset-x',type=float,default=0.,help='Experimental stance-transition compensation in world x (metres)')
    parser.add_argument('--carry-arrival-offset-y',type=float,default=0.,help='Experimental stance-transition compensation in world y (metres)')
    parser.add_argument('--carry-recovery',action='store_true')
    parser.add_argument('--strict-physics-clock',action='store_true')
    parser.add_argument('--audit-render-clock',action='store_true')
    parser.add_argument('--audit-leg-contacts',action='store_true',help='Read-only per-calf/foot contacts against both low tables')
    parser.add_argument('--carry-reference',choices=('mount','level'),default='mount')
    parser.add_argument('--carry-stop-reference',choices=('world','mount'),default='world')
    parser.add_argument('--lift-m',type=float,default=.12,help='Planned physical lift; original acceptance remains 9 cm')
    parser.add_argument('--stop-support-enter-seconds',type=float,default=None)
    parser.add_argument('--gait-checkpoint',type=Path,
        default=ROOT/'artifacts/gait_adaptation/run_20261021_183701/mounted_gait.pt')
    parser.add_argument('--jaw-sample-x',type=float,default=.13,help='Parcel centre along the original jaw depth, [.10,.17] metres')
    parser.add_argument('--spawn-offset-x',type=float,default=0.,help='Initial-only unobserved payload x offset, within a 4mm XY radius')
    parser.add_argument('--spawn-offset-y',type=float,default=0.)
    args = parser.parse_args()
    if args.analyze:
        report = analyze_run(args.analyze.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report['passed']:
            raise SystemExit(2)
        return
    output = (args.output or ROOT/'artifacts/real_arm_z1'/datetime.now().strftime('contact_%Y%m%d_%H%M%S_%f')).resolve()
    if args.prepare_only:
        prepared = prepare_run(output, args.import_report.resolve(), ProbeConfig(gravity_compensation=args.gravity_compensation,
            base_height_m=args.arm_base_height if args.arm_base_height is not None else (.43 if args.mounted_go2 else .20),
            table_top_m=args.table_height if args.table_height is not None else (.12 if args.mounted_go2 else .18),
            parcel_xy_m=(args.parcel_x if args.parcel_x is not None else (.42 if args.mounted_go2 else .32),0.),
            lateral_approach_m=args.lateral_approach, contact_feedback=args.contact_feedback,
            arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping, close_advance_m=args.close_advance,
            collision_mode=args.collision_mode,close_seconds=args.close_seconds,
            arm_velocity_feedforward=args.arm_velocity_feedforward,grip_force_target_n=args.grip_force,
            grip_force_gain=args.grip_gain,motion_seconds=args.motion_seconds,wrist_contact_feedback=args.wrist_contact_feedback,
            contact_wrench=args.contact_wrench,jaw_sample_x_m=args.jaw_sample_x,
            precontact_normal_feedback=args.precontact_normal_feedback,
            grip_tracking_antiwindup=args.grip_tracking_antiwindup,approach_seconds=args.approach_seconds,
            retreat_seconds=args.retreat_seconds,support_enter_seconds=args.support_enter_seconds,
            loaded_carry_seconds=args.carry_seconds if args.loaded_carry else 0.,loaded_carry_distance_m=args.carry_distance,
            loaded_delivery=args.loaded_delivery,carry_control=args.carry_control,carry_recovery=args.carry_recovery,
            carry_route=args.carry_route,
            carry_arrival_stop=args.carry_arrival_stop,
            carry_brake_seconds=args.carry_brake_seconds,
            carry_arrival_offset_xy_m=(args.carry_arrival_offset_x,args.carry_arrival_offset_y),
            strict_physics_clock=args.strict_physics_clock,audit_render_clock=args.audit_render_clock,
            audit_leg_contacts=args.audit_leg_contacts,
            carry_reference=args.carry_reference,carry_stop_reference=args.carry_stop_reference,
            lift_m=args.lift_m,stop_support_enter_seconds=args.stop_support_enter_seconds,
            parcel_spawn_offset_xy_m=(args.spawn_offset_x,args.spawn_offset_y)),mounted_go2=args.mounted_go2)
        print('Z1_CONTACT_PREFLIGHT '+json.dumps(dict(passed=True, output=str(output),
            duration_s=prepared['plan']['duration_s'], geometry=prepared['geometry'],
            scope='geometry, mass/inertia/limit audit and IK only; physics not run')), flush=True)
        return
    # Kit must initialize its USD bindings before geometry preparation imports
    # pxr. Loading the conda USD binary first prevents Kit's notice wrappers
    # from registering and can terminate the process without a result file.
    output.mkdir(parents=True, exist_ok=False)
    os.environ['WARP_CACHE_PATH'] = str(output/'warp_cache')
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=not args.visual, enable_cameras=args.video, device='cpu',
        kit_args=f'--/log/file="{output.as_posix()}/kit.log"')
    try:
        prepared = prepare_run(output, args.import_report.resolve(), ProbeConfig(gravity_compensation=args.gravity_compensation,
            base_height_m=args.arm_base_height if args.arm_base_height is not None else (.43 if args.mounted_go2 else .20),
            table_top_m=args.table_height if args.table_height is not None else (.12 if args.mounted_go2 else .18),
            parcel_xy_m=(args.parcel_x if args.parcel_x is not None else (.42 if args.mounted_go2 else .32),0.),
            lateral_approach_m=args.lateral_approach, contact_feedback=args.contact_feedback,
            arm_stiffness=args.arm_stiffness, arm_damping=args.arm_damping, close_advance_m=args.close_advance,
            collision_mode=args.collision_mode,close_seconds=args.close_seconds,
            arm_velocity_feedforward=args.arm_velocity_feedforward,grip_force_target_n=args.grip_force,
            grip_force_gain=args.grip_gain,motion_seconds=args.motion_seconds,wrist_contact_feedback=args.wrist_contact_feedback,
            contact_wrench=args.contact_wrench,jaw_sample_x_m=args.jaw_sample_x,
            precontact_normal_feedback=args.precontact_normal_feedback,
            grip_tracking_antiwindup=args.grip_tracking_antiwindup,approach_seconds=args.approach_seconds,
            retreat_seconds=args.retreat_seconds,support_enter_seconds=args.support_enter_seconds,
            loaded_carry_seconds=args.carry_seconds if args.loaded_carry else 0.,loaded_carry_distance_m=args.carry_distance,
            loaded_delivery=args.loaded_delivery,carry_control=args.carry_control,carry_recovery=args.carry_recovery,
            carry_route=args.carry_route,
            carry_arrival_stop=args.carry_arrival_stop,
            carry_brake_seconds=args.carry_brake_seconds,
            carry_arrival_offset_xy_m=(args.carry_arrival_offset_x,args.carry_arrival_offset_y),
            strict_physics_clock=args.strict_physics_clock,audit_render_clock=args.audit_render_clock,
            audit_leg_contacts=args.audit_leg_contacts,
            carry_reference=args.carry_reference,carry_stop_reference=args.carry_stop_reference,
            lift_m=args.lift_m,stop_support_enter_seconds=args.stop_support_enter_seconds,
            parcel_spawn_offset_xy_m=(args.spawn_offset_x,args.spawn_offset_y)), directory_reserved=True,mounted_go2=args.mounted_go2)
    except BaseException:
        launcher.app.close()
        raise
    print('Z1_CONTACT_OUTPUT '+str(output), flush=True)
    if not run_physics(prepared, args, launcher.app)['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
