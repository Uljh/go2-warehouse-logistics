"""Loaded Z1 walking experiment, separate from stationary/release acceptance.

Uses the existing RSL-RL actor and the warehouse's 48-value observation ABI.
No rigid-body state writes, bindings, external forces or recovery resets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass,field
from pathlib import Path

import numpy as np


def planar_mount_transform(root_position, root_quaternion_xyzw):
    """Measured XY/yaw frame excluding base bounce, roll and pitch.

    Its targets follow the robot across the floor at a fixed world height.
    This returns a motor-control reference, never a rigid-body state write.
    """
    from scipy.spatial.transform import Rotation
    position=np.asarray(root_position,dtype=float)
    quaternion=np.asarray(root_quaternion_xyzw,dtype=float)
    if (position.shape!=(3,) or quaternion.shape!=(4,)
            or not np.isfinite(np.r_[position,quaternion]).all()
            or abs(np.linalg.norm(quaternion)-1.)>1e-3):
        raise ValueError('Invalid measured root pose for planar carry reference')
    matrix=Rotation.from_quat(quaternion).as_matrix()
    yaw=math.atan2(matrix[1,0],matrix[0,0])
    frame=np.eye(4)
    frame[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    frame[:2,3]=position[:2]
    return frame


def carry_velocity(position_xy, yaw, destination_xy, previous, dt):
    """World position feedback to bounded body velocities for a short aisle."""
    position=np.asarray(position_xy,dtype=float)
    destination=np.asarray(destination_xy,dtype=float)
    previous=np.asarray(previous,dtype=float)
    if (position.shape!=(2,) or destination.shape!=(2,) or previous.shape!=(3,)
            or not np.isfinite(np.r_[position,destination,previous,yaw,dt]).all()
            or not 0<dt<=.02):
        raise ValueError('Invalid measured carry pose, command or step')
    error=destination-position
    world=1.5*error
    world*=min(1.,.18/max(np.linalg.norm(world),1e-12))
    rotation=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
    desired=np.r_[rotation@world,np.clip(-.8*yaw,-.15,.15)]
    change=np.clip(desired-previous,-.4*dt,.4*dt)
    return previous+change


@dataclass
class CarryVelocityFeedback:
    """Bounded position/velocity cascade for a gait with steady speed error.

    The inner integral estimates the missing *command*, not a pose correction.
    All movement still comes from the same actor and original leg motors.
    """
    bias_w: np.ndarray = field(default_factory=lambda:np.zeros(2))

    def update(self,position_xy,yaw,destination_xy,velocity_w,previous,dt):
        position=np.asarray(position_xy,dtype=float);destination=np.asarray(destination_xy,dtype=float)
        measured=np.asarray(velocity_w,dtype=float);previous=np.asarray(previous,dtype=float)
        if (position.shape!=(2,) or destination.shape!=(2,) or measured.shape!=(2,) or previous.shape!=(3,)
                or not np.isfinite(np.r_[position,destination,measured,previous,yaw,dt]).all()
                or not 0<dt<=.02):
            raise ValueError('Invalid measured carry feedback')
        desired=1.5*(destination-position)
        desired*=min(1.,.18/max(np.linalg.norm(desired),1e-12))
        error=desired-measured
        self.bias_w=np.clip(self.bias_w+.8*error*dt,-.12,.12)
        world=desired+.4*error+self.bias_w
        world*=min(1.,.30/max(np.linalg.norm(world),1e-12))
        rotation=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
        target=np.r_[rotation@world,np.clip(-.8*yaw,-.15,.15)]
        return previous+np.clip(target-previous,-.4*dt,.4*dt)


@dataclass
class CarryStanceGuard:
    """Request the existing motor-torque stance controller before crouch stalls.

    Recovery is bounded to three seconds and requires measured contact and
    0.2 seconds of settled posture. It never resets or moves a rigid body.
    """
    active: bool=False
    elapsed_s: float=0.
    settled_s: float=0.
    count: int=0

    def update(self,height,gravity_z,speed,angular_speed,contacts,dt):
        if not (all(math.isfinite(x) for x in (height,gravity_z,speed,angular_speed,dt))
                and 0<dt<=.02 and len(contacts)==4):
            raise ValueError('Invalid measured loaded stance')
        if not self.active and (height<.295 or gravity_z>-math.cos(math.radians(8.))):
            self.active=True;self.elapsed_s=0.;self.settled_s=0.;self.count+=1
        if self.active:
            self.elapsed_s+=dt
            settled=height>=.305 and gravity_z<=-.995 and speed<=.06 and angular_speed<=.12 and sum(contacts)>=3
            self.settled_s=self.settled_s+dt if settled else 0.
            if self.settled_s>=.2-1e-10:
                self.active=False
            elif self.elapsed_s>3.+1e-10:
                raise RuntimeError('Loaded stance recovery exceeded the original three-second bound')
        return self.active


def carry_route_waypoints(distance, route='direct'):
    """Known-layout motor references; payload destination stays unchanged."""
    if not math.isfinite(distance) or not .3<=distance<=.8:
        raise ValueError('Invalid loaded carry distance')
    if route=='direct':
        return [[0.,distance]]
    if route=='retreat_aisle':
        return [[-.18,0.],[-.18,distance],[-.10,distance]]
    if route=='offset_aisle':
        return [[-.10,0.],[-.10,distance]]
    raise ValueError('Unknown loaded carry route')


@dataclass
class CarryWaypointFollower:
    waypoints: list
    index: int=0

    def update(self, position):
        position=np.asarray(position,dtype=float)
        points=np.asarray(self.waypoints,dtype=float)
        if (position.shape!=(2,) or points.ndim!=2 or points.shape[1]!=2
                or not len(points) or not np.isfinite(np.r_[position,points.ravel()]).all()):
            raise ValueError('Invalid measured carry waypoint input')
        changed=False
        if self.index<len(points)-1 and np.linalg.norm(position-points[self.index])<=.025:
            self.index+=1;changed=True
        return points[self.index].copy(),changed


@dataclass
class CarryArrivalStop:
    """Latch motor stance at the measured final dock, without shortening the run."""
    arrived: bool=False
    brake_seconds: float=0.
    braking: bool=False
    elapsed_s: float=0.
    settled_s: float=0.

    def __post_init__(self):
        if not math.isfinite(self.brake_seconds) or not 0<=self.brake_seconds<=1.5:
            raise ValueError('Arrival gait braking must be within 0..1.5 seconds')

    def update(self, position, destination, velocity_xy, final_waypoint, *, dt=None, stance_ready=None):
        position,destination,velocity=map(lambda x:np.asarray(x,dtype=float),
            (position,destination,velocity_xy))
        if (any(x.shape!=(2,) for x in (position,destination,velocity))
                or not np.isfinite(np.r_[position,destination,velocity]).all()):
            raise ValueError('Invalid measured carry arrival state')
        tolerance,speed_limit=(.025,.04) if self.brake_seconds else (.05,.08)
        near_dock=final_waypoint and np.linalg.norm(position-destination)<=tolerance and np.linalg.norm(velocity)<=speed_limit
        if not self.brake_seconds:
            if near_dock:
                self.arrived=True
            return self.arrived
        if dt is None or not math.isfinite(dt) or not 0<dt<=.02 or stance_ready is None:
            raise ValueError('Gait braking requires measured stance readiness and a finite control step')
        if near_dock:
            self.braking=True
        if self.braking and not self.arrived:
            self.elapsed_s+=dt
            settled=stance_ready and np.linalg.norm(velocity)<=.05
            self.settled_s=self.settled_s+dt if settled else 0.
            if self.elapsed_s>=self.brake_seconds and self.settled_s>=.2-1e-10:
                self.arrived=True
            elif self.elapsed_s>3.+1e-10:
                raise RuntimeError('Arrival gait braking exceeded the original three-second stop bound')
        return self.arrived


def extend_with_carry(commands, plan, dt, seconds, distance, *, delivery=False, route='direct',arrival_offset_xy=(0.,0.)):
    """Retain the validated pick/hold prefix, then walk and stop holding.

    This deliberately has its own acceptance; it does not claim table release
    or warehouse delivery. The original stationary plan remains unchanged.
    """
    if not (10<=seconds<=20 and .3<=distance<=.8 and 0<dt<=.01):
        raise ValueError('Loaded carry is bounded to 10..20 seconds and .3...8 metres')
    first_lower=next(i for i,c in enumerate(commands) if c['phase']=='lower')
    selected=commands[:first_lower]
    last=selected[-1]
    for phase,duration in (('carry',seconds),('carry_settle',3.)):
        selected.extend(dict(phase=phase,arm_rad=last['arm_rad'].copy(),
            gripper_rad=last['gripper_rad']) for _ in range(round(duration/dt)))
    if delivery:
        selected.extend(dict(phase='delivery_align',arm_rad=last['arm_rad'].copy(),
            gripper_rad=last['gripper_rad']) for _ in range(round(8./dt)))
        selected.extend(commands[first_lower:])
    offset=np.asarray(arrival_offset_xy,dtype=float)
    if offset.shape!=(2,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.10:
        raise ValueError('Carry arrival compensation must be finite and within 0.10 metres')
    waypoints=carry_route_waypoints(distance,route)
    nominal_destination=np.asarray(waypoints[-1],dtype=float)
    waypoints[-1]=(nominal_destination+offset).tolist()
    report=dict(plan,ticks=len(selected),duration_s=len(selected)*dt,
        carry_route=route,carry_waypoints_root_xy_m=waypoints,
        carry_destination_root_xy_m=nominal_destination.tolist(),
        carry_arrival_target_root_xy_m=waypoints[-1],carry_arrival_offset_xy_m=offset.tolist(),
        scope='Original-mass Z1 contact pick, short lateral loaded walk, stop; no release or warehouse delivery')
    if delivery:
        report.update(loaded_delivery=True,delivery_translation_w_m=[0.,distance,0.],
            scope='Original-mass Z1 contact pick, short loaded walk, known destination low-table contact release; not warehouse/fleet acceptance')
    return selected,report


def interpolate_world_goal(start, end, fraction):
    """Smooth motor reference for stopped alignment, never a pose writer."""
    from scipy.spatial.transform import Rotation
    from go2_logistics.z1_world_target import validate_transform
    start=validate_transform(start);end=validate_transform(end)
    if not math.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Alignment fraction must be within [0,1]')
    f=fraction*fraction*(3.-2.*fraction)
    result=np.eye(4)
    result[:3,3]=(1.-f)*start[:3,3]+f*end[:3,3]
    delta=Rotation.from_matrix(end[:3,:3]@start[:3,:3].T).as_rotvec()
    result[:3,:3]=Rotation.from_rotvec(delta*f).as_matrix()@start[:3,:3]
    return result


def load_gait_actor(checkpoint, agent_config):
    """Instantiate RSL-RL's own model with its recorded configuration."""
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel

    config=yaml.safe_load(Path(agent_config).read_text(encoding='utf-8'))
    actor_config=dict(config['actor'])
    if actor_config.pop('class_name')!='MLPModel':
        raise ValueError('This probe requires the recorded nonrecurrent MLPModel actor')
    actor=MLPModel(TensorDict({'policy':torch.zeros((1,48))},batch_size=[1]),
        obs_groups={'actor':['policy']},obs_set='actor',output_dim=12,**actor_config)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    actor.load_state_dict(saved['actor_state_dict'],strict=True)
    actor.eval()
    return actor


def evaluate_loaded_carry_trace(samples,config,integrity,plan):
    """Predeclared measured acceptance for the isolated loaded walking probe."""
    expected=['initial_settle','approach','close','lift','hold','carry','carry_settle']
    if plan.get('loaded_delivery'):
        expected+=['delivery_align','lower','release','retreat','settle']
    sequence=[]
    for row in samples:
        if not sequence or sequence[-1]!=row['phase']:
            sequence.append(row['phase'])
    checks=dict(phase_order=sequence==expected,
        complete_control_timing=len(samples)==plan['ticks'] and all(
            row['tick']==i+1 and abs(row['time_s']-(i+1)*config.dt_s)<1e-8
            and row['physics_step_count']==i+1 for i,row in enumerate(samples)),
        original_mass_and_limits=bool(integrity.get('passed')),
        floating_base_without_world_anchor=integrity.get('fixed_base') is False
            and integrity.get('robot_world_joints')==[],
        unassisted_payload=integrity.get('payload_joints')==[] and integrity.get('payload_pose_writes')==0
            and integrity.get('resets_during_probe')==0 and integrity.get('payload_kinematic') is False)
    metrics={}
    if not samples:
        return dict(passed=False,checks=checks,metrics=metrics,scope=plan['scope'])
    forces=np.asarray([r.get('jaw_total_contact_n',[math.inf]*2) for r in samples])
    normals=np.asarray([r.get('jaw_contact_n',[math.inf]*2) for r in samples])
    checks['original_20n_force_guard']=bool(np.isfinite(forces).all() and np.isfinite(normals).all()
        and np.min(np.r_[forces.ravel(),normals.ravel()])>=0
        and max(forces.max(),normals.max())<=20.)
    metrics['maximum_total_contact_n']=float(forces.max())
    close=[r for r in samples if r['phase']=='close']
    run=longest=0
    for row in close:
        run=run+1 if min(row['jaw_contact_n'])>.05 else 0
        longest=max(longest,run)
    checks['paired_before_lift']=longest*config.dt_s>=.1
    held=[r for r in samples if r['phase'] in ('lift','hold','carry','carry_settle','delivery_align')]
    run=longest=0
    for row in held:
        run=run+1 if min(row['jaw_contact_n'])<=.05 else 0
        longest=max(longest,run)
    checks['no_lost_pair_over_025s']=bool(held) and longest*config.dt_s<=.25
    metrics['longest_pair_loss_s']=longest*config.dt_s
    baseline=[r['parcel_pos_w'][2] for r in samples if r['phase']=='initial_settle']
    baseline=float(np.median(baseline[-round(.5/config.dt_s):]))
    hold=[r for r in samples if r['phase']=='hold'][round(.5/config.dt_s):]
    moving=[r for r in samples if r['phase']=='carry']
    stopped_all=[r for r in samples if r['phase']=='carry_settle']
    stopped=stopped_all[-round(1./config.dt_s):]
    stable=[r for r in samples if r['time_s']>=.5]
    roots=np.asarray([r['root_pos_w'] for r in stable])
    gravity=np.asarray([r['projected_gravity_b'] for r in stable])
    checks['upright_free_base']=bool(np.isfinite(roots).all() and np.isfinite(gravity).all()
        and roots[:,2].min()>=.25 and (-gravity[:,2]).min()>=math.cos(math.radians(20.)))
    metrics.update(minimum_base_height_m=float(roots[:,2].min()),
        maximum_base_tilt_deg=float(np.degrees(np.arccos(np.clip(-gravity[:,2],-1,1))).max()))
    checks['world_arm_tracking']=all(r['world_arm_tracking']['position_error_m']<=.02
        and r['world_arm_tracking']['orientation_error_rad']<=.08 for r in stable)
    metrics['maximum_world_arm_position_error_m']=max(r['world_arm_tracking']['position_error_m'] for r in stable)
    if config.carry_recovery:
        recovery=[r.get('carry_recovery') for r in moving]
        checks['bounded_motor_stance_recovery']=bool(recovery) and all(
            value is not None and (not value['active'] or value['elapsed_s']<=3.+1e-8) for value in recovery)
        metrics['carry_recoveries']=max((value['count'] for value in recovery if value),default=0)
    checks['sustained_contact_hold']=bool(hold) and len(hold)*config.dt_s>=2.4 and all(
        r['parcel_pos_w'][2]-baseline>=.09 and r['table_contact_n']<.05 for r in hold)
    loaded=hold+moving+stopped_all+[r for r in samples if r['phase']=='delivery_align']
    if loaded:
        offsets=np.asarray([r['parcel_in_stator_m'] for r in loaded])
        drift=float(np.linalg.norm(offsets-offsets[0],axis=1).max())
        fraction=sum(min(r['jaw_contact_n'])>.05 for r in loaded)/len(loaded)
        checks['loaded_contact_and_clearance']=bool(moving) and bool(stopped) and drift<=.02 and fraction>=.9 and all(
            r['parcel_pos_w'][2]-baseline>=.09 and r['table_contact_n']<.05 for r in loaded)
        metrics.update(loaded_drift_m=drift,loaded_contact_fraction=fraction,
            minimum_loaded_lift_m=min(r['parcel_pos_w'][2]-baseline for r in loaded))
    else:
        checks['loaded_contact_and_clearance']=False
    destination=np.asarray(plan['carry_destination_root_xy_m'])
    checks['measured_walk_and_stop']=False
    if moving and len(stopped)*config.dt_s>=.99:
        origin=np.asarray(moving[0]['root_pos_w'][:2])
        ends=np.asarray([r['root_pos_w'][:2] for r in stopped])
        errors=np.linalg.norm(ends-destination,axis=1)
        checks['measured_walk_and_stop']=bool(errors.max()<=.05
            and np.linalg.norm(ends.mean(0)-origin)>=.30 and all(
            np.linalg.norm(r.get('root_lin_vel_w',r['stationary_support']['root_velocity_w']))<=.05 for r in stopped))
        metrics.update(maximum_stopped_endpoint_error_m=float(errors.max()),
            measured_walk_distance_m=float(np.linalg.norm(ends.mean(0)-origin)))
    if plan.get('loaded_delivery'):
        settled=[r for r in samples if r['phase']=='settle'][-round(1./config.dt_s):]
        destination=np.asarray(config.parcel_xy_m)+np.asarray(plan['delivery_translation_w_m'][:2])
        checks['unassisted_destination_release']=len(settled)*config.dt_s>=.99 and all(
            abs(r['parcel_pos_w'][2]-(config.table_top_m+config.parcel_size_xyz_m[2]/2))<=.012
            and np.linalg.norm(np.asarray(r['parcel_pos_w'][:2])-destination)<=.04
            and np.linalg.norm(r['parcel_lin_vel_w'])<=.025 and np.linalg.norm(r['parcel_ang_vel_w'])<=.15
            and r.get('destination_table_contact_n',0.)>.1 and r['table_contact_n']<.05
            and max(r['jaw_contact_n'])<.05 and r['gripper_rad']<-1.2
            and r['tcp_payload_distance_m']>=.075 for r in settled)
        if settled:
            metrics['destination_placement_error_m']=max(np.linalg.norm(np.asarray(r['parcel_pos_w'][:2])-destination) for r in settled)
    return dict(passed=all(checks.values()),checks=checks,metrics=metrics,scope=plan['scope'])
