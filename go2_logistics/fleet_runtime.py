"""Shared-world Go2 fleet: one physics clock, cooperative pick/place missions.

The official 48-observation / 12-action gait is evaluated in a robot batch.
Only env.step(primary_actions) advances physics; secondary targets are written
before that step. num_envs stays one, so dogs share parcels and collisions.
"""
import copy
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from .fleet import CooperativeMissions, FleetDispatcher, traffic_commands, observed_grasp, verified_release, west_return_route, clear_placement_station, slew_velocity, placement_standoff, parked_home_peer_positions, z1_placement_route, limit_planar_speed, navigation_step_budget, return_speed_limit, spine_release_clear
from .receiving_pipeline import (wait_for_receiving, select_receiving_approach, home_entry_blocker,
                                 settle_receiving_arrival)
from .fleet import receiving_approach_route
from .interfaces import Pose2D
from .fleet_gui import FleetGuiControl
from .status_io import publish_fleet_status
from .support_exit import SupportExitRamp
from .isaac_policy_runner import BridgeConfig, Go2PolicyBridge, _tensor


def configure_fleet(cfg, count):
    if count != 2:
        raise ValueError('Initial shared-world fleet supports exactly two dogs')
    cfg.scene.robot.init_state.pos = (0., -1., .4)
    secondary = copy.deepcopy(cfg.scene.robot)
    secondary.prim_path = '{ENV_REGEX_NS}/Robot2'
    secondary.init_state.pos = (0., 1., .4)
    cfg.scene.robot2 = secondary
    # Scene duplication does not duplicate manager events: apply the same
    # startup contact material to both articulations, including the fingers.
    material=copy.deepcopy(cfg.events.physics_material)
    material.params['asset_cfg'].name='robot2'
    cfg.events.dog2_physics_material=material
    # Reset manager terms also resolve only "robot" by default. Without the
    # secondary joint reset its Panda branch stays at USD zero angles.
    for term_name in ('reset_robot_joints','reset_base'):
        term=copy.deepcopy(getattr(cfg.events,term_name))
        asset_cfg=copy.deepcopy(term.params.get('asset_cfg',material.params['asset_cfg']))
        asset_cfg.name='robot2'
        term.params['asset_cfg']=asset_cfg
        setattr(cfg.events,'dog2_'+term_name,term)
    for side in ('left', 'right'):
        sensor = copy.deepcopy(getattr(cfg.scene, f'{side}_grip_contact'))
        sensor.prim_path = sensor.prim_path.replace('/Robot/', '/Robot2/', 1)
        setattr(cfg.scene, f'dog2_{side}_grip_contact', sensor)
    if getattr(cfg.scene, 'recovery_camera', None) is not None:
        camera = copy.deepcopy(cfg.scene.recovery_camera)
        camera.prim_path = '{ENV_REGEX_NS}/Dog2RecoveryCamera'
        cfg.scene.dog2_recovery_camera = camera
    if os.environ.get('GO2_FLEET_RECOVERY_CONTROLLER') == 'force_balance':
        contact = copy.deepcopy(cfg.scene.contact_forces)
        contact.prim_path = contact.prim_path.replace('/Robot/', '/Robot2/')
        cfg.scene.dog2_balance_contacts = contact
    # Fleet performs measured fall checks; avoid automatic primary reset.
    cfg.terminations.base_contact = None


class _SceneView:
    def __init__(self, scene, robot_key):
        self.scene, self.robot_key = scene, robot_key

    def __getitem__(self, key):
        return self.scene[self.robot_key if key == 'robot' else key]

    def __getattr__(self, key):
        return getattr(self.scene, key)


class FleetBridge(Go2PolicyBridge):
    def __init__(self, env, robot_key, scheduler, max_steps, *, vector_tracking=False):
        import torch
        self.command = torch.zeros((1, 3), device=env.unwrapped.device)
        view = SimpleNamespace(scene=_SceneView(env.unwrapped.scene, robot_key),
                               command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(command=self.command)))
        super().__init__(SimpleNamespace(unwrapped=view), None,
                         BridgeConfig(max_control_steps=max_steps, position_tolerance=.04, yaw_tolerance=.12,
                                      max_linear_speed=.5, max_lateral_speed=.35,
                                      minimum_longitudinal_command=0. if vector_tracking else .22,
                                      minimum_lateral_command=0.,minimum_docking_lateral_command=0. if vector_tracking else .22))
        self.scheduler = scheduler
        self.robot_key = robot_key
        self.error = ''
        self.localization_model = None

    def true_pose(self):
        """Reference pose for physics evaluation and centralized traffic only."""
        return super().pose()

    def pose_observation(self):
        if self.localization_model is None:
            return None
        observation = self.localization_model.observation
        if observation is None:
            raise RuntimeError('Localization has not been sampled for this control tick')
        return observation

    def pose(self):
        observation = self.pose_observation()
        return observation.estimate if observation is not None else self.true_pose()

    def _policy_step(self):
        self.scheduler.yield_tick()
        if self.error:
            error, self.error = self.error, ''
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError(error)

    def set_body_velocity_command(self, vx, vy=0., yaw_rate=0.):
        """Set velocity in the robot frame, without waypoint conversion."""
        values=(float(vx),float(vy),float(yaw_rate))
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Recovery body velocity must be finite')
        if not 0.<=values[0]<=.50 or abs(values[1])>1e-9 or abs(values[2])>1e-9:
            raise ValueError('Recovery command exceeds the bounded forward-only experiment')
        self.command[0]=self.command.new_tensor(values)


def gait_observation(robot, leg_ids, command, last_action):
    import torch
    data = robot.data
    return torch.cat((_tensor(data.root_lin_vel_b), _tensor(data.root_ang_vel_b),
                      _tensor(data.projected_gravity_b), command,
                      _tensor(data.joint_pos)[:, leg_ids]-_tensor(data.default_joint_pos)[:, leg_ids],
                      _tensor(data.joint_vel)[:, leg_ids]-_tensor(data.default_joint_vel)[:, leg_ids], last_action), dim=-1)


def run_shared_fleet(env, policy, args, pickup, codes, *, return_policy=None):
    import torch
    import omni.usd
    from tensordict import TensorDict
    from .mounted_robot_asset import LEG_JOINT_NAMES
    from .physical_manipulation import PhysicalPandaArm, PhysicalPandaConfig
    from .camera_perception import RedParcelPerception
    from .sensor_navigation import SensorNavigator, PhysXRangeScanner
    from .conveyor_control import ReceivingGate
    from .feed_readiness import ReceivingReadiness
    from .policy_provenance import checkpoint_provenance
    from .mission import PayloadLost
    from .fleet_drop_recovery import FleetDropRecovery, wait_for_reservation
    from .sorting_layout import shelf_for_code, SHELF_DECK_SIZE
    from .streaming_feed import StreamingFeedAudit

    z1_backend = getattr(args, 'arm_model', 'panda_surrogate') == 'z1'
    gait_inference_stride = 4 if z1_backend else 1
    telemetry_stride=10*gait_inference_stride
    if z1_backend and abs(env.unwrapped.step_dt-.005)>1e-8:
        raise RuntimeError('Experimental Z1 fleet requires the validated 5 ms contact-control period')

    scheduler, dispatcher = CooperativeMissions(), FleetDispatcher()
    recover_drops = bool(getattr(args, 'recover_drops', False))
    inject_drop = bool(getattr(args, 'inject_drop', False))
    drop_guard = FleetDropRecovery(dispatcher) if recover_drops else None
    drop_handlers = {}
    reserve_transit = os.environ.get('GO2_FLEET_RESERVE_TRANSIT') == '1'
    segment_transit = os.environ.get('GO2_FLEET_SEGMENT_TRANSIT') == '1'
    if reserve_transit and segment_transit:
        raise ValueError('Full transit and segment transit are separate experiments')
    short_spine = os.environ.get('GO2_FLEET_SHORT_SPINE_RESERVATION') == '1'
    if short_spine and not segment_transit:
        raise ValueError('Short spine reservation requires segment transit')
    spine_x = -7.4 if short_spine else -6.6
    transit_reservation_events = []
    segment_reservation_events = []
    receiving_staging_events = []
    receiving_reservation_events = []
    pipeline_receiving = os.environ.get('GO2_FLEET_PIPELINE_RECEIVING') == '1'
    compact_receiving = os.environ.get('GO2_FLEET_COMPACT_RECEIVING') == '1'
    receiving_approach_x = float(os.environ.get('GO2_FLEET_RECEIVING_APPROACH_X') or '2.4')
    receiving_approach_staged_only = os.environ.get('GO2_FLEET_RECEIVING_APPROACH_STAGED_ONLY') == '1'
    if compact_receiving:
        if not (pipeline_receiving and segment_transit and z1_backend
                and receiving_approach_x==2.4 and not receiving_approach_staged_only):
            raise ValueError('Compact receiving requires original Z1, pipeline/segment transit and original approach options')
        from .compact_receiving import stage_route, exit_route, release_clearance
        stage_route(pickup)  # Reject unsupported fixtures before starting physics.
        receiving_approach_x,receiving_approach_staged_only=3.1,True
    if receiving_approach_staged_only and not (pipeline_receiving and receiving_approach_x > 2.4):
        raise ValueError('Staged-only receiving approach requires PipelineReceiving and alignment x > 2.4')
    receiving_approach_events = []
    receiving_entry_wait_events = []
    receiving_arrival_settle = os.environ.get('GO2_FLEET_SETTLE_RECEIVING') == '1'
    if receiving_arrival_settle and not z1_backend:
        raise ValueError('Receiving arrival settling is scoped to the original Z1 backend')
    receiving_arrival_events = []
    receiving_approach = receiving_approach_route(pickup, original_z1=z1_backend,
        lateral_x=receiving_approach_x)
    scene = env.unwrapped.scene
    stage = omni.usd.get_context().get_stage()
    output = getattr(args, 'run_dir', None) or Path('artifacts/fleet')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_publish_denials=0
    def publish_status(status):
        nonlocal status_publish_denials
        if not publish_fleet_status(output,status):
            status_publish_denials+=1
            if status_publish_denials==1 or status_publish_denials%25==0:
                print(f'FLEET_STATUS_SHARED_LOCKED skipped={status_publish_denials}; run status preserved',flush=True)
    policy_identity = checkpoint_provenance(args.checkpoint)
    (output/'policy_checkpoint.pt').write_bytes(Path(args.checkpoint).read_bytes())
    policy_identity['saved_checkpoint'] = 'policy_checkpoint.pt'
    if policy_identity['training_manifest_verified']:
        (output/'gait_training_manifest.json').write_bytes(Path(policy_identity['training_manifest']).read_bytes())
    return_policy_identity=None
    if return_policy is not None:
        return_path=os.environ.get('GO2_Z1_RETURN_CHECKPOINT')
        if not z1_backend or not return_path:
            raise ValueError('Empty return specialist requires an explicit original-Z1 checkpoint')
        return_policy_identity=checkpoint_provenance(return_path)
        (output/'return_policy_checkpoint.pt').write_bytes(Path(return_path).read_bytes())
        return_policy_identity['saved_checkpoint']='return_policy_checkpoint.pt'
        if return_policy_identity['training_manifest_verified']:
            (output/'return_gait_training_manifest.json').write_bytes(Path(return_policy_identity['training_manifest']).read_bytes())
    # This workspace has no Git history. Preserve the loaded implementation
    # beside each run so later edits cannot obscure which controller ran.
    source_dir=output/'sources'
    source_dir.mkdir()
    source_manifest={}
    for filename in ('fleet_runtime.py','run_isolation.py','touchdown_ik.py','sensor_navigation.py','range_measurement.py','localization.py','video_recording.py','waypoint_ramp.py','fleet.py','status_io.py','support_exit.py','physics_resolution.py','yaw_brake.py',
                     'physical_manipulation.py','mounted_robot_asset.py','isaac_policy_runner.py','return_gait_policy.py','stow_brake_support.py','motor_target_support.py',
                     'local_ground.py','local_ground_importer.py',
                     'conveyor.py','conveyor_control.py','streaming_feed.py','feed_readiness.py','observation_archive.py',
                     'run_localized_demo.ps1','run_streaming_demo.ps1','run_z1_warehouse_demo.ps1','analyze_fleet.py','analyze_segment_transit.py','receiving_pipeline.py','compact_receiving.py','analyze_receiving_pipeline.py',
                     'run_go2_mission.ps1','run_warehouse_demo.ps1','check_fleet_result.ps1','balance_recovery.py','recovery_targets.py','arm_kinematics.py','stance_kinematics.py','travel_heading.py','leg_target_guard.py','policy_provenance.py','fleet_drop_recovery.py','drop_recovery.py','camera_perception.py','recovery_views.py','sorting_scene.py','parcel_geometry.py','contact_righting.py'):
        content=Path(__file__).with_name(filename).read_bytes()
        (source_dir/filename).write_bytes(content)
        source_manifest[filename]=hashlib.sha256(content).hexdigest()
    (output/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf-8')
    if z1_backend:
        for filename in ('z1_fleet_arm.py','z1_warehouse_backend.py','mounted_z1_asset.py',
                         'z1_asset.py','z1_kinematics.py','z1_contact_control.py','z1_contact_wrench.py','z1_world_target.py',
                         'parcel_plane_geometry.py','parcel_geometry.py','sorting_layout.py','z1_contact_probe.py'):
            content=Path(__file__).with_name(filename).read_bytes()
            (source_dir/filename).write_bytes(content)
            source_manifest[filename]=hashlib.sha256(content).hexdigest()
        (output/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf-8')
    names = ('dog1', 'dog2')
    stow_braking = set()
    stow_braking_events = []
    settle_stow_enabled = os.environ.get('GO2_Z1_SETTLE_STOW') == '1'
    brake_support_enabled = os.environ.get('GO2_Z1_STOW_BRAKE_SUPPORT') == '1'
    if brake_support_enabled and not (z1_backend and settle_stow_enabled):
        raise ValueError('Stow brake support requires original Z1 and explicit SettleStow')
    from .stow_brake_support import StowBrakeSupport, loaded_brake_requested
    brake_support_gates={name:StowBrakeSupport() for name in names}
    loaded_brake_enabled=os.environ.get('GO2_Z1_LOADED_BRAKE_SUPPORT')=='1'
    if loaded_brake_enabled and not (z1_backend and os.environ.get('GO2_BALANCE_SCOPE','all') in ('settled','settledall')):
        raise ValueError('Loaded brake support requires original Z1 and a settled balance scope')
    receiving_brake_enabled=os.environ.get('GO2_Z1_RECEIVING_BRAKE_SUPPORT')=='1'
    if receiving_brake_enabled and not (z1_backend and loaded_brake_enabled):
        raise ValueError('Receiving brake support requires original Z1 and LoadedBrakeSupport')
    loaded_support_hysteresis=os.environ.get('GO2_Z1_LOADED_SUPPORT_HYSTERESIS')=='1'
    loaded_brake_gates={name:StowBrakeSupport(attitude_hysteresis=loaded_support_hysteresis) for name in names}
    loaded_recovery_qp_enabled=os.environ.get('GO2_Z1_LOADED_BRAKE_WRENCH_QP')=='1'
    if loaded_recovery_qp_enabled and not (z1_backend and loaded_brake_enabled
            and os.environ.get('GO2_BALANCE_ALLOCATOR','sequential')=='sequential'):
        raise ValueError('Loaded recovery QP requires original Z1, LoadedBrakeSupport and sequential base allocation')
    total_motor_qp_enabled=os.environ.get('GO2_Z1_TOTAL_MOTOR_WRENCH_QP')=='1'
    joint_motor_qp_enabled=os.environ.get('GO2_Z1_JOINT_MOTOR_WRENCH_QP')=='1'
    if joint_motor_qp_enabled and not total_motor_qp_enabled:
        raise ValueError('Joint motor QP requires original Z1 total motor QP')
    if total_motor_qp_enabled and not loaded_recovery_qp_enabled:
        raise ValueError('Total motor QP requires original Z1 and LoadedBrakeWrenchQP')
    if loaded_support_hysteresis and not (loaded_brake_enabled and joint_motor_qp_enabled):
        raise ValueError('Loaded support hysteresis requires loaded support and joint/total motor QP')
    from .return_gait_policy import ReturnPolicyBlend, return_policy_eligible, specialist_idle_control_requested
    empty_transit_specialist=os.environ.get('GO2_Z1_EMPTY_TRANSIT_SPECIALIST')=='1'
    resource_wait_specialist=os.environ.get('GO2_Z1_RESOURCE_WAIT_SPECIALIST')=='1'
    repeat_pickup_specialist=os.environ.get('GO2_Z1_REPEAT_PICKUP_SPECIALIST')=='1'
    specialist_idle_control=os.environ.get('GO2_Z1_SPECIALIST_IDLE_CONTROL')=='1'
    if specialist_idle_control and not repeat_pickup_specialist:
        raise ValueError('Specialist idle control requires measured repeat-pickup specialist selection')
    if (empty_transit_specialist or resource_wait_specialist or repeat_pickup_specialist) and (not z1_backend or return_policy is None):
        raise ValueError('Empty transit specialist requires original Z1 and an explicit return checkpoint')
    return_blends={name:ReturnPolicyBlend() for name in names} if return_policy is not None else {}
    return_blend_stats={name:dict(eligible_ticks=0,active_ticks=0,full_specialist_ticks=0) for name in return_blends}
    specialist_idle_frames={name:0 for name in names}
    keys = ('robot', 'robot2')
    prims = ('/World/envs/env_0/Robot', '/World/envs/env_0/Robot2')
    robots = {name:scene[key] for name,key in zip(names,keys)}
    vector_tracking=z1_backend and os.environ.get('GO2_FLEET_VECTOR_TRACKING')=='1'
    navigation_seconds=os.environ.get('GO2_FLEET_NAVIGATION_BUDGET_S') or None
    navigation_steps=navigation_step_budget(args.steps,env.unwrapped.step_dt,navigation_seconds)
    navigation_budget=dict(mode='seconds' if navigation_seconds is not None else 'explicit_raw_steps',
        control_steps=navigation_steps,control_period_s=env.unwrapped.step_dt,
        nominal_seconds=navigation_steps*env.unwrapped.step_dt,
        scope='outer navigation iterations; nested stance recovery has a separate original guard')
    print(f'FLEET_NAVIGATION_BUDGET {navigation_budget}',flush=True)
    bridges = {name:FleetBridge(env,key,scheduler,navigation_steps,vector_tracking=vector_tracking) for name,key in zip(names,keys)}
    localization_profile = os.environ.get('GO2_FLEET_LOCALIZATION_PROFILE', 'truth')
    if localization_profile not in ('truth', 'perturbed'):
        raise ValueError('Unknown localization profile')
    if localization_profile == 'perturbed':
        from .localization import LocalizationModel
        for index, name in enumerate(names):
            bridges[name].localization_model = LocalizationModel(int(args.seed)+2003*index)
    leg_ids = {name:robots[name].find_joints(list(LEG_JOINT_NAMES),preserve_order=True)[0] for name in names}
    last_actions = {name:torch.zeros((1,12),device=env.unwrapped.device) for name in names}
    standing_samples={name:[] for name in names}
    standing_actions={}
    stationary_mode = os.environ.get('GO2_FLEET_STATIONARY_MODE', 'captured')
    if stationary_mode not in ('captured', 'policy', 'balance'):
        raise ValueError('Unknown stationary mode')
    standing_activations={name:0 for name in names}
    recovery_forward=float(os.environ.get('GO2_FLEET_RECOVERY_FORWARD','0.0'))
    recovery_controller = os.environ.get('GO2_FLEET_RECOVERY_CONTROLLER', 'policy')
    balance_scope = os.environ.get('GO2_BALANCE_SCOPE','all')
    yaw_brake_enabled = os.environ.get('GO2_FLEET_PRE_SUPPORT_YAW_BRAKE') == '1'
    if yaw_brake_enabled and balance_scope not in ('settled','settledall'):
        raise ValueError('Pre-support yaw brake requires settled support')
    if balance_scope not in ('all','stationary','settled','settledall'):
        raise ValueError('Unknown balance support scope')
    if balance_scope in ('stationary','settled','settledall') and (stationary_mode!='balance' or recovery_controller!='force_balance'):
        raise ValueError('Stationary support scope requires Balance parking and force_balance support')
    if recovery_controller not in ('policy', 'force_balance'):
        raise ValueError('Unknown fleet recovery controller')
    if stationary_mode == 'balance' and recovery_controller != 'force_balance':
        raise ValueError('Balance stationary support requires force_balance')
    if recovery_controller == 'force_balance' and recovery_forward:
        raise ValueError('Force balance recovery requires zero velocity, not the forward experiment')
    if not math.isfinite(recovery_forward) or not 0.0<=recovery_forward<=.50:
        raise ValueError('GO2_FLEET_RECOVERY_FORWARD must be finite in [0,.50]')
    for name in names:
        if recovery_forward:
            bridges[name].recovery_velocity_body=(recovery_forward,0.,0.)
    was_standing={name:False for name in names}
    arms, perceptions, navigators = {}, {}, {}
    phases = {name:'starting' for name in names}
    home_positions={name:(0.,-1. if index==0 else 1.) for index,name in enumerate(names)}
    plan_peer_clearance=os.environ.get('GO2_FLEET_PLAN_PEER_CLEARANCE')=='1'
    if plan_peer_clearance and not (reserve_transit or segment_transit):
        raise ValueError('Peer clearance planning requires reserved transport')
    current_codes = {name:None for name in names}
    returned = {name:False for name in names}
    endurance_cycles=getattr(args,'fleet_endurance_cycles',0)
    completed_cycles={name:0 for name in names}
    completed_home_returns={name:0 for name in names}
    home_return_events=[]
    delivery_evidence=[]
    manipulation_attempts=[]
    objects = {code:scene['parcel' if i==0 else f'parcel_{i}'] for i,code in enumerate(codes)}
    streaming_feed_enabled=bool(getattr(args,'streaming_feed',False))
    feed_audit=(StreamingFeedAudit(codes,run_id=output.name,
                                   receiving_xy=(pickup.x,pickup.y))
                if streaming_feed_enabled else None)
    forward_floor=float(os.environ.get('GO2_FLEET_FORWARD_FLOOR','0.30'))
    leg_target_margin=float(os.environ.get('GO2_FLEET_LEG_TARGET_MARGIN','0'))
    if not math.isfinite(leg_target_margin) or not 0.<=leg_target_margin<=.20:
        raise ValueError('Leg target margin must be finite in [0,.20]')
    leg_guard_stats={name:dict(corrected_frames=0,max_target_correction_rad=0.,max_corrected_joints=0) for name in names}
    carry_retraction=float(os.environ.get('GO2_FLEET_CARRY_RETRACTION','0'))
    retreat_height=float(os.environ.get('GO2_FLEET_RELEASE_RETREAT','0.08'))
    if not math.isfinite(retreat_height) or not .08<=retreat_height<=.18:
        raise ValueError('Fleet release retreat height must be finite in [.08,.18]')
    if not math.isfinite(forward_floor) or not .30<=forward_floor<=.50:
        raise ValueError('GO2_FLEET_FORWARD_FLOOR must be finite in [.30,.50]')
    object_paths = {code:'/World/envs/env_0/Parcel'+(str(i) if i else '') for i,code in enumerate(codes)}
    for index,name in enumerate(names):
        folder = output/name
        folder.mkdir()
        if z1_backend:
            from .z1_fleet_arm import PhysicalZ1Arm, Z1FleetConfig
            from .z1_warehouse_backend import (warehouse_carry_lift, warehouse_approach_seconds,
                warehouse_close_seconds, warehouse_jaw_sample_x, warehouse_grasp_heading_reference,
                warehouse_pregrasp_seconds, warehouse_lift_seconds)
            from .mounted_z1_asset import z1_gripper_body_path
            arms[name] = PhysicalZ1Arm(env,stage,objects[codes[0]],
                hand_path=z1_gripper_body_path(prims[index],'gripperStator'),
                box_path=object_paths[codes[0]],robot_key=keys[index],
                contact_sensor_prefix='' if index==0 else 'dog2_',
                config=Z1FleetConfig(dropoff_position=(0.,0.,.46),
                    parcel_size_xyz_m=tuple(objects[codes[0]].cfg.spawn.size),
                    lift_height=warehouse_carry_lift(os.environ.get('GO2_Z1_CARRY_LIFT','.44')),
                    approach_seconds=warehouse_approach_seconds(os.environ.get('GO2_Z1_APPROACH_SECONDS','8')),
                    close_seconds=warehouse_close_seconds(os.environ.get('GO2_Z1_CLOSE_SECONDS','24')),
                    close_on_stable_contact=os.environ.get('GO2_Z1_CLOSE_ON_STABLE_CONTACT')=='1',
                    pregrasp_seconds=warehouse_pregrasp_seconds(os.environ.get('GO2_Z1_PREGRASP_SECONDS','')),
                    lift_seconds=warehouse_lift_seconds(os.environ.get('GO2_Z1_LIFT_SECONDS','12')),
                    grasp_heading_reference_rad=warehouse_grasp_heading_reference(
                        os.environ.get('GO2_Z1_GRASP_HEADING_REFERENCE_DEG','')),
                    jaw_sample_x_m=warehouse_jaw_sample_x(os.environ.get('GO2_Z1_JAW_SAMPLE_X','.15')),
                    precontact_limit_n=2.,
                    placement_approach_height=.04,
                    release_clearance_backoff_m=.04 if os.environ.get('GO2_Z1_DEPARTURE_CLEARANCE')=='1' else 0.,
                    release_clearance_rise_m=.04 if os.environ.get('GO2_Z1_DEPARTURE_CLEARANCE')=='1' else 0.,
                    placement_surface_z=shelf_for_code(codes[0]).surface[2],
                    placement_deck_size_xy=SHELF_DECK_SIZE[:2]),
                step_fn=bridges[name]._policy_step)
        else:
            arms[name] = PhysicalPandaArm(env,stage,objects[codes[0]],
            hand_path=prims[index]+'/Arm/panda_hand',box_path=object_paths[codes[0]],
            robot_key=keys[index],contact_sensor_prefix='' if index==0 else 'dog2_',
            config=PhysicalPandaConfig(dropoff_position=(0.,0.,.46),grasp_mode='contact',
                # Bin center .06 + lift .44 - half height .04 = .46:
                # nominal 4 cm clearance above the .42 m shelf surface.
                lift_height=.44,lift_segments=5,ik_joint_limit_aware=True,
                carry_retraction_m=carry_retraction,
                ik_path_preflight=os.environ.get('GO2_FLEET_ARM_PREFLIGHT')=='1',
                ik_path_refinement=os.environ.get('GO2_FLEET_ARM_REFINEMENT')=='1',
                ik_grasp_symmetry=os.environ.get('GO2_FLEET_GRASP_SYMMETRY')=='1',
                reset_before_grasp=True,stow_after_release=False,
                placement_approach_height=.08,release_retreat_height=retreat_height,
                placement_planar_tolerance=.04,placement_height_tolerance=.015,
                placement_surface_z=shelf_for_code(codes[0]).surface[2],placement_deck_size_xy=SHELF_DECK_SIZE[:2],
                tool_offset=(0.,0.,.1034*.85),constraint_path='/World/FleetAssistedGrasp_'+name),
            step_fn=bridges[name]._policy_step)
        data=robots[name].data
        actual=_tensor(data.joint_pos)[0,arms[name].arm_ids]
        expected=_tensor(data.default_joint_pos)[0,arms[name].arm_ids]
        print(f'FLEET_INITIAL_ARM robot={name} actual={actual.tolist()} configured={expected.tolist()}',flush=True)
        if float((actual-expected).abs().max())>.10:
            raise RuntimeError(f'{name} arm not initialized to its configured reset posture')
        arms[name].prepare()
        bridges[name].recovery_payload_held=lambda arm=arms[name]: arm.held
        perceptions[name]=RedParcelPerception(env,arms[name],folder,decode_code=True)
        perceptions[name].estimate_geometry = z1_backend
        perceptions[name].archive_observations = z1_backend
        perceptions[name].geometry_max_tilt_deg = 12. if z1_backend else 0.
        navigators[name]=SensorNavigator(env,folder,dynamic_map=True,stalled_forward_command=.4,
            vector_tracking=vector_tracking,
            control_period_s=env.unwrapped.step_dt,
            peer_positions=(lambda name=name: parked_home_peer_positions(name,
                {peer:(bridges[peer].true_pose().x,bridges[peer].true_pose().y) for peer in names},phases,home_positions)
                if phases[name]=='transporting' and arms[name].held else []) if plan_peer_clearance else None,
            loaded_recovery_trigger_height=float(os.environ.get('GO2_FLEET_LOADED_RECOVERY_TRIGGER_HEIGHT','.285')),
            minimum_positive_forward_command=forward_floor,
            stalled_lateral_command=.30 if os.environ.get('GO2_FLEET_LATERAL_STALL')=='1' else None,
            axis_routes=os.environ.get('GO2_FLEET_AXIS_ROUTES')=='1',
            travel_heading=os.environ.get('GO2_FLEET_TRAVEL_HEADING')=='1')
        navigators[name].scanner=PhysXRangeScanner(env,robot_key=keys[index],self_path=prims[index],
            range_profile=os.environ.get('GO2_FLEET_RANGE_PROFILE','ideal'), seed=int(args.seed)+1009*index,
            measurement_log=folder/'range_measurements.jsonl',
            pose_observation=bridges[name].pose_observation if localization_profile == 'perturbed' else None)
        if recover_drops:
            from .drop_recovery import DroppedParcelRecovery
            drop_handlers[name] = DroppedParcelRecovery(env, arms[name], folder/'drop_recovery',
                inject=inject_drop and index == 0,
                camera_key='recovery_camera' if index == 0 else 'dog2_recovery_camera', clock=lambda: ticks,
                right_fallen=os.environ.get('GO2_FLEET_CONTACT_RIGHTING')=='1')
            # Recovery redocking must not recursively start another recovery.
            drop_handlers[name].bind(bridges[name],
                lambda target, name=name: navigators[name].navigate(bridges[name], target, transit=False))

    balances = {}
    if recovery_controller == 'force_balance':
        from .balance_recovery import ContactBalanceRecovery, BalanceConfig, support_scope_requested, workspace_scope_requested, SettledPayloadSupport
        balance_height=float(os.environ.get('GO2_BALANCE_HEIGHT','0.33'))
        if not math.isfinite(balance_height) or not .31<=balance_height<=.33:
            raise ValueError('Balance target height must be finite in [.31,.33]; navigation exit criteria are unchanged')
        balance_config = BalanceConfig(height_m=balance_height,allocator=os.environ.get('GO2_BALANCE_ALLOCATOR', 'sequential'),
            stationary_payload_qp=os.environ.get('GO2_Z1_PLACEMENT_WRENCH_QP')=='1',
            loaded_recovery_qp=loaded_recovery_qp_enabled,
            total_motor_qp=total_motor_qp_enabled,
            joint_motor_qp=joint_motor_qp_enabled,
            handoff_mode=os.environ.get('GO2_BALANCE_HANDOFF', 'legacy'),
            handoff_release_seconds=float(os.environ.get('GO2_BALANCE_HANDOFF_RELEASE_S', '0')),
            recenter_foot=os.environ.get('GO2_BALANCE_RECENTER')=='1',
            offload_foot=os.environ.get('GO2_BALANCE_OFFLOAD')=='1',
            measured_com=os.environ.get('GO2_BALANCE_MEASURED_COM')=='1',
            leg_ik=os.environ.get('GO2_BALANCE_LEG_IK','clipped'),
            stance_fit=os.environ.get('GO2_BALANCE_STANCE_FIT')=='1',
            target_update=os.environ.get('GO2_BALANCE_TARGET_UPDATE','legacy'),
            touchdown_ik=os.environ.get('GO2_BALANCE_TOUCHDOWN_IK')=='1',
            airborne_touchdown=os.environ.get('GO2_BALANCE_AIRBORNE_TOUCHDOWN')=='1',
            workspace_reference=os.environ.get('GO2_BALANCE_WORKSPACE_REFERENCE')=='1',
            workspace_entry_max_height_m=float(os.environ['GO2_BALANCE_WORKSPACE_MAX_HEIGHT'])
                if os.environ.get('GO2_BALANCE_WORKSPACE_MAX_HEIGHT') else None)
        from .stance_kinematics import StanceGeometry
        def stance_clearance(name,target):
            navigator=navigators[name]
            try:
                returns=navigator.scanner.scan()
                pose=bridges[name].pose()
                navigator.scans.append(dict(pose=(pose.x,pose.y),returns=returns,
                    purpose='workspace_reference' if balance_config.workspace_reference else 'stance_fit'))
                navigator.cleared_cells+=navigator.grid.update_scan(returns,navigator.scanner.free_segments)
                return navigator.grid.clear_line((pose.x,pose.y),tuple(target[:2]))
            except Exception as error:
                print(f'STANCE_FIT_SCAN_REJECTED robot={name} error={error}',flush=True)
                return False
        for index, name in enumerate(names):
            bridges[name].recovery_settle_motion = balance_config.handoff_mode == 'settled_slew'
            sensor = scene.sensors['contact_forces' if index == 0 else 'dog2_balance_contacts']
            balances[name] = ContactBalanceRecovery(robots[name], sensor, leg_ids[name],
                env.unwrapped.cfg.actions.joint_pos.scale, payload_mass=float(objects[codes[0]].cfg.spawn.mass_props.mass),
                payload_state=lambda arm=arms[name]: ((_tensor(arm.box.data.body_com_pose_w)[0,0,:3].cpu().numpy(),
                    float(arm.box.cfg.spawn.mass_props.mass)) if arm.held else None),
                config=balance_config,
                stance_geometry=StanceGeometry.from_stage(stage,prims[index]) if (balance_config.stance_fit or balance_config.workspace_reference) else None,
                stance_clearance_check=lambda target,name=name:stance_clearance(name,target))
            if balance_config.stance_fit or balance_config.workspace_reference:
                checked=balances[name].check_stance_geometry()
                print(f'FLEET_STANCE_GEOMETRY robot={name} check={checked}',flush=True)
            bridges[name].recovery_step_ready=lambda name=name:(not balances[name].recenter_active
                and not balances[name].stance_fit_active and not balances[name].workspace_active)

    def phase(name, value):
        if phases[name]==value:
            return
        phases[name]=value
        print(f'FLEET_PHASE robot={name} phase={value} code={current_codes[name]}',flush=True)

    def recover_payload(name):
        original_phase=phases[name]
        phase(name, 'recovering_dropped_parcel')
        bridges[name]._zero_command()
        drop_guard.recover(name, current_codes[name], drop_handlers[name], arms[name], lambda: ticks)
        phase(name, original_phase)

    def navigate(name, target, transit=True, *, position_tolerance=None):
        if os.environ.get('GO2_FLEET_SETTLE_TRANSITIONS') == '1':
            if name not in balances:
                raise ValueError('Transition settling requires force-balance recovery')
            navigators[name].recover_stance(bridges[name], settle_motion=True)
        while True:
            try:
                if not navigators[name].navigate(bridges[name],target,transit=transit,
                                                 position_tolerance=position_tolerance):
                    raise RuntimeError(f'{name}: navigation timeout at {target}')
                return
            except PayloadLost:
                if drop_guard is None:
                    raise
                recover_payload(name)

    def brake_for_reservation(name,resource):
        if (z1_backend and arms[name].held and balance_scope in ('settled','settledall')
                and os.environ.get('GO2_FLEET_SETTLE_RESERVATION')=='1'):
            print(f'FLEET_RESERVATION_BRAKE robot={name} resource={resource} tick={ticks}',flush=True)
            while True:
                try:
                    navigators[name].recover_stance(bridges[name],settle_motion=True,minimum_duration_s=.3)
                    break
                except PayloadLost:
                    if drop_guard is None:
                        raise
                    recover_payload(name)

    def reserve(name, resource, *, prebraked=False):
        if segment_transit:
            segment_reservation_events.append(dict(event='requested',resource=resource,robot=name,tick=ticks))
        def brake_before_wait():
            if not prebraked:
                brake_for_reservation(name,resource)
        wait_for_reservation(dispatcher,bridges[name],name,resource,
            lambda:phase(name,'waiting_'+resource),
            (lambda:recover_payload(name)) if drop_guard else None,
            before_wait=brake_before_wait)
        if resource == 'delivery_transit':
            transit_reservation_events.append(dict(event='acquired',robot=name,tick=ticks))
            print(f'FLEET_TRANSIT_RESERVED robot={name} tick={ticks}',flush=True)
        if segment_transit:
            segment_reservation_events.append(dict(event='acquired',resource=resource,robot=name,tick=ticks))

    def release_segment(name, resource, **evidence):
        dispatcher.release(resource,name)
        segment_reservation_events.append(dict(event='released',resource=resource,robot=name,tick=ticks,**evidence))

    def handoff(name,previous,following,release):
        from .fleet import handoff_owned_resource
        handoff_owned_resource(dispatcher,name,previous,
            lambda:brake_for_reservation(name,following),release,
            lambda:phase(name,'waiting_'+following))
        reserve(name,following,prebraked=True)

    def record_attempt(name,event):
        event=dict(event,robot=name,tick=ticks)
        if event.get('operation')=='grasp' and perceptions[name].last_archive is not None:
            event['observation_snapshot']=dict(perceptions[name].last_archive)
        manipulation_attempts.append(event)
        print(f'FLEET_MANIPULATION {event}',flush=True)

    def mission(name,index):
        home = Pose2D(0.,-1. if index==0 else 1.)
        try:
            # Let both physical articulations settle under their own gait.
            for _ in range(60*gait_inference_stride):
                bridges[name]._policy_step()
            if args.fleet_smoke:
                smoke_goal_x = -3. if os.environ.get('GO2_FLEET_SMOKE_OBSTACLE_ROUTE')=='1' else 1.5
                for cycle in range(endurance_cycles or 1):
                    phase(name,'parallel_navigation')
                    navigate(name,Pose2D(smoke_goal_x,home.y))
                    if endurance_cycles:
                        navigate(name,Pose2D(smoke_goal_x,home.y-.6))
                        navigate(name,Pose2D(0.,home.y-.6))
                    phase(name,'returning')
                    # Home is an acceptance endpoint, including the short GUI
                    # smoke. A 20 cm transit exit can drift beyond the 21 cm
                    # final acceptance radius while the other dog is braking.
                    navigate(name,home,transit=False)
                    completed_cycles[name]+=1
                    print(f'FLEET_ROUTE_COMPLETE robot={name} cycle={cycle+1}/{endurance_cycles or 1}',flush=True)
                returned[name]=True
                phase(name,'smoke_complete')
                return
            while len(dispatcher.delivered)<len(codes):
                phase(name,'waiting_receiving')
                acquired=wait_for_receiving(dispatcher,bridges[name],name,codes=codes,
                    phases=phases,pickup=pickup,navigate=lambda point:navigate(name,point),
                    withdraw=lambda:navigate(name,home,False,position_tolerance=.02 if z1_backend else None),
                    phase=lambda value:phase(name,value),tick=lambda:ticks,
                    staging_events=receiving_staging_events,reservation_events=receiving_reservation_events,
                    enabled=pipeline_receiving,compact_layout=compact_receiving,
                    entry_blocker=(lambda:home_entry_blocker(dispatcher.resources,name,
                        (bridges[name].true_pose().x,bridges[name].true_pose().y),home_positions[name],
                        reservation_events=segment_reservation_events))
                        if segment_transit and plan_peer_clearance else None,
                    entry_wait_events=receiving_entry_wait_events)
                if not acquired:
                    phase(name,'waiting_receiving')
                    bridges[name]._zero_command()
                    bridges[name]._policy_step()
                    continue
                # Only one dog enters the pickup approach. Reservation remains
                # held until its trunk exits west after the grasp.
                phase(name,'approaching_receiving')
                staged = receiving_reservation_events[-1].get('staged') is True
                selected_approach = select_receiving_approach(pickup, original_z1=z1_backend,
                    lateral_x=receiving_approach_x, staged_only=receiving_approach_staged_only, staged=staged)
                receiving_approach_events.append(dict(robot=name,tick=ticks,staged=staged,
                    lateral_x_m=selected_approach[0].x,waypoints=[vars(point) for point in selected_approach]))
                for point_index,point in enumerate(selected_approach):
                    # Complete the lateral maneuver west of the low rim;
                    # retain the original short-arm dock and precision stops.
                    navigate(name,point,transit=point_index==0)
                bridges[name]._zero_command()
                if receiving_arrival_settle:
                    # Stay in approaching_receiving so the zero-command gait
                    # brakes before waiting_visible_parcel enables static PD.
                    settle_receiving_arrival(bridges[name],navigators[name],name,selected_approach[-1],
                        tick=lambda:ticks,events=receiving_arrival_events)
                phase(name,'waiting_visible_parcel')
                code=None
                readiness=ReceivingReadiness((pickup.x,pickup.y))
                for attempt in range(240):
                    if attempt%5==0:
                        detection=perceptions[name].last_detection if perceptions[name].locate('box_red') else None
                        if readiness.observe(detection,conveyor_running=gate.running):
                            candidate=detection['parcel_code']
                            if candidate not in objects:
                                raise RuntimeError('Observed parcel is absent from the physical registry')
                            if feed_audit is not None:
                                feed_audit.observe_receiving(detection,tick=ticks)
                            if dispatcher.claim(candidate,name):
                                code=candidate
                                current_codes[name]=code
                                if feed_audit is not None:
                                    feed_audit.claim_visual(code,tick=ticks)
                                print(f'FLEET_PICKUP_READY robot={name} code={code} stable_samples={readiness.count} belt_running={gate.running}',flush=True)
                                break
                    for _ in range(5*gait_inference_stride):
                        bridges[name]._policy_step()
                if code is None:
                    raise RuntimeError('No unclaimed camera-decoded parcel at the receiving station')
                current_codes[name]=code
                arm=arms[name]
                arm.box,arm.box_path=objects[code],object_paths[code]
                navigators[name].scanner.payload_path=object_paths[code]
                shelf=shelf_for_code(code)
                arm.config=replace(arm.config,dropoff_position=(*shelf.surface[:2],shelf.surface[2]+.04),
                    placement_surface_z=shelf.surface[2])
                phase(name,'grasping')
                if not observed_grasp(arm,perceptions[name],code,emit=lambda event:record_attempt(name,event)):
                    raise RuntimeError(arm.last_error)
                dispatcher.transition(code,name,'carrying')
                if feed_audit is not None:
                    feed_audit.mark_carrying(code,tick=ticks)
                phase(name,'leaving_receiving')
                if compact_receiving:
                    departure=exit_route(pickup)
                    navigate(name,departure[0],False)
                else:
                    navigate(name,Pose2D(2.2,pickup.y))
                    navigate(name,Pose2D(2.2,-.3))
                def release_receiving():
                    pose=bridges[name].true_pose()
                    detail={}
                    if compact_receiving:
                        detail=dict(compact_release=True,
                            incoming_path_clearance_m=release_clearance(pose,pickup))
                    dispatcher.release('receiving',name)
                    receiving_reservation_events.append(dict(event='released',resource='receiving',robot=name,tick=ticks,
                        root_xy=[pose.x,pose.y],**detail))
                if reserve_transit:
                    # The unreserved approach to the western waiting point
                    # crosses a peer's return clearance route. Wait here, east
                    # of that route, until its entire delivery/return finishes.
                    # The peer may still navigate to receiving and grasp while
                    # this dog transports. Existing traffic veto stays active.
                    handoff(name,'receiving','delivery_transit',release_receiving)
                elif segment_transit:
                    handoff(name,'receiving','approach_corridor',release_receiving)
                else:
                    release_receiving()
                phase(name,'transporting')
                if compact_receiving:
                    # The old receiving lease was kept through measured braking.
                    # The remaining exit uses the acquired approach corridor.
                    navigate(name,departure[1])
                # Narrow picking rows cannot support opposite-direction dogs.
                # Wait outside the western spine, then reserve row + spine in
                # this consistent order. Other rows can operate concurrently.
                navigate(name,Pose2D(-4.7,-1.3))
                if recover_drops and drop_handlers[name].inject_pending:
                    drop_guard.begin_injection(name, code, ticks)
                    phase(name, 'drop_injection')
                    bridges[name]._zero_command()
                    try:
                        drop_handlers[name].inject_after_transit()
                    except PayloadLost:
                        recover_payload(name)
                    phase(name, 'transporting')
                if segment_transit:
                    handoff(name,'approach_corridor',f'row_{shelf.row+1}',
                        lambda:release_segment(name,'approach_corridor'))
                else:
                    reserve(name,f'row_{shelf.row+1}')
                reserve(name,'west_spine',prebraked=segment_transit)
                phase(name,'transporting')
                for point in (Pose2D(spine_x,-.3),Pose2D(spine_x,shelf.dock.y-.85)):
                    navigate(name,point)
                # A walking payload must not overlap the deck: trunk height
                # changes exhausted the nominal lift clearance in round 31.
                # Finish south of the deck, then reach in while standing.
                if z1_backend:
                    approach=z1_placement_route(shelf)
                    for point in approach:
                        navigate(name,point,False,
                                 position_tolerance=.015 if point==approach[-1] else None)
                else:
                    navigate(name,Pose2D(shelf.dock.x,shelf.dock.y-.85))
                    navigate(name,placement_standoff(shelf),False)
                if not segment_transit:
                    dispatcher.release('west_spine',name)
                bridges[name]._zero_command()
                if z1_backend and balance_scope in ('settled','settledall'):
                    # Stay in the navigation phase while braking: entering
                    # "placing" enables stationary support immediately. Let
                    # the gait stop first and the measured support gate latch.
                    navigators[name].recover_stance(bridges[name],settle_motion=True,
                        minimum_duration_s=.6,
                        precision_manipulation=os.environ.get('GO2_FLEET_PLACEMENT_PRECISION')=='1')
                    pose=bridges[name].pose()
                    dock=z1_placement_route(shelf)[-1]
                    docking_error=math.hypot(pose.x-dock.x,pose.y-dock.y)
                    if docking_error>bridges[name].config.position_tolerance:
                        raise RuntimeError(f'{name}: stopped outside placement dock after braking '
                                           f'({docking_error:.4f}m)')
                    print(f'FLEET_PLACEMENT_STOP robot={name} tick={ticks} '
                          f'dock_error_m={docking_error:.6f}',flush=True)
                if short_spine:
                    stopped_pose=bridges[name].true_pose()
                    if not spine_release_clear(stopped_pose,spine_x=spine_x):
                        raise RuntimeError(f'{name}: dock does not clear western transit lane')
                    release_segment(name,'west_spine',reason='inside_exclusive_row_at_dock',
                                    root_xy=[stopped_pose.x,stopped_pose.y])
                phase(name,'placing')
                if not verified_release(arm,code,emit=lambda event:record_attempt(name,event)):
                    raise RuntimeError(arm.last_error)
                phase(name,'returning')
                if short_spine or not segment_transit:
                    reserve(name,'west_spine')
                phase(name,'returning')
                return_route=west_return_route(shelf,original_z1=z1_backend,spine_x=spine_x)
                def depart_station():
                    navigate(name,return_route[0],not z1_backend)
                def before_stow():
                    if z1_backend and settle_stow_enabled:
                        print(f'FLEET_STOW_BRAKE robot={name} tick={ticks}',flush=True)
                        event=dict(robot=name,start_tick=ticks,completed=False)
                        stow_braking_events.append(event)
                        stow_braking.add(name)
                        try:
                            navigators[name].recover_stance(bridges[name],settle_motion=True,minimum_duration_s=.3)
                            event.update(completed=True,end_tick=ticks,
                                root_velocity_w=_tensor(robots[name].data.root_lin_vel_w)[0].cpu().tolist(),
                                root_angular_velocity_b=_tensor(robots[name].data.root_ang_vel_b)[0].cpu().tolist())
                        finally:
                            stow_braking.discard(name)
                    phase(name,'stowing_after_departure')
                clear_placement_station(arm,depart_station,before_stow=before_stow)
                delivery_evidence.append(dict(code=code,robot=name,shelf=shelf.number,
                    verified_lift_m=arm.verified_lift,max_hold_error_m=arm.max_hold_error,
                    constraint_created=arm.joint is not None,placement=copy.deepcopy(arm.placement_metrics)))
                dispatcher.transition(code,name,'delivered')
                if feed_audit is not None:
                    feed_audit.mark_delivered(code,tick=ticks)
                phase(name,'returning')
                for point in return_route[1:]:
                    if segment_transit and point==return_route[-1]:
                        # Do not hold this approach reservation while waiting
                        # for a row/spine. Outbound traffic releases it at the
                        # southern staging point, so no circular wait forms.
                        reserve(name,'approach_corridor')
                        phase(name,'returning')
                    navigate(name,point)
                if segment_transit:
                    release_segment(name,'west_spine')
                    release_segment(name,f'row_{shelf.row+1}')
                else:
                    dispatcher.release('west_spine',name)
                    dispatcher.release(f'row_{shelf.row+1}',name)
                # Leave reserve for small stance drift while the peer finishes;
                # the independent original-Z1 home check remains 4 cm.
                navigate(name,home,False,position_tolerance=.02 if z1_backend else None)
                current_codes[name]=None
                arm.prepare()
                navigators[name].scanner.payload_path=None
                if reserve_transit:
                    dispatcher.release('delivery_transit',name)
                    transit_reservation_events.append(dict(event='released',robot=name,tick=ticks))
                    print(f'FLEET_TRANSIT_RELEASED robot={name} tick={ticks}',flush=True)
                elif segment_transit:
                    release_segment(name,'approach_corridor')
                completed_home_returns[name]+=1
                home_return_events.append(dict(robot=name,code=code,tick=ticks,
                    completed_returns=completed_home_returns[name],root_xy=[bridges[name].true_pose().x,bridges[name].true_pose().y]))
            returned[name]=True
            phase(name,'complete')
        except Exception as error:
            bridges[name]._zero_command()
            code=current_codes[name]
            if code and dispatcher.orders[code].state in ('claimed','carrying','recovering'):
                dispatcher.transition(code,name,'failed',str(error))
            if feed_audit is not None and code in feed_audit.state and feed_audit.state[code] not in ('delivered','failed'):
                feed_audit.mark_failed(code,tick=ticks,reason=str(error))
            # Do not release station reservations blindly: a failed dog can
            # still physically obstruct the protected area.
            phase(name,'failed')
            raise

    for index,name in enumerate(names):
        scheduler.add(name,lambda name=name,index=index:mission(name,index))
    gate=ReceivingGate(env,pickup)
    if feed_audit is not None:
        gate.attach_stream_audit(feed_audit,objects,tick=0,sample_interval=telemetry_stride)
    camera=scene['warehouse_camera']
    camera.set_world_poses_from_view(torch.tensor([[11.,-10.,7.]],device=env.unwrapped.device),
                                     torch.tensor([[0.,3.,.4]],device=env.unwrapped.device))
    if args.visual:
        env.unwrapped.sim.set_camera_view((11.,-10.,7.),(0.,3.,.4))
    app=timeline=hud=gui=None
    camera_mode='overview'
    if args.visual:
        import omni.kit.app
        import omni.timeline
        import omni.ui as ui
        app=omni.kit.app.get_app()
        timeline=omni.timeline.get_timeline_interface()
        gui=FleetGuiControl(env.unwrapped.sim,app,timeline,wait_for_start=args.wait_for_start)
        window=ui.Window('Shared warehouse fleet',width=390,height=205)
        with window.frame:
            with ui.VStack(spacing=8):
                hud=ui.Label('Two physical Go2 robots | shared world',word_wrap=True)
                # SimulationContext.play/pause call app.update internally.
                # Defer these operations until the renderer callback returns.
                ui.Button('START / PAUSE / RESUME',clicked_fn=gui.request_toggle)
                def select_camera(mode):
                    nonlocal camera_mode
                    camera_mode=mode
                    if mode=='overview':
                        env.unwrapped.sim.set_camera_view((11.,-10.,7.),(0.,3.,.4))
                    else:
                        p=_tensor(robots[mode].data.root_pos_w)[0].cpu().tolist()
                        env.unwrapped.sim.set_camera_view((p[0]+2.4,p[1]-2.8,2.1),(p[0]+.2,p[1],.4))
                    print(f'FLEET_UI camera={mode}',flush=True)
                with ui.HStack(spacing=4):
                    ui.Button('WAREHOUSE',clicked_fn=lambda:gui.request_camera('overview'))
                    ui.Button('DOG 1',clicked_fn=lambda:gui.request_camera('dog1'))
                    ui.Button('DOG 2',clicked_fn=lambda:gui.request_camera('dog2'))
                ui.Label('Close the app to exit. STOP requires relaunch.',word_wrap=True)
        gui.synchronize()
        if args.wait_for_start:
            hud.text='WAITING FOR START | press START / PAUSE / RESUME\nTwo physical Go2 robots | shared world'
    import imageio.v2 as imageio
    from .video_recording import get_mission_writer
    recording_fps=1./(telemetry_stride*env.unwrapped.step_dt)
    writer=get_mission_writer(output/'fleet.mp4',fps=recording_fps)
    detail_enabled=os.environ.get('GO2_FLEET_DETAIL_VIDEO') == '1'
    detail_camera=scene['mission_camera'] if detail_enabled else None
    detail_writer=(get_mission_writer(output/'fleet_detail.mp4',fps=recording_fps)
                   if detail_enabled else None)
    detail_focus='dog1'
    detail_switch_tick=0
    detail_freeze_checks={}
    ticks=0
    overlap_ticks=0
    navigation_overlap_ticks=0
    traffic_stops=0
    minimum_distance=float('inf')
    failure=''
    abi_checked=False
    abi_checks=0
    abi_max_error=0.
    previous_commands={name:(0.,0.,0.) for name in names}
    velocity_slew_enabled=os.environ.get('GO2_FLEET_VELOCITY_SLEW')=='1'
    loaded_speed_limit=float(os.environ.get('GO2_FLEET_LOADED_SPEED_LIMIT','0'))
    if not math.isfinite(loaded_speed_limit) or not (loaded_speed_limit==0. or .20<=loaded_speed_limit<=.61):
        raise ValueError('Loaded planar command limit must be disabled or within [.20,.61] m/s')
    empty_return_limit=float(os.environ.get('GO2_FLEET_RETURN_SPEED_LIMIT','0'))
    return_speed_limit((0.,0.,0.),empty_return_limit,original_z1=z1_backend,payload_held=False,phase='starting')
    from .waypoint_ramp import WaypointCommandRamp, nominal_stop_segment
    waypoint_ramp_enabled=os.environ.get('GO2_FLEET_WAYPOINT_RAMP')=='1'
    waypoint_ramps={name:WaypointCommandRamp() for name in names}
    if waypoint_ramp_enabled and velocity_slew_enabled:
        raise ValueError('Waypoint ramp and global velocity slew are separate experiments')
    support_exit_enabled=os.environ.get('GO2_FLEET_SUPPORT_EXIT_RAMP')=='1'
    support_exit_ramps={name:SupportExitRamp() for name in names}
    trace=(output/'trace.jsonl').open('w',encoding='utf-8')
    control_trace=(output/'control_trace.jsonl').open('w',encoding='utf-8')
    published_gui_state=gui.state if gui else None
    def progress_label():
        if args.fleet_smoke:
            target=endurance_cycles or 1
            return 'UNLOADED ROUTES | '+ ' | '.join(f'{name} {completed_cycles[name]}/{target}' for name in names)
        return f'delivered {len(dispatcher.delivered)}/{len(codes)}'
    def final_state_snapshot():
        return dict(
            final_roots={name:_tensor(robot.data.root_pos_w)[0].cpu().tolist() for name,robot in robots.items()},
            final_parcels={code:_tensor(obj.data.root_pos_w)[0].cpu().tolist() for code,obj in objects.items()},
            final_parcel_quaternions_xyzw={code:_tensor(obj.data.root_quat_w)[0].cpu().tolist() for code,obj in objects.items()},
            final_parcel_velocities={code:_tensor(obj.data.root_lin_vel_w)[0].cpu().tolist() for code,obj in objects.items()})
    last_valid_final_state=final_state_snapshot()
    last_valid_state_tick=0
    gui_check_enabled=getattr(args,'fleet_gui_check',False)
    gui_checks={}
    def verify_gui_freeze(label, *, camera_checks=False, toolbar_play=False):
        from .fleet_gui_check import backend_snapshot, check_frozen
        def pump():
            gui.synchronize()
            gui.pump_ui()
        snapshot=lambda:backend_snapshot(env.unwrapped.sim,robots,objects)
        gui_checks[label]=check_frozen(snapshot,pump)
        gui_checks[label]['tick']=ticks
        if camera_checks:
            for mode in ('dog1','dog2','overview'):
                gui.request_camera(mode)
                selected=gui.take_camera_request()
                if selected!=mode:
                    raise RuntimeError('Camera callback request lost')
                def camera_pump():
                    select_camera(selected)
                    pump()
                gui_checks[label+'/'+mode]=check_frozen(snapshot,camera_pump,pumps=2)
        if toolbar_play:
            def toolbar_pump():
                if gui.state=='finished':
                    gui.request_toggle()
                timeline.play()
                pump()
            gui_checks[label+'/toolbar_play']=check_frozen(snapshot,toolbar_pump,pumps=2)
        print(f'FLEET_GUI_CHECK {label} passed tick={ticks}',flush=True)

    def service_gui():
        nonlocal published_gui_state
        while app.is_running():
            playing=gui.synchronize()
            selected=gui.take_camera_request()
            if selected is not None:
                select_camera(selected)
            if gui.state!=published_gui_state:
                published_gui_state=gui.state
                status=dict(tick=ticks,sim_time_s=ticks*env.unwrapped.step_dt,state=gui.state,
                    sim_playing=env.unwrapped.sim.is_playing(),timeline_playing=timeline.is_playing(),
                    phases=phases,orders=dispatcher.snapshot(),run_dir=str(output.resolve()))
                publish_status(status)
                print(f'FLEET_UI state={gui.state} tick={ticks}',flush=True)
                if not playing:
                    hud.text=f'{gui.state.upper()} | {ticks*env.unwrapped.step_dt:.1f}s\nPress START / PAUSE / RESUME to continue.'
            if playing:
                return
            gui.pump_ui()
            time.sleep(.02)
        raise RuntimeError('Application closed before mission completion')

    settled_payload_support={name:SettledPayloadSupport() for name in names} if balance_scope in ('settled','settledall') else {}
    from .yaw_brake import PreSupportYawBrake
    yaw_brakes = {name:PreSupportYawBrake() for name in names}
    print(f'FLEET_START shared_world=True num_envs=1 robots=2 output={output}',flush=True)
    publish_status(dict(tick=0,sim_time_s=0.,
        state='waiting_for_start' if args.visual and args.wait_for_start else 'running',
        phases=phases,orders=dispatcher.snapshot(),run_dir=str(output.resolve())))
    try:
        if gui_check_enabled:
            verify_gui_freeze('waiting',camera_checks=True,toolbar_play=True)
            gui.request_toggle()
        while ticks<args.fleet_ticks:
            if gui_check_enabled and ticks==120:
                gui.request_toggle()
                if gui.synchronize() or gui.state!='paused':
                    raise RuntimeError('GUI pause did not hold the mission')
                verify_gui_freeze('paused',camera_checks=True)
                gui.request_toggle()
            if app:
                service_gui()
            for name in names:
                if bridges[name].localization_model is not None:
                    bridges[name].localization_model.update(ticks, ticks*env.unwrapped.step_dt, bridges[name].true_pose())
            scheduler.advance(active=drop_guard.active_robots() if drop_guard else None)
            if scheduler.errors:
                raise RuntimeError(f'Mission coroutine failed: {scheduler.errors}')
            if scheduler.finished:
                break
            positions={name:(bridges[name].true_pose().x,bridges[name].true_pose().y) for name in names}
            commands={name:(torch.tensor([slew_velocity(previous_commands[name],
                bridges[name].command[0].tolist(),env.unwrapped.step_dt)],
                device=env.unwrapped.device,dtype=bridges[name].command.dtype)
                if velocity_slew_enabled else bridges[name].command.clone()) for name in names}
            if loaded_speed_limit:
                for name in names:
                    if arms[name].held:
                        commands[name][0]=commands[name].new_tensor(
                            limit_planar_speed(commands[name][0].tolist(),loaded_speed_limit))
            if support_exit_enabled:
                for name in names:
                    ramped=support_exit_ramps[name].update(commands[name][0].tolist(),
                        previous_commands[name],name in balances and balances[name].active,
                        env.unwrapped.step_dt)
                    commands[name][0]=commands[name].new_tensor(ramped)
            if waypoint_ramp_enabled:
                for name in names:
                    guard_record={}
                    def clearance(candidate):
                        start,end=nominal_stop_segment(bridges[name].pose(), candidate, env.unwrapped.step_dt)
                        clear=navigators[name].grid.clear_line(start,end)
                        guard_record.update(start=list(start),end=list(end),clear=clear)
                        return clear
                    filtered=waypoint_ramps[name].update(commands[name][0].tolist(),previous_commands[name],
                        navigators[name].target_revision,env.unwrapped.step_dt,clearance)
                    commands[name][0]=commands[name].new_tensor(filtered)
                    waypoint_ramps[name].telemetry['nominal_clearance']=guard_record
            if empty_return_limit:
                for name in names:
                    commands[name][0]=commands[name].new_tensor(return_speed_limit(
                        commands[name][0].tolist(),empty_return_limit,original_z1=z1_backend,
                        payload_held=arms[name].held,phase=phases[name]))
            quarantined = {name for name in names if drop_guard and drop_guard.owner is not None
                           and drop_guard.owner != name}
            for name in quarantined:
                commands[name].zero_()
            world_commands={}
            for name in names:
                pose=bridges[name].true_pose()
                vx,vy=commands[name][0,:2].tolist()
                world_commands[name]=(math.cos(pose.yaw)*vx-math.sin(pose.yaw)*vy,math.sin(pose.yaw)*vx+math.cos(pose.yaw)*vy)
            _,stopped=traffic_commands(positions,world_commands)
            traffic_stops+=len(stopped)
            # Use a copy: traffic veto must not erase the coroutine's desired
            # command while it is suspended inside an IK or navigation call.
            for name in stopped:
                commands[name].zero_()
            previous_commands={name:tuple(commands[name][0].tolist()) for name in names}
            primary_command=env.unwrapped.command_manager.get_term('base_velocity').command
            primary_command[:]=commands['dog1']
            observations=[gait_observation(robots[name],leg_ids[name],commands[name],last_actions[name]) for name in names]
            # Startup can hide velocity/order mistakes because many values are
            # zero. Recheck while moving against the official primary manager.
            if not abi_checked or ticks%100==0:
                official=env.get_observations()['policy']
                error=float((observations[0]-official).abs().max()) if official.shape==observations[0].shape else float('inf')
                abi_checks+=1
                abi_max_error=max(abi_max_error,error)
                if official.shape[-1]!=48 or not torch.allclose(observations[0],official,atol=1e-5):
                    raise RuntimeError(f'Fleet observation ABI differs from the official task: {official.shape}, max error={error}')
                if not abi_checked:
                    print('FLEET_ABI official primary observation matched (48 values, 12 leg actions); periodic moving checks enabled',flush=True)
                abi_checked=True
            if ticks % gait_inference_stride == 0:
                with torch.inference_mode():
                    inferred_actions=policy(TensorDict({'policy':torch.cat(observations,dim=0)},batch_size=[2]))
                    if return_policy is not None:
                        return_actions=return_policy(TensorDict({'policy':torch.cat(observations,dim=0)},batch_size=[2]))
            actions=inferred_actions
            actions=actions.clone()
            if return_policy is not None:
                from .mounted_z1_asset import Z1_HOME
                for index,name in enumerate(names):
                    eligible=return_policy_eligible(original_z1=z1_backend,payload_held=arms[name].held,
                        phase=phases[name],arm_state=arms[name].state,
                        measured_arm=_tensor(robots[name].data.joint_pos)[0,arms[name].arm_ids].cpu().tolist(),
                        home_arm=Z1_HOME,empty_transit=empty_transit_specialist,
                        resource_wait=resource_wait_specialist,repeat_pickup=repeat_pickup_specialist,
                        completed_returns=completed_home_returns[name])
                    weight=return_blends[name].update(eligible,env.unwrapped.step_dt)
                    stats=return_blend_stats[name]
                    stats['eligible_ticks']+=int(eligible);stats['active_ticks']+=int(weight>0.)
                    stats['full_specialist_ticks']+=int(weight==1.)
                    actions[index]=(1.-weight)*actions[index]+weight*return_actions[index]
            if env.clip_actions is not None:
                actions=actions.clamp(-env.clip_actions,env.clip_actions)
            # Keep the captured policy support targets: measured-foot residual
            # feedback remains experimental after destabilizing an idle dog.
            # PD support does not alter the floating root or parcel state.
            for index,name in enumerate(names):
                data=robots[name].data
                navigating=(name not in quarantined and phases[name] in
                    ('staging_receiving','withdrawing_receiving','approaching_receiving','leaving_receiving',
                     'transporting','returning','parallel_navigation'))
                guard_correction=None
                if leg_target_margin and navigating and not navigators[name].recovering:
                    from .leg_target_guard import project_leg_targets
                    default=_tensor(data.default_joint_pos)[0,leg_ids[name]]
                    scale=env.unwrapped.cfg.actions.joint_pos.scale
                    requested=default+scale*actions[index]
                    guarded=project_leg_targets(requested.cpu().numpy(),
                        _tensor(data.joint_pos_limits)[0,leg_ids[name]].cpu().numpy(),leg_target_margin)
                    guarded=torch.as_tensor(guarded,device=requested.device,dtype=requested.dtype)
                    guard_correction=guarded-requested
                    actions[index]=(guarded-default)/scale
                    corrected=int((guard_correction.abs()>1e-6).sum())
                    stats=leg_guard_stats[name]
                    stats['corrected_frames']+=int(corrected>0)
                    stats['max_target_correction_rad']=max(stats['max_target_correction_rad'],float(guard_correction.abs().max()))
                    stats['max_corrected_joints']=max(stats['max_corrected_joints'],corrected)
                leg_guard_stats[name]['last_correction_rad']=guard_correction.cpu().tolist() if guard_correction is not None else None
                if 40*gait_inference_stride<=ticks<60*gait_inference_stride and float(_tensor(data.root_pos_w)[0,2])>.31 and float(_tensor(data.projected_gravity_b)[0,2])<-.99:
                    standing_samples[name].append(actions[index].clone())
                if ticks>=60*gait_inference_stride and name not in standing_actions and len(standing_samples[name])>=10*gait_inference_stride:
                    standing_actions[name]=torch.stack(standing_samples[name]).mean(dim=0)
                    print(f'FLEET_STAND_CAPTURE robot={name} samples={len(standing_samples[name])}',flush=True)
                idle_expert=(return_blends[name].telemetry if return_policy is not None else {})
                idle_control=specialist_idle_control_requested(enabled=specialist_idle_control,
                    completed_returns=completed_home_returns[name],phase=phases[name],payload_held=arms[name].held,
                    arm_state=arms[name].state,expert_eligible=idle_expert.get('eligible'),
                    expert_weight=idle_expert.get('weight'),command=commands[name][0].cpu().tolist())
                specialist_idle_frames[name]+=int(idle_control)
                stationary=(stationary_mode in ('captured','balance') and name in standing_actions
                    and not navigating and float(commands[name].abs().max())<.05 and not idle_control)
                if stationary and stationary_mode == 'captured':
                    actions[index]=standing_actions[name]
                if name in balances:
                    settled_active=False
                    if balance_scope in ('settled','settledall'):
                        payload_gate=settled_payload_support[name]
                        settled_active=payload_gate.update(navigators[name].recovering and (arms[name].held or balance_scope=='settledall' or name in stow_braking),
                            _tensor(data.root_lin_vel_w)[0].cpu().tolist(),
                            _tensor(data.root_ang_vel_b)[0].cpu().tolist(),
                            float(_tensor(data.projected_gravity_b)[0,2]),
                            _tensor(balances[name].sensor.data.net_forces_w)[0,balances[name].contact_ids,2].cpu().tolist(),
                            env.unwrapped.step_dt)
                    support_active=support_scope_requested(navigators[name].recovering,
                        stationary and stationary_mode=='balance',balance_scope,settled_active)
                    brake_active=brake_support_gates[name].update(
                        brake_support_enabled and name in stow_braking and navigators[name].recovering
                            and phases[name]=='returning' and not arms[name].held,
                        _tensor(data.root_lin_vel_w)[0].cpu().tolist(),
                        _tensor(data.root_ang_vel_b)[0].cpu().tolist(),
                        float(_tensor(data.projected_gravity_b)[0,2]),
                        _tensor(balances[name].sensor.data.net_forces_w)[0,balances[name].contact_ids,2].cpu().tolist(),
                        env.unwrapped.step_dt)
                    support_active=support_active or brake_active
                    loaded_brake_active=loaded_brake_gates[name].update(
                        loaded_brake_enabled and loaded_brake_requested(navigators[name].recovering,
                            phases[name],arms[name].held,arms[name].state,commands[name][0].cpu().tolist(),
                            receiving_brake=receiving_brake_enabled),
                        _tensor(data.root_lin_vel_w)[0].cpu().tolist(),
                        _tensor(data.root_ang_vel_b)[0].cpu().tolist(),
                        float(_tensor(data.projected_gravity_b)[0,2]),
                        _tensor(balances[name].sensor.data.net_forces_w)[0,balances[name].contact_ids,2].cpu().tolist(),
                        env.unwrapped.step_dt)
                    support_active=support_active or loaded_brake_active
                    if idle_control:support_active=False
                    actions[index], support_effort = balances[name].apply(support_active,
                        actions[index], last_actions[name][0], standing_actions.get(name), env.unwrapped.step_dt,
                        workspace_eligible=workspace_scope_requested(navigators[name].recovering,arms[name].held,phases[name]),
                        stationary_payload=(z1_backend and stationary and arms[name].held and phases[name]=='placing'),
                        loaded_recovery=loaded_brake_requested(navigators[name].recovering,
                            phases[name],arms[name].held,arms[name].state,commands[name][0].cpu().tolist(),
                            receiving_brake=receiving_brake_enabled))
                    balances[name].telemetry['scope']=balance_scope
                    if specialist_idle_control:
                        balances[name].telemetry['specialist_idle_control']=dict(requested=idle_control,
                            support_requested=bool(support_active))
                    if brake_support_enabled:
                        balances[name].telemetry['empty_stow_brake_support']=brake_support_gates[name].telemetry.copy()
                    if loaded_brake_enabled:
                        balances[name].telemetry['loaded_brake_support']=loaded_brake_gates[name].telemetry.copy()
                    if balance_config.workspace_reference:
                        balances[name].telemetry['workspace_reference_scope']='empty_returning_only'
                    if yaw_brake_enabled:
                        brake_effort, brake_telemetry = yaw_brakes[name].apply(balances[name],
                            navigators[name].recovering and not support_active and (arms[name].held or balance_scope=='settledall'),
                            env.unwrapped.step_dt)
                        support_effort = support_effort + brake_effort
                        balances[name].telemetry['pre_support_yaw_brake'] = brake_telemetry
                    if balance_scope in ('settled','settledall'):
                        balances[name].telemetry['settled_payload_gate']=dict(active=payload_gate.active,
                            entry_eligible=payload_gate.eligible,stable_time_s=payload_gate.stable_time_s)
                    robots[name].set_joint_effort_target_index(target=support_effort.unsqueeze(0), joint_ids=leg_ids[name])
                standing_activations[name]+=int(stationary and not was_standing[name])
                was_standing[name]=stationary
            for name in names:
                arms[name].apply_targets()
            secondary=robots['dog2']
            target=_tensor(secondary.data.default_joint_pos)[:,leg_ids['dog2']]+env.unwrapped.cfg.actions.joint_pos.scale*actions[1:2]
            secondary.set_joint_position_target_index(target=target,joint_ids=leg_ids['dog2'])
            _,_,dones,_=env.step(actions[0:1])
            ticks+=1
            z1_contact_samples={name:arms[name]._read_contact() for name in names} if z1_backend else {}
            # Every control frame is recorded separately: the 5 Hz video/trace
            # can alias short gait dips and cannot explain recovery chatter.
            control_trace.write(json.dumps(dict(tick=ticks,sim_time=ticks*env.unwrapped.step_dt,
                robots={name:dict(height_m=float(_tensor(robot.data.root_pos_w)[0,2]),
                    active_order_code=current_codes[name],
                    navigation_pose_observation=(bridges[name].pose_observation().record()
                        if localization_profile == 'perturbed' else None),
                    root_position_w=_tensor(robot.data.root_pos_w)[0].cpu().tolist(),
                    root_quaternion_xyzw=_tensor(robot.data.root_quat_w)[0].cpu().tolist(),
                    gravity_z=float(_tensor(robot.data.projected_gravity_b)[0,2]),
                    root_velocity_w=_tensor(robot.data.root_lin_vel_w)[0].cpu().tolist(),
                    root_angular_velocity_b=_tensor(robot.data.root_ang_vel_b)[0].cpu().tolist(),
                    commanded_velocity=commands[name][0].cpu().tolist(),phase=phases[name],
                    support_exit_ramp=(support_exit_ramps[name].telemetry if support_exit_enabled else None),
                    waypoint_ramp=(waypoint_ramps[name].telemetry if waypoint_ramp_enabled else None),
                    recovering=navigators[name].recovering,
                    standing_pd=was_standing[name],
                    balance_recovery=balances[name].telemetry if name in balances else None,
                    foot_positions_w=(_tensor(robot.data.body_link_pose_w)[0,balances[name].feet,:3].cpu().tolist()
                        if name in balances else None),
                    foot_contact_normal_n=(_tensor(balances[name].sensor.data.net_forces_w)[0,balances[name].contact_ids,2].cpu().tolist()
                        if name in balances else None),
                    leg_position_rad=_tensor(robot.data.joint_pos)[0,leg_ids[name]].cpu().tolist(),
                    leg_commanded_position_rad=(_tensor(robot.data.default_joint_pos)[0,leg_ids[name]]
                        +env.unwrapped.cfg.actions.joint_pos.scale*actions[names.index(name)]).cpu().tolist(),
                    leg_target_guard_correction_rad=leg_guard_stats[name]['last_correction_rad'],
                    leg_velocity_radps=_tensor(robot.data.joint_vel)[0,leg_ids[name]].cpu().tolist(),
                    leg_computed_torque_nm=_tensor(robot.data.computed_torque)[0,leg_ids[name]].cpu().tolist(),
                    leg_applied_torque_nm=_tensor(robot.data.applied_torque)[0,leg_ids[name]].cpu().tolist(),
                    arm_position_rad=_tensor(robot.data.joint_pos)[0,arms[name].arm_ids].cpu().tolist(),
                    original_z1=(dict(state=arms[name].state,
                        close_completion=arms[name].close_completion,
                        return_gait_specialist=return_blends[name].telemetry if return_policy is not None else None,
                        active_payload_code=current_codes[name],active_payload_path=arms[name].box_path,
                        arm_target_rad=arms[name]._arm_target.cpu().tolist(),
                        arm_velocity_radps=_tensor(robot.data.joint_vel)[0,arms[name].arm_ids].cpu().tolist(),
                        arm_applied_torque_nm=_tensor(robot.data.applied_torque)[0,arms[name].arm_ids].cpu().tolist(),
                        nominal_world_reference=(arms[name]._nominal_world_reference.tolist()
                            if arms[name]._nominal_world_reference is not None else None),
                        gripper_position_rad=arms[name]._joint_position(arms[name].gripper_ids).cpu().tolist(),
                        gripper_target_rad=arms[name]._gripper_target,
                        fixed_jaw_admittance=asdict(arms[name]._normal_controller),
                        jaw_normal_n=z1_contact_samples[name]['normal_magnitudes'],
                        jaw_total_n=z1_contact_samples[name]['total_magnitudes'],
                        jaw_normal_vectors_w=[v.cpu().tolist() for v in z1_contact_samples[name]['normal_vectors']],
                        jaw_friction_vectors_w=[v.cpu().tolist() for v in z1_contact_samples[name]['friction_vectors']],
                        contact_wrenches=arms[name].contact_wrench_diagnostic(z1_contact_samples[name]),
                        payload_sleep=arms[name].payload_sleep_diagnostic(),
                        active_payload_position_w=_tensor(arms[name].box.data.root_pos_w)[0].cpu().tolist(),
                        active_payload_quaternion_xyzw=_tensor(arms[name].box.data.root_quat_w)[0].cpu().tolist(),
                        active_payload_velocity_w=_tensor(arms[name].box.data.root_lin_vel_w)[0].cpu().tolist())
                        if z1_backend else None),
                    tool_position_w=arms[name]._tool_pose()[0].cpu().tolist(),
                    tool_quaternion_xyzw=arms[name]._tool_pose()[1].cpu().tolist(),
                    recovery_velocity_body=getattr(bridges[name],'recovery_velocity_body',(0.,0.,0.)) if navigators[name].recovering else (0.,0.,0.),
                    recovery_requested_velocity_body=getattr(bridges[name],'recovery_requested_velocity_body',(0.,0.,0.)),
                    payload_held=arms[name].held,
                    drop_quarantined=name in quarantined,
                    traffic_veto=name in stopped)
                    for name,robot in robots.items()}))+'\n')
            for index,name in enumerate(names):
                last_actions[name]=actions[index:index+1].clone()
                if z1_backend:
                    # Navigation after release does not resume arm._step()
                    # and sync() intentionally skips unheld parcels. Guard
                    # the same freshly recorded force sample in every phase.
                    arms[name]._hard_contact_guard(z1_contact_samples[name])
                data=robots[name].data
                if float(_tensor(data.projected_gravity_b)[0,2])>-.85 or float(_tensor(data.root_pos_w)[0,2])<.22:
                    raise RuntimeError(f'{name} stance unsafe; stopping fleet without a world reset')
                try:
                    arms[name].sync()
                except PayloadLost as error:
                    if drop_guard is None:
                        raise
                    drop_guard.detect(name, current_codes[name], phases[name], error, ticks,
                                      copy.deepcopy(arms[name].loss_events[-1]))
                    # Deliver this typed exception through the suspended
                    # robot's stack on its next tick, never the peer's stack.
                    bridges[name].error=error
                    bridges[name]._zero_command()
            if bool(dones[0]):
                raise RuntimeError('Unexpected primary episode termination')
            gate.step(arms['dog1'],output)
            distance=math.dist(positions['dog1'],positions['dog2'])
            minimum_distance=min(minimum_distance,distance)
            active=[phases[name] not in ('starting','waiting_receiving','complete','smoke_complete','failed') for name in names]
            overlap_ticks+=int(all(active))
            navigation_overlap_ticks+=int(all(math.hypot(*commands[name][0,:2].tolist())>.05 for name in names))
            if ticks%telemetry_stride==0:
                # Preserve an explicitly timestamped snapshot for user-close:
                # Kit destroys asset views before result finalization runs.
                last_valid_final_state=final_state_snapshot()
                last_valid_state_tick=ticks
                finger_forces={}
                for index,name in enumerate(names):
                    code=current_codes[name]
                    if code is None:
                        finger_forces[name]=None
                        continue
                    parcel_index=codes.index(code)
                    prefix='' if index==0 else 'dog2_'
                    finger_forces[name]={side:float(_tensor(scene.sensors[prefix+f'{side}_grip_contact'].data.force_matrix_w)[0,:,parcel_index].norm(dim=-1).sum()) for side in ('left','right')}
                record=dict(tick=ticks,sim_time=ticks*env.unwrapped.step_dt,phases=dict(phases),codes=dict(current_codes),
                    roots={name:_tensor(robots[name].data.root_pos_w)[0].cpu().tolist() for name in names},
                    gravity={name:_tensor(robots[name].data.projected_gravity_b)[0].cpu().tolist() for name in names},
                    measured_velocity_b={name:_tensor(robots[name].data.root_lin_vel_b)[0].cpu().tolist() for name in names},
                    leg_actions={name:actions[index].cpu().tolist() for index,name in enumerate(names)},
                    held={name:arms[name].held for name in names},traffic_stopped=sorted(stopped),
                    parcels={code:_tensor(obj.data.root_pos_w)[0].cpu().tolist() for code,obj in objects.items()},
                    parcel_quaternions_xyzw={code:_tensor(obj.data.root_quat_w)[0].cpu().tolist() for code,obj in objects.items()},
                    finger_contact_force_n=finger_forces,
                    requested_velocities={name:bridges[name].command[0].cpu().tolist() for name in names},
                    commanded_velocities={name:commands[name][0].cpu().tolist() for name in names},dispatcher=dispatcher.snapshot())
                if detail_writer is not None:
                    # A second camera observes the exact same physics tick.
                    # Camera render/update never advances the simulation.
                    priorities={'drop_injection':6,'recovering_dropped_parcel':6,
                                'grasping':5,'placing':5,'stowing_after_departure':4,
                                'leaving_receiving':3,'transporting':3,'returning':2,
                                'approaching_receiving':1}
                    candidate=max(names,key=lambda name:priorities.get(phases[name],0))
                    if ticks-detail_switch_tick>=100 and candidate!=detail_focus:
                        detail_focus=candidate
                        detail_switch_tick=ticks
                    p=record['roots'][detail_focus]
                    eye=[p[0]+1.5,p[1]-2.,1.65]
                    look=[p[0]+.25,p[1],.38]
                    detail_image=[]
                    def capture_detail():
                        detail_camera.set_world_poses_from_view(torch.tensor([eye],device=env.unwrapped.device),
                                                               torch.tensor([look],device=env.unwrapped.device))
                        env.unwrapped.sim.render()
                        detail_camera.update(env.unwrapped.step_dt,force_recompute=True)
                        detail_image.append(detail_camera.data.output['rgb'][0,...,:3].cpu().numpy().copy())
                    check_key=detail_focus+'/'+phases[detail_focus]
                    if check_key not in detail_freeze_checks:
                        from .fleet_gui_check import backend_snapshot, check_frozen
                        detail_freeze_checks[check_key]=check_frozen(
                            lambda:backend_snapshot(env.unwrapped.sim,robots,objects),capture_detail,pumps=1)
                    else:
                        capture_detail()
                    detail_frame=detail_image[0]
                    import cv2
                    cv2.rectangle(detail_frame,(0,0),(detail_frame.shape[1],60),(25,25,25),-1)
                    cv2.putText(detail_frame,f'{detail_focus} | {phases[detail_focus]} | {ticks*env.unwrapped.step_dt:.1f}s',
                                (12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(100,220,255),1)
                    cv2.putText(detail_frame,'Rendered follow camera | 1x simulation time',
                                (12,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    detail_writer.append_data(detail_frame)
                    record['detail_camera']=dict(focus=detail_focus,eye=eye,look_at=look,physics_tick=ticks)
                trace.write(json.dumps(record)+'\n')
                if hud:
                    hud.text=(f'{ticks*env.unwrapped.step_dt:.1f}s | {progress_label()}\n'
                        +'\n'.join(f'{name}: {phases[name]} ({current_codes[name] or "-"})' for name in names))
                    if camera_mode!='overview':
                        p=_tensor(robots[camera_mode].data.root_pos_w)[0].cpu().tolist()
                        env.unwrapped.sim.set_camera_view((p[0]+2.4,p[1]-2.8,2.1),(p[0]+.2,p[1],.4))
                frame=camera.data.output['rgb'][0,...,:3].cpu().numpy().copy()
                import cv2
                cv2.rectangle(frame,(0,0),(frame.shape[1],88),(25,25,25),-1)
                cv2.putText(frame,f'SHARED WAREHOUSE | {ticks*env.unwrapped.step_dt:.1f}s | {progress_label()}',
                    (12,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(100,220,255),1)
                for line,name in enumerate(names):
                    cv2.putText(frame,f'{name} | {phases[name]} | {current_codes[name] or "-"}',(12,48+line*26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                writer.append_data(frame)
                if ticks==10 or ticks%500==0:
                    imageio.imwrite(output/f'fleet_{ticks:05d}.png',frame)
            if ticks%100==0:
                status=dict(tick=ticks,sim_time_s=ticks*env.unwrapped.step_dt,
                    completed_route_cycles=dict(completed_cycles),requested_endurance_cycles=endurance_cycles,
                    state=gui.state if gui else 'running',phases=phases,orders=dispatcher.snapshot(),run_dir=str(output.resolve()))
                publish_status(status)
                print(f'FLEET_PROGRESS tick={ticks} phases={phases} delivered={dispatcher.delivered} separation={distance:.2f}',flush=True)
                if feed_audit is not None:
                    feed_audit.write_json(output/'streaming_feed.json')
    except Exception as error:
        failure=str(error)
        rejected_problem=getattr(error,'support_allocation_diagnostics',None)
        if rejected_problem is not None:
            rejected_problem=dict(rejected_problem,robot=name,last_completed_tick=ticks,
                failed_pre_step_tick=ticks+1,phase=phases.get(name))
            (output/'support_allocation_failure.json').write_text(
                json.dumps(rejected_problem,indent=2,allow_nan=False),encoding='utf-8')
        print(f'FLEET_FAILED {failure}',flush=True)
    finally:
        trace.close()
        control_trace.close()
        writer.close()
        if detail_writer is not None:
            detail_writer.close()
        used_cached_final_state=bool(app and not app.is_running())
        if not used_cached_final_state:
            try:
                last_valid_final_state=final_state_snapshot()
                last_valid_state_tick=ticks
            except ReferenceError:
                used_cached_final_state=True
        final_roots=last_valid_final_state['final_roots']
        final_home_errors={name:math.dist(final_roots[name][:2],(0.,-1. if i==0 else 1.)) for i,name in enumerate(names)}
        complete=scheduler.finished and not scheduler.errors and not failure and all(returned.values())
        if complete and feed_audit is not None and not feed_audit.complete:
            complete=False
            failure='Fleet dispatch finished but finite streaming feed audit is incomplete'
        if used_cached_final_state:
            complete=False
            failure=failure or 'Application closed; only last valid physics snapshot remains'
        if complete and any(error>.21 for error in final_home_errors.values()):
            complete=False
            failure='Robot drifted outside the return area before the fleet finished'
        result=dict(state='complete' if complete else 'incomplete',mode=('shared_world_navigation_endurance' if endurance_cycles
                    else 'shared_world_navigation_smoke') if args.fleet_smoke else 'shared_world_contact_logistics',
            requested_endurance_cycles=endurance_cycles,completed_route_cycles=completed_cycles,
            launch_token=os.environ.get('GO2_MISSION_RUN_TOKEN'),
            seed=getattr(args,'seed',None),task=getattr(args,'task',None),
            gait_policy=policy_identity,
            empty_return_gait_specialist=dict(enabled=return_policy is not None,checkpoint=return_policy_identity,
                scope=('empty home-arm receiving approach/staging/waiting/withdrawal, navigation smoke and return'
                    if empty_transit_specialist else
                    'returning, empty parcel, stow_empty_arm and all six measured arm joints within .06 rad of home'),
                empty_transit_enabled=empty_transit_specialist,
                resource_wait_enabled=resource_wait_specialist,
                repeat_pickup_enabled=repeat_pickup_specialist,
                idle_control_enabled=specialist_idle_control,idle_control_frames=specialist_idle_frames,
                home_return_events=home_return_events,
                blend_seconds=.5,stats=return_blend_stats),
            empty_stow_braking=dict(enabled=settle_stow_enabled,events=stow_braking_events,
                measured_support_entry_enabled=brake_support_enabled,
                measured_support_entry_scope='empty returning after departure, during explicit stow braking only',
                measured_support_entry_config=(dict(planar_speed_below_mps=.14,vertical_speed_below_mps=.20,
                    angular_speed_below_radps=.50,gravity_z_below=-.99,minimum_contacts=3,
                    contact_normal_above_n=1.,entry_stable_time_s=.04) if brake_support_enabled else None),
                scope='original Z1 after shelf departure, before empty folding; measured settling, original three-second budget'),
            arm_model='official_unitree_z1' if z1_backend else 'lightweight_panda_surrogate',
            placement_wrench_qp=dict(enabled=z1_backend and os.environ.get('GO2_Z1_PLACEMENT_WRENCH_QP')=='1',
                scope='original Z1, stationary balance, held parcel, placing phase only',
                base_allocator=os.environ.get('GO2_BALANCE_ALLOCATOR','sequential'),
                feedforward_limit_nm=18.,actuator_limits_unchanged=True),
            airborne_touchdown=dict(enabled=os.environ.get('GO2_BALANCE_AIRBORNE_TOUCHDOWN')=='1',
                scope='zero-command original Z1 loaded transport recovery; airborne legs with downward foot error',
                position_slew_motor_and_recovery_limits_unchanged=True),
            loaded_recovery_wrench_qp=dict(enabled=loaded_recovery_qp_enabled,
                receiving_exit_enabled=receiving_brake_enabled,
                total_motor_enabled=total_motor_qp_enabled,total_motor_margin_nm=.1,
                joint_target_enabled=joint_motor_qp_enabled,
                scope='original Z1, transport or opt-in receiving exit, navigation recovery, held parcel, carried arm, zero command',
                base_allocator=os.environ.get('GO2_BALANCE_ALLOCATOR','sequential'),
                feedforward_limit_nm=18.,actuator_limits_unchanged=True),
            loaded_brake_support=dict(enabled=loaded_brake_enabled,
                receiving_exit_enabled=receiving_brake_enabled,
                attitude_hysteresis_enabled=loaded_support_hysteresis,
                retention_gravity_z_below=-.97 if loaded_support_hysteresis else -.99,
                scope='recovering, transport or opt-in receiving exit, held parcel, carried arm state, zero velocity command',
                planar_speed_below_mps=.14,vertical_speed_below_mps=.20,
                angular_speed_below_radps=.50,gravity_z_below=-.99,
                minimum_contacts=3,contact_normal_above_n=1.,entry_stable_time_s=.04),
            contact_wrench_diagnostic=dict(enabled=z1_backend and os.environ.get('GO2_Z1_CONTACT_WRENCH')=='1',
                phases=['closing_fingers','lift','failed'],
                scope='read-only active parcel jaw points and friction patches; moments about measured parcel center',
                source='PhysX native filtered contact buffers; no synthesized contact points'),
            arm_runtime_integrity={name:arm.runtime_integrity for name,arm in arms.items()} if z1_backend else None,
            gait_inference_period_s=env.unwrapped.step_dt*gait_inference_stride,
            control_trace_period_s=env.unwrapped.step_dt,
            navigation_budget=navigation_budget,
            physics_timing=dict(physics_dt_s=env.unwrapped.cfg.sim.dt,
                decimation=env.unwrapped.cfg.decimation,control_period_s=env.unwrapped.step_dt,
                render_interval=env.unwrapped.cfg.sim.render_interval,
                external_forces_every_iteration=env.unwrapped.cfg.sim.physics.enable_external_forces_every_iteration),
            detail_video=dict(enabled=detail_enabled,path='fleet_detail.mp4' if detail_enabled else None,
                              camera_source='rendered RGB follow camera',physics_advanced_for_camera=False,
                              container='fragmented_mp4',fragment_duration_us=2000000,
                              crash_recovery_scope='completed fragments only; current fragment may be lost',
                              sampled_render_freeze_checks=detail_freeze_checks),
            video_timing=dict(fps=recording_fps,sample_stride_ticks=telemetry_stride,
                              simulation_seconds_per_frame=telemetry_stride*env.unwrapped.step_dt),
            shared_world=True,num_envs=1,robot_count=2,world_reset_between_orders=False,
            configured_codes=codes,dispatcher=dispatcher.snapshot(),returned=returned,phases=phases,
            streaming_feed_enabled=streaming_feed_enabled,
            streaming_feed=feed_audit.snapshot() if feed_audit is not None else None,
            control_ticks=ticks,sim_time_s=ticks*env.unwrapped.step_dt,
            final_state_snapshot_tick=last_valid_state_tick,final_state_snapshot_cached=used_cached_final_state,
            concurrent_active_time_s=overlap_ticks*env.unwrapped.step_dt,
            concurrent_navigation_time_s=navigation_overlap_ticks*env.unwrapped.step_dt,
            minimum_trunk_separation_m=minimum_distance if math.isfinite(minimum_distance) else None,
            traffic_command_vetoes=traffic_stops,failure=failure or ('control tick budget exhausted' if not complete else ''),
            coroutine_errors=scheduler.errors,observation_abi_checked=abi_checked,
            observation_abi_checks=abi_checks,observation_abi_max_error=abi_max_error,
            grasp_model='contact',localization=('synthetic_perturbed_simulator_pose' if localization_profile == 'perturbed'
                else 'simulator_ground_truth'),traffic_telemetry='simulator_ground_truth',
            localization_observation={name:bridges[name].localization_model.metadata()
                for name in names if bridges[name].localization_model is not None},
            localization_reference_origin_w=_tensor(scene.env_origins)[0].cpu().tolist(),
            manipulation_controller=('original Z1 URDF world-target IK, measured contact admittance and gravity feedforward'
                if z1_backend else 'bounded damped local IK (BVLS); five Cartesian lift segments; optional inward carry path recorded in manipulation_config; no joint-centering/yaw override'),
            placement_docks={code:vars(Pose2D(shelf_for_code(code).surface[0]-.45,shelf_for_code(code).surface[1])
                if z1_backend else placement_standoff(shelf_for_code(code))) for code in codes},
            placement_strategy='depart from deck before stowing the empty arm; recheck parcel after departure and stow',
            loaded_reservation_braking=dict(enabled=os.environ.get('GO2_FLEET_SETTLE_RESERVATION')=='1',
                scope='original Z1 held parcel before first blocked reservation; measured stop before waiting support',
                minimum_duration_s=.3,recovery_budget_unchanged=True),
            manipulation_config={name:asdict(arms[name].config) for name in names},
            kinematic_plans={name:arms[name].kinematic_plan for name in names},
            kinematic_plan_history={name:arms[name].kinematic_plan_history for name in names},
            kinematic_plan_failures={name:arms[name].kinematic_plan_failures for name in names},
            stationary_controller={'policy':'zero-command feedback policy during idle/manipulation',
                'captured':'captured standing action + joint PD during idle/manipulation',
                'balance':'contact-force support + bounded leg IK from arrival targets during idle/manipulation'}[stationary_mode],
            stationary_mode=stationary_mode,
            transit_reservation=dict(enabled=reserve_transit,
                scope='receiving exit through precise home return and empty-arm preparation',
                events=transit_reservation_events),
            segment_reservation=dict(enabled=segment_transit,events=segment_reservation_events,
                short_spine_enabled=short_spine,spine_x_m=spine_x,station_release_clearance_m=1.45,
                scope=('shared approach and exclusive shelf rows retained; western spine released only at a measured row dock and reacquired before departure'
                    if short_spine else 'shared approach held outbound to southern staging or inbound from western return staging to home; spine held through placement and departure')),
            receiving_staging=dict(enabled=pipeline_receiving,events=receiving_staging_events,
                compact_layout_enabled=compact_receiving,
                compact_release_clearance_m=1.45 if compact_receiving else None,
                reservation_events=receiving_reservation_events,
                approach_alignment_x_m=receiving_approach_x,
                approach_waypoints=[vars(point) for point in receiving_approach],
                approach_staged_only=receiving_approach_staged_only,
                approach_selection_events=receiving_approach_events,
                home_entry_guard=segment_transit and plan_peer_clearance,
                home_entry_wait_events=receiving_entry_wait_events,
                arrival_settle_enabled=receiving_arrival_settle,
                arrival_settle_events=receiving_arrival_events,
                scope='peer stages in the marked buffer after the owner reaches the station, overlapping grasp; receiving stays exclusive until the owner exits; surplus standby dogs withdraw home'),
            peer_clearance_planning=dict(enabled=plan_peer_clearance,radius_m=1.50,
                traffic_guard_margin_m=1.25,source='simulator_ground_truth_peer_localization',
                peer_scope='waiting_receiving or complete within 0.21 m of configured home',home_tolerance_m=.21,
                scope='held-payload transporting phase; refreshed at each navigation scan; separate from sensor occupancy'),
            recovery_controller=recovery_controller,
            balance_scope=balance_scope,
            waypoint_ramp=dict(enabled=waypoint_ramp_enabled,linear_acceleration_mps2=.8,yaw_acceleration_radps2=2.,
                navigation_zero_gap_s=WaypointCommandRamp.zero_gap_s,
                scope='moving waypoint changes; exact stops immediate; startup unchanged',
                clearance='map check of nominal planar stopping segment; not measured braking guarantee'),
            navigation_smoke_goal_x=(-3. if os.environ.get('GO2_FLEET_SMOKE_OBSTACLE_ROUTE')=='1' else 1.5) if args.fleet_smoke else None,
            pre_support_yaw_brake=dict(enabled=yaw_brake_enabled, gain_nm_per_radps=3.,
                moment_cap_nm=.8, motor_increment_cap_nm=1., entry_blend_s=.1,
                source='simulator_contact_state_and_jacobians', policy_position_targets_unchanged=True),
            navigation_recovery_controller=({'stationary':'policy_after_balance_fade',
                'settled':'policy_then_measured_settled_payload_support',
                'settledall':'policy_then_measured_settled_support_any_payload'}.get(balance_scope,recovery_controller)),
            settled_payload_support_config=(dict(entry_stable_time_s=.10,linear_speed_below_mps=.06,
                angular_speed_below_radps=.15,gravity_z_below=-.99,minimum_contacts=3,
                contact_normal_above_n=1.,latch='until recovery exit, fewer than three contacts, or attitude gate lost',
                recovery_timeout_unchanged=True,payload_required=balance_scope=='settled') if balance_scope in ('settled','settledall') else None),
            settle_transitions=os.environ.get('GO2_FLEET_SETTLE_TRANSITIONS') == '1',
            balance_controller_config=asdict(next(iter(balances.values())).config) if balances else None,
            balance_controller_activations={name:item.activations for name,item in balances.items()},
            velocity_slew=dict(enabled=velocity_slew_enabled,linear_acceleration_mps2=.8,
                yaw_acceleration_radps2=2.,zero_and_traffic_veto='immediate zero command'),
            loaded_planar_speed_limit_mps=loaded_speed_limit,
            empty_return_planar_speed_limit_mps=empty_return_limit,
            empty_return_speed_limit_scope='original Z1, empty returning phase only; command norm, not measured speed guarantee; before traffic veto',
            placement_stop_precision=os.environ.get('GO2_FLEET_PLACEMENT_PRECISION')=='1',
            loaded_speed_limit_scope='held parcel only; command norm, not a measured speed guarantee',
            lateral_tracking='forward cruise with all-yaw swept clearance; original fixed-yaw docking' if os.environ.get('GO2_FLEET_TRAVEL_HEADING')=='1' else 'proportional correction in transit; .22 m/s minimum only inside .12 m docking radius; yaw approximately zero',
            minimum_positive_forward_command_mps=forward_floor,
            leg_target_guard=dict(margin_rad=leg_target_margin,scope='RL navigation before recovery blending; target projection only',robots=leg_guard_stats),
            stance_controller_activations=standing_activations,
            recovery_forward_command_mps=recovery_forward,
            recovery_forward_scope='experimental empty-gripper only; held parcels use zero command',
            source_manifest_file='source_manifest.json',
            support_exit_ramp=dict(enabled=support_exit_enabled,duration_s=SupportExitRamp.duration_s,
                linear_acceleration_mps2=SupportExitRamp.linear_acceleration,
                yaw_acceleration_radps2=SupportExitRamp.yaw_acceleration),
            recovery_safety=dict(scan_period_control_steps=navigators['dog1']._ticks(5),footprint='navigation low/high footprints in body frame',
                commanded_travel_cap_m=.15,drift_margin_m=.10,blocked_scan='latched zero until event ends',
                physical_displacement_guaranteed=False),
            standing_targets_captured=sorted(standing_actions),
            initial_arm_postures_verified=True,
            secondary_manager_events=['physics_material','reset_robot_joints','reset_base'],
            warehouse_asset=getattr(args,'indoor_warehouse',False),run_dir=str(output.resolve()),
            navigation={name:navigators[name].finish() for name in names},
            delivery_evidence=delivery_evidence,
            manipulation_attempts=manipulation_attempts,
            drop_recovery=dict(enabled=recover_drops,injection_requested=inject_drop,
                contact_righting_enabled=os.environ.get('GO2_FLEET_CONTACT_RIGHTING')=='1',
                events=drop_guard.events if drop_guard else [],
                sensor_events={name:handler.events for name,handler in drop_handlers.items()},
                loss_events={name:arm.loss_events for name,arm in arms.items()},
                active_owner=drop_guard.owner if drop_guard else None,
                scope='transport only; exclusive recovery; external per-robot RGB-D; active side-label fallback for isolated face-resting parcels; grasp must fit existing aperture; two losses per order'),
            final_roots=final_roots,final_home_error_m=final_home_errors,
            final_parcels=last_valid_final_state['final_parcels'],
            final_parcel_quaternions_xyzw=last_valid_final_state['final_parcel_quaternions_xyzw'],
            parcel_dimensions_m={code:list(obj.cfg.spawn.size) for code,obj in objects.items()},
            final_parcel_velocities=last_valid_final_state['final_parcel_velocities'],
            dispatcher_events=dispatcher.events,
            placement={name:arms[name].placement_metrics for name in names})
        publish_status(dict(state=result['state'],failure=result['failure'],tick=ticks,
            sim_time_s=result['sim_time_s'],phases=phases,orders=dispatcher.snapshot(),
            run_dir=str(output.resolve())))
        result['status_publish_denials']=status_publish_denials
        if feed_audit is not None:
            feed_audit.write_json(output/'streaming_feed.json')
        (output/'fleet_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        from .run_isolation import publish_json_pointer
        publish_json_pointer(Path('artifacts/fleet/latest_result.json'),result)
        print(f'FLEET_RESULT state={result["state"]} result={output/"fleet_result.json"}',flush=True)
    if app and app.is_running():
        gui.finish()
        hud.text=f'{result["state"].upper()} | {progress_label()}\n'+(result['failure'] or 'Both dogs returned. Evidence saved.')
        if gui_check_enabled:
            verify_gui_freeze('finished',camera_checks=True,toolbar_play=True)
            report=dict(passed=complete and all(v['passed'] for v in gui_checks.values()),
                method='native button callback requests and timeline Play; backend PhysX snapshots',
                physical_mouse_clicks_tested=False,checks=gui_checks,mission_state=result['state'])
            (output/'gui_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            if not report['passed']:
                raise RuntimeError('Native GUI check failed')
            return result
        while app.is_running():
            # Finished state ignores both panel resume and toolbar Play. Kit
            # still renders camera changes, with its own physics stepping off.
            gui.synchronize()
            selected=gui.take_camera_request()
            if selected is not None:
                select_camera(selected)
            gui.pump_ui()
            time.sleep(.02)
    if not complete:
        raise RuntimeError(f'Fleet run incomplete: {result["failure"]}')
    return result
