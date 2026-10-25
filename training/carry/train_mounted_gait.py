"""Fine-tune the official 48-observation/12-action gait on the mounted robot.

Run using run_gait_adaptation.ps1. Training episodes reset normally; the
separate deterministic evaluation disables resets and latches all unsafe
frames. There are no parcel bindings, base forces or modified motor limits.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_pose_bank(source):
    from go2_logistics.mounted_robot_asset import ARM_HOME
    poses = [list(ARM_HOME)]
    result = json.loads(source.read_text(encoding='utf-8'))
    for robot in sorted(result['kinematic_plan_history']):
        plans = [p for p in result['kinematic_plan_history'][robot] if p['operation'] == 'grasp']
        if plans:
            poses.append(plans[-1]['joints'][-1])
    return poses


def main():
    import gymnasium as gym
    import torch
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
    from isaaclab.envs import mdp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config
    from rsl_rl.runners import OnPolicyRunner
    from go2_logistics.mounted_robot_asset import configure_robot, LEG_JOINT_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--eval-seconds', type=float, default=40.)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/go2_official_checkpoint.pt')
    parser.add_argument('--pose-source', type=Path, default=ROOT/'artifacts/fleet/run_20261019_171441_849946/fleet_result.json')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--arm-model',choices=('panda_surrogate','z1'),default='panda_surrogate')
    parser.add_argument('--z1-pose-trace',type=Path,default=None,
                        help='Explicit extended-arm locomotion adaptation from measured contact-run targets; no payload')
    parser.add_argument('--command-trace', type=Path, action='append', default=[],
                        help='Optional 50/200 Hz warehouse control trace; resampled to 50 Hz, half the environments replay commands')
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 8 or args.iterations < 1 or args.eval_seconds < 5:
        parser.error('Use at least 8 environments, 1 iteration and 5 evaluation seconds')
    if args.z1_pose_trace and args.arm_model!='z1':
        parser.error('--z1-pose-trace requires --arm-model z1')
    args.headless, args.enable_cameras = True, False
    args.device = args.device or 'cuda:0'
    sys.argv = [sys.argv[0]]
    output = ROOT/'artifacts/gait_adaptation'/datetime.now().strftime('run_%Y%m%d_%H%M%S')
    output.mkdir(parents=True)
    os.environ['WARP_CACHE_PATH'] = str(ROOT/'artifacts/warp_cache')
    if args.arm_model=='z1':
        from go2_logistics.mounted_z1_asset import configure_z1_robot,build_mounted_z1_asset,Z1_HOME,Z1_OPEN
        from go2_logistics.z1_asset import ARM_JOINT_NAMES as arm_joint_names, GRIPPER_JOINT_NAME
        imported=json.loads((ROOT/'artifacts/real_arm_z1/usd/import_validation.json').read_text(encoding='utf-8'))
        if not imported.get('passed') or imported.get('fixed_base'):
            raise RuntimeError('Z1 training requires a validated floating-base official import')
        poses=[list(Z1_HOME),[0.,1.6,-2.4,0.,.8,0.]]
        if args.z1_pose_trace:
            import numpy as np
            from go2_logistics.gait_adaptation import measured_z1_pose_bank
            from go2_logistics.z1_kinematics import Z1Kinematics
            from go2_logistics.z1_asset import DEFAULT_OUTPUT
            measured_poses=measured_z1_pose_bank(args.z1_pose_trace)
            kin=Z1Kinematics.from_urdf(DEFAULT_OUTPUT)
            if any(not ((kin.limits[:,0]<=pose).all() and (pose<=kin.limits[:,1]).all()) for pose in map(np.asarray,measured_poses)):
                raise ValueError('Recorded arm target exceeds original Z1 limits')
            poses+=measured_poses
        finger_joint_names=(GRIPPER_JOINT_NAME,)
        finger_open=Z1_OPEN
    else:
        from go2_logistics.mounted_robot_asset import ARM_JOINT_NAMES as arm_joint_names,FINGER_JOINT_NAMES,FINGER_OPEN
        poses = arm_pose_bank(args.pose_source)
        finger_joint_names=FINGER_JOINT_NAMES
        finger_open=FINGER_OPEN
    replay_sequences, replay_sources = [], []
    if args.command_trace:
        from go2_logistics.gait_adaptation import load_warehouse_commands
        replay_sequences, replay_sources = load_warehouse_commands(args.command_trace)
    from go2_logistics.gait_adaptation import training_layout
    layout = training_layout(args.num_envs, len(poses), len(replay_sequences))
    cfg, agent = resolve_task_config('Isaac-Velocity-Flat-Unitree-Go2-v0', 'rsl_rl_cfg_entry_point')
    asset = output/'go2_z1.usda' if args.arm_model=='z1' else ROOT/'artifacts/go2_ground_panda.usda'
    if args.arm_model=='z1':
        configure_z1_robot(cfg,asset)
    else:
        configure_robot(cfg, asset)
    cfg.scene.num_envs, cfg.scene.env_spacing = args.num_envs, 6.
    cfg.sim.device = args.device
    cfg.sim.render_interval = 200
    cfg.seed = args.seed
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.
    cfg.commands.base_velocity.rel_standing_envs = .20
    cfg.commands.base_velocity.resampling_time_range = (3., 6.)
    cfg.commands.base_velocity.ranges.lin_vel_x = (-.5, .5)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-.35, .35)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-.15, .15)
    cfg.commands.base_velocity.debug_vis = False
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.reset_base.params['pose_range'] = {}
    cfg.events.reset_base.params['velocity_range'] = {}
    cfg.events.reset_robot_joints.params['position_range'] = (1., 1.)
    cfg.events.reset_robot_joints.params['velocity_range'] = (0., 0.)
    cfg.observations.policy.enable_corruption = False
    # Penalize only locomotion joints; the held arm has a different controller.
    leg_cfg = SceneEntityCfg('robot', joint_names=list(LEG_JOINT_NAMES), preserve_order=True)
    for term in ('dof_torques_l2', 'dof_acc_l2', 'dof_pos_limits'):
        getattr(cfg.rewards, term).params['asset_cfg'] = leg_cfg
    cfg.rewards.dof_pos_limits.weight = -2.
    cfg.rewards.warehouse_height = RewardTermCfg(func=mdp.base_height_l2, weight=-40., params={'target_height': .33})
    if args.z1_pose_trace:
        # Match the measured contact probe's arm drives. Keep leg motors and
        # original arm effort/rate limits; only change training objectives.
        cfg.scene.robot.actuators['z1_arm'].stiffness=800.
        cfg.scene.robot.actuators['z1_arm'].damping=34.
        cfg.rewards.warehouse_height.weight=-200.
        cfg.rewards.flat_orientation_l2.weight=-5.
        cfg.rewards.track_lin_vel_xy_exp.weight=3.
    cfg.terminations.warehouse_tilt = TerminationTermCfg(func=mdp.bad_orientation, params={'limit_angle': .5548110329800715})
    cfg.terminations.warehouse_height = TerminationTermCfg(func=mdp.root_height_below_minimum, params={'minimum_height': .22})
    agent = handle_deprecated_rsl_rl_cfg(agent, metadata.version('rsl-rl-lib'))
    agent.device, agent.seed = args.device, args.seed
    agent.algorithm.learning_rate, agent.algorithm.schedule = 1.e-4, 'fixed'
    agent.algorithm.entropy_coef = .003
    agent.save_interval = 50
    agent.obs_groups = {'actor':['policy'], 'critic':['policy']}
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
        source_checkpoint=str(args.checkpoint.resolve()), source_checkpoint_sha256=sha256(args.checkpoint),
        robot_asset=str(asset), robot_asset_sha256=sha256(asset) if asset.exists() else None,
        arm_model=args.arm_model,
        pose_source=str(args.pose_source) if args.arm_model!='z1' else 'explicit Z1 home and compact stow poses',
        pose_source_sha256=sha256(args.pose_source) if args.arm_model!='z1' else None,
        arm_target_poses_rad=poses, observation_dimension=48, action_dimension=12,
        seed=args.seed, num_envs=args.num_envs, iterations=args.iterations,
        training_scope=('flat ground, original 5.221 kg Z1 with Unitree gripper, home/compact stow poses; no payload/grasp training'
            if args.arm_model=='z1' else 'flat ground, mounted empty lightweight arm, home/available planned carry poses; no payload or parcel grasp training'),
        learning_rate=1.e-4, height_reward={'target_m':.33,'weight':cfg.rewards.warehouse_height.weight},
        joint_limit_reward_weight=-2., command_ranges={'vx':[-.5,.5],'vy':[-.35,.35],'yaw':[-.15,.15]},
        evaluation_scope='deterministic command cycles, original hard-stop thresholds, no recovery or resets',
        actual_training_completed=False,
        evaluation_seconds=args.eval_seconds, evaluation_warmup_seconds=5.,
        training_episode_resets_allowed=True,
        motor_limits_modified=False,
        command_replay=dict(environment_fraction=.5 if replay_sequences else 0.,
            environment_assignment=layout,
            sources=replay_sources, source_hashes={str(p):sha256(p) for p in args.command_trace},
            leg_actions_imitated=False, evaluation_replay_enabled=False),
        source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
            ('train_mounted_gait.py','gait_adaptation.py','mounted_robot_asset.py','mounted_z1_asset.py','run_gait_adaptation.ps1')})
    if args.z1_pose_trace:
        manifest.update(pose_source=str(args.z1_pose_trace.resolve()),pose_source_sha256=sha256(args.z1_pose_trace),
            arm_gravity_compensation=True,arm_drive_gains={'stiffness':800.,'damping':34.},
            training_scope='Original-mass Z1 home/stow plus measured extended carry poses; empty gripper, no parcel or grasp training',
            training_objective_changes={'height_weight':-200.,'flat_orientation_weight':-5.,'linear_velocity_weight':3.})
    for filename in ('train_mounted_gait.py','gait_adaptation.py','mounted_robot_asset.py','mounted_z1_asset.py','run_gait_adaptation.ps1'):
        shutil.copy2(Path(__file__).with_name(filename), output/filename)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from isaaclab.utils.io import dump_yaml
    dump_yaml(str(output/'env.yaml'), cfg)
    dump_yaml(str(output/'agent.yaml'), agent)
    print(f'GAIT_ADAPTATION output={output}',flush=True)

    with launch_simulation(cfg, args):
        if args.arm_model=='z1':
            # Load USD bindings only after Kit registers its bundled schemas.
            # Preloading standalone pxr here can cause incompatible C++ types.
            build_mounted_z1_asset(ROOT/'artifacts/go2_asset/go2.usd',imported['usd_path'],asset)
            manifest['robot_asset_sha256']=sha256(asset)
        # The policy still outputs only 12 leg targets. Hold all arm joints via
        # their existing drives every substep; no writes to measured state.
        from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

        class MountedLegAction(JointPositionAction):
            def __init__(self, action_cfg, env):
                super().__init__(action_cfg, env)
                self.arm_ids = self._asset.find_joints(list(arm_joint_names), preserve_order=True)[0]
                self.finger_ids = self._asset.find_joints(list(finger_joint_names), preserve_order=True)[0]
                bank = torch.tensor(poses, device=self.device)
                pose_ids = torch.tensor([group['arm_pose_index'] for group in layout], device=self.device)
                self.arm_targets = bank[pose_ids]
                self.finger_targets = torch.full((self.num_envs,len(finger_joint_names)), finger_open, device=self.device)
                self.arm_effort_limits=(torch.tensor(kin.efforts,device=self.device,dtype=torch.float32)
                    if args.z1_pose_trace else None)

            def apply_actions(self):
                super().apply_actions()
                self._asset.set_joint_position_target_index(target=self.arm_targets, joint_ids=self.arm_ids)
                self._asset.set_joint_position_target_index(target=self.finger_targets, joint_ids=self.finger_ids)
                if self.arm_effort_limits is not None:
                    gravity=self._asset.data.gravity_compensation_forces
                    gravity=getattr(gravity,'torch',gravity)[:,self._asset.num_base_dofs:][:,self.arm_ids]
                    bounded=torch.maximum(torch.minimum(gravity,self.arm_effort_limits),-self.arm_effort_limits)
                    self._asset.set_joint_effort_target_index(target=bounded,joint_ids=self.arm_ids)

        cfg.actions.joint_pos.class_type = MountedLegAction
        if replay_sequences:
            from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

            class WarehouseReplayCommand(UniformVelocityCommand):
                """PPO still learns actions; only the requested velocities replay."""

                def __init__(self, command_cfg, env):
                    super().__init__(command_cfg, env)
                    lengths = [len(sequence) for sequence in replay_sequences]
                    bank = torch.zeros((len(lengths), max(lengths), 3), device=self.device)
                    for i, sequence in enumerate(replay_sequences):
                        bank[i, :len(sequence)] = torch.tensor(sequence, device=self.device)
                    self.replay_bank = bank
                    self.replay_lengths = torch.tensor(lengths, device=self.device)
                    self.replay_sequence = torch.tensor([group['replay_sequence_index'] or 0
                        for group in layout], device=self.device)
                    self.replay_mask = torch.tensor([group['replay_enabled']
                        for group in layout], device=self.device)
                    self.replay_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
                    self.replay_enabled = True

                def _resample_command(self, env_ids):
                    super()._resample_command(env_ids)
                    lengths = self.replay_lengths[self.replay_sequence[env_ids]]
                    self.replay_offsets[env_ids] = (torch.rand(len(env_ids), device=self.device)*lengths).long()

                def _update_command(self):
                    super()._update_command()
                    if self.replay_enabled:
                        sample = (self._env.episode_length_buf + self.replay_offsets) % self.replay_lengths[self.replay_sequence]
                        replay = self.replay_bank[self.replay_sequence, sample]
                        self.vel_command_b[self.replay_mask] = replay[self.replay_mask]

            cfg.commands.base_velocity.class_type = WarehouseReplayCommand
        raw = gym.make('Isaac-Velocity-Flat-Unitree-Go2-v0', cfg=cfg)
        if args.arm_model=='z1':
            from go2_logistics.z1_asset import ARM_LINK_NAMES, GRIPPER_LINK_NAMES
            arm_bodies=raw.unwrapped.scene['robot'].find_bodies(list(ARM_LINK_NAMES)+list(GRIPPER_LINK_NAMES),preserve_order=True)[0]
            mass=raw.unwrapped.scene['robot'].data.default_mass.torch[:,arm_bodies].sum(-1)
            if len(arm_bodies)!=9 or not bool(torch.all((mass-5.22096983).abs()<.001)):
                raise RuntimeError('Z1 training mass or body count differs from official import')
            manifest['measured_arm_mass_kg']=mass.cpu().tolist()
            manifest['import_provenance']=imported['provenance']
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output/'training'), device=agent.device)
        # Deliberately start a fresh optimizer for the changed reward/dynamics.
        runner.load(str(args.checkpoint), load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False})

        def evaluate(label):
            from go2_logistics.gait_adaptation import SurvivalLedger, evaluation_command
            policy = runner.get_inference_policy(device=env.device)
            unwrapped = env.unwrapped
            saved_termination_cfg = unwrapped.cfg.terminations
            # Manager terms, not merely their config, must be disabled.
            old_terms = list(unwrapped.termination_manager.active_terms)
            saved_terms = {name:unwrapped.termination_manager.get_term_cfg(name) for name in old_terms}
            for name, term in saved_terms.items():
                disabled = term.copy()
                disabled.func = lambda env, **kwargs: torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
                unwrapped.termination_manager.set_term_cfg(name, disabled)
            command = unwrapped.command_manager.get_term('base_velocity')
            replay_was_enabled = getattr(command, 'replay_enabled', False)
            command.replay_enabled = False
            old_resample = command.cfg.resampling_time_range
            command.cfg.resampling_time_range = (1.e9, 1.e9)
            obs, _ = env.reset()
            ledger = SurvivalLedger(args.num_envs)
            minimum = torch.full((args.num_envs,),float('inf'),device=env.device)
            max_tilt = torch.zeros_like(minimum)
            low_frames = torch.zeros_like(minimum)
            squared_error = torch.zeros_like(minimum)
            frame_count = torch.zeros_like(minimum)
            path = output/(label+'_evaluation.jsonl')
            steps = round(args.eval_seconds/unwrapped.step_dt)
            warmup = round(5./unwrapped.step_dt)
            with path.open('w',encoding='utf-8') as trace, torch.inference_mode():
                for tick in range(steps+warmup):
                    elapsed = (tick-warmup)*unwrapped.step_dt
                    desired = ([evaluation_command(max(0.,elapsed),i) for i in range(args.num_envs)]
                               if tick>=warmup else [(0.,0.,0.)]*args.num_envs)
                    requested = torch.tensor(desired,device=env.device)
                    command.command[:] = requested
                    # Recompute observations after writing the benchmark input.
                    obs = env.get_observations()
                    actions = policy(obs)
                    _, _, dones, _ = env.step(actions)
                    robot = unwrapped.scene['robot']
                    height = robot.data.root_pos_w.torch[:,2]
                    gravity = robot.data.projected_gravity_b.torch[:,2]
                    ledger.record((tick+1)*unwrapped.step_dt,height.tolist(),gravity.tolist(),dones.tolist())
                    if tick>=warmup:
                        valid = torch.tensor([r is None for r in ledger.reason],device=env.device)
                        minimum = torch.minimum(minimum,height)
                        max_tilt = torch.maximum(max_tilt,torch.acos((-gravity).clamp(-1.,1.)))
                        low_frames += (height<.285).float()
                        error = ((robot.data.root_lin_vel_b.torch[:,:2]-requested[:,:2])**2).sum(-1)
                        squared_error += torch.where(valid,error,0.)
                        frame_count += valid.float()
                    if tick%10==0:
                        trace.write(json.dumps(dict(tick=tick+1,time_s=(tick+1)*unwrapped.step_dt,
                            height_m=height.tolist(),gravity_z=gravity.tolist(),command=requested.tolist(),
                            velocity_b=robot.data.root_lin_vel_b.torch.tolist(),dones=dones.tolist()))+'\n')
            report = dict(label=label,**ledger.summary(),warmup_s=5.,evaluated_s=args.eval_seconds,
                minimum_height_m=minimum.tolist(),max_tilt_deg=torch.rad2deg(max_tilt).tolist(),
                low_clearance_fraction=(low_frames/steps).tolist(),
                velocity_rmse_before_failure_mps=(squared_error/frame_count.clamp_min(1.)).sqrt().tolist(),
                valid_tracking_frames=frame_count.tolist(),resets_allowed=False,recovery_controller=False,
                note='Isolated flat-ground mounted-empty-arm benchmark, not logistics success or payload validation')
            (output/(label+'_evaluation.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
            for name, term in saved_terms.items():
                unwrapped.termination_manager.set_term_cfg(name, term)
            unwrapped.cfg.terminations = saved_termination_cfg
            command.cfg.resampling_time_range = old_resample
            command.replay_enabled = replay_was_enabled
            print(f'GAIT_EVALUATION label={label} survived={report["survived"]}/{args.num_envs}',flush=True)
            env.reset()
            return report

        baseline = evaluate('before')
        manifest['baseline_evaluation_survived'] = baseline['survived']
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if not args.eval_only:
            start = time.monotonic()
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
            checkpoint = output/'mounted_gait.pt'
            runner.save(str(checkpoint))
            manifest.update(actual_training_completed=True,training_wall_seconds=time.monotonic()-start,
                trained_checkpoint=str(checkpoint),trained_checkpoint_sha256=sha256(checkpoint),
                environment_transitions=args.num_envs*agent.num_steps_per_env*args.iterations)
            after = evaluate('after')
            manifest['evaluation_survived']={'before':baseline['survived'],'after':after['survived'],'total':args.num_envs}
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        env.close()
    if not args.eval_only and not manifest.get('evaluation_survived'):
        raise RuntimeError('Training or post-training evaluation did not finish; manifest is incomplete')


if __name__ == '__main__':
    main()
