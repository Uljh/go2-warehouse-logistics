"""Measured simulator evidence and third-person frames for physical missions."""
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import asdict


class MissionRecorder:
    def __init__(self, env, arm, output_dir):
        self.env = env.unwrapped
        self.arm = arm
        self.camera = self.env.scene["mission_camera"]
        self.base_dir = Path(output_dir)
        self.output_dir = self.base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.previous_phase = None
        self.started = time.perf_counter()
        self.video = None
        self.batch = None
        self.feed_video = None
        self.recovery = None
        self.overview_video=None
        self.navigator=None
        self.dynamic_obstacle=None
        self.receiving_gate=None

    def begin(self):
        self.output_dir = self.base_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.previous_phase = None
        self.started = time.perf_counter()
        import imageio.v2 as imageio
        self.video = imageio.get_writer(str(self.output_dir / "mission.mp4"), fps=5, codec="libx264", quality=8, macro_block_size=8)
        if 'warehouse_camera' in self.env.scene.sensors:
            import torch
            camera=self.env.scene['warehouse_camera']
            camera.set_world_poses_from_view(torch.tensor([[13.,-13.,17.]],device=self.env.device),
                                             torch.tensor([[0.,4.,0.]],device=self.env.device))
            self.overview_video=imageio.get_writer(str(self.output_dir/'warehouse_overview.mp4'),fps=5,codec='libx264',quality=8,macro_block_size=8)
        if self.batch is not None:
            import torch
            camera=self.env.scene['feed_camera']
            if self.batch.mode == 'two_parcel_active_uncovering':
                belt=torch.tensor((self.batch.pickup.x,self.batch.pickup.y-.2,.02),device=self.env.device)
            else:
                belt=self.env.scene['conveyor'].data.root_pos_w[0]
            eye=belt+torch.tensor((.7,-1.4,1.2),device=self.env.device)
            target=belt+torch.tensor((-.25,0.,0.),device=self.env.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0),target.unsqueeze(0))
            self.feed_video=imageio.get_writer(str(self.output_dir/'conveyor.mp4'),fps=5,codec='libx264',quality=8,macro_block_size=8)

    def step(self, pose=None):
        if self.receiving_gate is not None:
            self.receiving_gate.step(self.arm,self.output_dir)
        if self.dynamic_obstacle is not None:
            self.dynamic_obstacle.step(self.arm,self.output_dir)
        from .mission import PayloadLost
        loss = None
        try:
            self.arm.sync(pose)
        except PayloadLost as exc:
            loss = exc
        robot = self.env.scene["robot"]
        root = robot.data.root_pos_w[0].cpu().tolist()
        gravity = robot.data.projected_gravity_b[0].cpu().tolist()
        box = self.arm.box.data.root_pos_w[0].cpu().tolist()
        contacts = {}
        for side in ("left", "right"):
            sensor = self.env.scene.sensors.get(f"{side}_grip_contact")
            if sensor is not None:
                forces = getattr(sensor.data.force_matrix_w, "torch", sensor.data.force_matrix_w)
                contacts[side] = forces[0].cpu().tolist() if forces is not None else None
        blocker_force = None
        sensor = self.env.scene.sensors.get('blocker_contact')
        if sensor is not None:
            forces = getattr(sensor.data.net_forces_w_history,'torch',sensor.data.net_forces_w_history)
            blocker_force = float(forces[0].norm(dim=-1).max()) if forces is not None else None
        data = self.arm.robot.data.joint_pos
        joints = getattr(data, "torch", data)[0, self.arm.finger_ids].cpu().tolist()
        self.records.append(dict(step=len(self.records), phase=self.arm.state, root=root, gravity=gravity, parcel=box, held=self.arm.held, finger_joints=joints, finger_contact_forces=contacts, blocker_contact_force_n=blocker_force))
        if self.dynamic_obstacle is not None:
            self.records[-1]['dynamic_obstacle_position']=list(self.dynamic_obstacle.position)
        if self.receiving_gate is not None:
            self.records[-1]['belt_running']=self.receiving_gate.running
        command=self.env.command_manager.get_term('base_velocity').command
        self.records[-1]['command_body_velocity']=getattr(command,'torch',command)[0].cpu().tolist()
        self.records[-1]['parcel_quaternion_xyzw']=self.arm.box.data.root_quat_w[0].cpu().tolist()
        self.records[-1]['tool_position']=self.arm._tool_pose()[0].cpu().tolist()
        if self.batch is not None:
            self.records[-1]['parcels']=self.batch.positions()
            self.records[-1]['belt']=None if self.batch.mode == 'two_parcel_active_uncovering' else self.env.scene['conveyor'].data.root_pos_w[0].cpu().tolist()
            if len(self.records) in (1,20,50):
                print(f"CONVEYOR_STATE step={len(self.records)} belt={self.records[-1]['belt']} parcels={self.records[-1]['parcels']}",flush=True)
        if len(self.records)%200==0 or self.arm.state!=self.previous_phase:
            (self.output_dir/'live_state.json').write_text(json.dumps(self.records[-1],indent=2),encoding='utf-8')
            (self.base_dir/'warehouse_status.json').write_text(json.dumps(dict(
                run_dir=str(self.output_dir.resolve()),phase=self.arm.state,
                observed_parcel=getattr(self.arm,'observed_parcel_code',None),
                destination=list(self.arm.config.dropoff_position),
                completed=sorted(self.batch.completed) if self.batch else [],
                configured_orders=len(self.batch.codes) if self.batch else 1,
                belt_running=self.receiving_gate.running if self.receiving_gate else None,
                scans=len(self.navigator.scans) if self.navigator else 0,
                path_invalidations=self.navigator.path_invalidations if self.navigator else 0,
                blocked_wait_scans=self.navigator.wait_steps if self.navigator else 0),indent=2),encoding='utf-8')
        if self.arm.state != self.previous_phase:
            self.snapshot(self.arm.state)
            self.previous_phase = self.arm.state
        elif len(self.records) % 200 == 0:
            self.snapshot(f"{self.arm.state}_{len(self.records):05d}")
        if self.video is not None and len(self.records) % 10 == 0:
            import cv2
            rgb = self._image(render_frames=1)
            cv2.putText(rgb, f"{self.arm.state} | t={len(self.records)*self.env.step_dt:.1f}s", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            self.video.append_data(rgb)
            if self.overview_video is not None:
                overview=self.env.scene['warehouse_camera']
                overview.update(self.env.step_dt,force_recompute=True)
                frame=overview.data.output['rgb'][0,...,:3].cpu().numpy().copy()
                code=getattr(self.arm,'observed_parcel_code','not observed')
                completed=len(self.batch.completed) if self.batch else 0
                total=len(self.batch.codes) if self.batch else 1
                cv2.putText(frame,f'{self.arm.state} | delivered {completed}/{total} | {code}',(15,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                if self.navigator is not None:
                    from .navigation_overlay import draw_navigation
                    draw_navigation(frame,self.navigator,root)
                self.overview_video.append_data(frame)
                if len(self.records)==10:
                    from PIL import Image
                    Image.fromarray(frame).save(self.output_dir/'warehouse_overview.png')
            if self.feed_video is not None:
                camera=self.env.scene['feed_camera']
                camera.update(self.env.step_dt,force_recompute=True)
                self.feed_video.append_data(camera.data.output['rgb'][0,...,:3].cpu().numpy())
        if loss is not None:
            raise loss

    def snapshot(self, name):
        from PIL import Image
        Image.fromarray(self._image()).save(self.output_dir / (name + ".png"))

    def _image(self, render_frames=3):
        import torch
        root = self.env.scene["robot"].data.root_pos_w[0]
        eye = root + torch.tensor((1.3, -1.5, 1.0), device=self.env.device)
        target = root + torch.tensor((.3, 0., .25), device=self.env.device)
        self.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        for _ in range(render_frames):
            self.env.sim.render()
            self.camera.update(self.env.step_dt, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        return rgb.copy()

    def finish(self, mission, detection=None):
        self.snapshot("final_" + mission.state.value)
        if self.overview_video is not None:
            self.overview_video.close()
            self.overview_video=None
        if self.video is not None:
            self.video.close()
            self.video = None
        if self.feed_video is not None:
            self.feed_video.close()
            self.feed_video=None
        robot = self.env.scene["robot"]
        root = robot.data.root_pos_w[0].cpu().tolist()
        gravity = robot.data.projected_gravity_b[0].cpu().tolist()
        robot_prim = self.arm.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        result = dict(state=mission.state.value, failure_reason=mission.failure_reason, metrics=asdict(mission.metrics), wall_seconds=time.perf_counter()-self.started,
                      root_final=root, gravity_final=gravity, detection=detection,
                      parcel_final=self.arm.box.data.root_pos_w[0].cpu().tolist(),
                      physical_control_steps=len(self.records),
                      min_height=min((r["root"][2] for r in self.records), default=0),
                      grasp_model=self.arm.config.grasp_mode,
                      simulation_device=str(self.env.device),
                      conveyor_batch=self.batch.codes if self.batch is not None else None,
                      navigation=getattr(mission, 'navigation', None),
                      blocker_max_contact_force_n=max((r['blocker_contact_force_n'] for r in self.records if r['blocker_contact_force_n'] is not None),default=None),
                      target_dropoff=list(self.arm.config.dropoff_position),
                      tool_offset=list(self.arm.config.tool_offset),
                      arm_scale=robot_prim.GetCustomDataByKey("arm_scale"),
                      arm_mass_kg=robot_prim.GetCustomDataByKey("arm_mass_kg"),
                      mount_pitch_degrees=robot_prim.GetCustomDataByKey("mount_pitch_degrees"),
                      grasp_constraint_created=self.arm.joint is not None,
                      visual_geometry=getattr(self.arm,'visual_geometry',False),
                      wide_gripper=robot_prim.GetCustomDataByKey('wide_gripper'),
                      finger_open_target_m=self.arm.config.finger_open,
                      verified_lift_m=self.arm.verified_lift, max_hold_error_m=self.arm.max_hold_error,
                      placement=self.arm.placement_metrics,
                      drop_recovery_events=self.recovery.events if self.recovery is not None else [],
                      occlusion_events=getattr(self.batch,'events',[]),
                      payload_loss_events=self.arm.loss_events,
                      feed_events=self.receiving_gate.events if self.receiving_gate else [],
                      dynamic_obstacle_events=self.dynamic_obstacle.events if self.dynamic_obstacle else [],
                      arm_model="lightweight_scaled_panda_surrogate")
        (self.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (self.output_dir / "trajectory.json").write_text(json.dumps(self.records), encoding="utf-8")
        (self.base_dir / "latest_result.json").write_text(json.dumps(dict(result=result, run_dir=str(self.output_dir.resolve())), indent=2), encoding="utf-8")
        print(f"PHYSICAL_EVIDENCE result={self.output_dir / 'result.json'}", flush=True)
