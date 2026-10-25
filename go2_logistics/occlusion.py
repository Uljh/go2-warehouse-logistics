"""Two-parcel active uncovering baseline with a physical temporary pad.

Side labels provide an inventory and lowest-shelf-number dispatch priority.
The top identity is discovered separately, staged when it blocks that target,
then the lower top face is reobserved and grasped. No hidden label is handed to
the controller by the scene. This is a two-layer baseline with readable side
labels, not arbitrary buried-target retrieval or a learned rummaging policy.
"""
import copy
import json
from dataclasses import replace
from pathlib import Path

from .conveyor import ConveyorBatch
from .interfaces import Pose2D


def configure_occlusion(cfg,pickup,codes):
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils
    if len(codes)!=2 or len(set(codes))!=2:
        raise ValueError('Occlusion baseline requires exactly two distinct labels')
    cfg.sim.device='cpu'
    # Two upright parcels stacked; initialization only. Gravity/contact determine
    # their subsequent state. The lower top label is physically occluded.
    for index in range(2):
        parcel=cfg.scene.parcel if index==0 else copy.deepcopy(cfg.scene.parcel)
        parcel.prim_path='{ENV_REGEX_NS}/Parcel'+(str(index) if index else '')
        parcel.init_state.pos=(pickup.x,pickup.y,.0605+index*.0805)
        parcel.init_state.rot=(0.,0.,0.,1.)
        if index:
            cfg.scene.parcel_1=parcel
    for side in ('left','right'):
        getattr(cfg.scene,f'{side}_grip_contact').filter_prim_paths_expr=['{ENV_REGEX_NS}/Parcel','{ENV_REGEX_NS}/Parcel1']
    cfg.scene.staging_pad=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/StagingPad',
        spawn=sim_utils.CuboidCfg(size=(.30,.30,.02),collision_props=sim_utils.CollisionPropertiesCfg(),
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.55,.18))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(pickup.x,pickup.y-.65,.01)))
    camera=cfg.scene.mission_camera.copy()
    camera.prim_path='{ENV_REGEX_NS}/FeedCamera'
    cfg.scene.feed_camera=camera
    inspector=cfg.scene.parcel_camera.copy()
    inspector.prim_path='{ENV_REGEX_NS}/InventoryCamera'
    inspector.update_latest_camera_pose=True
    cfg.scene.inventory_camera=inspector
    cfg.scene.parcel_camera.update_latest_camera_pose=True


class OcclusionBatch(ConveyorBatch):
    mode='two_parcel_active_uncovering'
    def __init__(self,env,arm,codes,pickup,perception,*,stack_feed=False):
        super().__init__(env,arm,codes)
        self.pickup=pickup
        self.perception=perception
        self.temporary=(pickup.x,pickup.y-.65,.06)
        self.staged_code=None
        self.stage_status='initial'
        self.events=[]
        self.output_dir=None
        self.navigate=None
        self.cycle=0
        self.target_code=None
        self.stack_feed=stack_feed
        self.feed_ready=False
        if stack_feed:
            self.mode='conveyor_two_parcel_active_uncovering'

    def begin_cycle(self,cycle,output_dir):
        self.cycle=cycle
        self.output_dir=Path(output_dir)
        if cycle==0:
            self.stage_status='initial'
            self.staged_code=None
            self.events=[]
            self.target_code=None
            self.feed_ready=False
        self.aim_camera(self.pickup.x,self.temporary[1] if cycle>0 and self.staged_code is not None else self.pickup.y)

    def aim_camera(self,x,y):
        import torch
        camera=self.perception.camera
        # The incoming layer is higher than a floor parcel; frame its complete
        # top face rather than cropping it at the image edge.
        aim_z=.12 if self.stack_feed and self.cycle==0 and self.stage_status=='initial' else .06
        target=torch.tensor((x,y,aim_z),device=self.env.device)
        eye=torch.tensor((x+.35,y,.8),device=self.env.device)
        camera.set_world_poses_from_view(eye.unsqueeze(0),target.unsqueeze(0))

    def source_dock(self):
        return Pose2D(self.pickup.x-.58,self.temporary[1] if self.cycle>0 and self.staged_code is not None else self.pickup.y)

    def bind(self,bridge,navigate):
        self.navigate=navigate

    def log(self,**event):
        self.events.append(event)
        (self.output_dir/'occlusion_events.json').write_text(json.dumps(self.events,indent=2),encoding='utf-8')
        print(f'OCCLUSION {event}',flush=True)

    def inspect_inventory(self):
        """Read side labels at two fixed bin inspection heights, not body poses."""
        import torch
        from PIL import Image
        from .sorting_scene import read_rendered_code
        camera=self.env.scene['inventory_camera']
        observed=[]
        for level in range(2):
            z=.06+.08*level
            eye=torch.tensor((self.pickup.x+(0. if self.stack_feed else .35),self.pickup.y+(.35 if self.stack_feed else 0.),z+.22),device=self.env.device)
            target=torch.tensor((self.pickup.x+(0. if self.stack_feed else .05),self.pickup.y,z),device=self.env.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0),target.unsqueeze(0))
            for _ in range(4):
                self.env.sim.render();camera.update(self.env.step_dt,force_recompute=True)
            rgb=camera.data.output['rgb'][0,...,:3].cpu().numpy()
            Image.fromarray(rgb).save(self.output_dir/f'inventory_side_{level}.png')
            try:
                code=read_rendered_code(rgb)
            except ValueError:
                continue
            if code not in observed:
                observed.append(code)
        if not observed:
            self.arm.last_error='No side labels available for visual priority selection'
            return False
        # Deterministic dispatch priority: lowest destination number first.
        # No label value is taken from the scene-generation batch order.
        self.target_code=min(observed,key=lambda code:int(code.split('-')[0]))
        self.log(event='select_visual_target',observed_codes=observed,target_code=self.target_code,
                 policy='lowest_shelf_number_among_camera_labels')
        return True

    def locate(self,object_id):
        if self.cycle>0:
            self.perception.output_dir=self.output_dir/'staged_pickup'
            return self.perception.locate(self.staged_code or 'box_red')
        if self.stage_status=='initial':
            if self.stack_feed and not self.feed_ready and not self.wait_for_feed():
                return False
            if self.target_code is None and not self.inspect_inventory():
                return False
            self.perception.output_dir=self.output_dir/'occlusion_before'
            if not self.perception.locate('box_red'):
                return False
            self.staged_code=self.perception.last_detection['parcel_code']
            if self.staged_code==self.target_code:
                self.log(event='target_directly_accessible',code=self.target_code)
                self.staged_code=None
                return True
            self.log(event='observe_occluder',code=self.staged_code,detection=self.perception.last_detection)
            geometry=self.arm.detected_geometry
            # Two layers are inferred from the observed surface above a known
            # 2cm bin floor, not from reading the lower parcel body.
            if geometry['center_world'][2]-geometry['height_m']/2 < .045:
                self.arm.last_error='No elevated parcel layer observed'
                return False
            if not self.arm.grasp(self.staged_code):
                return False
            self.stage_status='holding_occluder'
            self.log(event='lift_occluder',code=self.staged_code)
        if self.stage_status=='holding_occluder':
            self.arm._set_phase('uncover_move_to_temporary')
            if not self.navigate(Pose2D(self.pickup.x-.58,self.temporary[1])):
                return False
            self.arm.config=replace(self.arm.config,dropoff_position=self.temporary)
            if not self.arm.release():
                return False
            self.stage_status='occluder_staged'
            self.log(event='stage_occluder',code=self.staged_code,placement=self.arm.placement_metrics)
            self.arm.prepare()
        self.arm._set_phase('uncover_return_to_bin')
        if not self.navigate(self.source_dock()):
            return False
        self.aim_camera(self.pickup.x,self.pickup.y)
        self.perception.output_dir=self.output_dir/'occlusion_after'
        if not self.perception.locate('box_red'):
            return False
        code=self.perception.last_detection['parcel_code']
        if code!=self.target_code:
            self.arm.last_error='Uncovering did not expose the selected target parcel'
            return False
        self.log(event='observe_lower',code=code,detection=self.perception.last_detection)
        return True

    def wait_for_feed(self):
        """Gate manipulation on stable RGB-D observations, never parcel truth."""
        from .feed_readiness import StackReadiness
        gate=StackReadiness((self.pickup.x,self.pickup.y))
        self.arm._set_phase('wait_for_stacked_feed')
        self.perception.output_dir=self.output_dir/'feed_observation'
        # Sample after physical time advances; repeat rendering alone is not
        # evidence that an incoming parcel has stopped moving.
        for sample in range(80):
            for _ in range(15):
                self.arm._step()
            detection=self.perception.last_detection if self.perception.locate('box_red') else None
            if gate.observe(detection):
                self.feed_ready=True
                self.log(event='stack_feed_ready',source='rendered_rgb_depth',samples=sample+1,
                         detection=detection,temporary_position=list(self.temporary))
                return True
        self.arm.last_error='Conveyor stack did not become visually stable within 24 seconds'
        return False
