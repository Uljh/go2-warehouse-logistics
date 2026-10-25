"""Finite-batch conveyor: contact transport and gravity feed, no parcel teleport.

Parcels start spaced on the belt. Surface velocity supplies frictional motion;
all delivered objects remain in the same simulation across mission cycles.
"""
import copy
import json
from pathlib import Path
from .sorting_layout import ParcelCode
from .streaming_feed import StreamingFeedPlan, DEFAULT_STREAM_SPACING_M


def spawn_belt(prim_path,cfg,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.spawners.shapes import spawn_cuboid
    from pxr import PhysxSchema,Gf
    prim=spawn_cuboid(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
    api=PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
    api.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(-.04,0.,0.))
    api.CreateSurfaceVelocityEnabledAttr().Set(True)
    api.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
    return prim


def configure_conveyor(cfg, pickup, codes, *, stack_feed=False, streaming=False,
                       spacing_m=None):
    """Author a contact-driven conveyor and its finite parcel queue.

    ``streaming=True`` selects the bounded 6--10 parcel recipe used for
    continuous-feed trials.  Every rigid parcel is still created during scene
    construction; the longer queue and receiving gate provide replenishment by
    contact, so this option never spawns, teleports, or resets an item during a
    mission.  The existing finite/stack modes retain their original geometry.

    The returned :class:`StreamingFeedPlan` is an identity/geometry manifest
    for the runtime and audit layer.  It is deliberately not used to claim a
    parcel: claims must follow a camera-decoded code.
    """
    from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    if streaming and stack_feed:
        raise ValueError('streaming and stack_feed are separate physical recipes')
    plan = (StreamingFeedPlan(tuple(codes), spacing_m=(
        DEFAULT_STREAM_SPACING_M if spacing_m is None else spacing_m))
            if streaming else None)
    # Isaac Sim 6's surface-velocity contact modifier penetrated the thin belt
    # with our GPU pipeline. CPU PhysX passed the stationary/moving comparison.
    cfg.sim.device='cpu'
    if not codes or len(set(codes)) != len(codes):
        raise ValueError('Conveyor needs distinct parcel identities')
    for code in codes:
        ParcelCode.parse(code)
    length = (plan.queue_length_m if plan is not None
              else (.6 if stack_feed else .6+1.0*(len(codes)-1)))
    end_x = pickup.x+(.060 if stack_feed else .015)
    belt_top = .101 if stack_feed else .025
    cfg.scene.bin_wall_3=None  # Open receiving side; the belt bridges this rim.
    cfg.scene.feed_camera=CameraCfg(prim_path='{ENV_REGEX_NS}/FeedCamera',update_period=.2,
        height=600,width=800,data_types=['rgb'],renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=20.955,clipping_range=(.05,100.)),
        offset=CameraCfg.OffsetCfg(pos=(pickup.x+1.,pickup.y-1.4,1.2),convention='world'))
    cfg.scene.conveyor = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/Conveyor',
        spawn=sim_utils.CuboidCfg(func=spawn_belt,size=(length,.16,.025),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.16,.20))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(end_x+length/2,pickup.y,belt_top-.0125)))
    # Physical guide rails limit lateral drift; no joints constrain parcels.
    for side in (-1,1):
        setattr(cfg.scene,f'conveyor_rail_{side+1}',AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/ConveyorRail'+str(side+1),
            spawn=sim_utils.CuboidCfg(size=(length,.015,.025),collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.45,.5))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(end_x+length/2,pickup.y+side*.0875,belt_top+.0125))))
    if stack_feed:
        # A passive stop aligns the arriving upright box over the preloaded one.
        # This is a controlled two-piece feeder, not free-fall random piling.
        cfg.scene.stack_stop=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/StackStop',
            spawn=sim_utils.CuboidCfg(size=(.025,.06,.16),collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.4,.45,.5))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(pickup.x-.0675,pickup.y,.10)))
    for index,code in enumerate(codes):
        if stack_feed and index==0:
            continue
        parcel=cfg.scene.parcel if index==0 else copy.deepcopy(cfg.scene.parcel)
        parcel.prim_path='{ENV_REGEX_NS}/Parcel'+(str(index) if index else '')
        if plan is not None:
            # These are ordinary rigid bodies placed on a long physical queue.
            # Their later motion comes from belt contact; this is not a release
            # command and is never repeated after the simulation starts.
            queue_offset = plan.parcels[index].queue_x_m
        else:
            queue_offset = .20+(0. if stack_feed else index*1.)
        parcel.init_state.pos=(pickup.x+queue_offset,pickup.y,belt_top+.041)
        if index:
            setattr(cfg.scene,f'parcel_{index}',parcel)
    for side in ('left','right'):
        getattr(cfg.scene,f'{side}_grip_contact').filter_prim_paths_expr=['{ENV_REGEX_NS}/Parcel'+(str(i) if i else '') for i in range(len(codes))]
    return plan


def author_conveyor(stage, codes, assets_dir):
    from .sorting_scene import author_parcel_label
    for index,code in enumerate(codes):
        if index:
            author_parcel_label(stage,code,assets_dir,'/World/envs/env_0/Parcel'+str(index))


class ConveyorBatch:
    mode='finite_batch_surface_velocity_conveyor'
    def __init__(self,env,arm,codes,*,streaming_audit=None):
        self.env=env.unwrapped
        self.arm=arm
        self.codes=codes
        self.objects={code:self.env.scene['parcel' if i==0 else f'parcel_{i}'] for i,code in enumerate(codes)}
        self.completed=set()
        self.runs=[]
        self.perception=None
        self.streaming_audit=streaming_audit

    def bind_stream_audit(self, audit):
        """Attach an optional finite-feed ledger without changing claim logic."""
        self.streaming_audit = audit
        return audit

    def observe_receiving(self, detection, *, tick):
        if self.streaming_audit is None:
            raise RuntimeError('streaming feed audit is not bound')
        return self.streaming_audit.observe_receiving(detection, tick=tick)

    def claim_visual(self, code, *, tick):
        if self.streaming_audit is None:
            raise RuntimeError('streaming feed audit is not bound')
        return self.streaming_audit.claim_visual(code, tick=tick)

    def mark_carrying(self, code, *, tick):
        if self.streaming_audit is None:
            raise RuntimeError('streaming feed audit is not bound')
        return self.streaming_audit.mark_carrying(code, tick=tick)

    def mark_delivered(self, code, *, tick):
        if self.streaming_audit is None:
            raise RuntimeError('streaming feed audit is not bound')
        return self.streaming_audit.mark_delivered(code, tick=tick)

    def bind_perception(self, perception):
        """Continuous feeding observes whichever registered parcel is at the bin."""
        self.perception = perception

    def locate(self, _ignored_object_id):
        if self.perception is None:
            raise RuntimeError('Conveyor perception adapter is not bound')
        # Do not pass the previous cycle's code as an expectation. The camera
        # must decode the parcel currently occupying the receiving area.
        return self.perception.locate('box_red')

    def bind_visible(self,code):
        if code in self.completed or code not in self.objects:
            raise RuntimeError(f'Observed parcel unavailable: {code}')
        index=self.codes.index(code)
        self.arm.box=self.objects[code]
        self.arm.box_path='/World/envs/env_0/Parcel'+(str(index) if index else '')
        self.arm.observed_parcel_code=code

    def positions(self):
        return {code:obj.data.root_pos_w[0].cpu().tolist() for code,obj in self.objects.items()}

    def finish_cycle(self,mission,run_dir):
        if mission.state.value=='complete':
            if mission.object_id in self.completed:
                raise RuntimeError('Duplicate delivery')
            self.completed.add(mission.object_id)
        self.runs.append(dict(code=mission.object_id,state=mission.state.value,run_dir=str(Path(run_dir).resolve())))
        result=dict(mode=self.mode,requested=self.codes,completed=sorted(self.completed),
                    state='complete' if len(self.completed)==len(self.codes) else 'incomplete',runs=self.runs,
                    final_positions=self.positions(),world_reset_between_cycles=False)
        if self.streaming_audit is not None:
            result['streaming_feed']=self.streaming_audit.snapshot()
        if hasattr(self,'pickup'):
            result['receiving_layout']=dict(bin_xy=[self.pickup.x,self.pickup.y],
                                             temporary_position=list(self.temporary))
        Path('artifacts/physical_mission/latest_batch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        if self.runs:
            Path(self.runs[0]['run_dir'],'batch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        status_path=Path('artifacts/physical_mission/warehouse_status.json')
        if status_path.exists():
            status=json.loads(status_path.read_text(encoding='utf-8'))
            if status.get('run_dir')==str(Path(run_dir).resolve()):
                status.update(completed=sorted(self.completed),batch_state=result['state'],
                              phase='batch_complete' if result['state']=='complete' else mission.state.value)
                status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
