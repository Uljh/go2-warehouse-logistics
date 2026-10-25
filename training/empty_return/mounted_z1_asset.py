"""Mount an unscaled, original-inertia Unitree Z1 on the Go2 articulation."""
from __future__ import annotations

from pathlib import Path
import math

from .z1_asset import ARM_JOINT_NAMES, GRIPPER_JOINT_NAME
from .mounted_robot_asset import LEG_JOINT_NAMES

Z1_HOME=(0.,1.2,-1.2,0.,0.,0.)
Z1_OPEN=-.8


def build_mounted_z1_asset(go2_path,z1_path,output_path,*,mount_offset=(0.,0.,.10),
                           gripper_collision_mode=None):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
    sources=[Path(value).resolve() for value in (go2_path,z1_path)]
    if not all(value.is_file() for value in sources):
        raise FileNotFoundError(sources)
    if len(mount_offset)!=3 or not all(math.isfinite(value) for value in mount_offset):
        raise ValueError('mount_offset must contain three finite values')
    if gripper_collision_mode not in (None,'convexHull','convexDecomposition'):
        raise ValueError('Unsupported original gripper collision approximation')
    destination=Path(output_path).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root=UsdGeom.Xform.Define(stage,'/Robot').GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(sources[0].as_posix())
    root.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
    base=stage.GetPrimAtPath('/Robot/base')
    if not base or not base.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Go2 reference does not expose /base rigid body')
    UsdPhysics.ArticulationRootAPI.Apply(base)
    arm=UsdGeom.Xform.Define(stage,'/Robot/Arm')
    arm.GetPrim().GetReferences().AddReference(sources[1].as_posix())
    variants=arm.GetPrim().GetVariantSets().GetVariantSet('Physics')
    if 'physx' in variants.GetVariantNames():
        variants.SetVariantSelection('physx')
    arm.ClearXformOpOrder()
    arm.AddTranslateOp().Set(Gf.Vec3d(*mount_offset))
    links=[]
    disabled=[]
    for prim in list(Usd.PrimRange(arm.GetPrim())):
        schemas=prim.GetMetadata('apiSchemas')
        applied=schemas.GetAppliedItems() if schemas is not None else []
        # The Isaac 6 multi-physics import also carries Newton's root schema.
        # Its registered inheritance can revive a second USD articulation root
        # in Kit even after the direct PhysicsArticulationRootAPI was removed.
        kept=[token for token in applied if not token.endswith('ArticulationRootAPI')]
        if kept!=applied:
            prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(kept))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                disabled.append(str(prim.GetPath()))
                prim.SetActive(False)
    arm_base=next((prim for prim in links if prim.GetName()=='link00'),None)
    if arm_base is None or len(links)!=9:
        raise ValueError(f'Expected nine official Z1+gripper rigid links; got {len(links)}')
    cache=UsdGeom.XformCache()
    relative=cache.GetLocalToWorldTransform(arm_base)*cache.GetLocalToWorldTransform(base).GetInverse()
    rotation=Gf.Transform(relative).GetRotation().GetQuat()
    joint=UsdPhysics.FixedJoint.Define(stage,'/Robot/z1_mount')
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([arm_base.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(relative.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(rotation.GetReal(),Gf.Vec3f(rotation.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    UsdPhysics.FilteredPairsAPI.Apply(base).CreateFilteredPairsRel().AddTarget(arm_base.GetPath())
    masses={prim.GetName():float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()) for prim in links}
    total=sum(masses.values())
    if abs(total-5.22096983)>1e-5:
        raise ValueError(f'Unscaled official arm mass changed: {total}')
    collisions=[]
    if gripper_collision_mode is not None:
        paths=[prim.GetPath() for prim in Usd.PrimRange(arm.GetPrim(),Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and 'gripper' in str(prim.GetPath()).lower()]
        for path in paths:
            prim=stage.GetPrimAtPath(path)
            parent=prim
            while parent and parent.IsInstanceProxy():
                parent=parent.GetParent()
            if parent and parent.IsInstance():
                parent.SetInstanceable(False)
            prim=stage.GetPrimAtPath(path)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(gripper_collision_mode)
            collisions.append(dict(path=str(path),approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()))
        if len(collisions)!=2 or any(row['approximation']!=gripper_collision_mode for row in collisions):
            raise RuntimeError(f'Expected both original jaw collision meshes: {collisions}')
    root.SetCustomDataByKey('arm_model','official_unitree_z1_with_unitree_gripper')
    root.SetCustomDataByKey('arm_mass_kg',total)
    root.SetCustomDataByKey('arm_scale',1.)
    stage.GetRootLayer().Export(str(destination))
    return dict(asset=str(destination),arm_mass_kg=total,link_masses_kg=masses,
                world_joints_disabled=disabled,mount_offset_m=list(mount_offset),scale=1.,
                body_paths={prim.GetName():str(prim.GetPath()) for prim in links},
                gripper_collisions=collisions,added_payload_joints=[])


def configure_z1_robot(cfg,asset_path):
    """Keep the original leg observation/action order; use real Z1 joint limits."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.managers import SceneEntityCfg
    robot=cfg.scene.robot
    robot.spawn.usd_path=str(Path(asset_path).resolve())
    robot.spawn.variants={'Physics':'physx'}
    robot.spawn.articulation_props.solver_position_iteration_count=12
    robot.spawn.articulation_props.solver_velocity_iteration_count=4
    robot.init_state.joint_pos.update(dict(zip(ARM_JOINT_NAMES,Z1_HOME)))
    robot.init_state.joint_pos[GRIPPER_JOINT_NAME]=Z1_OPEN
    robot.actuators['z1_arm']=ImplicitActuatorCfg(joint_names_expr=list(ARM_JOINT_NAMES),
        effort_limit_sim={name:60. if name=='joint2' else 30. for name in ARM_JOINT_NAMES},
        velocity_limit_sim=math.pi,stiffness=160.,damping=12.)
    robot.actuators['z1_gripper']=ImplicitActuatorCfg(joint_names_expr=[GRIPPER_JOINT_NAME],
        effort_limit_sim=30.,velocity_limit_sim=3.1415,stiffness=80.,damping=8.)
    cfg.actions.joint_pos.joint_names=list(LEG_JOINT_NAMES)
    cfg.actions.joint_pos.preserve_order=True
    for name in ('policy','critic'):
        group=getattr(cfg.observations,name,None)
        if group is not None:
            for field in ('joint_pos','joint_vel'):
                term=getattr(group,field,None)
                if term is not None:
                    term.params['asset_cfg']=SceneEntityCfg('robot',joint_names=list(LEG_JOINT_NAMES),preserve_order=True)
