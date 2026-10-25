"""Build a physically connected Go2 and a lightweight Panda surrogate.

The arm is one branch of the Go2 articulation, joined by a fixed joint.  It is
not a mesh follower. Dimensions are 62% of a Panda and total arm mass is 0.35 kg:
this is an explicitly idealized mobile arm, not a hardware-accurate Panda/Go2
combination.  The existing 12-action gait checkpoint sees only the original
leg joints in its original order; arm targets are provided independently.

USD imports are lazy, so importing the constants does not require Isaac Sim.
The builder also works with a standalone Python installation containing pxr.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


ARM_SCALE = 0.62
ARM_MASS_KG = 0.35
MOUNT_OFFSET = (0.08, 0.0, 0.12)
ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
HAND_BODY_NAME = "panda_hand"
HAND_TIP_OFFSET = (0.0, 0.0, 0.1034 * ARM_SCALE)
ARM_HOME = (0.0, -0.45, 0.0, -2.2, 0.0, 1.8, math.pi / 4.0)
# Keep the reset pose inside the float32-authored 0.0248 m joint limit.
FINGER_OPEN = 0.024
# Confirmed against the unmodified checkpoint runner's resolved action names.
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for joint in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
)
ASSET_VERSION = 1


def build_mounted_robot_asset(
    go2_path: str | Path,
    arm_path: str | Path,
    output_path: str | Path,
    *,
    arm_scale: float = ARM_SCALE,
    arm_mass_kg: float = ARM_MASS_KG,
    mount_offset: tuple[float, float, float] = MOUNT_OFFSET,
    mount_pitch_degrees: float = 0.0,
    wide_gripper: bool = False,
) -> Path:
    """Author a USD override layer with one floating, 21-DOF articulation.

    All Panda link translations, geometry, and joint anchor distances are
    scaled together.  There is no scale on a rigid body or its ancestors,
    avoiding the different scale conventions of USD and PhysX tensor poses.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    go2_path, arm_path, output_path = (
        Path(p).resolve() for p in (go2_path, arm_path, output_path)
    )
    for source in (go2_path, arm_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if arm_scale <= 0.0 or arm_mass_kg <= 0.0:
        raise ValueError("Arm scale and mass must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(go2_path.as_posix())
    root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
    root.SetCustomDataByKey("mounted_asset_version", ASSET_VERSION)
    root.SetCustomDataByKey("arm_scale", arm_scale)
    root.SetCustomDataByKey("arm_mass_kg", arm_mass_kg)
    root.SetCustomDataByKey("mount_pitch_degrees", mount_pitch_degrees)
    root.SetCustomDataByKey("arm_model", "lightweight_scaled_panda_surrogate")
    root.SetCustomDataByKey("wide_gripper", wide_gripper)

    base = stage.GetPrimAtPath("/Robot/base")
    if not base or not base.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Go2 asset must expose rigid body /base in Physics=physx")
    UsdPhysics.ArticulationRootAPI.Apply(base)

    arm = UsdGeom.Xform.Define(stage, "/Robot/Arm")
    arm.GetPrim().GetReferences().AddReference(arm_path.as_posix())
    arm.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    # The source rootJoint constrains link0 to the world.  Keeping it would pin
    # the whole robot to the floor or split the intended articulation.
    world_joint = stage.GetPrimAtPath("/Robot/Arm/rootJoint")
    if world_joint:
        world_joint.SetActive(False)
    arm.ClearXformOpOrder()
    arm.AddTranslateOp().Set(Gf.Vec3d(*mount_offset))
    arm.AddRotateYOp().Set(mount_pitch_degrees)

    bodies = [
        p for p in Usd.PrimRange(arm.GetPrim())
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(bodies) != 11:
        raise ValueError(f"Expected Panda's 11 rigid links; found {len(bodies)}")
    source_mass = sum(float(UsdPhysics.MassAPI(p).GetMassAttr().Get()) for p in bodies)
    mass_factor = arm_mass_kg / source_mass
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for body in bodies:
        xf = UsdGeom.Xformable(body)
        transform = xf.GetLocalTransformation()
        transform.SetTranslateOnly(transform.ExtractTranslation() * arm_scale)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="mounted").Set(transform)
        body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(False)

        # Scale the complete visual/collision subtree of each rigid link.
        # Geometry roots may be instanceable; their root xform is still editable.
        for child in body.GetChildren():
            if child.IsA(UsdGeom.Xformable) and not child.IsA(UsdPhysics.Joint):
                child_xf = UsdGeom.Xformable(child)
                child_transform = child_xf.GetLocalTransformation()
                child_transform = child_transform * Gf.Matrix4d().SetScale(arm_scale)
                child_xf.ClearXformOpOrder()
                child_xf.AddTransformOp(opSuffix="mounted").Set(child_transform)

        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass = float(mass_api.GetMassAttr().Get()) * mass_factor
        mass_api.CreateMassAttr(mass)
        # Source USD carries near-zero legacy inertias.  Use finite box-inertia
        # approximations of each scaled link, including its collision geometry.
        bbox.Clear()
        bounds = bbox.ComputeLocalBound(body).ComputeAlignedBox()
        extent = bounds.GetSize()
        inertia = Gf.Vec3f(*(
            max(1e-6, mass / 12.0 * (extent[(i + 1) % 3] ** 2 + extent[(i + 2) % 3] ** 2))
            for i in range(3)
        ))
        mass_api.CreateDiagonalInertiaAttr(inertia)
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        center = mass_api.GetCenterOfMassAttr().Get()
        if center is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(center) * arm_scale)

    for prim in Usd.PrimRange(arm.GetPrim()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for attr in (joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()):
            value = attr.Get()
            if value is not None:
                attr.Set(Gf.Vec3f(value) * arm_scale)
        if prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic = UsdPhysics.PrismaticJoint(prim)
            for attr in (prismatic.GetLowerLimitAttr(), prismatic.GetUpperLimitAttr()):
                attr.Set(float(attr.Get()) * arm_scale)
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            drive.CreateTargetPositionAttr(FINGER_OPEN)
            if wide_gripper:
                # Explicit custom long-stroke surrogate, not stock Panda.
                prismatic.CreateUpperLimitAttr(.05)

    mount = UsdPhysics.FixedJoint.Define(stage, "/Robot/joints/panda_mount_fixed")
    mount.CreateBody0Rel().SetTargets([base.GetPath()])
    mount.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/Arm/panda_link0")])
    mount.CreateLocalPos0Attr(Gf.Vec3f(*mount_offset))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    mount.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,1,0), mount_pitch_degrees).GetQuat()))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0))
    mount.CreateCollisionEnabledAttr(False)
    mount.CreateExcludeFromArticulationAttr(False)

    # A visible mounting pedestal belongs to the physical Go2 base link.
    deck = UsdGeom.Cube.Define(stage, "/Robot/base/PayloadMount")
    deck.CreateSizeAttr(1.0)
    deck.AddTranslateOp().Set(Gf.Vec3d(mount_offset[0], mount_offset[1], 0.085))
    deck.AddScaleOp().Set(Gf.Vec3f(0.16, 0.13, 0.07))
    deck.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.16, 0.21)])
    UsdPhysics.CollisionAPI.Apply(deck.GetPrim())

    friction = UsdShade.Material.Define(stage, "/Robot/GripMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(friction.GetPrim())
    material_api.CreateStaticFrictionAttr(1.5)
    material_api.CreateDynamicFrictionAttr(1.2)
    material_api.CreateRestitutionAttr(0.0)
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        prim = stage.GetPrimAtPath(f"/Robot/Arm/{finger}")
        if wide_gripper:
            # Flat parallel pads replace the stock tapered collision surfaces
            # for thin, side-lying boxes. This is a custom gripper geometry.
            for child in list(prim.GetChildren()):
                if child.IsInstanceable():
                    child.SetInstanceable(False)
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
            pad = UsdGeom.Mesh.Define(stage,str(prim.GetPath())+'/ParallelPad')
            sy=1. if finger=='panda_leftfinger' else -1.
            pad.CreatePointsAttr([Gf.Vec3f(x,y+sy*.006,z+.04) for x,y,z in (
                (-.0125,-.006,-.014),(.0125,-.006,-.014),(.0125,.006,-.014),(-.0125,.006,-.014),
                (-.0125,-.006,.014),(.0125,-.006,.014),(.0125,.006,.014),(-.0125,.006,.014))])
            pad.CreateFaceVertexCountsAttr([4]*6)
            pad.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
            pad.CreateSubdivisionSchemeAttr('none')
            pad.CreateDisplayColorAttr([Gf.Vec3f(.1,.55,.3)])
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(pad.GetPrim()).CreateApproximationAttr('convexHull')
            from pxr import PhysxSchema
            pad_collision=PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
            pad_collision.CreateContactOffsetAttr(.001)
            pad_collision.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            friction, UsdShade.Tokens.strongerThanDescendants, "physics"
        )

    stage.GetRootLayer().Export(str(output_path))
    validate_mounted_robot_asset(output_path)
    return output_path


def validate_mounted_robot_asset(path: str | Path) -> dict[str, Any]:
    """Reject disconnected roots, world joints, or incorrect active DOF counts."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    roots, movable, world_joints = [], [], []
    arm_mass = 0.0
    transform_cache = UsdGeom.XformCache()
    max_anchor_error = 0.0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()).startswith("/Robot/Arm/"):
            arm_mass += float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets0 = joint.GetBody0Rel().GetTargets()
            targets1 = joint.GetBody1Rel().GetTargets()
            if not targets0 or not targets1:
                world_joints.append(str(prim.GetPath()))
            else:
                anchors = []
                for target, position in (
                    (targets0[0], joint.GetLocalPos0Attr().Get()),
                    (targets1[0], joint.GetLocalPos1Attr().Get()),
                ):
                    transform = transform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target))
                    anchors.append(transform.Transform(Gf.Vec3d(position)))
                max_anchor_error = max(max_anchor_error, (anchors[0] - anchors[1]).GetLength())
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                movable.append(prim.GetName())
    expected = set(LEG_JOINT_NAMES + ARM_JOINT_NAMES + FINGER_JOINT_NAMES)
    if roots != ["/Robot/base"] or world_joints or set(movable) != expected or len(movable) != len(expected):
        raise ValueError(f"Invalid mounted articulation: roots={roots}, world={world_joints}, joints={movable}")
    if max_anchor_error > 1e-4:
        raise ValueError(f"Mounted joint anchors disagree by {max_anchor_error:.6f} m")
    return {
        "articulation_roots": roots,
        "dof_count": len(movable),
        "arm_mass_kg": arm_mass,
        "max_anchor_error_m": max_anchor_error,
    }


def ensure_mounted_robot_asset(project_dir: str | Path | None = None, *, ground_reach: bool = False, wide_gripper: bool = False, output_path: str | Path | None = None) -> Path:
    """Build once, rebuilding after builder or source USD changes."""
    project = Path(project_dir) if project_dir else Path(__file__).resolve().parents[1]
    go2 = project / "artifacts/go2_asset/go2.usd"
    panda = project / "artifacts/franka_asset/franka.usd"
    output = project / ("artifacts/go2_ground_panda.usda" if ground_reach else "artifacts/go2_mounted_panda.usda")
    if wide_gripper:
        output = project / 'artifacts/go2_ground_panda_wide.usda'
    if output_path is not None:
        output = Path(output_path).resolve()
    sources = (Path(__file__), go2, panda)
    if not output.is_file() or any(p.stat().st_mtime > output.stat().st_mtime for p in sources):
        build_mounted_robot_asset(go2, panda, output, arm_scale=.85 if ground_reach else ARM_SCALE,
                                  mount_pitch_degrees=30. if ground_reach else 0., wide_gripper=wide_gripper)
    return output


def configure_robot(env_cfg: Any, asset_path: str | Path) -> None:
    """Keep the 48-value locomotion observation and 12-action checkpoint ABI."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.managers import SceneEntityCfg

    robot = env_cfg.scene.robot
    robot.spawn.usd_path = str(Path(asset_path).resolve())
    robot.spawn.variants = {"Physics": "physx"}
    robot.spawn.articulation_props.enabled_self_collisions = False
    robot.spawn.articulation_props.solver_position_iteration_count = 12
    robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    robot.init_state.joint_pos.update(dict(zip(ARM_JOINT_NAMES, ARM_HOME)))
    robot.init_state.joint_pos["panda_finger_joint.*"] = FINGER_OPEN
    robot.actuators["mounted_arm"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        effort_limit_sim=18.0,
        velocity_limit_sim=1.6,
        stiffness=240.0,
        damping=14.0,
        armature=0.002,
    )
    robot.actuators["mounted_fingers"] = ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint[12]"],
        effort_limit_sim=15.0,
        velocity_limit_sim=0.12,
        stiffness=1200.0,
        damping=20.0,
        armature=0.001,
    )
    action = env_cfg.actions.joint_pos
    action.joint_names = list(LEG_JOINT_NAMES)
    action.preserve_order = True
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            continue
        for term_name in ("joint_pos", "joint_vel"):
            term = getattr(group, term_name, None)
            if term is not None:
                term.params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=list(LEG_JOINT_NAMES), preserve_order=True
                )
    # Repeatable demonstration resets must not randomize the arm into a table.
    reset = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset is not None:
        reset.params["position_range"] = (1.0, 1.0)
        reset.params["velocity_range"] = (0.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path)
    args = parser.parse_args()
    path = ensure_mounted_robot_asset(args.project_dir)
    print(f"MOUNTED_ROBOT_ASSET path={path} validation={validate_mounted_robot_asset(path)}")


if __name__ == "__main__":
    main()
