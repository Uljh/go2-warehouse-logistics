"""Scene and factory hooks for the experimental original-mass Z1 backend."""
from pathlib import Path
import json
import math

from .mounted_z1_asset import configure_z1_robot, z1_gripper_body_path, build_mounted_z1_asset

ROOT = Path(__file__).resolve().parents[1]


def warehouse_carry_lift(value=.44):
    """Bound the opt-in clearance experiment; physical preflight still applies."""
    height=float(value)
    if not math.isfinite(height) or not .44<=height<=.52:
        raise ValueError('Z1 warehouse carry lift must be finite in [.44,.52] metres')
    return height


def warehouse_approach_seconds(value=8.):
    """Opt-in unloaded-approach timing; contact/lift/release guards are unchanged."""
    seconds=float(value)
    if not math.isfinite(seconds) or not 4.<=seconds<=8.:
        raise ValueError('Z1 warehouse approach seconds must be finite in [4,8]')
    return seconds


def warehouse_close_seconds(value=24.):
    """Bound a timing-only experiment; opposing contact is still measured."""
    seconds=float(value)
    if not math.isfinite(seconds) or not 8.<=seconds<=24.:
        raise ValueError('Z1 warehouse close seconds must be finite in [8,24]')
    return seconds


def warehouse_grasp_heading_reference(value=''):
    """An optional equivalent-side reference, leaving observed box pose intact."""
    if value is None or value=='':return None
    degrees=float(value)
    if not math.isfinite(degrees) or not -180.<=degrees<=180.:
        raise ValueError('Z1 grasp heading reference must be finite in [-180,180] degrees')
    return math.radians(degrees)


def warehouse_jaw_sample_x(value=.15):
    """A motor-target geometry experiment; original jaw meshes are unchanged."""
    reference=float(value)
    if not math.isfinite(reference) or not .14 <= reference <= .16:
        raise ValueError('Z1 jaw insertion reference must be finite in [.14,.16] metres')
    return reference


def configure_z1_warehouse(cfg, asset_path):
    """Keep contact control at 200 Hz, while the fleet actor stays at 50 Hz."""
    from isaaclab_physx.sensors import ContactSensorCfg
    configure_z1_robot(cfg, asset_path, arm_stiffness=800., arm_damping=34.,
                       gripper_stiffness=8., gripper_damping=.4, gripper_open=-1.45)
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.sim.dt = .005
    cfg.decimation = 1
    cfg.sim.render_interval = 20
    if getattr(cfg.scene, 'parcel_camera', None) is not None:
        cfg.scene.parcel_camera.update_latest_camera_pose = True
    for side, link in (('left', 'gripperStator'), ('right', 'gripperMover')):
        setattr(cfg.scene, side+'_grip_contact', ContactSensorCfg(
            prim_path=z1_gripper_body_path('{ENV_REGEX_NS}/Robot', link),
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Parcel'], update_period=0.,
            track_friction_forces=True, max_contact_data_count_per_prim=128))


def ensure_z1_warehouse_asset(destination, *, import_report=None):
    """Build a fresh reference layer and explicitly activate both nested jaws."""
    import isaaclab.sim as sim_utils
    from pxr import Usd
    from .z1_asset import prepare_z1_urdf
    from .import_z1_isaac import audit_import
    report_path = Path(import_report or ROOT/'artifacts/real_arm_z1/usd/import_validation.json')
    imported = json.loads(report_path.read_text(encoding='utf-8'))
    if not imported.get('passed') or imported.get('fixed_base'):
        raise RuntimeError('Z1 warehouse requires a validated original floating-base import')
    report = build_mounted_z1_asset(ROOT/'artifacts/go2_asset/go2.usd', imported['usd_path'], destination,
                                    gripper_collision_mode='convexDecomposition')
    # Generic recursive activation stops at the first rigid body in the nested
    # importer hierarchy. Author jaw contact-report APIs on the saved layer.
    stage = Usd.Stage.Open(str(destination))
    for name in ('gripperStator', 'gripperMover'):
        sim_utils.activate_contact_sensors(report['body_paths'][name], stage=stage)
    stage.GetRootLayer().Save()
    _, spec = prepare_z1_urdf(overwrite=False)
    physical = audit_import(destination, spec)
    if not physical['passed']:
        raise RuntimeError('Mounted Z1 no longer matches original mass, inertia or joint limits')
    result = dict(mount=report, physical_asset_audit=physical,
                  scope='asset and sensor configuration only; task not physically accepted')
    Path(destination).with_suffix('.audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
