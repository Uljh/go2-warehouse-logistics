"""Unitree Z1 asset preparation and Isaac importer configuration.

This module is intentionally independent of Isaac Kit.  The repository's
official ``unitree_ros`` checkout contains a generated Z1 URDF whose default
xacro argument omits the Unitree gripper.  :func:`prepare_z1_urdf` expands the
same official gripper links into a reproducible URDF layer, preserving every
mass, centre of mass, inertia, mesh and joint limit from the upstream files.

The actual USD conversion is kept as a small adapter around Isaac Sim's
official URDF importer.  It is only callable from an already running Kit
process; importing this module never starts Isaac.  No scale or mass override
is applied to the Z1 model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "assets" / "real_arm_z1" / "upstream"
DESCRIPTION = UPSTREAM / "unitree_ros" / "robots" / "z1_description"
DEFAULT_URDF = DESCRIPTION / "xacro" / "z1.urdf"
DEFAULT_XACRO = DESCRIPTION / "xacro" / "robot.xacro"
DEFAULT_CONST = DESCRIPTION / "xacro" / "const.xacro"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "real_arm_z1" / "z1_with_unitree_gripper.urdf"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_JOINT_NAME = "jointGripper"
ARM_LINK_NAMES = tuple(f"link0{i}" for i in range(7))
GRIPPER_LINK_NAMES = ("gripperStator", "gripperMover")
TIP_LINK_NAME = "gripperMover"
GRIPPER_STATOR_MASS_KG = 0.52603655
GRIPPER_MOVER_MASS_KG = 0.27621302


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in text.split())
    if len(values) != 3:
        raise ValueError(f"expected xyz/rpy vector, got {text!r}")
    return values


def _fmt(values: tuple[float, float, float]) -> str:
    return " ".join(f"{v:.12g}" for v in values)


def _origin(parent: ET.Element, *, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    return ET.SubElement(parent, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))


def _materialize_gripper(root: ET.Element) -> None:
    """Append the exact gripper block from the official xacro constants.

    ``robot.xacro`` is deliberately not run through an external ROS/xacro
    executable so the asset preparation remains usable on this Windows host.
    Values below are copied from the checked-out official ``const.xacro`` and
    the corresponding geometry/poses from ``robot.xacro``.
    """
    names = {link.attrib.get("name") for link in root.findall("link")}
    joints = {joint.attrib.get("name") for joint in root.findall("joint")}
    if "gripperMover" in names or GRIPPER_JOINT_NAME in joints:
        return

    fixed = ET.SubElement(root, "joint", name="gripperStator", type="fixed")
    _origin(fixed, xyz=(0.051, 0.0, 0.0))
    ET.SubElement(fixed, "parent", link="link06")
    ET.SubElement(fixed, "child", link="gripperStator")

    stator = ET.SubElement(root, "link", name="gripperStator")
    visual = ET.SubElement(stator, "visual")
    ET.SubElement(visual, "geometry").append(ET.Element("mesh", filename="package://z1_description/meshes/visual/z1_GripperStator.dae", scale="1 1 1"))
    collision = ET.SubElement(stator, "collision")
    ET.SubElement(collision, "geometry").append(ET.Element("mesh", filename="package://z1_description/meshes/collision/z1_GripperStator.STL", scale="1 1 1"))
    inertial = ET.SubElement(stator, "inertial")
    _origin(inertial, xyz=(0.04764427, -0.00035819, -0.00249162))
    ET.SubElement(inertial, "mass", value=str(GRIPPER_STATOR_MASS_KG))
    ET.SubElement(inertial, "inertia", ixx="0.00038683", ixy="0.00000359", ixz="-0.00007662", iyy="0.00068614", iyz="-0.00000209", izz="0.00066293")

    moving_joint = ET.SubElement(root, "joint", name=GRIPPER_JOINT_NAME, type="revolute")
    _origin(moving_joint, xyz=(0.049, 0.0, 0.0))
    ET.SubElement(moving_joint, "parent", link="gripperStator")
    ET.SubElement(moving_joint, "child", link="gripperMover")
    ET.SubElement(moving_joint, "axis", xyz="0 1 0")
    ET.SubElement(moving_joint, "dynamics", damping="1.0", friction="1.0")
    ET.SubElement(moving_joint, "limit", effort="30.0", velocity="3.1415", lower=str(-math.pi / 2.0), upper="0.0")

    mover = ET.SubElement(root, "link", name="gripperMover")
    visual = ET.SubElement(mover, "visual")
    ET.SubElement(visual, "geometry").append(ET.Element("mesh", filename="package://z1_description/meshes/visual/z1_GripperMover.dae", scale="1 1 1"))
    collision = ET.SubElement(mover, "collision")
    ET.SubElement(collision, "geometry").append(ET.Element("mesh", filename="package://z1_description/meshes/collision/z1_GripperMover.STL", scale="1 1 1"))
    inertial = ET.SubElement(mover, "inertial")
    _origin(inertial, xyz=(0.01320633, 0.00476708, 0.00380534))
    ET.SubElement(inertial, "mass", value=str(GRIPPER_MOVER_MASS_KG))
    ET.SubElement(inertial, "inertia", ixx="0.00017716", ixy="-0.00001683", ixz="0.00001786", iyy="0.00026787", iyz="-0.00000262", izz="0.00035728")


def _indent(element: ET.Element, level: int = 0) -> None:
    # ET.indent is unavailable on a few Python versions used by Isaac tools.
    pad = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            _indent(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = "\n" + (level - 1) * "  "


@dataclass(frozen=True)
class Z1AssetSpec:
    urdf_path: str
    package_name: str
    package_path: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_name: str | None
    tip_link: str
    arm_mass_kg: float
    gripper_mass_kg: float
    total_mass_kg: float
    joint_limits: dict[str, tuple[float, float]]
    joint_efforts_nm: dict[str, float]
    source_sha256: str
    generated_sha256: str
    provenance: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["arm_joint_names"] = list(self.arm_joint_names)
        result["joint_limits"] = {k: list(v) for k, v in self.joint_limits.items()}
        return result


def _spec_from_urdf(path: Path, *, source_sha256: str, generated_sha256: str, provenance: dict[str, str]) -> Z1AssetSpec:
    root = ET.parse(path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    def mass(name: str) -> float:
        inertial = links[name].find("inertial/mass")
        return float(inertial.attrib["value"]) if inertial is not None else 0.0
    arm_mass = sum(mass(name) for name in ARM_LINK_NAMES)
    gripper_mass = sum(mass(name) for name in GRIPPER_LINK_NAMES if name in links)
    arm_joints = tuple(name for name in ARM_JOINT_NAMES if name in joints)
    gripper_joint = GRIPPER_JOINT_NAME if GRIPPER_JOINT_NAME in joints else None
    selected = arm_joints + ((gripper_joint,) if gripper_joint else ())
    limits = {}
    efforts = {}
    for name in selected:
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"missing limit for {name}")
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        efforts[name] = float(limit.attrib.get("effort", "nan"))
    return Z1AssetSpec(
        urdf_path=str(path.resolve()), package_name="z1_description", package_path=str(DESCRIPTION.resolve()),
        arm_joint_names=arm_joints, gripper_joint_name=gripper_joint, tip_link=TIP_LINK_NAME if gripper_joint else "link06",
        arm_mass_kg=arm_mass, gripper_mass_kg=gripper_mass, total_mass_kg=arm_mass + gripper_mass,
        joint_limits=limits, joint_efforts_nm=efforts, source_sha256=source_sha256,
        generated_sha256=generated_sha256, provenance=provenance,
    )


def prepare_z1_urdf(output_path: str | Path = DEFAULT_OUTPUT, *, source_urdf: str | Path = DEFAULT_URDF, overwrite: bool = False) -> tuple[Path, Z1AssetSpec]:
    """Create a reproducible official Z1+Unitree-gripper URDF and manifest.

    The source remains untouched.  ``package://z1_description`` is resolved by
    the Isaac importer through :func:`isaac_import_config` below.
    """
    source = Path(source_urdf).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        spec = _spec_from_urdf(output, source_sha256=_sha256(source), generated_sha256=_sha256(output), provenance={"source_urdf": str(source), "source_xacro": str(DEFAULT_XACRO), "source_constants": str(DEFAULT_CONST)})
        return output, spec
    root = ET.parse(source).getroot()
    root = copy.deepcopy(root)
    _materialize_gripper(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    provenance = {"source_urdf": str(source), "source_xacro": str(DEFAULT_XACRO), "source_constants": str(DEFAULT_CONST), "upstream_repo": "https://github.com/unitreerobotics/unitree_ros.git", "upstream_revision": "8ebecd883a842a4631fcfa98933fae586aeb6709", "gripper_expansion": "official robot.xacro UnitreeGripper=true block; no mass/inertia scaling"}
    spec = _spec_from_urdf(output, source_sha256=_sha256(source), generated_sha256=_sha256(output), provenance=provenance)
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps(spec.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output, spec


def isaac_import_config(urdf_path: str | Path, output_dir: str | Path, *, fix_base: bool = True) -> dict[str, Any]:
    """Return keyword arguments for Isaac Sim's official ``URDFImporterConfig``.

    This is data-only and does not import Isaac modules.  ``fix_base`` should
    be true for standalone arm validation; the Go2 mounted builder must use a
    floating Z1 conversion and add its own fixed joint to the Go2 base.
    """
    urdf = Path(urdf_path).resolve()
    return {
        "urdf_path": os.fspath(urdf), "usd_path": os.fspath(Path(output_dir).resolve()),
        "merge_fixed_joints": False, "merge_mesh": False,
        "collision_from_visuals": False, "collision_type": "Convex Hull",
        "allow_self_collision": True, "ros_package_paths": [{"name": "z1_description", "path": os.fspath(DESCRIPTION.resolve())}],
        "robot_type": "Manipulator", "fix_base": bool(fix_base), "link_density": None,
        "joint_drive_type": "force", "joint_target_type": "position", "override_joint_stiffness": None,
        "override_joint_damping": None, "run_asset_transformer": True, "run_multi_physics_conversion": True,
        "debug_mode": False,
    }


def build_z1_usd_with_official_importer(urdf_path: str | Path, output_dir: str | Path, *, fix_base: bool = True) -> Path:
    """Run the Isaac Sim URDF importer from an existing Kit process.

    It is intentionally a late import so unit tests and asset preparation do
    not launch Isaac.  The importer preserves URDF inertial data because no
    density or mass override is supplied.
    """
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    kwargs = isaac_import_config(urdf_path, output_dir, fix_base=fix_base)
    generated = URDFImporter(URDFImporterConfig(**kwargs)).import_urdf()
    if not generated:
        raise RuntimeError("Isaac URDF importer returned no USD path")
    return Path(generated).resolve()


if __name__ == "__main__":
    path, spec = prepare_z1_urdf(overwrite=True)
    print(json.dumps({"urdf": str(path), "spec": spec.to_json()}, ensure_ascii=False, indent=2))
