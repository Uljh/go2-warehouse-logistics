"""Convert the unscaled official Z1 and audit its imported physical data.

Run serially with C:/isaacsim/python.bat; this creates one Kit process.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def audit_import(usd_path, spec):
    import numpy as np
    from pxr import Usd, UsdPhysics, UsdGeom
    stage=Usd.Stage.Open(str(usd_path))
    root=stage.GetDefaultPrim()
    variants=root.GetVariantSets().GetVariantSet('Physics')
    if 'physx' in variants.GetVariantNames():
        variants.SetVariantSelection('physx')
    source=ET.parse(spec.urdf_path).getroot()
    expected={link.attrib['name']:link for link in source.findall('link') if link.find('inertial') is not None}
    bodies={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    comparisons={}
    for name,link in expected.items():
        prim=bodies.get(name)
        if prim is None:
            comparisons[name]=dict(passed=False,reason='missing imported rigid link')
            continue
        inertia=link.find('inertial/inertia').attrib
        matrix=np.array([[float(inertia['ixx']),float(inertia['ixy']),float(inertia['ixz'])],
                         [float(inertia['ixy']),float(inertia['iyy']),float(inertia['iyz'])],
                         [float(inertia['ixz']),float(inertia['iyz']),float(inertia['izz'])]])
        source_mass=float(link.find('inertial/mass').attrib['value'])
        api=UsdPhysics.MassAPI(prim)
        imported_mass=api.GetMassAttr().Get()
        diagonal=api.GetDiagonalInertiaAttr().Get()
        principal=sorted(float(v) for v in diagonal) if diagonal is not None else []
        eigenvalues=np.linalg.eigvalsh(matrix).tolist()
        mass_ok=imported_mass is not None and abs(float(imported_mass)-source_mass)<1e-5
        inertia_ok=(len(principal)==3 and min(principal)>0
                    and np.allclose(principal,eigenvalues,rtol=1e-3,atol=1e-7))
        comparisons[name]=dict(path=str(prim.GetPath()),source_mass_kg=source_mass,
            imported_mass_kg=imported_mass,source_principal_inertia=eigenvalues,
            imported_principal_inertia=principal,center_of_mass=list(api.GetCenterOfMassAttr().Get()),
            mass_check=bool(mass_ok),inertia_check=bool(inertia_ok),passed=bool(mass_ok and inertia_ok))
    joint_report={}
    for name,limits in spec.joint_limits.items():
        prim=joints.get(name)
        if prim is None:
            joint_report[name]=dict(passed=False,reason='missing imported revolute joint')
            continue
        joint=UsdPhysics.RevoluteJoint(prim)
        lower,upper=joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()
        expected_degrees=np.degrees(limits).tolist()
        joint_report[name]=dict(path=str(prim.GetPath()),limits_degrees=[lower,upper],
            expected_degrees=expected_degrees,
            passed=bool(np.allclose([lower,upper],expected_degrees,atol=1e-3)))
    return dict(usd_path=str(usd_path),default_prim=str(root.GetPath()),
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        rigid_body_count=len(bodies),revolute_joint_count=len(joints),links=comparisons,joints=joint_report,
        imported_total_mass_kg=sum(row.get('imported_mass_kg') or 0 for row in comparisons.values()),
        expected_total_mass_kg=spec.total_mass_kg,
        passed=bool(comparisons) and all(row['passed'] for row in comparisons.values())
            and all(row['passed'] for row in joint_report.values()),
        scope='asset schema/inertial/limit validation only; no locomotion or grasp success implied')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/real_arm_z1/usd')
    parser.add_argument('--fix-base',action='store_true')
    args=parser.parse_args()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    try:
        import omni.kit.app
        manager=omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf',True)
        from go2_logistics.z1_asset import prepare_z1_urdf,build_z1_usd_with_official_importer
        urdf,spec=prepare_z1_urdf(overwrite=True)
        usd=build_z1_usd_with_official_importer(urdf,args.output,fix_base=args.fix_base)
        report=audit_import(usd,spec)
        report.update(created_at=datetime.now().astimezone().isoformat(),fixed_base=args.fix_base,
            urdf_sha256=spec.generated_sha256,source_sha256=spec.source_sha256,
            usd_root_sha256=hashlib.sha256(usd.read_bytes()).hexdigest(),provenance=spec.provenance)
        destination=args.output/'import_validation.json'
        destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Z1_IMPORT_VALIDATION '+json.dumps(report),flush=True)
        if not report['passed']:
            raise RuntimeError('Imported Z1 differs from official mass/inertia/limits; see '+str(destination))
    finally:
        app.close()


if __name__=='__main__':
    main()
