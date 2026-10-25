"""Adapt NVIDIA's furnished indoor warehouse to the reachable task layout.

Shell meshes retain authored collision. Original furniture intersecting the
task map is removed to make room for low numbered shelves, while perimeter
storage is retained. This is an adapted simulation asset, not a digital twin
of a measured real warehouse.
"""
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'artifacts/isaac_assets/Isaac/Environments/Simple_Warehouse/warehouse.usd'
OUTPUT = PROJECT/'artifacts/indoor_warehouse.usda'
REMOVED_PREFIXES = ('SM_Rack', 'SM_CardBox', 'SM_Palette', 'SM_Pushcart', 'KLT_Bins',
                    'SM_FloorDecal', 'S_AisleSign', 'S_Barcode', 'SM_Sign')


def build_indoor_warehouse():
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf
    manifest=PROJECT/'artifacts/isaac_assets/warehouse.source.json'
    if not manifest.is_file() or not SOURCE.is_file():
        raise RuntimeError('Warehouse asset cache incomplete. Run Isaac Python -m go2_logistics.prepare_warehouse_asset first.')
    if OUTPUT.exists():
        OUTPUT.unlink()
    stage=Usd.Stage.CreateNew(str(OUTPUT))
    root=UsdGeom.Xform.Define(stage,'/IndoorWarehouse')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    root.GetPrim().GetReferences().AddReference(str(SOURCE),'/Root')
    # Expand width for the existing west logistics spine; keep floor height and
    # unit scale in z. Translation/scale are scene authoring, never body control.
    xform=UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(-.5,1.,0.))
    xform.AddScaleOp().Set(Gf.Vec3f(1.4,1.,1.))
    removed=[]
    retained=[]
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    for child in root.GetPrim().GetChildren():
        if child.GetName().startswith(REMOVED_PREFIXES):
            box=bounds.ComputeWorldBound(child).ComputeAlignedBox()
            low,high=box.GetMin(),box.GetMax()
            # Keep original furnished storage beyond the operating region.
            # Remove only items intersecting the existing task map, including
            # trunk/arm clearance around its outermost routes.
            intersects=not (high[0]<-8.6 or low[0]>8.6 or high[1]<-2. or low[1]>11.5)
            if intersects:
                child.SetActive(False)
                removed.append(child.GetName())
            else:
                retained.append(child.GetName())
    # Original baked training navmesh is not the project's sensing map.
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() in ('NavMesh', 'NavMeshBoundsVolume'):
            prim.SetActive(False)
        if prim.HasAPI(UsdLux.LightAPI):
            light=UsdLux.LightAPI(prim)
            light.CreateNormalizeAttr().Set(True)
            light.CreateIntensityAttr().Set(3000. if prim.GetParent()!=root.GetPrim() else 10000.)
        # Avoid two coincident collision floors with different friction. The
        # official floor remains rendered; Isaac's ground supplies collision.
        if 'floor' in str(prim.GetPath()).lower() and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    # The authored fixtures give good local highlights but leave the low
    # picking area dark in RTX. Add an indoor area source below the roof;
    # increasing an outdoor dome would be blocked by the building shell.
    fill=UsdLux.RectLight.Define(stage,'/IndoorWarehouse/TaskAreaFill')
    fill.CreateWidthAttr().Set(12.)
    fill.CreateHeightAttr().Set(13.)
    fill.CreateNormalizeAttr().Set(True)
    fill.CreateIntensityAttr().Set(120000.)
    fill.CreateColorAttr().Set(Gf.Vec3f(1.,.96,.90))
    # Adapter coordinates: the parent already supplies x scale/translation.
    UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(.36,3.,5.2))
    stage.GetRootLayer().Save()
    report=dict(source=str(SOURCE),source_manifest=str(manifest),output=str(OUTPUT),
                translation_m=[-.5,1.,0.],scale=[1.4,1.,1.],removed_furniture=removed,retained_furniture=retained,
                floor_collision='Isaac ground only; official floor is rendered',
                lighting='normalized indoor fixtures + below-roof task area fill (120000 normalized intensity)',
                task_shelf_map='sorting_layout.numbered_shelves',
                note='Official simulation warehouse shell adapted to 30 low reachable shelves; physics acceptance recorded separately.')
    (PROJECT/'artifacts/indoor_warehouse.adapter.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'INDOOR_WAREHOUSE built removed_furniture={len(removed)} output={OUTPUT}',flush=True)
    return OUTPUT


def configure_indoor_warehouse(cfg):
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim
    if not OUTPUT.is_file():
        raise RuntimeError('Missing indoor_warehouse.usda. Run Isaac Python -m go2_logistics.indoor_warehouse before launching.')
    cfg.scene.indoor_warehouse=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/IndoorWarehouse',
        spawn=sim.UsdFileCfg(usd_path=str(OUTPUT)))
    # Replace the simple wall cuboids with the official architectural meshes.
    for name in ('north','west','east'):
        if hasattr(cfg.scene,'warehouse_'+name):
            setattr(cfg.scene,'warehouse_'+name,None)


if __name__=='__main__':
    build_indoor_warehouse()
