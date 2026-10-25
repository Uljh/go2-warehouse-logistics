"""Procedural warehouse integration layer; task poses stay in sorting_layout.

Fixtures carry real collision geometry. Painted traffic markings and signs
are visual only. The scene configuration is never supplied to the planner.
"""
from .sorting_layout import numbered_shelves


def configure_warehouse(cfg):
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim
    def cuboid(name,position,size,color,collision=True):
        setattr(cfg.scene,'warehouse_'+name,AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/Warehouse/'+name,
            spawn=sim.CuboidCfg(size=size,
                collision_props=sim.CollisionPropertiesCfg() if collision else None,
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=color)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position)))
    # Enclose the existing reachable task map, leaving cameras an open roof.
    for name,pos,size in [('north',(0,11.5,1.5),(18,.15,3)),
                          ('west',(-8.7,4.5,1.5),(.15,14,3)),
                          ('east',(8.7,4.5,1.5),(.15,14,3))]:
        cuboid(name,pos,size,(.48,.53,.57))
    # Rear rack uprights retain the reachable low picking deck and west access.
    for shelf in numbered_shelves():
        x,y,z=shelf.surface
        for side in (-1,1):
            cuboid(f'rack_{shelf.number}_{side+1}',(x+side*.17,y+.65,.70),(.035,.035,1.40),(.12,.22,.32))
        cuboid(f'rack_beam_{shelf.number}',(x,y+.65,1.38),(.37,.035,.035),(.95,.46,.08))
        cuboid(f'rack_rear_deck_{shelf.number}',(x,y+.40,.40),(.30,.53,.035),(.12,.22,.32))
    for row in range(3):
        y=2.15+row*3
        for side in (-1,1):
            cuboid(f'lane_{row}_{side+1}',(.1,y+side*.38,.002),(13.8,.025,.003),(.95,.72,.12),False)
    # Storage blocks make the receiving-side perimeter legible, away from docks.
    for index,(x,y) in enumerate(((6.8,-.7),(7.1,4.6),(7.1,7.6),(-7.7,10.3))):
        cuboid(f'pallet_{index}',(x,y,.075),(.65,.65,.15),(.45,.29,.12))
        cuboid(f'carton_{index}',(x,y,.40),(.55,.55,.50),(.60,.42,.23))
    cfg.scene.warehouse_camera=cfg.scene.mission_camera.copy()
    cfg.scene.warehouse_camera.prim_path='{ENV_REGEX_NS}/WarehouseCamera'


def author_warehouse(stage,assets_dir):
    from pathlib import Path
    from PIL import Image,ImageDraw,ImageFont
    from .sorting_scene import _textured_quad
    assets_dir=Path(assets_dir);assets_dir.mkdir(parents=True,exist_ok=True)
    signs=[('receiving','RECEIVING / CONVEYOR',(5,-.6,.006),(2.4,.45)),
           ('staging','TEMPORARY BUFFER',(4,-.15,.006),(2.4,.35)),
           ('obstacle','OBSTACLE / REPLAN',(-1,-.35,.006),(2.6,.35))]
    signs += [(f'row{i}',f'ROW {i+1} / {i*10+1:02d}-{i*10+10:02d}',(-3.5,3.9+i*3,.006),(3,.4)) for i in range(3)]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',42)
    for name,text,pos,size in signs:
        img=Image.new('RGB',(800,120),'#243442');draw=ImageDraw.Draw(img)
        draw.text((400,60),text,font=font,anchor='mm',fill='#f5cf68')
        path=assets_dir/(name+'.png');img.save(path)
        _textured_quad(stage,'/World/envs/env_0/Warehouse/Sign_'+name,path,size,pos)
