"""Thirty reachable numbered shelves and a camera-readable parcel label."""
from pathlib import Path
import math
from .sorting_layout import numbered_shelves, ParcelCode, SHELF_DECK_SIZE


def configure_sorting_scene(cfg, pickup, ground_bin):
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils

    cfg.scene.dropoff_table = None
    camera = cfg.scene.parcel_camera
    camera.height = camera.width = 1024
    camera.spawn.focal_length = 75.
    height = .80 if ground_bin else 1.20
    # A vertical camera could end up inside the parked gripper. Observe from
    # the far side of the bin, keeping the label visible across docking error.
    camera.offset.pos = (pickup.x+.35, pickup.y, height)
    tilt = math.atan2(.35, height-(.06 if ground_bin else .46))
    camera.offset.rot = (math.cos(tilt/2), 0., -math.sin(tilt/2), 0.)
    for shelf in numbered_shelves():
        x,y,z = shelf.surface
        parts = [("Deck", (x,y,z-SHELF_DECK_SIZE[2]/2), SHELF_DECK_SIZE)]
        for index, (dx,dy) in enumerate(((.12,-.17),(.12,.17),(-.12,-.17),(-.12,.17))):
            parts.append((f"Post{index}", (x+dx,y+dy,(z-.03)/2), (.025,.025,z-.03)))
        for part,position,size in parts:
            setattr(cfg.scene, f"shelf_{shelf.number}_{part}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Shelf{shelf.number:02d}/{part}",
                spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12+.12*shelf.row,.34,.44))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position)))


def _textured_quad(stage, path, image_path, size, position, rotation=None):
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    mesh = UsdGeom.Mesh.Define(stage, path)
    w,h = size
    mesh.CreatePointsAttr([(-w/2,-h/2,0),(w/2,-h/2,0),(w/2,h/2,0),(-w/2,h/2,0)])
    mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateDoubleSidedAttr(True)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotation is not None:
        mesh.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set([(0,0),(1,0),(1,1),(0,1)])
    material = UsdShade.Material.Define(stage, path + "/Material")
    surface = UsdShade.Shader.Define(stage, path + "/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.)
    tex = UsdShade.Shader.Define(stage, path + "/Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(image_path).resolve()))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, path + "/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def author_parcel_label(stage, parcel_code, assets_dir, parcel_path='/World/envs/env_0/Parcel', all_faces=False):
    import cv2
    ParcelCode.parse(parcel_code)
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    qr = cv2.QRCodeEncoder_create().encode(parcel_code)
    qr = cv2.copyMakeBorder(qr, 4,4,4,4,cv2.BORDER_CONSTANT,value=255)
    qr_path = assets_dir / (parcel_code + ".png")
    cv2.imwrite(str(qr_path), cv2.resize(qr,(512,512),interpolation=cv2.INTER_NEAREST))
    if not stage.GetPrimAtPath(parcel_path+'/CodeLabel').IsValid():
        _textured_quad(stage, parcel_path+"/CodeLabel", qr_path, (.032,.032), (0,0,.0403))
    if all_faces:
        for name, position, rotation in (
            ('Bottom',(0,0,-.0403),(180,0,0)),
            ('Left',(0,.0178,0),(90,0,0)),
            ('Right',(0,-.0178,0),(90,0,0)),
            ('Front',(.0503,0,0),(0,90,0)),
            ('Back',(-.0503,0,0),(0,90,0)),
        ):
            _textured_quad(stage,parcel_path+'/CodeLabel'+name,qr_path,(.032,.032),position,rotation)


def author_sorting_labels(stage, parcel_code, assets_dir):
    from PIL import Image, ImageDraw, ImageFont
    assets_dir=Path(assets_dir)
    author_parcel_label(stage,parcel_code,assets_dir)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 86)
    for shelf in numbered_shelves():
        image = Image.new("RGB", (320,140), "#e8f0ee")
        draw = ImageDraw.Draw(image)
        draw.text((160,70), str(shelf.number), anchor="mm", font=font, fill="#152d3d")
        file = assets_dir / f"shelf_{shelf.number:02d}.png"
        image.save(file)
        x,y,z = shelf.surface
        _textured_quad(stage, f"/World/envs/env_0/Shelf{shelf.number:02d}/Number", file, (.25,.11), (x+.14,y,z+.08), (90,0,-90))


def read_rendered_code(rgb):
    """Decode only camera pixels; there is no configured ID fallback."""
    return read_rendered_label(rgb)['parcel_code']


def read_rendered_label(rgb):
    """Return identity and corners in the ORIGINAL image coordinates."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.QRCodeDetector()
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    height,width=gray.shape
    variants=[(gray,lambda p:p),(binary,lambda p:p),
        (cv2.rotate(gray,cv2.ROTATE_90_CLOCKWISE),lambda p:np.column_stack((p[:,1],height-1-p[:,0]))),
        (cv2.flip(gray,1),lambda p:np.column_stack((width-1-p[:,0],p[:,1])))]
    for image,restore in variants:
        text, points, _ = detector.detectAndDecode(image)
        if text:
            return dict(parcel_code=str(ParcelCode.parse(text)),corners_px=restore(points.reshape(4,2)).tolist())
    # The conventional detector can select the wrong corner on a tilted,
    # antialiased rendered code. ArUco's alternative QR corner detector reads
    # the same pixels; it receives no registry, expected ID or destination.
    aruco_factory=getattr(cv2,'QRCodeDetectorAruco',None)
    if aruco_factory is not None:
        aruco=aruco_factory()
        for image in (gray,binary):
            text,points,_=aruco.detectAndDecode(image)
            if text:
                return dict(parcel_code=str(ParcelCode.parse(text)),corners_px=points.reshape(4,2).tolist())
    # Rendered module edges can fall unfavorably on the pixel grid. A small
    # deterministic image pyramid retries sampling, without supplying an ID.
    for scale,interpolation in [(s,cv2.INTER_LINEAR) for s in (.75,1.25,1.5)]+[(1.25,cv2.INTER_NEAREST)]:
        # The recorded righted-box label decodes at 1.25x nearest sampling;
        # linear interpolation blurs its already antialiased module edges.
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
        text, points, _ = detector.detectAndDecode(resized)
        if text:
            # OpenCV resize maps pixel centers, including a half-pixel offset.
            ratios=np.array([resized.shape[1]/width,resized.shape[0]/height])
            corners=(points.reshape(4,2)+.5)/ratios-.5
            return dict(parcel_code=str(ParcelCode.parse(text)),corners_px=corners.tolist())
    raise ValueError("No valid warehouse QR code visible in rendered image")
