"""RGB-D red parcel localization from an overhead warehouse camera.

Uses rendered color/depth and camera calibration; never reads a parcel prim or
rigid-body pose. The baseline assumes an 8 cm upright height; visual-geometry
mode fits a horizontal face and matches known cuboid dimensions for height and
grasp yaw. This controlled-scene detector is not a learned VLA.
"""
from pathlib import Path
import json


class RedParcelPerception:
    def __init__(self, env, arm, output_dir, decode_code=False):
        self.env = env.unwrapped
        self.camera = self.env.scene["parcel_camera"]
        self.arm = arm
        self.output_dir = Path(output_dir)
        self.last_detection = None
        self.decode_code = decode_code
        self.estimate_geometry = False
        self.geometry_max_tilt_deg = 0.
        self.archive_observations = False
        self.last_archive = None
        self._observation_index = 0

    def locate(self, object_id):
        return self._locate(object_id,decode_identity=self.decode_code)

    def locate_geometry(self):
        """Depth-only observation, deliberately without a parcel identity."""
        return self._locate('box_red',decode_identity=False)

    def inspect_label(self):
        """Read a side label with depth, without claiming a grasp pose."""
        import numpy as np
        from PIL import Image
        from .sorting_scene import read_rendered_label
        from .recovery_views import label_world_center
        for _ in range(4):
            self.env.sim.render()
            self.camera.update(self.env.step_dt,force_recompute=True)
        rgb=self.camera.data.output['rgb'][0,...,:3].cpu().numpy()
        depth=self.camera.data.output['distance_to_image_plane'][0].cpu().numpy().squeeze()
        self.output_dir.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rgb).save(self.output_dir/'label_rgb.png')
        np.save(self.output_dir/'label_depth.npy',depth)
        try:
            observed=read_rendered_label(rgb)
            observed['center_world']=label_world_center(observed['corners_px'],depth,
                self.camera.data.intrinsic_matrices[0].cpu().numpy(),
                self.camera.data.pos_w[0].cpu().numpy(),self.camera.data.quat_w_ros[0].cpu().numpy())
        except ValueError as error:
            (self.output_dir/'label_failure.json').write_text(json.dumps(dict(reason=str(error))),encoding='utf-8')
            return None
        observed.update(source='fresh_rendered_label_rgb_depth',
            intrinsics=self.camera.data.intrinsic_matrices[0].cpu().tolist(),
            camera_position=self.camera.data.pos_w[0].cpu().tolist(),
            camera_quaternion_ros=self.camera.data.quat_w_ros[0].cpu().tolist())
        (self.output_dir/'label_observation.json').write_text(json.dumps(observed,indent=2),encoding='utf-8')
        return observed

    def _locate(self, object_id, *, decode_identity):
        import numpy as np
        import torch
        from PIL import Image
        from scipy.ndimage import label
        from isaaclab.utils.math import quat_apply

        self.last_detection = None
        self.last_archive = None
        self.arm.detected_position = None
        self.arm.detected_geometry = None
        if object_id != "box_red" and not self.decode_code:
            return False
        # Render several fresh frames after navigation. Physics need not move.
        for _ in range(4):
            self.env.sim.render()
            self.camera.update(self.env.step_dt, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        depth = self.camera.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(self.output_dir / "pickup_rgb.png")
        np.save(self.output_dir / "pickup_depth.npy", depth)
        # Persist calibration even for a rejected observation, so failures can
        # be replayed from actual pixels/depth without reconstructing a pose.
        (self.output_dir/'observation.json').write_text(json.dumps(dict(
            camera_position=self.camera.data.pos_w[0].cpu().tolist(),
            camera_quaternion_ros=self.camera.data.quat_w_ros[0].cpu().tolist(),
            intrinsics=self.camera.data.intrinsic_matrices[0].cpu().tolist()),indent=2),encoding='utf-8')
        colors = rgb.astype(np.float32)
        # Warehouse fill lights desaturate the red top face toward orange.
        # Require both red dominance and chroma, allowing that exposure shift.
        mask = (colors[..., 0] > 65) & (colors[..., 0] > colors[..., 1] * 1.25) & (colors[..., 0] > colors[..., 2] * 1.6)
        mask &= np.isfinite(depth) & (depth > .1) & (depth < 3)
        components, count = label(mask)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if count == 0 or sizes.max() < 12:
            Image.fromarray((mask * 255).astype(np.uint8)).save(self.output_dir / "pickup_red_mask.png")
            print("RGBD_PERCEPTION failed=no_red_parcel", flush=True)
            self.arm.detected_position = None
            return False
        k = self.camera.data.intrinsic_matrices[0].cpu().numpy()
        def project(component):
            rows,cols=np.nonzero(components==component)
            z=depth[rows,cols]
            points=np.stack(((cols-k[0,2])*z/k[0,0],(rows-k[1,2])*z/k[1,1],z),axis=1)
            points=torch.tensor(points,dtype=torch.float32,device=self.env.device)
            quat=self.camera.data.quat_w_ros[0].expand(len(points),4)
            world=(quat_apply(quat,points)+self.camera.data.pos_w[0]).cpu().numpy()
            return component,rows,cols,world
        geometry=None;selection=None
        if self.estimate_geometry:
            from .parcel_geometry import select_visible_box
            try:
                candidates=np.flatnonzero(sizes>=30)
                if len(candidates)>16:
                    raise ValueError('Too many red components for isolated-parcel geometry')
                fitter=None
                if self.geometry_max_tilt_deg:
                    from functools import partial
                    from .parcel_plane_geometry import fit_sloped_visible_box
                    fitter=partial(fit_sloped_visible_box,max_tilt_deg=self.geometry_max_tilt_deg)
                selected,selection=select_visible_box((project(i) for i in candidates),rgb.shape,fit_function=fitter)
                component,rows,cols,world,geometry=selected
            except ValueError as exc:
                (self.output_dir/'geometry_failure.json').write_text(json.dumps(dict(reason=str(exc))),encoding='utf-8')
                print(f'RGBD_PERCEPTION failed={exc}',flush=True)
                return False
            self.arm.detected_geometry = geometry
            center=geometry['center_world']
        else:
            component,rows,cols,world=project(sizes.argmax())
            center=visible_parcel_center(world)
        mask=components==component
        self.arm.detected_position = tuple(float(x) for x in center)
        self.last_detection = dict(object_id=object_id, center_world=list(self.arm.detected_position), pixels=len(rows), source="rendered_rgb_depth",
                                   camera_position=self.camera.data.pos_w[0].cpu().tolist(),
                                   camera_quaternion_ros=self.camera.data.quat_w_ros[0].cpu().tolist(),
                                   intrinsics=k.tolist())
        self.last_detection['geometry'] = geometry
        if selection is not None:self.last_detection['geometry_selection']=selection
        if decode_identity:
            from .sorting_scene import read_rendered_code
            try:
                decode_rgb = rgb
                if geometry is not None:
                    # Associate identity with the fitted top face. Decoding the
                    # entire frame can bind a lower/neighbor parcel's label.
                    from .parcel_geometry import visible_face_mask
                    top = visible_face_mask(world,geometry)
                    rr,cc = rows[top],cols[top]
                    decode_rgb = rgb[max(0,rr.min()-5):rr.max()+6,max(0,cc.min()-5):cc.max()+6]
                self.last_detection["parcel_code"] = read_rendered_code(decode_rgb)
                if object_id != "box_red" and self.last_detection["parcel_code"] != object_id:
                    raise ValueError(f"Expected {object_id}, observed {self.last_detection['parcel_code']}")
            except ValueError as exc:
                self.arm.detected_position = None
                print(f"RGBD_PERCEPTION failed={exc}", flush=True)
                return False
            if callable(getattr(self,'bind_visible',None)):
                self.bind_visible(self.last_detection['parcel_code'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(self.output_dir / "pickup_rgb.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(self.output_dir / "pickup_red_mask.png")
        np.save(self.output_dir / "pickup_depth.npy", depth)
        (self.output_dir / "detection.json").write_text(json.dumps(self.last_detection, indent=2), encoding="utf-8")
        if self.archive_observations and 'parcel_code' in self.last_detection:
            from .observation_archive import archive_observation
            self._observation_index += 1
            self.last_archive=archive_observation(self.output_dir,self.last_detection,self._observation_index)
        print(f"RGBD_PERCEPTION {self.last_detection}", flush=True)
        return True


def visible_parcel_center(world, half_height=.04):
    """Known upright cuboid: estimate its top before subtracting half height.

    An oblique camera sees side pixels as well as the top. Averaging all
    visible heights would bias the grasp down toward the bin floor.
    """
    import numpy as np
    top_z = np.percentile(world[:,2],95)
    top = world[world[:,2] >= top_z-.005]
    center = (np.percentile(top,5,axis=0)+np.percentile(top,95,axis=0))/2
    center[2] = top_z-half_height
    return center
