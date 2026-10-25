"""Visible top-face geometry for known rectangular parcels resting on a face.

No simulator imports or object poses. A rectangle fit estimates yaw and the
known three dimensions identify the unobserved thickness. Severe occlusion,
nonhorizontal faces and implausible dimensions are rejected.
"""
import math
import numpy as np


def visible_face_mask(world, geometry):
    """Keep the pixels belonging to the measured face, including a tilted one."""
    if 'surface_normal_world' in geometry:
        return np.abs((world-np.asarray(geometry['surface_center_world']))
                      @np.asarray(geometry['surface_normal_world']))<.0007
    return world[:,2]>geometry['top_z_m']-.003


def select_visible_box(candidates, image_shape, *, fit_function=None):
    """Select one depth-valid red component; area alone cannot reject glare.

    Candidates carry (component ID, pixel rows, pixel columns, world points).
    No identity or object truth is inferred. Multiple plausible boxes are
    rejected rather than arbitrarily binding a neighbor's geometry.
    """
    valid=[];rejected=[]
    for component,rows,cols,world in candidates:
        try:
            geometry=(fit_function or fit_visible_box)(world)
            top_pixels=visible_face_mask(world,geometry)
            if (rows[top_pixels].min()<2 or rows[top_pixels].max()>=image_shape[0]-2
                    or cols[top_pixels].min()<2 or cols[top_pixels].max()>=image_shape[1]-2):
                raise ValueError('Visible top face clipped by camera image boundary')
            valid.append((component,rows,cols,world,geometry))
        except ValueError as error:
            rejected.append(dict(component=int(component),pixels=len(rows),reason=str(error)))
    if len(valid)!=1:
        raise ValueError(f'Expected one depth-valid parcel component, found {len(valid)}; rejected={rejected}')
    selected=valid[0]
    return selected,dict(selected_component=int(selected[0]),valid_components=len(valid),rejected=rejected)


def fit_visible_box(world, dimensions=(.10,.08,.035)):
    import cv2
    world = np.asarray(world, dtype=np.float64)
    if len(world) < 30 or not np.isfinite(world).all():
        raise ValueError('Insufficient finite parcel points')
    top_z = float(np.percentile(world[:,2],95))
    top = world[world[:,2] > top_z-.003]
    if len(top) < 30:
        raise ValueError('No visible horizontal face')
    centered=top[:,:2]-top[:,:2].mean(axis=0)
    plane=np.linalg.lstsq(np.column_stack((centered,np.ones(len(top)))),top[:,2],rcond=None)[0]
    if float(np.linalg.norm(plane[:2]))>.025:
        raise ValueError('Visible face tilted beyond horizontal grasp baseline')
    # A robust rectangle boundary retains the full shape despite a white QR
    # hole. PCA alone is biased by asymmetric label occlusion and side pixels.
    rectangle = cv2.minAreaRect(top[:,:2].astype(np.float32))
    corners = cv2.boxPoints(rectangle).astype(np.float64)
    edges = (corners[1]-corners[0],corners[2]-corners[1])
    lengths = [float(np.linalg.norm(e)) for e in edges]
    long_index = int(np.argmax(lengths))
    long_axis = edges[long_index]/lengths[long_index]
    observed = sorted(lengths,reverse=True)
    candidates = []
    for thickness_index, height in enumerate(dimensions):
        face = sorted([d for i,d in enumerate(dimensions) if i != thickness_index],reverse=True)
        error = float(np.linalg.norm(np.array(observed)-face))
        candidates.append((error,height,face))
    error,height,face = min(candidates)
    if error > .012 or abs(observed[1]-face[1]) > .009:
        raise ValueError(f'Visible face incomplete or unknown size: {observed}')
    yaw = math.atan2(long_axis[1],long_axis[0])
    # Parallel-jaw orientation is symmetric under 180 degrees; select the
    # nearest forward-facing branch to keep the wrist away from its limits.
    yaw = (yaw+math.pi/2)%math.pi-math.pi/2
    center = [float(rectangle[0][0]),float(rectangle[0][1]),top_z-height/2]
    return dict(center_world=center,height_m=height,grasp_width_m=face[1],yaw_rad=yaw,
                top_z_m=top_z,observed_face_m=observed,fit_error_m=error,
                hand_quaternion_xyzw=[math.cos(yaw/2),math.sin(yaw/2),0.,0.],
                source='rendered_depth_visible_rectangle_known_dimensions')
