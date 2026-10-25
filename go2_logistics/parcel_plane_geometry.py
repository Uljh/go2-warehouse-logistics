"""Fit a mildly tilted known cuboid face from rendered depth, without object truth."""
import math

import numpy as np


def fit_sloped_visible_box(world, *, max_tilt_deg=12., dimensions=(.10, .08, .035)):
    """RANSAC rejects side pixels; a rectangle in the measured plane fixes the TCP.

    The previous horizontal fitter intentionally remains unchanged. This
    separate Z1 path measures face normal/axes and supplies them to the arm,
    rather than treating a tilted parcel as a horizontal one.
    """
    import cv2
    world=np.asarray(world,dtype=np.float64)
    if world.ndim!=2 or world.shape[1]!=3 or len(world)<30 or not np.isfinite(world).all():
        raise ValueError('Insufficient finite parcel points')
    if not 0 < max_tilt_deg <= 15.:
        raise ValueError('Sloped box baseline supports a measured tilt of at most 15 degrees')
    sample=world[np.linspace(0,len(world)-1,min(len(world),4000)).astype(int)]
    rng=np.random.default_rng(0)
    triples=sample[rng.integers(0,len(sample),size=(384,3))]
    normals=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0])
    norms=np.linalg.norm(normals,axis=1)
    valid=norms>1.e-8
    normals[valid]/=norms[valid,None]
    normals[normals[:,2]<0]*=-1.
    valid &= normals[:,2]>=math.cos(math.radians(max_tilt_deg))
    normals=normals[valid]
    origins=triples[valid,0]
    if not len(normals):
        raise ValueError('No upward parcel face within the supported tilt range')
    offsets=np.einsum('ij,ij->i',origins,normals)
    residual=np.abs(sample@normals.T-offsets)
    best=int(np.argmax((residual<.0007).sum(axis=0)))
    normal=normals[best]
    inliers=np.abs(world@normal-offsets[best])<.0007
    if inliers.sum()<30 or inliers.mean()<.20:
        raise ValueError('Too little depth evidence for a coherent upper face')
    for _ in range(2):
        points=world[inliers]
        midpoint=points.mean(axis=0)
        _,_,vh=np.linalg.svd(points[::max(1,len(points)//10000)]-midpoint,full_matrices=False)
        normal=vh[-1]
        if normal[2]<0:normal=-normal
        if normal[2]<math.cos(math.radians(max_tilt_deg)):
            raise ValueError('Measured parcel face exceeds supported tilt range')
        inliers=np.abs((world-midpoint)@normal)<.0007
    points=world[inliers]
    rms=float(np.sqrt(np.mean(((points-midpoint)@normal)**2)))
    if len(points)<30 or rms>.0005:
        raise ValueError('Measured parcel face is not planar enough')
    axis=np.array([1.,0.,0.]);axis-=np.dot(axis,normal)*normal;axis/=np.linalg.norm(axis)
    other=np.cross(normal,axis)
    local=np.column_stack(((points-midpoint)@axis,(points-midpoint)@other))
    rectangle=cv2.minAreaRect(local.astype(np.float32))
    corners=cv2.boxPoints(rectangle).astype(float)
    edges=[corners[1]-corners[0],corners[2]-corners[1]]
    lengths=[float(np.linalg.norm(edge)) for edge in edges]
    long_i=int(np.argmax(lengths));long_uv=edges[long_i]/lengths[long_i]
    long_axis=long_uv[0]*axis+long_uv[1]*other
    yaw=math.atan2(long_axis[1],long_axis[0])
    if not -math.pi/2<=yaw<math.pi/2:
        long_axis=-long_axis
    yaw=math.atan2(long_axis[1],long_axis[0])
    observed=sorted(lengths,reverse=True)
    choices=[]
    for i,thickness in enumerate(dimensions):
        face=sorted([d for j,d in enumerate(dimensions) if i!=j],reverse=True)
        choices.append((float(np.linalg.norm(np.asarray(observed)-face)),thickness,face))
    error,height,face=min(choices)
    if error>.012 or abs(observed[1]-face[1])>.009:
        raise ValueError(f'Visible sloped face incomplete or unknown size: {observed}')
    center_uv=np.asarray(rectangle[0])
    surface_center=midpoint+axis*center_uv[0]+other*center_uv[1]
    center=surface_center-normal*(height/2)
    return dict(center_world=center.tolist(),height_m=height,grasp_width_m=face[1],yaw_rad=yaw,
        top_z_m=float(np.max(points[:,2])),observed_face_m=observed,fit_error_m=error,
        surface_center_world=surface_center.tolist(),surface_normal_world=normal.tolist(),
        long_axis_world=long_axis.tolist(),face_plane_rms_m=rms,
        face_inlier_fraction=float(inliers.mean()),face_tilt_deg=math.degrees(math.acos(np.clip(normal[2],-1.,1.))),
        source='rendered_depth_visible_rectangle_sloped_known_dimensions')
