"""Associate a freshly decoded side label with a depth-fitted cuboid.

No simulator state or configured identity is used to select an observation.
Only a single isolated red cuboid is supported, not crowded-scene tracking.
"""
import math
import numpy as np


def label_world_center(corners, depth, intrinsics, position, quaternion):
    import cv2
    from scipy.spatial.transform import Rotation
    corners=np.asarray(corners,dtype=float)
    depth=np.asarray(depth)
    k=np.asarray(intrinsics,dtype=float)
    if corners.shape!=(4,2) or not np.isfinite(corners).all() or depth.ndim!=2:
        raise ValueError('Invalid label image geometry')
    mask=np.zeros(depth.shape,dtype=np.uint8)
    cv2.fillConvexPoly(mask,np.rint(corners).astype(np.int32),1)
    # Avoid mixed-depth label boundaries; require a real interior observation.
    mask=cv2.erode(mask,np.ones((5,5),dtype=np.uint8))>0
    mask &= np.isfinite(depth)&(depth>.1)&(depth<3.)
    rows,cols=np.nonzero(mask)
    if len(rows)<30:
        raise ValueError('Insufficient valid label depth')
    z=depth[rows,cols]
    camera=np.column_stack(((cols-k[0,2])*z/k[0,0],(rows-k[1,2])*z/k[1,1],z))
    world=Rotation.from_quat(quaternion).apply(camera)+np.asarray(position)
    return np.median(world,axis=0).tolist()


def label_on_observed_box(label, geometry, tolerance=.012):
    """Require the QR center to lie on this box's observed 3-D surface."""
    center=np.asarray(label['center_world'],dtype=float)
    origin=np.asarray(geometry['center_world'],dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        return False
    delta=center-origin
    yaw=geometry['yaw_rad']; c,s=math.cos(yaw),math.sin(yaw)
    local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2]])
    half=np.array([*sorted(geometry['observed_face_m'],reverse=True),geometry['height_m']])/2
    return bool(np.all(np.abs(local)<=half+tolerance)
                and np.min(np.abs(np.abs(local)-half))<=tolerance)


def side_view_poses(anchor):
    """Eight bounded views around the last tool/observed parcel location."""
    x,y=anchor[:2]
    target=(x,y,.035)
    return [((x+.32*math.cos(angle),y+.32*math.sin(angle),.14),target)
            for angle in np.arange(8)*math.pi/4]


def geometry_view_pose(anchor, last_visual_center=None):
    """Wide initial view; accept a bounded depth hint without claiming its ID."""
    center=np.asarray(anchor,dtype=float)
    hint=np.asarray(last_visual_center,dtype=float) if last_visual_center is not None else None
    if hint is not None and hint.shape==(3,) and np.isfinite(hint).all() and np.linalg.norm(hint[:2]-center[:2])<=.20:
        center=hint
    target=(float(center[0]),float(center[1]),.035)
    return (target[0]+.22,target[1],1.20),target
