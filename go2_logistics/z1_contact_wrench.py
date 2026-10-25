"""Reconstruct pair contact wrenches from PhysX contact/patch buffers.

All forces are reported on the sensor body. Their reactions on the other
body have opposite signs. This module neither advances nor writes physics.
"""
import numpy as np


def _array(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    elif hasattr(value,'numpy'):
        value=value.numpy()
    return np.asarray(value)


def _pair_range(counts,starts,pair):
    count,start=_array(counts),_array(starts)
    if count.shape!=start.shape or count.ndim!=2:
        raise ValueError('Contact counts and starts must have matching sensor/filter shapes')
    if pair is None:
        if count.size!=1:
            raise ValueError('Wrench diagnostic requires exactly one sensor/filter pair or explicit pair selection')
        pair=(0,0)
    if (not isinstance(pair,(tuple,list)) or len(pair)!=2
            or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in pair)):
        raise ValueError('Pair selection must contain integer sensor and filter indices')
    if any(i<0 or i>=size for i,size in zip(pair,count.shape)):
        raise ValueError('Sensor/filter pair selection is out of range')
    n,first=count[tuple(pair)],start[tuple(pair)]
    if not np.isfinite([n,first]).all() or n!=int(n) or first!=int(first):
        raise ValueError('Contact buffer indices must be finite integers')
    return int(n),int(first)


def _active(buffer,counts,starts,pair=None):
    n,first=_pair_range(counts,starts,pair)
    data=_array(buffer)
    if n==0:
        return data[:0].astype(float)
    if n<0 or first<0 or first+n>len(data):
        raise ValueError('Contact buffer range is invalid')
    selected=data[first:first+n]
    if not np.isfinite(selected).all():
        raise ValueError('Non-finite active contact buffer')
    return selected.astype(float)


def read_pair_wrench(view,dt,origin,expected_normal,expected_friction,*,pair=None):
    """Copy before the next query: PhysX reuses normal/friction count buffers."""
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('Wrench diagnostic requires a positive finite physics timestep')
    normal_data=tuple(_array(value).copy() for value in view.get_contact_data(dt=dt))
    friction_data=tuple(_array(value).copy() for value in view.get_friction_data(dt=dt))
    return reconstruct_pair_wrench(normal_data,friction_data,origin,expected_normal,expected_friction,pair=pair)


def reconstruct_pair_wrench(normal_data,friction_data,origin,expected_normal,expected_friction,*,pair=None):
    """Cross-check force sums, then sum each measured point's r cross force."""
    force_buffer,points,normals,separations,counts,starts=normal_data
    friction_buffer,friction_points,friction_counts,friction_starts=friction_data
    if _array(counts).shape!=_array(friction_counts).shape:
        raise ValueError('Normal and friction pair layouts differ')
    magnitudes=_active(force_buffer,counts,starts,pair).reshape(-1)
    points=_active(points,counts,starts,pair).reshape(-1,3)
    directions=_active(normals,counts,starts,pair).reshape(-1,3)
    separation=_active(separations,counts,starts,pair).reshape(-1)
    friction=_active(friction_buffer,friction_counts,friction_starts,pair).reshape(-1,3)
    friction_points=_active(friction_points,friction_counts,friction_starts,pair).reshape(-1,3)
    origin=np.asarray(origin,dtype=float)
    expected_normal=np.asarray(expected_normal,dtype=float)
    expected_friction=np.asarray(expected_friction,dtype=float)
    if origin.shape!=(3,) or expected_normal.shape!=(3,) or expected_friction.shape!=(3,):
        raise ValueError('Expected finite three-dimensional wrench reference')
    if not np.isfinite(np.r_[origin,expected_normal,expected_friction]).all():
        raise ValueError('Non-finite wrench reference')
    forces=magnitudes[:,None]*directions
    summed=forces.sum(axis=0)
    # PhysX backend conventions may reverse point normals. Resolve only
    # this sign convention against the independently exposed pair vector.
    sign=1 if np.linalg.norm(summed-expected_normal)<=np.linalg.norm(-summed-expected_normal) else -1
    forces*=sign
    normal_error=float(np.linalg.norm(forces.sum(axis=0)-expected_normal))
    friction_error=float(np.linalg.norm(friction.sum(axis=0)-expected_friction))
    if max(normal_error,friction_error)>1e-3:
        raise RuntimeError(f'Point forces do not reconstruct aggregate sensors: normal={normal_error}, friction={friction_error}, '
            f'point_sum={forces.sum(axis=0).tolist()}, expected={expected_normal.tolist()}, count={len(forces)}')
    total_force=forces.sum(axis=0)+friction.sum(axis=0)
    torque=np.cross(points-origin,forces).sum(axis=0)+np.cross(friction_points-origin,friction).sum(axis=0)
    return dict(force_on_sensor_w_n=total_force.tolist(),moment_on_sensor_about_origin_w_nm=torque.tolist(),
        origin_w_m=origin.tolist(),normal_direction_sign=sign,
        selected_pair=list(pair) if pair is not None else [0,0],
        normal_buffer_capacity=len(force_buffer),friction_buffer_capacity=len(friction_buffer),
        normal_buffer_slots_used=int(_array(counts).sum()),
        friction_buffer_slots_used=int(_array(friction_counts).sum()),
        normal_reconstruction_error_n=normal_error,friction_reconstruction_error_n=friction_error,
        normal_contacts=[dict(point_w_m=p.tolist(),force_w_n=f.tolist(),separation_m=float(s))
            for p,f,s in zip(points,forces,separation)],
        friction_patches=[dict(point_w_m=p.tolist(),force_w_n=f.tolist()) for p,f in zip(friction_points,friction)])
