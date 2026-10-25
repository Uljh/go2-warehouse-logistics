"""Additional measured acceptance for free-base, original-mass Z1 grasping.

Physics uses the same contact probe/controller as the fixed-support baseline:
python -m go2_logistics.z1_contact_probe --mounted-go2 ...
This is an isolated low-table manipulation experiment, not fleet acceptance.
"""
from __future__ import annotations

import math
import numpy as np

from go2_logistics.z1_contact_probe import evaluate_trace


def evaluate_mounted_trace(samples, config, integrity):
    result=evaluate_trace(samples,config,integrity)
    checks=result['checks']
    checks['floating_base_without_world_anchor']=bool(
        integrity.get('fixed_base') is False and integrity.get('robot_world_joints')==[])
    supported=[s for s in samples if s.get('time_s',0.)>=.5]
    required=('root_pos_w','projected_gravity_b','foot_normal_n','world_arm_tracking')
    measured=bool(supported) and all(all(k in s for k in required) for s in supported)
    checks['floating_base_measurements_present']=measured
    checks['base_stays_upright']=False
    checks['stationary_base_holds_position']=False
    checks['measured_foot_support']=False
    checks['world_arm_tracking']=False
    if measured:
        roots=np.asarray([s['root_pos_w'] for s in supported],dtype=float)
        gravity=np.asarray([s['projected_gravity_b'] for s in supported],dtype=float)
        feet=np.asarray([s['foot_normal_n'] for s in supported],dtype=float)
        finite=bool(roots.shape==(len(supported),3) and gravity.shape==(len(supported),3)
            and feet.shape==(len(supported),4) and np.isfinite(np.r_[roots.ravel(),gravity.ravel(),feet.ravel()]).all())
        if finite:
            minimum_height=float(roots[:,2].min())
            maximum_tilt=float(np.degrees(np.arccos(np.clip(-gravity[:,2],-1.,1.))).max())
            maximum_drift=float(np.linalg.norm(roots[:,:2],axis=1).max())
            support_fraction=float(np.mean((feet>1.).sum(axis=1)>=3))
            checks['base_stays_upright']=minimum_height>=.25 and maximum_tilt<=20.
            checks['stationary_base_holds_position']=maximum_drift<=.05
            checks['measured_foot_support']=support_fraction>=.98
            result['metrics'].update(minimum_base_height_m=minimum_height,maximum_base_tilt_deg=maximum_tilt,
                maximum_base_xy_drift_m=maximum_drift,three_foot_support_fraction=support_fraction)
        tracking=[s['world_arm_tracking'] for s in supported]
        valid=all(isinstance(v,dict) and all(isinstance(v.get(k),(int,float)) and math.isfinite(v[k])
            for k in ('position_error_m','orientation_error_rad')) for v in tracking)
        if valid:
            position=max(v['position_error_m'] for v in tracking)
            orientation=max(v['orientation_error_rad'] for v in tracking)
            checks['world_arm_tracking']=position<=.02 and orientation<=.08
            result['metrics'].update(maximum_world_arm_position_error_m=position,
                maximum_world_arm_orientation_error_rad=orientation)
    result['passed']=all(checks.values())
    result['scope']=('free-base Go2 with original-mass Z1 contact lift/hold/release on a low table; '
        'simulation pose/contact feedback, known initial parcel; no perception, walking or fleet acceptance')
    return result
