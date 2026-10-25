"""Rate-constrained touchdown targets for an airborne leg near a joint limit.

This optional candidate chooses the *final* position command inside both the
joint box and the previous command's slew box. Vertical foot motion has first
priority; tangential error and joint motion choose among that vertical fit.
It is local linear kinematics, not a contact/dynamics guarantee.
"""
import itertools
import math

import numpy as np


def touchdown_joint_target(measured, previous, lower, upper, jacobian_world,
                           foot_error_world, max_step, *, contact_normal_n,
                           joint_margin_rad=.05, contact_threshold_n=1., damping=.03,
                           near_limit_only=True):
    """Return ``(target, diagnostic)`` or abstain with ``target is None``.

    ``lower``/``upper`` are the caller's unchanged command bounds. If an
    inherited command lies so far outside those bounds that no legal slew
    step exists, abstain; never silently widen either constraint. The caller
    retains its pre-existing recovery/failure path and elapsed time.
    """
    measured,previous,lower,upper,error=(np.asarray(v,dtype=float) for v in
        (measured,previous,lower,upper,foot_error_world))
    jac=np.asarray(jacobian_world,dtype=float)
    if any(v.shape!=(3,) for v in (measured,previous,lower,upper,error)) or jac.shape!=(3,3):
        raise ValueError('Touchdown IK requires a three-joint leg and a 3x3 world Jacobian')
    scalars=(max_step,contact_normal_n,joint_margin_rad,contact_threshold_n,damping)
    if (not all(np.isfinite(v).all() for v in (measured,previous,lower,upper,error,jac))
            or not all(math.isfinite(v) for v in scalars) or not np.all(lower<upper)
            or max_step<=0 or joint_margin_rad<=0 or contact_threshold_n<0 or damping<=0):
        raise ValueError('Touchdown IK inputs must be finite with ordered bounds and positive limits')
    if not isinstance(near_limit_only,bool):
        raise ValueError('Touchdown entry scope must be an explicit boolean')
    margin=float(np.minimum(measured-lower,upper-measured).min())
    diagnostic=dict(active=False,measured_command_margin_rad=margin,
        contact_normal_n=float(contact_normal_n),near_limit_only=near_limit_only)
    if contact_normal_n>contact_threshold_n:
        return None,dict(diagnostic,reason='measured_contact_present')
    if near_limit_only and margin>=joint_margin_rad:
        return None,dict(diagnostic,reason='joint_margin_clear')
    if error[2]>=-1e-6:
        return None,dict(diagnostic,reason='touchdown_not_requested')
    command_lower=np.maximum(lower,previous-max_step)
    command_upper=np.minimum(upper,previous+max_step)
    if np.any(command_lower>command_upper):
        return None,dict(diagnostic,reason='empty_position_and_slew_intersection',
            inherited_bound_violation_rad=float(np.maximum(lower-previous,previous-upper).clip(min=0).max()))
    lo,hi=command_lower-measured,command_upper-measured
    vertical=jac[2]
    z_min=float(np.where(vertical>=0,vertical*lo,vertical*hi).sum())
    z_max=float(np.where(vertical>=0,vertical*hi,vertical*lo).sum())
    z_target=float(np.clip(error[2],z_min,z_max))
    matrix=np.vstack((jac[:2],damping*np.eye(3)))
    desired=np.r_[error[:2],np.zeros(3)]
    hessian=matrix.T@matrix
    linear=matrix.T@desired
    best=None
    # Only three joints: enumerate box faces and solve their tiny equality
    # constrained least-squares problems. No iterative optimizer or warm state.
    for face in itertools.product((-1,0,1),repeat=3):
        fixed=np.array([i for i,state in enumerate(face) if state],dtype=int)
        free=np.array([i for i,state in enumerate(face) if not state],dtype=int)
        value=np.zeros(3)
        if len(fixed):
            value[fixed]=[lo[i] if face[i]<0 else hi[i] for i in fixed]
        remaining=z_target-float(vertical[fixed]@value[fixed])
        if len(free):
            hff=hessian[np.ix_(free,free)]
            rhs=linear[free]-hessian[np.ix_(free,fixed)]@value[fixed]
            constraint=vertical[free]
            if np.linalg.norm(constraint)<1e-12:
                if abs(remaining)>1e-9:
                    continue
                value[free]=np.linalg.solve(hff,rhs)
            else:
                kkt=np.block([[hff,constraint[:,None]],
                    [constraint[None,:],np.zeros((1,1))]])
                value[free]=np.linalg.solve(kkt,np.r_[rhs,remaining])[:-1]
        elif abs(remaining)>1e-9:
            continue
        if (np.any(value<lo-1e-9) or np.any(value>hi+1e-9)
                or abs(float(vertical@value)-z_target)>1e-8):
            continue
        objective=float(np.linalg.norm(matrix@value-desired)**2)
        if best is None or objective<best[0]:
            best=(objective,value.copy())
    if best is None:
        return None,dict(diagnostic,reason='numerical_candidate_unavailable')
    command=np.clip(measured+best[1],command_lower,command_upper)
    predicted=jac@(command-measured)
    return command,dict(diagnostic,active=True,reason='rate_constrained_touchdown',
        predicted_foot_delta_world_m=predicted.tolist(),
        requested_vertical_step_m=float(error[2]),
        minimum_possible_vertical_step_m=z_min,
        vertical_request_within_command_bounds=bool(z_min-1e-9<=error[2]<=z_max+1e-9),
        predicted_upward_motion_unavoidable=bool(z_min>1e-8),
        max_target_step_rad=float(np.abs(command-previous).max()))
