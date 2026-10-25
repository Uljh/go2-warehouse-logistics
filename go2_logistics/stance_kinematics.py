"""Fit a level body stance to four stationary, measured feet.

Planning only: this module never writes simulator state. Feasibility here is
kinematic; real contacts, torque limits and recovery timeout remain essential.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .arm_kinematics import ArmChain, pose_matrix


@dataclass
class StanceGeometry:
    legs: list

    @classmethod
    def from_stage(cls, stage, robot_path):
        from pxr import UsdPhysics
        def frame(position, quaternion):
            return pose_matrix(position, (*quaternion.GetImaginary(), quaternion.GetReal()))
        legs=[]
        for leg in ('FL','FR','RL','RR'):
            before,after,axes=[],[],[]
            parent=f'{robot_path}/base'
            for name in ('hip','thigh','calf'):
                prim=stage.GetPrimAtPath(f'{robot_path}/joints/{leg}_{name}_joint')
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    raise ValueError(f'Missing revolute leg joint {leg}_{name}')
                joint=UsdPhysics.RevoluteJoint(prim)
                child=f'{robot_path}/{leg}_{name}'
                if list(joint.GetBody0Rel().GetTargets())!=[parent] or list(joint.GetBody1Rel().GetTargets())!=[child]:
                    raise ValueError('Unexpected leg joint connectivity')
                before.append(frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()))
                after.append(np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())))
                axes.append(np.eye(3)['XYZ'.index(joint.GetAxisAttr().Get())])
                parent=child
            prim=stage.GetPrimAtPath(f'{robot_path}/joints/{leg}_foot_joint')
            if not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError('Expected a fixed foot frame')
            joint=UsdPhysics.FixedJoint(prim)
            if list(joint.GetBody0Rel().GetTargets())!=[parent] or list(joint.GetBody1Rel().GetTargets())!=[f'{robot_path}/{leg}_foot']:
                raise ValueError('Unexpected foot connectivity')
            fixed=frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get())@np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get()))
            legs.append(ArmChain(before,after,axes,fixed))
        return cls(legs)

    def feet(self, positions):
        q=np.asarray(positions,dtype=float)
        if q.shape!=(12,):
            raise ValueError('Expected hip/thigh/calf-major 12 leg angles')
        return np.array([chain.forward(q[[leg,leg+4,leg+8]])[:3,3] for leg,chain in enumerate(self.legs)])

    def fit(self, feet_world, root_position, yaw, positions, limits, *, height=.33,
            margin=.08, max_shift=.10, support_margin=.015):
        feet=np.asarray(feet_world,dtype=float)
        root=np.asarray(root_position,dtype=float)
        q=np.asarray(positions,dtype=float)
        limits=np.asarray(limits,dtype=float)
        if feet.shape!=(4,3) or root.shape!=(3,) or q.shape!=(12,) or limits.shape!=(12,2):
            raise ValueError('Invalid stance fit dimensions')
        if not all(np.isfinite(v).all() for v in (feet,root,q,limits)) or not np.isfinite(yaw):
            raise ValueError('Non-finite stance inputs')
        if not .305<height<=.33 or not 0<margin<.25 or not 0<max_shift<=.12 or support_margin<0:
            raise ValueError('Invalid bounded stance parameters')
        lower,upper=limits[:,0]+margin,limits[:,1]-margin
        if np.any(lower>=upper):
            raise ValueError('No joint range remains after the stance margin')
        try:
            hull=ConvexHull(feet[:,:2])
        except QhullError:
            return dict(success=False,reason='degenerate_support_polygon',height_m=height)
        halfplanes=hull.equations
        rotation=Rotation.from_euler('z',yaw).as_matrix()
        reference=np.clip(q,lower+1e-6,upper-1e-6)
        def error(value):
            candidate=np.r_[root[:2]+value[12:],height]
            predicted=self.feet(value[:12])@rotation.T+candidate
            outside=np.maximum(halfplanes[:,:2]@candidate[:2]+halfplanes[:,2]+support_margin,0.)
            return np.r_[(predicted-feet).ravel(),
                .0005*(value[:12]-reference),.003*value[12:],5.*outside,
                5.*max(np.linalg.norm(value[12:])-max_shift,0.)]
        result=least_squares(error,np.r_[reference,0.,0.],
            bounds=(np.r_[lower,-max_shift,-max_shift],np.r_[upper,max_shift,max_shift]),
            max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        candidate=np.r_[root[:2]+result.x[12:],height]
        predicted=self.feet(result.x[:12])@rotation.T+candidate
        residual=np.linalg.norm(predicted-feet,axis=1)
        clearance=float(-(halfplanes[:,:2]@candidate[:2]+halfplanes[:,2]).max())
        shift=float(np.linalg.norm(result.x[12:]))
        success=bool(result.success and max(residual)<.003 and shift<=max_shift+1e-6 and clearance>=support_margin-1e-5)
        return dict(success=success,root_target_w=candidate.tolist(),joints=result.x[:12].tolist(),
            foot_residual_m=residual.tolist(),shift_m=shift,support_clearance_m=clearance,
            minimum_joint_margin_rad=float(np.minimum(result.x[:12]-limits[:,0],limits[:,1]-result.x[:12]).min()),
            height_m=height,scope='four fixed feet, level yaw; kinematics only, no dynamics guarantee')
