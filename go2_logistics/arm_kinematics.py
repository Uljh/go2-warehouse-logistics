"""USD joint-frame kinematics and bounded multi-start arm-path preflight.

No simulator state is modified here. A feasible kinematic path is not a
collision-free plan or proof a floating base can execute it dynamically.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


class ArmPlanningError(RuntimeError):
    """A rejected plan plus the exact, read-only query needed to replay it."""

    def __init__(self, diagnostics):
        super().__init__('No bounded continuous arm path found by kinematic preflight')
        self.diagnostics = diagnostics


def replay_preflight(diagnostics):
    """Repeat a recorded query without starting Isaac or touching robot state."""
    model = diagnostics['chain']
    chain = ArmChain([np.asarray(x) for x in model['before']],
                     [np.asarray(x) for x in model['after']],
                     [np.asarray(x) for x in model['axes']], np.asarray(model['tool']))
    query = diagnostics['query']
    return chain.plan([(x['position'], x['quaternion_xyzw']) for x in query['world_targets']],
                      query['base_pose'], query['current_rad'], query['limits_rad'],
                      attempts=query['attempts'], first_minimum_world_z=query['first_minimum_world_z'])


def pose_matrix(position, quaternion_xyzw):
    value=np.eye(4)
    value[:3,:3]=Rotation.from_quat(quaternion_xyzw).as_matrix()
    value[:3,3]=position
    return value


@dataclass
class ArmChain:
    before: list
    after: list
    axes: list
    tool: np.ndarray

    @classmethod
    def from_stage(cls, stage, arm_path, tool_offset):
        from pxr import UsdPhysics
        def frame(position,quat):
            return pose_matrix(position,(*quat.GetImaginary(),quat.GetReal()))
        before,after,axes=[],[],[]
        for index in range(1,8):
            prim=stage.GetPrimAtPath(f'{arm_path}/panda_link{index-1}/panda_joint{index}')
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f'Missing serial arm joint {index}')
            joint=UsdPhysics.RevoluteJoint(prim)
            if str(joint.GetBody0Rel().GetTargets()[0])!=f'{arm_path}/panda_link{index-1}' or str(joint.GetBody1Rel().GetTargets()[0])!=f'{arm_path}/panda_link{index}':
                raise ValueError('Unexpected arm joint connectivity')
            before.append(frame(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get()))
            after.append(np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())))
            axes.append(np.eye(3)['XYZ'.index(joint.GetAxisAttr().Get())])
        hand=UsdPhysics.Joint(stage.GetPrimAtPath(f'{arm_path}/panda_link7/panda_hand_joint'))
        fixed=frame(hand.GetLocalPos0Attr().Get(),hand.GetLocalRot0Attr().Get())@np.linalg.inv(frame(hand.GetLocalPos1Attr().Get(),hand.GetLocalRot1Attr().Get()))
        offset=np.eye(4);offset[:3,3]=tool_offset
        return cls(before,after,axes,fixed@offset)

    def forward(self, positions, *, with_jacobian=False):
        transform=np.eye(4)
        origins,axes=[],[]
        for angle,before,after,axis in zip(positions,self.before,self.after,self.axes):
            transform=transform@before
            origins.append(transform[:3,3].copy());axes.append(transform[:3,:3]@axis)
            motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix()
            transform=transform@motion@after
        transform=transform@self.tool
        if with_jacobian:
            jacobian=np.array([np.r_[np.cross(axis,transform[:3,3]-origin),axis] for axis,origin in zip(axes,origins)]).T
            return transform,jacobian
        return transform

    def plan_refined(self, world_targets, base_pose, current, limits, *, attempts=12,
                     first_minimum_world_z=None, cartesian_step=.02):
        """Retry a rejected interpolation using closer Cartesian IK targets.

        The original bounds, residual tolerances and tool corridor remain in
        force. Only segments after the first approach are subdivided: this is
        not a new collision-free route from the initial arm posture.
        """
        if not np.isfinite(cartesian_step) or not 0 < cartesian_step <= .02:
            raise ValueError('Refinement step must be finite in (0,.02]')
        try:
            plan = self.plan(world_targets, base_pose, current, limits, attempts=attempts,
                             first_minimum_world_z=first_minimum_world_z)
        except ArmPlanningError as error:
            coarse = error.diagnostics
            if not any(x['reason'] == 'tool_corridor' for x in coarse['candidates']):
                raise
        else:
            plan['refinement'] = dict(used=False, cartesian_step_m=cartesian_step)
            return plan

        dense, groups = [world_targets[0]], [[0]]
        for (pa, qa), (pb, qb) in zip(world_targets, world_targets[1:]):
            pa, pb = np.asarray(pa), np.asarray(pb)
            rotations = Rotation.from_quat([qa, qb])
            angle = (rotations[1] * rotations[0].inv()).magnitude()
            count = max(1, int(np.ceil(np.linalg.norm(pb-pa)/cartesian_step)),
                        int(np.ceil(angle/.10)))
            interpolate = Slerp([0., 1.], rotations)
            group = []
            for fraction in np.linspace(0., 1., count+1)[1:]:
                group.append(len(dense))
                dense.append((pa+fraction*(pb-pa), interpolate(fraction).as_quat()))
            groups.append(group)
        try:
            plan = self.plan(dense, base_pose, current, limits, attempts=attempts,
                             first_minimum_world_z=first_minimum_world_z)
        except ArmPlanningError as error:
            error.diagnostics['coarse_rejection'] = coarse
            raise
        dense_joints, dense_residuals = plan['joints'], plan['residuals']
        plan['refined_path'] = dict(world_targets=plan['world_targets'], joints=dense_joints,
                                    residuals=dense_residuals)
        plan['segment_joint_paths'] = [[dense_joints[i] for i in group] for group in groups]
        plan['joints'] = [dense_joints[group[-1]] for group in groups]
        plan['residuals'] = [dense_residuals[group[-1]] for group in groups]
        plan['world_targets'] = [dict(position=np.asarray(p).tolist(), quaternion_xyzw=np.asarray(q).tolist())
                                 for p, q in world_targets]
        plan['refinement'] = dict(used=True, cartesian_step_m=cartesian_step,
                                  original_waypoints=len(world_targets), refined_waypoints=len(dense),
                                  coarse_rejection=coarse)
        return plan

    def plan_parallel_grasp(self, world_targets, base_pose, current, limits, *,
                            refine=False, attempts=12, first_minimum_world_z=None):
        """Try equivalent finger orientations before closing symmetric jaws.

        A local tool-Z half turn exchanges fingers, preserving the approach
        axis and opposing grasp faces. Never use this for loaded placement.
        Both options retain all plan checks.
        """
        planner = self.plan_refined if refine else self.plan
        options = dict(attempts=attempts, first_minimum_world_z=first_minimum_world_z)
        try:
            plan = planner(world_targets, base_pose, current, limits, **options)
        except ArmPlanningError as error:
            original = error.diagnostics
        else:
            plan['grasp_symmetry'] = dict(used=False, local_tool_z_rotation_rad=0.)
            return plan
        half_turn = Rotation.from_rotvec([0., 0., np.pi])
        equivalent = [(p, (Rotation.from_quat(q) * half_turn).as_quat())
                      for p, q in world_targets]
        try:
            plan = planner(equivalent, base_pose, current, limits, **options)
        except ArmPlanningError as error:
            error.diagnostics['original_orientation_rejection'] = original
            raise
        plan['grasp_symmetry'] = dict(used=True, local_tool_z_rotation_rad=float(np.pi),
                                      original_orientation_rejection=original)
        return plan

    def plan(self, world_targets, base_pose, current, limits, *, attempts=12, first_minimum_world_z=None):
        base=pose_matrix(*base_pose)
        inverse=np.linalg.inv(base)
        targets=[inverse@pose_matrix(*target) for target in world_targets]
        limits=np.asarray(limits,dtype=float)
        lower,upper=limits[:,0]+.08,limits[:,1]-.08
        current=np.asarray(current,dtype=float)
        if current.shape!=(7,) or len(self.axes)!=7 or not np.isfinite(limits).all() or np.any(lower>=upper):
            raise ValueError('Invalid arm planning state')
        rng=np.random.default_rng(17)
        seeds=[np.clip(current,lower,upper), (lower+upper)/2]
        seeds.extend(rng.uniform(lower+.01,upper-.01) for _ in range(max(0,attempts-len(seeds))))
        best=None
        diagnostics=dict(schema_version=1,
            chain=dict(before=[x.tolist() for x in self.before],after=[x.tolist() for x in self.after],
                       axes=[x.tolist() for x in self.axes],tool=self.tool.tolist()),
            query=dict(world_targets=[dict(position=np.asarray(p).tolist(),quaternion_xyzw=np.asarray(q).tolist())
                                      for p,q in world_targets],
                       base_pose=[np.asarray(x).tolist() for x in base_pose],current_rad=current.tolist(),
                       limits_rad=limits.tolist(),attempts=attempts,first_minimum_world_z=first_minimum_world_z),
            thresholds=dict(position_m=.004,angle_rad=.06,joint_margin_rad=.08,tool_corridor_m=.03),
            candidates=[])
        def error(q,target,prior):
            actual=self.forward(q)
            rotation=Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).as_rotvec()
            return np.r_[target[:3,3]-actual[:3,3],.18*rotation,.0008*(q-prior)]
        for seed_index,seed in enumerate(seeds):
            reference=seed.copy();path=[];residuals=[]
            candidate=dict(seed_index=seed_index,accepted=False)
            diagnostics['candidates'].append(candidate)
            for target_index,target in enumerate(targets):
                result=least_squares(error,reference,args=(target,reference),bounds=(lower,upper),
                    max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                value=error(result.x,target,result.x)
                position=float(np.linalg.norm(value[:3]));angle=float(np.linalg.norm(value[3:6])/.18)
                if position>.004 or angle>.06:
                    candidate.update(reason='waypoint_pose_residual',waypoint_index=target_index,
                        position_m=position,angle_rad=angle,solver_status=int(result.status),
                        function_evaluations=int(result.nfev),joints_rad=result.x.tolist())
                    break
                path.append(result.x.copy());residuals.append(dict(position_m=position,angle_rad=angle))
                reference=result.x
            if len(path)!=len(targets):
                continue
            # Sample the actual joint interpolation that will be commanded.
            # The pregrasp transition must stay above the bin; later steps
            # stay in a 3cm corridor around their Cartesian segment.
            previous=current
            safe=True
            for index,q in enumerate(path):
                start=self.forward(previous)[:3,3];end=targets[index][:3,3]
                for fraction in np.linspace(0.,1.,25):
                    point=self.forward(previous+fraction*(q-previous))[:3,3]
                    world=(base@np.r_[point,1.])[:3]
                    if index==0:
                        floor=world_targets[0][0][2]-.03 if first_minimum_world_z is None else first_minimum_world_z
                        if world[2]<floor:
                            candidate.update(reason='first_transition_height',waypoint_index=index,
                                fraction=float(fraction),world_z_m=float(world[2]),minimum_world_z_m=float(floor))
                            safe=False;break
                    else:
                        direction=end-start
                        along=np.clip(np.dot(point-start,direction)/max(np.dot(direction,direction),1e-12),0.,1.)
                        deviation=float(np.linalg.norm(point-(start+along*direction)))
                        if deviation>.03:
                            candidate.update(reason='tool_corridor',waypoint_index=index,
                                fraction=float(fraction),deviation_m=deviation)
                            safe=False;break
                if not safe:break
                previous=q
            if not safe:continue
            margin=float(np.minimum(np.asarray(path)-limits[:,0],limits[:,1]-np.asarray(path)).min())
            travel=float(np.abs(np.diff(np.vstack([current,*path]),axis=0)).sum())
            score=travel-.5*margin
            candidate.update(accepted=True,score=score,minimum_joint_margin_rad=margin)
            if best is None or score<best['score']:
                best=dict(score=score,seed_index=seed_index,joints=[q.tolist() for q in path],
                    residuals=residuals,minimum_joint_margin_rad=margin,
                    world_targets=[dict(position=list(pos),quaternion_xyzw=list(quat)) for pos,quat in world_targets],
                    collision_scope='tool-point corridor only; not full-link collision planning')
        if best is None:
            raise ArmPlanningError(diagnostics)
        return best
