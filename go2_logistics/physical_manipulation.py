"""Joint-controlled Panda pick/place on a floating Go2 articulation.

Contact mode relies solely on finger collision and friction; assisted mode
uses a proximity-gated external fixed constraint. Neither mode overwrites
the payload pose during a mission. Ground-truth poses serve IK and monitoring.

Isaac imports are deliberately delayed until an initialized simulator calls
the controller.  All quaternion tensor values use Isaac Lab 3's xyzw order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PhysicalPandaConfig:
    dropoff_position: tuple[float, float, float]
    grasp_mode: str = "assisted"
    arm_joint_names: tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))
    finger_joint_names: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
    hand_body_name: str = "panda_hand"
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.064108)
    # Keep a small margin below the .0248 m baked prismatic joint limit.
    finger_open: float = 0.024
    box_half_height: float = 0.04
    box_width: float = 0.035
    grasp_height_offset: float = 0.012
    approach_height: float = 0.13
    placement_approach_height: float = 0.13
    release_retreat_height: float = 0.12
    placement_planar_tolerance: float = 0.075
    placement_height_tolerance: float = 0.035
    placement_surface_z: float | None = None
    placement_deck_size_xy: tuple[float, float] | None = None
    lift_height: float = 0.15
    lift_segments: int = 1
    carry_retraction_m: float = 0.0
    carry_retraction_clearance_m: float = 0.12
    ik_path_preflight: bool = False
    ik_path_refinement: bool = False
    ik_grasp_symmetry: bool = False
    grasp_yaw_offset: float = 0.0
    position_tolerance: float = 0.012
    # Transport clearance tolerates base/arm compliance; approach keeps the
    # tighter grasp tolerance. Physical parcel lift is verified separately.
    lift_position_tolerance: float = 0.025
    orientation_tolerance: float = 0.20
    grasp_distance_limit: float = 0.029
    ik_damping: float = 0.055
    ik_joint_centering_gain: float = 0.0
    ik_joint_limit_aware: bool = False
    max_joint_increment: float = 0.045
    max_ik_steps: int = 260
    close_steps: int = 45
    settle_steps: int = 50
    constraint_path: str = "/World/AssistedParcelGrasp"
    reset_before_grasp: bool = False
    stow_after_release: bool = False

    def __post_init__(self):
        if self.grasp_mode not in ("contact", "assisted"):
            raise ValueError("grasp_mode must be contact or assisted")
        if self.ik_path_refinement and not self.ik_path_preflight:
            raise ValueError('Arm path refinement requires kinematic preflight')
        if self.ik_grasp_symmetry and not self.ik_path_preflight:
            raise ValueError('Symmetric grasp selection requires kinematic preflight')
        if not isinstance(self.lift_segments,int) or self.lift_segments<1:
            raise ValueError('lift_segments must be a positive integer')
        if not 0<=self.ik_joint_centering_gain<=.1:
            raise ValueError('ik_joint_centering_gain must be between zero and .1')
        import math
        if not all(math.isfinite(v) and v>0 for v in
                   (self.release_retreat_height,self.placement_planar_tolerance,self.placement_height_tolerance)):
            raise ValueError('Placement tolerances and retreat height must be finite and positive')
        if (self.placement_surface_z is None)!=(self.placement_deck_size_xy is None):
            raise ValueError('Shelf verification needs both surface height and deck dimensions')
        if self.placement_surface_z is not None and (
                not math.isfinite(self.placement_surface_z) or len(self.placement_deck_size_xy)!=2
                or not all(math.isfinite(v) and v>0 for v in self.placement_deck_size_xy)):
            raise ValueError('Invalid shelf verification geometry')
        if not math.isfinite(self.carry_retraction_m) or not 0<=self.carry_retraction_m<=.15:
            raise ValueError('carry_retraction_m must be finite in [0,.15]')
        if self.carry_retraction_m and not 0<self.carry_retraction_clearance_m<self.lift_height:
            raise ValueError('Carry retraction requires clearance below final lift height')


def carry_lift_waypoints(start, root_xy, *, height, segments, retraction=0., clearance=.12):
    """Cartesian references: clear the bin first, then bring the load inward.

    The root is measured once at grasp time; no body states are written. This
    local path has no general collision guarantee, so it remains opt-in.
    """
    import math
    if len(start)!=3 or len(root_xy)!=2 or not all(math.isfinite(float(v)) for v in (*start,*root_xy)):
        raise ValueError('Invalid carry geometry')
    if not isinstance(segments,int) or segments<1 or not math.isfinite(height) or height<=0:
        raise ValueError('Invalid lift height or segments')
    if not math.isfinite(retraction) or not 0<=retraction<=.15:
        raise ValueError('Invalid carry retraction')
    if retraction and not 0<clearance<height:
        raise ValueError('Retraction must start after positive bin clearance')
    dx,dy=start[0]-root_xy[0],start[1]-root_xy[1]
    radius=math.hypot(dx,dy)
    if retraction and radius<=retraction+.20:
        raise ValueError('Carry retraction would enter the trunk envelope')
    points=[]
    for segment in range(1,segments+1):
        rise=height*segment/segments
        fraction=max(0.,min(1.,(rise-clearance)/(height-clearance))) if retraction else 0.
        offset=retraction*fraction*fraction*(3.-2.*fraction)
        scale=offset/max(radius,1e-9)
        points.append((start[0]-scale*dx,start[1]-scale*dy,start[2]+rise))
    return points


def author_grasp_joint(
    stage: Any,
    hand_path: str,
    box_path: str,
    joint_path: str = "/World/AssistedParcelGrasp",
) -> Any:
    """Author a disabled external constraint, preferably before sim.reset().

    Excluding this joint from the articulation is essential: the parcel is an
    independently simulated rigid body, not a new articulated robot link.
    """
    from pxr import Gf, Sdf, UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(box_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(False)
    joint.GetPrim().SetCustomDataByKey("grasp_model", "proximity_gated_assisted_constraint")
    return joint


def _tensor(value: Any) -> Any:
    return getattr(value, "torch", value)


def joint_centering_increment(jacobian, positions, limits, gain):
    """Use redundant joint motion to leave limits without moving the tool.

    Exact pseudoinverse gives J(I-J+J)=0 to numerical precision; ordinary
    damped task IK remains responsible for the requested Cartesian motion.
    This is local redundancy resolution, not a global collision-free IK solver.
    """
    import torch
    midpoint=limits.mean(dim=1)
    half_range=(limits[:,1]-limits[:,0])/2
    gradient=(-(positions-midpoint)/half_range.clamp_min(1e-6)).clamp(-1.,1.)
    projector=torch.eye(len(positions),device=jacobian.device,dtype=jacobian.dtype)-torch.linalg.pinv(jacobian)@jacobian
    return gain*(projector@gradient)


def bounded_damped_increment(jacobian, error, lower, upper, damping):
    """Solve the local damped IK task with joint bounds inside the solve.

    BVLS minimizes ||J dq - error||^2 + damping^2 ||dq||^2.
    Unlike clipping an unconstrained solution, a saturated column's missing
    motion can be assigned to the remaining free joints. This local solver
    does not promise global reachability or collision avoidance.
    """
    import numpy as np
    import torch
    from scipy.optimize import lsq_linear

    matrix=jacobian.detach().cpu().double().numpy()
    target=error.detach().cpu().double().numpy()
    lo=lower.detach().cpu().double().numpy()
    hi=upper.detach().cpu().double().numpy()
    if np.any(lo>hi):
        raise ValueError('Inconsistent IK increment bounds')
    count=matrix.shape[1]
    augmented=np.vstack((matrix,damping*np.eye(count)))
    rhs=np.concatenate((target,np.zeros(count)))
    fixed=hi-lo<=1e-12
    delta=np.zeros(count)
    delta[fixed]=(lo[fixed]+hi[fixed])/2
    if (~fixed).any():
        solution=lsq_linear(augmented[:,~fixed],rhs-augmented[:,fixed]@delta[fixed],
                            bounds=(lo[~fixed],hi[~fixed]),method='bvls',tol=1e-8,max_iter=100)
        if not solution.success:
            raise RuntimeError(f'Bounded IK failed: {solution.message}')
        delta[~fixed]=solution.x
    return torch.as_tensor(delta,device=jacobian.device,dtype=jacobian.dtype)


class PhysicalPandaArm:
    """Measured-pose IK and contact/assisted grasp for one physical parcel.

    ``step_fn`` must advance one environment control step, holding the Go2 at
    its docking pose with the locomotion policy.  It must preserve the arm
    targets: the environment action term must address only the 12 leg joints.
    ``sync`` can be the normal navigation post-step hook; it never moves bodies.
    """

    def __init__(
        self,
        env: Any,
        stage: Any,
        box: Any,
        *,
        hand_path: str,
        box_path: str,
        config: PhysicalPandaConfig,
        step_fn: Callable[[], None] | None = None,
        phase_fn: Callable[[str], None] | None = None,
        robot_key: str = 'robot',
        contact_sensor_prefix: str = '',
    ) -> None:
        import torch
        from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
        from pxr import UsdPhysics

        self.torch = torch
        self.quat_apply = quat_apply
        self.quat_apply_inverse = quat_apply_inverse
        self.quat_conjugate = quat_conjugate
        self.quat_mul = quat_mul
        self.axis_angle_from_quat = axis_angle_from_quat
        self.env = env
        self.robot = env.unwrapped.scene[robot_key]
        self.contact_sensor_prefix = contact_sensor_prefix
        self.stage = stage
        self.box = box
        self.box_path = box_path
        self.hand_path = hand_path
        self.config = config
        self.step_fn = step_fn
        self.phase_fn = phase_fn
        self.device = self.robot.device
        self.arm_ids, arm_names = self.robot.find_joints(list(config.arm_joint_names), preserve_order=True)
        self.finger_ids, finger_names = self.robot.find_joints(list(config.finger_joint_names), preserve_order=True)
        hand_ids, _ = self.robot.find_bodies(config.hand_body_name)
        if len(self.arm_ids) != 7 or len(self.finger_ids) != 2 or len(hand_ids) != 1:
            raise RuntimeError(f"Expected mounted Panda (7+2 joints, one hand), got {arm_names}, {finger_names}, {hand_ids}")
        self.hand_id = hand_ids[0]
        self.jacobian_body_id = self.hand_id - int(self.robot.is_fixed_base)
        self.jacobian_joint_ids = [joint_id + self.robot.num_base_dofs for joint_id in self.arm_ids]
        self.tool_offset = torch.tensor(config.tool_offset, device=self.device, dtype=torch.float32)
        self._downward_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        self._eye6 = torch.eye(6, device=self.device)
        self._arm_target = _tensor(self.robot.data.joint_pos)[0, self.arm_ids].clone()
        self._home_target = self._arm_target.clone()
        self._finger_target = config.finger_open
        self.held = False
        self.state = "waiting"
        self.last_error = ""
        self._lost_grasp_steps = 0
        self._grasp_tool_offset = None
        self._released_once = False
        self.detected_position = None
        self.detected_geometry = None
        self.contact_loss_detection = False
        self.loss_events = []
        prim = stage.GetPrimAtPath(config.constraint_path)
        self.joint = None
        if config.grasp_mode == "assisted":
            self.joint = UsdPhysics.FixedJoint(prim) if prim.IsValid() else author_grasp_joint(
                stage, hand_path, box_path, config.constraint_path
            )
        elif prim.IsValid():
            raise RuntimeError("Contact mode requires a fresh scene without the assisted grasp joint")
        self.verified_lift = 0.0
        self.max_hold_error = 0.0
        self.placement_metrics = None
        self.kinematic_plan = None
        self.kinematic_plan_history = []
        self.kinematic_plan_failures = []
        self._arm_chain = None
        self._planned_moves = []
        self._planned_carry_targets = None
        self._set_phase("waiting")
        print(
            f"PHYSICAL_ARM ready hand={hand_path} arm_joints={arm_names} fingers={finger_names} "
            f"grasp_model={config.grasp_mode} constraint_created={self.joint is not None}",
            flush=True,
        )

    def bind_step(self, step_fn: Callable[[], None]) -> None:
        self.step_fn = step_fn

    def _set_phase(self, phase: str) -> None:
        if self.state != phase:
            print(f"PHYSICAL_ARM phase={phase}", flush=True)
        self.state = phase
        prim = self.stage.GetPrimAtPath(self.box_path)
        if prim.IsValid():
            prim.SetCustomDataByKey("payload_state", phase)
        if self.phase_fn is not None:
            self.phase_fn(phase)

    def _fail(self, reason: str) -> bool:
        self.last_error = reason
        self._set_phase("failed")
        print(f"PHYSICAL_ARM failed reason={reason}", flush=True)
        return False

    def _step(self) -> None:
        if self.step_fn is None:
            raise RuntimeError("PhysicalPandaArm requires a Go2 hold-policy step callback")
        self.apply_targets()
        self.step_fn()

    def apply_targets(self) -> None:
        """Write only arm targets; env.step writes the leg targets separately."""
        self.robot.set_joint_position_target_index(target=self._arm_target.unsqueeze(0), joint_ids=self.arm_ids)
        fingers = self.torch.full((1, 2), self._finger_target, device=self.device)
        self.robot.set_joint_position_target_index(target=fingers, joint_ids=self.finger_ids)

    def prepare(self) -> None:
        """Clear logical grasp state; preserve the world's physical objects."""
        self.disable_constraint()
        self.held = False
        self._lost_grasp_steps = 0
        self._released_once = False
        self._grasp_tool_offset = None
        self.detected_position = None
        self.detected_geometry = None
        self._finger_target = self.config.finger_open
        self._arm_target = _tensor(self.robot.data.joint_pos)[0, self.arm_ids].clone()
        self.last_error = ""
        self.verified_lift = 0.0
        self.max_hold_error = 0.0
        self.placement_metrics = None
        self.apply_targets()
        self.loss_events.clear()
        self.kinematic_plan=None
        self._planned_moves=[]
        self._planned_carry_targets=None
        self._set_phase("waiting")

    def disable_constraint(self) -> None:
        if self.joint is not None:
            self.joint.GetJointEnabledAttr().Set(False)

    def stow_empty(self, steps=120, phase='stow_empty_arm') -> None:
        """Fold with joint targets before observing again; never teleport an arm or parcel."""
        if self.held:
            raise RuntimeError('Cannot fold an arm that still holds a parcel')
        self._finger_target=self.config.finger_open
        self._set_phase(phase)
        start=_tensor(self.robot.data.joint_pos)[0,self.arm_ids].clone()
        for index in range(steps):
            fraction=(index+1)/steps
            blend=fraction*fraction*(3-2*fraction)
            self._arm_target=start+blend*(self._home_target-start)
            self._step()

    def _hand_pose(self) -> tuple[Any, Any]:
        pose = _tensor(self.robot.data.body_link_pose_w)[0, self.hand_id]
        return pose[:3], pose[3:7]

    def _tool_pose(self) -> tuple[Any, Any]:
        position, orientation = self._hand_pose()
        return position + self.quat_apply(orientation, self.tool_offset), orientation

    def _box_pose(self) -> tuple[Any, Any]:
        return _tensor(self.box.data.root_pos_w)[0], _tensor(self.box.data.root_quat_w)[0]

    def _tool_jacobian(self) -> Any:
        _, orientation = self._hand_pose()
        # Lab3's Jacobian is at the link origin in world axes.  Translate its
        # linear rows to the point between the two fingertips.
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[0, self.jacobian_body_id]
        jacobian = jacobian[:, self.jacobian_joint_ids].clone()
        offset_w = self.quat_apply(orientation, self.tool_offset)
        jacobian[:3] += self.torch.cross(
            jacobian[3:].transpose(0, 1), offset_w.expand(7, 3), dim=-1
        ).transpose(0, 1)
        return jacobian

    def _pose_error(self, target_pos: Any, target_quat: Any) -> tuple[Any, Any]:
        position, orientation = self._tool_pose()
        position_error = target_pos - position
        rotation_error = self.axis_angle_from_quat(self.quat_mul(target_quat, self.quat_conjugate(orientation)))
        return position_error, rotation_error

    def _ik_step(self, target_pos: Any, target_quat: Any) -> tuple[float, float]:
        position_error, rotation_error = self._pose_error(target_pos, target_quat)
        position_norm = float(self.torch.linalg.vector_norm(position_error).item())
        rotation_norm = float(self.torch.linalg.vector_norm(rotation_error).item())
        # Meter/radian weighting gives translation enough influence on the
        # small scaled arm.  Clamp Cartesian and joint increments for a visible,
        # continuous approach instead of commanding the end configuration.
        translation = position_error * min(1.0, 0.025 / max(position_norm, 1e-8))
        rotation = rotation_error * min(1.0, 0.12 / max(rotation_norm, 1e-8))
        jacobian = self._tool_jacobian()
        jacobian[3:] *= 0.18
        error = self.torch.cat((translation, rotation * 0.18))
        positions = _tensor(self.robot.data.joint_pos)[0, self.arm_ids]
        limits = _tensor(self.robot.data.joint_pos_limits)[0, self.arm_ids]
        if self.config.ik_joint_limit_aware:
            low=self.torch.maximum(limits[:,0]+.015,positions-.12)
            high=self.torch.minimum(limits[:,1]-.015,positions+.12)
            origin=self._arm_target.clamp(min=low,max=high)
            increment=self.config.max_joint_increment*.5
            lower=(low-origin).clamp(min=-increment)
            upper=(high-origin).clamp(max=increment)
            delta=bounded_damped_increment(jacobian,error*.5,lower,upper,self.config.ik_damping)
            self._arm_target=origin+delta
            self._step()
            return position_norm,rotation_norm
        delta = jacobian.transpose(0, 1) @ self.torch.linalg.solve(
            jacobian @ jacobian.transpose(0, 1) + self.config.ik_damping**2 * self._eye6, error
        )
        delta = delta.clamp(-self.config.max_joint_increment, self.config.max_joint_increment)
        if self.config.ik_joint_centering_gain:
            delta += joint_centering_increment(jacobian,positions,limits,self.config.ik_joint_centering_gain)
            delta = delta.clamp(-self.config.max_joint_increment,self.config.max_joint_increment)
        # Integrate a bounded joint target to remove gravity/PD steady-state
        # error. Resetting the target to measured q each iteration stalls IK.
        desired = self._arm_target + delta * .5
        desired = self.torch.maximum(self.torch.minimum(desired, positions + .12), positions - .12)
        self._arm_target = self.torch.maximum(
            self.torch.minimum(desired, limits[:, 1] - 0.015), limits[:, 0] + 0.015
        )
        self._step()
        return position_norm, rotation_norm

    def _move_tool(self, target_pos: Any, target_quat: Any, phase: str, *, position_tolerance: float | None = None) -> bool:
        self._set_phase(phase)
        if self._planned_moves:
            planned=self._planned_moves.pop(0)
            if float((target_pos-self.torch.tensor(planned['position'],device=self.device)).norm())>1e-5:
                return self._fail('Kinematic path waypoint mismatch')
            import math
            for joint_goal in planned.get('joint_path',[planned['joints']]):
                start=_tensor(self.robot.data.joint_pos)[0,self.arm_ids].clone()
                goal=self.torch.tensor(joint_goal,device=self.device,dtype=start.dtype)
                count=max(12,math.ceil(float((goal-start).abs().max())/(.8*self.env.unwrapped.step_dt)))
                for index in range(count):
                    fraction=(index+1)/count
                    blend=fraction*fraction*(3.-2.*fraction)
                    self._arm_target=start+blend*(goal-start)
                    self._step()
        tolerance=self.config.position_tolerance if position_tolerance is None else position_tolerance
        stable_steps = 0
        for step in range(self.config.max_ik_steps):
            distance, angle = self._ik_step(target_pos, target_quat)
            if distance < tolerance and angle < self.config.orientation_tolerance:
                stable_steps += 1
                if stable_steps >= 6:
                    print(f"PHYSICAL_ARM reached phase={phase} distance={distance:.4f} angle={angle:.3f}", flush=True)
                    return True
            else:
                stable_steps = 0
            if step % 50 == 0:
                tool, _ = self._tool_pose()
                print(
                    f"PHYSICAL_ARM progress phase={phase} step={step} distance={distance:.4f} "
                    f"angle={angle:.3f} tool={tool.detach().cpu().tolist()} joints={_tensor(self.robot.data.joint_pos)[0,self.arm_ids].tolist()}",
                    flush=True,
                )
        return self._fail(f"{phase}: IK timeout; position error={distance:.3f}m orientation error={angle:.3f}rad")

    def _preflight_tool_path(self, points, target_quat, operation, *, first_minimum_world_z=None):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from .arm_kinematics import ArmChain,ArmPlanningError,pose_matrix
        if self._arm_chain is None:
            self._arm_chain=ArmChain.from_stage(self.stage,self.hand_path.rsplit('/',1)[0],self.config.tool_offset)
        base_ids,_=self.robot.find_bodies('panda_link0')
        pose=_tensor(self.robot.data.body_link_pose_w)[0,base_ids[0]].cpu().numpy()
        positions=_tensor(self.robot.data.joint_pos)[0,self.arm_ids].cpu().numpy()
        base=pose_matrix(pose[:3],pose[3:])
        predicted,local_jacobian=self._arm_chain.forward(positions,with_jacobian=True)
        predicted=base@predicted
        tool,orientation=self._tool_pose()
        position_error=float(np.linalg.norm(predicted[:3,3]-tool.cpu().numpy()))
        angle_error=float(np.linalg.norm(Rotation.from_matrix(predicted[:3,:3]@Rotation.from_quat(orientation.cpu().numpy()).as_matrix().T).as_rotvec()))
        predicted_jacobian=np.vstack((base[:3,:3]@local_jacobian[:3],base[:3,:3]@local_jacobian[3:]))
        jacobian_error=float(np.abs(predicted_jacobian-self._tool_jacobian().cpu().numpy()).max())
        verification=dict(position_m=position_error,angle_rad=angle_error,jacobian_max_error=jacobian_error)
        print(f'ARM_KINEMATICS_CHECK {verification}',flush=True)
        if position_error>.002 or angle_error>.01 or jacobian_error>.005:
            raise RuntimeError(f'USD arm kinematics disagree with live articulation: {verification}')
        limits=_tensor(self.robot.data.joint_pos_limits)[0,self.arm_ids].cpu().numpy()
        try:
            planner=self._arm_chain.plan_refined if self.config.ik_path_refinement else self._arm_chain.plan
            options={}
            if operation in ('grasp','contact_righting') and self.config.ik_grasp_symmetry:
                planner=self._arm_chain.plan_parallel_grasp
                options['refine']=self.config.ik_path_refinement
            plan=planner([(point,target_quat.cpu().tolist()) for point in points],
                (pose[:3],pose[3:]),positions,limits,attempts=12,first_minimum_world_z=first_minimum_world_z,**options)
        except ArmPlanningError as error:
            report=dict(operation=operation,live_model_verification=verification,**error.diagnostics)
            self.kinematic_plan_failures.append(report)
            print(f'ARM_KINEMATIC_REJECT operation={operation} candidates={report["candidates"]}',flush=True)
            raise
        plan['live_model_verification']=verification
        plan['operation']=operation
        plan['first_minimum_world_z']=first_minimum_world_z
        plan['measured_arm_base_pose_xyzw']=pose.tolist()
        plan['joint_limits_rad']=limits.tolist()
        self.kinematic_plan=plan
        paths=plan.get('segment_joint_paths',[[q] for q in plan['joints']])
        self._planned_moves=[dict(position=point,joints=joints,joint_path=path)
                             for point,joints,path in zip(points,plan['joints'],paths)]
        print(f'ARM_KINEMATIC_PLAN operation={operation} seed={plan["seed_index"]} margin_rad={plan["minimum_joint_margin_rad"]:.4f} targets={len(points)}',flush=True)
        if plan.get('refinement',{}).get('used'):
            print(f'ARM_PATH_REFINED operation={operation} waypoints={plan["refinement"]["refined_waypoints"]}',flush=True)
        if plan.get('grasp_symmetry',{}).get('used'):
            print('ARM_GRASP_SYMMETRY local_tool_z_rotation_rad=3.141592653589793',flush=True)
        return plan

    def _preflight_grasp_path(self, above, target, target_quat):
        root=_tensor(self.robot.data.root_pos_w)[0,:2].cpu().tolist()
        self._planned_carry_targets=carry_lift_waypoints(target.cpu().tolist(),root,
            height=self.config.lift_height,segments=self.config.lift_segments,
            retraction=self.config.carry_retraction_m,clearance=self.config.carry_retraction_clearance_m)
        points=[above.cpu().tolist()]+[(above+fraction*(target-above)).cpu().tolist() for fraction in (.25,.5,.75,1.)]+self._planned_carry_targets
        plan=self._preflight_tool_path(points,target_quat,'grasp')
        return self.torch.tensor(plan['world_targets'][0]['quaternion_xyzw'],
                                 device=self.device,dtype=target_quat.dtype)

    def _enable_grasp(self) -> None:
        if self.config.grasp_mode == "contact":
            if self.stage.GetPrimAtPath(self.config.constraint_path).IsValid():
                raise RuntimeError("Unexpected assisted constraint in contact mode")
            box_pos, _ = self._box_pose()
            tool_pos, hand_quat = self._tool_pose()
            self._grasp_tool_offset = self.quat_apply_inverse(hand_quat, box_pos - tool_pos).clone()
            self.held = True
            self._lost_grasp_steps = 0
            return
        from pxr import Gf

        hand_pos, hand_quat = self._hand_pose()
        box_pos, box_quat = self._box_pose()
        local_pos = self.quat_apply_inverse(hand_quat, box_pos - hand_pos)
        local_quat = self.quat_mul(self.quat_conjugate(hand_quat), box_quat)
        xyz = [float(v) for v in local_pos.detach().cpu().tolist()]
        qx, qy, qz, qw = [float(v) for v in local_quat.detach().cpu().tolist()]
        # The joint frames use the current measured poses.  Enabling the joint
        # therefore preserves the parcel's position rather than snapping it.
        self.joint.GetLocalPos0Attr().Set(Gf.Vec3f(*xyz))
        self.joint.GetLocalRot0Attr().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
        self.joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        self.joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0))
        self.joint.GetJointEnabledAttr().Set(True)
        tool_pos, _ = self._tool_pose()
        self._grasp_tool_offset = self.quat_apply_inverse(hand_quat, box_pos - tool_pos).clone()
        self.held = True
        self._lost_grasp_steps = 0

    def grasp(self, object_id: str) -> bool:
        if self.held:
            # A retained grip after an IK timeout is not a completed grasp.
            return self.verified_lift>.07
        if not object_id:
            return self._fail("empty object id")
        self.last_error = ""
        self._finger_target = self.config.finger_open
        if self.config.reset_before_grasp:
            self.stow_empty(phase='prepare_grasp_posture')
        for step in range(60):
            self._step()
            if step % 10 == 0:
                print(f"ARM_STAND step={step} pose={_tensor(self.robot.data.root_pos_w)[0].tolist()} gravity={_tensor(self.robot.data.projected_gravity_b)[0].tolist()}", flush=True)
        print(f"PHYSICAL_ARM standing_height={float(_tensor(self.robot.data.root_pos_w)[0,2]):.3f}", flush=True)
        box_pos, box_quat = self._box_pose()
        initial_box_height = float(box_pos[2].item())
        target_quat = self.quat_mul(box_quat.clone(), self._downward_quat)
        if self.detected_geometry is not None:
            target_quat = self.torch.tensor(self.detected_geometry['hand_quaternion_xyzw'],device=self.device)
            if self.detected_geometry['grasp_width_m'] > 2*self.config.finger_open + .014:
                return self._fail('Observed parcel width exceeds gripper aperture')
        if self.config.grasp_yaw_offset:
            import math
            yaw=self.config.grasp_yaw_offset/2
            yaw_quat=self.torch.tensor((0.,0.,math.sin(yaw),math.cos(yaw)),device=self.device)
            target_quat=self.quat_mul(yaw_quat,target_quat)
        if self.detected_position is None:
            return self._fail("RGB-D parcel localization required before grasp")
        target = self.torch.tensor(self.detected_position, device=self.device)
        grasp_offset = self.config.grasp_height_offset
        if self.detected_geometry is not None and self.detected_geometry['height_m'] < .05:
            # Thin face-down parcels need the finger pads below the top face;
            # the upright offset would merely press their top while closing.
            grasp_offset = .002
        target[2] += grasp_offset
        above = target.clone()
        above[2] += self.config.approach_height
        if self.config.ik_path_preflight:
            target_quat=self._preflight_grasp_path(above,target,target_quat)
            self.kinematic_plan['object_id']=object_id
            self.kinematic_plan_history.append(self.kinematic_plan)
        if not self._move_tool(above, target_quat, "pregrasp"):
            return False
        # Follow a short Cartesian descent: a single distant IK target can
        # sweep the fingertips sideways into the narrow parcel before closing.
        for fraction in (.25,.50,.75,1.):
            descent = above + fraction*(target-above)
            if not self._move_tool(descent, target_quat, "approach"):
                return False
        self._set_phase("closing_fingers")
        if self.detected_geometry is not None and self.config.finger_open > .04:
            for side in ('left','right'):
                ids,_=self.robot.find_bodies('panda_'+side+'finger')
                finger_pose=_tensor(self.robot.data.body_link_pose_w)[0,ids[0]]
                hp,hq=self._hand_pose()
                relative=self.quat_apply_inverse(hq,finger_pose[:3]-hp)
                print(f'GRIP_GEOMETRY side={side} position_in_hand={relative.tolist()} quat={finger_pose[3:].tolist()}',flush=True)
        self._finger_target = 0.0
        for _ in range(self.config.close_steps):
            self._ik_step(target, target_quat)
        tool_pos, hand_quat = self._tool_pose()
        box_pos, _ = self._box_pose()
        local_error = self.quat_apply_inverse(hand_quat, box_pos - tool_pos)
        distance = float(self.torch.linalg.vector_norm(box_pos - tool_pos).item())
        finger_positions = _tensor(self.robot.data.joint_pos)[0, self.finger_ids]
        finger_gap = float(finger_positions.sum().item())
        rotation_error = float(self.torch.linalg.vector_norm(self._pose_error(target, target_quat)[1]).item())
        # Gate on actual joint closure and geometry, not the commanded target.
        # A parcel displaced outside the finger channel cannot become attached.
        aligned = abs(float(local_error[1].item())) < 0.014 and abs(float(local_error[0].item())) < 0.028
        closed = finger_gap < 2.0 * self.config.finger_open - 0.004
        lateral_contact = True
        if self.detected_geometry is not None:
            forces = []
            for side in ('left','right'):
                sensor = self.env.unwrapped.scene.sensors[self.contact_sensor_prefix+f'{side}_grip_contact']
                matrix = _tensor(sensor.data.force_matrix_w)
                forces.append(float(matrix[0,...,:2].norm(dim=-1).sum()))
            lateral_contact = min(forces) > .2
            if self.config.finger_open > .04:
                # The custom parallel pads close at approximately the parcel
                # width. A nearly closed jaw below a wide box is wedging, not
                # a valid opposing-face grasp, even if it can raise the box.
                lateral_contact = lateral_contact and abs(finger_gap-self.detected_geometry['grasp_width_m']) < .015
            print(f'PHYSICAL_GRASP lateral_contact_n={forces}',flush=True)
        print(
            f"PHYSICAL_GRASP gate distance={distance:.4f} finger_gap={finger_gap:.4f} "
            f"orientation={rotation_error:.3f} aligned={aligned} closed={closed}",
            flush=True,
        )
        if not (distance < self.config.grasp_distance_limit and aligned and closed and rotation_error < 0.30 and lateral_contact):
            self._finger_target = self.config.finger_open
            self.apply_targets()
            return self._fail("parcel not enclosed by closed gripper; constraint remains disabled")
        self._enable_grasp()
        print(f"PHYSICAL_GRASP mode={self.config.grasp_mode} constraint_enabled={self.joint is not None}", flush=True)
        # Long single-target IK can approach a wrist limit while prioritizing
        # translation. Short vertical targets preserve orientation throughout
        # the lift instead of trying to correct it only at the final height.
        root_xy=_tensor(self.robot.data.root_pos_w)[0,:2].detach().cpu().tolist()
        waypoints=self._planned_carry_targets or carry_lift_waypoints(target.detach().cpu().tolist(),root_xy,
            height=self.config.lift_height,segments=self.config.lift_segments,
            retraction=self.config.carry_retraction_m,clearance=self.config.carry_retraction_clearance_m)
        print(f'PHYSICAL_CARRY root_xy={root_xy} retraction_m={self.config.carry_retraction_m} targets={waypoints}',flush=True)
        for waypoint in waypoints:
            lifted = self.torch.tensor(waypoint,device=self.device,dtype=target.dtype)
            if not self._move_tool(lifted, target_quat, "lifting", position_tolerance=self.config.lift_position_tolerance):
                # A failed trajectory is not proof the fingers are empty.
                # Preserve the grip and stop; the mission must not fold a
                # loaded arm or create a drop as an ordinary retry side effect.
                return False
        box_pos, _ = self._box_pose()
        rise = float(box_pos[2].item()) - initial_box_height
        self.verified_lift = rise
        if rise < 0.07:
            self.disable_constraint()
            self.held = False
            return self._fail(f"gripper did not lift physical parcel (rise={rise:.3f}m)")
        self._set_phase("carried")
        print(f"PHYSICAL_GRASP verified_lift={rise:.3f}m parcel={box_pos.detach().cpu().tolist()}", flush=True)
        return True

    def sync(self, pose: Any = None) -> None:
        """Check grasp integrity during walking, without writing body poses."""
        del pose
        if not self.held or self._grasp_tool_offset is None:
            return
        tool_pos, hand_quat = self._tool_pose()
        box_pos, _ = self._box_pose()
        expected = tool_pos + self.quat_apply(hand_quat, self._grasp_tool_offset)
        error = float(self.torch.linalg.vector_norm(expected - box_pos).item())
        self.max_hold_error = max(self.max_hold_error, error)
        threshold = .04 if self.config.grasp_mode == "contact" else .09
        if self.contact_loss_detection:
            # Filtered finger-to-parcel contact sensors. No parcel pose enters
            # the loss decision; pose discrepancy above remains evaluation only.
            force = 0.0
            for side in ('left', 'right'):
                sensor = self.env.unwrapped.scene.sensors[self.contact_sensor_prefix+f'{side}_grip_contact']
                matrix = _tensor(sensor.data.force_matrix_w)
                force += float(matrix[0].norm(dim=-1).sum())
            lost = force < .05
            debounce = 10
            reason = f'finger contact absent for 0.20s; force={force:.4f}N'
        else:
            lost = error > threshold
            debounce = 5
            reason = f'physical grasp lost; tool-to-parcel displacement={error:.3f}m'
        self._lost_grasp_steps = self._lost_grasp_steps + 1 if lost else 0
        if self._lost_grasp_steps >= debounce:
            self.held = False
            self.loss_events.append(dict(reason=reason, tool_position=tool_pos.detach().cpu().tolist(),
                                         source='finger_contact' if self.contact_loss_detection else 'pose_discrepancy',
                                         contact_force_n=force if self.contact_loss_detection else None,
                                         debounce_steps=debounce,
                                         evaluation_displacement_m=error))
            self._fail(reason)
            from .mission import PayloadLost
            raise PayloadLost(self.last_error)

    def release(self) -> bool:
        for step in range(60):
            self._step()
            if step % 10 == 0:
                print(f"PLACE_STAND step={step} pose={_tensor(self.robot.data.root_pos_w)[0].tolist()} gravity={_tensor(self.robot.data.projected_gravity_b)[0].tolist()}", flush=True)
        if not self.held:
            if self._released_once:
                return self._verify_and_stow()
            return self._fail("no physical parcel held")
        tool_pos, hand_quat = self._tool_pose()
        box_pos, box_quat = self._box_pose()
        desired_box = self.torch.tensor(self.config.dropoff_position, device=self.device)
        # Preserve measured box-in-hand rotation and request an upright box.
        # Contact transport can rotate the parcel slightly within the fingers;
        # blindly restoring the hand's nominal downward pose preserves that tilt.
        box_in_hand = self.quat_mul(self.quat_conjugate(hand_quat),box_quat)
        # Keep current parcel yaw: straightening roll/pitch does not require
        # twisting the wrist to a warehouse-fixed heading after belt feeding.
        import math
        qx,qy,qz,qw=(float(v) for v in box_quat)
        yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        upright=self.torch.tensor((0.,0.,math.sin(yaw/2),math.cos(yaw/2)),device=self.device)
        target_quat = self.quat_mul(upright,self.quat_conjugate(box_in_hand))
        if self.detected_geometry is not None and self.detected_geometry['height_m'] < .05:
            # Preserve the face selected from RGB-D instead of rotating every
            # package to its original authored upright orientation. Correct
            # hand tilt while retaining its present yaw, avoiding a 90deg flip.
            hx,hy,hz,hw = (float(v) for v in hand_quat)
            hand_yaw = math.atan2(2*(hx*hy+hw*hz),1-2*(hy*hy+hz*hz))
            target_quat = self.torch.tensor((math.cos(hand_yaw/2),math.sin(hand_yaw/2),0.,0.),device=self.device)
        # Preserve the measured tool-to-parcel offset; place the physical parcel
        # slightly above the pedestal and let contact/gravity settle it.
        box_from_tool = self.quat_apply_inverse(hand_quat, box_pos - tool_pos)
        target = desired_box - self.quat_apply(target_quat, box_from_tool)
        target[2] += 0.008
        above = target.clone()
        above[2] += self.config.placement_approach_height
        retreat = target.clone()
        retreat[2] += self.config.release_retreat_height
        if self.config.ik_path_preflight:
            # The held parcel must clear the deck throughout the first joint
            # transition. This point corridor is not full-link collision QA.
            minimum=max(float(target[2])+.01,min(float(tool_pos[2]),float(above[2]))-.015)
            identity=self.kinematic_plan_history[-1].get('object_id') if self.kinematic_plan_history else None
            self._preflight_tool_path([point.cpu().tolist() for point in (above,target,retreat)],target_quat,
                'place',first_minimum_world_z=minimum)
            self.kinematic_plan['object_id']=identity
            self.kinematic_plan_history.append(self.kinematic_plan)
        # This is a clearance waypoint 13cm above release, sometimes near the
        # elbow limit. Final lowering and placement keep their tighter checks.
        if not self._move_tool(above, target_quat, "place_preapproach",position_tolerance=.025):
            return False
        if not self._move_tool(target, target_quat, "lowering"):
            return False
        self._set_phase("opening_fingers")
        self.disable_constraint()
        self.held = False
        self._released_once = True
        self._finger_target = self.config.finger_open
        for _ in range(25):
            self._ik_step(target,target_quat)
        retreat = target.clone()
        retreat[2] += self.config.release_retreat_height
        if not self._move_tool(retreat, target_quat, "retreating"):
            return False
        for _ in range(self.config.settle_steps):
            self._step()
        return self._verify_and_stow()

    def _verify_and_stow(self) -> bool:
        if not self._verify_placement():
            return False
        if self.config.stow_after_release:
            self.stow_empty(steps=150)
            # Folding must not knock a previously placed parcel off its pad.
            return self._verify_placement()
        return True

    def _verify_placement(self) -> bool:
        box_pos, box_quat = self._box_pose()
        desired = self.torch.tensor(self.config.dropoff_position, device=self.device)
        planar_error = float(self.torch.linalg.vector_norm(box_pos[:2] - desired[:2]).item())
        height_error = abs(float((box_pos[2] - desired[2]).item()))
        speed = float(self.torch.linalg.vector_norm(_tensor(self.box.data.root_lin_vel_w)[0]).item())
        nominal_height_error=height_error
        footprint_fits=None
        if self.config.placement_surface_z is not None:
            # Use the authored cuboid dimensions and measured orientation.
            # A side-resting parcel has a different support height; its center
            # alone cannot prove contact with the shelf or a fitting footprint.
            dimensions=self.torch.as_tensor(self.box.cfg.spawn.size,device=self.device,dtype=box_pos.dtype)
            if not bool(self.torch.isfinite(dimensions).all()) or not bool((dimensions>0).all()):
                return self._fail('Invalid measured shelf parcel dimensions')
            if not bool(self.torch.isfinite(box_quat).all()) or abs(float(box_quat.norm())-1.)>.01:
                return self._fail('Invalid measured shelf parcel orientation')
            axes=self.quat_apply(box_quat.expand(3,4),self.torch.eye(3,device=self.device,dtype=box_pos.dtype))
            extents=axes.abs().T @ (dimensions*.5)
            height_error=abs(float(box_pos[2]-extents[2])-self.config.placement_surface_z)
            footprint_fits=all(abs(float(box_pos[i]-desired[i]))+float(extents[i])
                <=self.config.placement_deck_size_xy[i]/2 for i in (0,1))
        success=(planar_error < self.config.placement_planar_tolerance
            and height_error < self.config.placement_height_tolerance and speed < .10
            and footprint_fits is not False)
        self.placement_metrics = dict(success=success, planar_error_m=planar_error, height_error_m=height_error,
            nominal_center_height_error_m=nominal_height_error,speed_mps=speed,footprint_within_deck=footprint_fits)
        print(
            f"PHYSICAL_RELEASE verified={success} xy_error={planar_error:.4f} "
            f"height_error={height_error:.4f} speed={speed:.4f} parcel={box_pos.detach().cpu().tolist()}",
            flush=True,
        )
        if not success:
            return self._fail("parcel did not settle on the destination pedestal")
        self._set_phase("released")
        return True
