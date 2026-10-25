"""Experimental, bounded fingertip push to expose a fallen parcel's top label.

Planning uses a fresh associated QR and visible box dimensions. Execution is
only through existing joint targets, with contact-force and iteration bounds.
A completed push is never declared a successful righting without reobserving.
"""
import math
import numpy as np


def plan_contact_righting(detection):
    geometry=detection['geometry']
    center=np.asarray(geometry['center_world'],dtype=float)
    label=np.asarray(detection['side_label']['center_world'],dtype=float)
    height=geometry['height_m']
    if not .025<=height<=.045 or abs(center[2]-height/2)>.015:
        raise ValueError('Righting supports an isolated thin face on the floor')
    yaw=geometry['yaw_rad']
    short=np.array([-math.sin(yaw),math.cos(yaw)])
    side=float(np.dot(label[:2]-center[:2],short))
    if abs(side)<.025 or abs(label[2]-center[2])>.025:
        raise ValueError('A freshly observed short-edge side label is required')
    outward=short*math.copysign(1.,side)
    width=min(geometry['observed_face_m'])
    # Horizontal pushing slid the real box without tipping. Lift the labelled
    # near edge about the opposite floor edge instead: friction at the closed
    # tapered fingers may supply upward traction while the far edge supports
    # the box. This is a reference hypothesis, not an assumed contact pivot.
    # The pusher stays vertical; actual slip and rotation must be measured.
    floor_z=float(center[2]-height/2)
    pivot=np.r_[center[:2]-outward*width/2,floor_z]
    # The previous top-corner contact disappeared on the first upward step.
    # Engage the side face below its top; the stock fingertip extends 7.6 mm
    # below the configured tool point, still leaving >15 mm floor clearance.
    start=np.r_[center[:2]+outward*(width/2+.010),center[2]+.006]
    outside=start.copy();outside[:2]+=outward*.025
    above=outside.copy();above[2]+=.13
    radius=width+.010
    initial_z=start[2]-floor_z
    # Stop just before the reference's geometric high point. The 100-degree
    # experiment physically righted the parcel near 75 degrees, then pressed
    # the upright top and overloaded. Retract vertically after this release
    # hypothesis; only fresh RGB-D can accept righting.
    terminal_angle=math.radians(75.)
    angles=np.r_[np.linspace(0.,math.radians(40.),21)[1:],
                 np.linspace(math.radians(40.),terminal_angle,9)[1:]]
    arc=[np.r_[pivot[:2]+outward*(radius*math.cos(a)-initial_z*math.sin(a)),
               floor_z+radius*math.sin(a)+initial_z*math.cos(a)] for a in angles]
    clear=arc[-1].copy();clear[2]=max(.30,float(clear[2])+.16)
    points=[above,outside,start]+arc+[clear]
    normals=[np.r_[outward,0.]]*3+[
        np.r_[outward*math.cos(a),math.sin(a)] for a in angles]+[np.r_[outward,0.]]
    return dict(points=[p.tolist() for p in points],
        point_roles=['approach','approach','contact']+['edge_lift']*len(arc)+['clear'],
        surface_normals=[n.tolist() for n in normals],
        hand_quaternion_xyzw=geometry['hand_quaternion_xyzw'],
        strategy='force_regulated_side_face_upward_arc',
        hypothetical_pivot_world=pivot.tolist(),arc_angle_rad=terminal_angle,
        minimum_edge_contact_force_n=.2,
        normal_force_target_n=.55,normal_admittance_m_per_ns=.025,
        normal_offset_bounds_m=[-.004,.012],maximum_offset_step_m=.00025,
        contact_loss_timeout_s=.5,maximum_reference_step_m=.001,
        max_contact_force_n=20.,max_steps_per_point=100,
        scope='one contact arc; no physical pivot, righting or grasp success implied')


def force_regulated_offset(offset, normal_force, dt, plan):
    """Bounded position correction from measured contact, without pose truth."""
    change=plan['normal_admittance_m_per_ns']*(plan['normal_force_target_n']-normal_force)*dt
    bound=plan['maximum_offset_step_m']
    return float(np.clip(offset+np.clip(change,-bound,bound),*plan['normal_offset_bounds_m']))


def execute_contact_righting(arm,detection,record):
    try:
        plan=plan_contact_righting(detection)
    except ValueError as error:
        record(dict(event='contact_righting',success=False,reason=str(error)))
        return False
    arm._finger_target=0.
    orientation=arm.torch.tensor(plan['hand_quaternion_xyzw'],device=arm.device)
    # The empty parallel fingers can exchange sides. Plan the whole path
    # before contact so a wrist limit cannot strand the pusher at floor level.
    from .arm_kinematics import ArmPlanningError
    try:
        kinematics=arm._preflight_tool_path(plan['points'],orientation,'contact_righting',
                                          first_minimum_world_z=.10)
    except ArmPlanningError as error:
        record(dict(event='contact_righting',success=False,reason='kinematic_preflight_rejected',
                    plan=plan,kinematics=error.diagnostics,steps=0,completed_points=0,
                    maximum_finger_force_n=0.))
        return False
    orientation=arm.torch.tensor(kinematics['world_targets'][0]['quaternion_xyzw'],device=arm.device)
    # Consume only this operation's preflight, never leave stale waypoints for
    # a later grasp. Every interpolation step below retains the force guard.
    moves=arm._planned_moves
    arm._planned_moves=[]
    arm._set_phase('contact_righting_close')
    for _ in range(30):arm._step()
    maximum_force=0.;steps=0;reached=0;reason='reference_completed'
    errors=[];feedback=[];offset=0.;lost_time=0.
    def contact_measurement():
        force=0.;vector=np.zeros(3)
        for side in ('left','right'):
            sensor=arm.env.unwrapped.scene.sensors[arm.contact_sensor_prefix+f'{side}_grip_contact']
            matrix=getattr(sensor.data.force_matrix_w,'torch',sensor.data.force_matrix_w)
            force+=float(matrix[0].norm(dim=-1).sum())
            vector+=matrix[0].reshape(-1,3).sum(dim=0).detach().cpu().numpy()
        return force,vector
    def finger_force():return contact_measurement()[0]
    for index,point in enumerate(plan['points']):
        role=plan['point_roles'][index]
        arm._set_phase('contact_righting_'+role)
        target=arm.torch.tensor(point,device=arm.device)
        if role=='edge_lift':
            # Nominal joints establish reachability, but replaying them here
            # would erase force corrections. Continue with the same bounded
            # differential IK and a slow Cartesian reference instead.
            previous=np.asarray(plan['points'][index-1])
            nominal=np.asarray(point);normal=np.asarray(plan['surface_normals'][index])
            fraction=0.;stable=0;distance=angle=float('inf')
            for _ in range(plan['max_steps_per_point']):
                force,vector=contact_measurement()
                maximum_force=max(maximum_force,force)
                if force>plan['max_contact_force_n']:
                    reason='contact_force_guard';break
                normal_force=abs(float(np.dot(vector,normal)))
                dt=arm.env.unwrapped.step_dt
                lost_time=lost_time+dt if force<.05 else 0.
                if lost_time>=plan['contact_loss_timeout_s']-1e-9:
                    reason='edge_contact_lost';break
                offset=force_regulated_offset(offset,normal_force,dt,plan)
                fraction=min(1.,fraction+plan['maximum_reference_step_m']/max(float(np.linalg.norm(nominal-previous)),1e-9))
                corrected=previous+fraction*(nominal-previous)-normal*offset
                distance,angle=arm._ik_step(arm.torch.tensor(corrected,device=arm.device,dtype=target.dtype),orientation)
                steps+=1
                after_force=finger_force();maximum_force=max(maximum_force,after_force)
                feedback.append(dict(step=steps,index=index,fraction=fraction,normal_force_n=normal_force,
                    total_force_n=force,force_vector_w_n=vector.tolist(),normal_offset_m=offset,
                    target_world=corrected.tolist()))
                if after_force>plan['max_contact_force_n']:
                    reason='contact_force_guard';break
                stable=stable+1 if fraction>=1. and distance<.003 and angle<arm.config.orientation_tolerance else 0
                if stable>=6:
                    reached+=1;break
            else:reason='bounded_ik_timeout'
            errors.append(dict(index=index,role=role,position_error_m=distance,
                orientation_error_rad=angle,position_tolerance_m=.003))
            if reason!='reference_completed':break
            continue
        for joint_goal in moves[index].get('joint_path',[moves[index]['joints']]):
            positions=getattr(arm.robot.data.joint_pos,'torch',arm.robot.data.joint_pos)
            start=positions[0,arm.arm_ids].clone()
            goal=arm.torch.tensor(joint_goal,device=arm.device,dtype=start.dtype)
            count=max(12,math.ceil(float((goal-start).abs().max())/(.8*arm.env.unwrapped.step_dt)))
            for tick in range(count):
                fraction=(tick+1)/count
                blend=fraction*fraction*(3-2*fraction)
                arm._arm_target=start+blend*(goal-start)
                arm._step();steps+=1
                force=finger_force();maximum_force=max(maximum_force,force)
                if force>plan['max_contact_force_n']:
                    reason='contact_force_guard';break
            if reason!='reference_completed':break
        if reason!='reference_completed':break
        tolerance=arm.config.position_tolerance if role in ('approach','clear') else .003
        stable=0
        for _ in range(plan['max_steps_per_point']):
            distance,angle=arm._ik_step(target,orientation)
            steps+=1
            force=finger_force()
            maximum_force=max(maximum_force,force)
            if force>plan['max_contact_force_n']:
                reason='contact_force_guard'
                break
            stable=stable+1 if distance<tolerance and angle<arm.config.orientation_tolerance else 0
            if stable>=6:
                reached+=1
                break
        else:reason='bounded_ik_timeout'
        errors.append(dict(index=index,role=role,position_error_m=distance,
                           orientation_error_rad=angle,position_tolerance_m=tolerance))
        if role=='contact' and reason=='reference_completed' and finger_force()<plan['minimum_edge_contact_force_n']:
            reason='edge_contact_not_established'
        if reason!='reference_completed':break
    completed=reached==len(plan['points'])
    record(dict(event='contact_righting',success=completed,reason=reason,plan=plan,
        completed_points=reached,steps=steps,maximum_finger_force_n=maximum_force,
        waypoint_errors=errors,kinematics=kinematics,contact_feedback=feedback,
        scope='reference execution only; fresh RGB-D must verify the outcome'))
    return completed
