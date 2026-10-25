"""Independently check scoped placement and loaded-recovery allocation."""
import json
import math
from pathlib import Path


def motor_envelope_matches(motor,state,previous):
    """Recompute original Go2 DC/PD equations without importing the controller."""
    fields=('stiffness','damping','effort_limit','velocity_limit','saturation_effort',
        'position_rad','velocity_radps','position_target_rad','velocity_target_radps','pd_bias_nm',
        'lower_nm','upper_nm','feedforward_nm','predicted_total_nm')
    if not isinstance(motor,dict) or previous is None:return False
    if any(not isinstance(motor.get(k),list) or len(motor[k])!=12
           or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in motor[k]) for k in fields):return False
    if motor.get('active') is not True or motor.get('margin_nm')!=.1:return False
    for key,expected in [('stiffness',25.),('damping',.5),('effort_limit',23.5),
                         ('velocity_limit',30.),('saturation_effort',23.5)]:
        if any(abs(x-expected)>1e-6 for x in motor[key]):return False
    for key,measured in [('position_rad',previous.get('leg_position_rad')),
                          ('velocity_radps',previous.get('leg_velocity_radps')),
                          ('position_target_rad',state.get('leg_commanded_position_rad'))]:
        if measured is None or len(measured)!=12 or any(abs(x-y)>2e-6 for x,y in zip(motor[key],measured)):return False
    actual=state.get('leg_computed_torque_nm');applied=state.get('leg_applied_torque_nm')
    if (actual is None or applied is None or len(actual)!=12 or len(applied)!=12
            or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (*actual,*applied))):return False
    for i in range(12):
        # Zero is the original locomotion actuator's velocity target.
        if abs(motor['velocity_target_radps'][i])>1e-9:return False
        velocity=max(-60.,min(60.,motor['velocity_radps'][i]))
        lo=max(23.5*(-1.-velocity/30.),-23.5);hi=min(23.5*(1.-velocity/30.),23.5)
        bias=25.*(motor['position_target_rad'][i]-motor['position_rad'][i])-.5*motor['velocity_radps'][i]
        total=bias+motor['feedforward_nm'][i]
        if (abs(bias-motor['pd_bias_nm'][i])>1e-5
                or abs(lo-motor['lower_nm'][i])>1e-5 or abs(hi-motor['upper_nm'][i])>1e-5
                or abs(total-motor['predicted_total_nm'][i])>1e-5
                or not lo+.1-1e-4<=total<=hi-.1+1e-4
                or abs(total-actual[i])>.002 or abs(actual[i]-applied[i])>1e-4):return False
    return True


def joint_target_matches(report,state,previous,period):
    """Independently recompute the unchanged recovery/exit rate and position box."""
    if not isinstance(report,dict) or previous is None:return False
    fields=('previous_target_rad','nominal_target_rad','selected_target_rad','position_lower_rad','position_upper_rad')
    if (report.get('enabled') is not True or report.get('qp_status')!=0
            or report.get('feasibility_lp_status')!=0 or report.get('qp_solution_feasible') is not True
            or report.get('constraint_set_feasible') is not True
            or any(not isinstance(report.get(k),list) or len(report[k])!=12
                or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in report[k]) for k in fields)):
        return False
    old=previous.get('leg_commanded_position_rad');current=state.get('leg_commanded_position_rad')
    if old is None or current is None or len(old)!=12 or len(current)!=12:return False
    rate=1.2 if (state.get('balance_recovery') or {}).get('active') is True else 4.
    step=rate*period
    if abs(report.get('max_step_rad',float('inf'))-step)>1e-9:return False
    for i in range(12):
        lo,hi=report['position_lower_rad'][i],report['position_upper_rad'][i]
        if (lo>=hi or not math.isfinite(old[i]) or not math.isfinite(current[i])
                or abs(report['previous_target_rad'][i]-old[i])>2e-6
                or abs(report['selected_target_rad'][i]-current[i])>2e-6
                or abs(current[i]-old[i])>step+2e-6
                or not min(lo,old[i])-2e-6<=current[i]<=max(hi,old[i])+2e-6):return False
    return True


def replay(rows,period, *, placement_qp=True, loaded_recovery_qp=False, receiving_brake=False,
           total_motor_qp=False,joint_motor_qp=False):
    count=0;issues=[];active={};loaded_active={};clipping={};previous={};motor_active={}
    for row in rows:
        count+=1
        if (row['tick']!=count or not math.isfinite(row['sim_time'])
                or abs(row['sim_time']-count*period)>1.e-8):
            if len(issues)<12:issues.append(dict(tick=row['tick'],reason='control clock'))
        for name,r in row['robots'].items():
            command=r['commanded_velocity'];balance=r.get('balance_recovery') or {}
            placement=bool(placement_qp and r['phase']=='placing' and r['payload_held'] and len(command)==3
                and all(math.isfinite(v) for v in command) and max(abs(v) for v in command)<.05)
            loaded_phase=r['phase']=='transporting' or receiving_brake and r['phase']=='leaving_receiving'
            loaded=bool(loaded_recovery_qp and r.get('recovering') is True
                and loaded_phase and r['payload_held']
                and (r.get('original_z1') or {}).get('state')=='carried' and len(command)==3
                and all(math.isfinite(v) and abs(v)<1e-9 for v in command))
            eligible=placement or loaded
            # Once the controller has faded out it does not run allocation.
            executed=balance.get('active') is True or balance.get('blend',0.)>0.
            active.setdefault(name,0);loaded_active.setdefault(name,0);clipping.setdefault(name,0.)
            prior=previous.get(name);previous[name]=r;motor_active.setdefault(name,0)
            if not executed:continue
            expected='wrench_qp' if eligible else 'sequential'
            valid=(balance.get('allocator')==expected and balance.get('base_allocator')=='sequential'
                and balance.get('stationary_payload_qp_scope') is placement
                and balance.get('loaded_recovery_qp_scope',False) is loaded)
            raw=balance.get('raw_feedforward_max_nm');clip=balance.get('feedforward_clipping_max_nm')
            valid=valid and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0. for v in (raw,clip))
            if valid:
                clipping[name]=max(clipping[name],clip)
                if eligible:valid=raw<=18.+1.e-5 and clip<=1.e-5
            if not valid and len(issues)<12:
                issues.append(dict(tick=row['tick'],robot=name,reason='scoped QP or feedforward bound'))
            if total_motor_qp and loaded:
                motor_active[name]+=1
                if not motor_envelope_matches(balance.get('total_motor_qp'),r,prior) and len(issues)<12:
                    issues.append(dict(tick=row['tick'],robot=name,reason='total motor prediction, native limits or measured torque'))
            joint=balance.get('joint_motor_target')
            if ((joint_motor_qp and loaded and not joint_target_matches(joint,r,prior,period))
                    or joint is not None and not (joint_motor_qp and loaded)) and len(issues)<12:
                issues.append(dict(tick=row['tick'],robot=name,reason='joint motor target scope, original slew or position bounds'))
            active[name]+=int(placement)
            loaded_active[name]+=int(loaded)
    return dict(passed=bool(count) and not issues,observed_ticks=count,issues=issues,
        independently_scoped_qp_frames=active,maximum_observed_ff_clipping_nm=clipping,
        independently_scoped_loaded_recovery_qp_frames=loaded_active,
        independently_checked_total_motor_frames=motor_active,
        scope='Allocator selection and requested feedforward only; no total-PD torque or delivery success claim')


def audit_placement_wrench(folder,result):
    config=result.get('placement_wrench_qp') or {}
    loaded_config=result.get('loaded_recovery_wrench_qp') or {}
    placement_enabled=config.get('enabled',False) is True
    loaded_enabled=loaded_config.get('enabled',False) is True
    if not (placement_enabled or loaded_enabled):return dict(enabled=False,passed=True)
    configs=[c for c in (config,loaded_config) if c.get('enabled') is True]
    brake_config=result.get('loaded_brake_support') or {}
    if (any(c.get('base_allocator')!='sequential' or c.get('feedforward_limit_nm')!=18.
            or c.get('actuator_limits_unchanged') is not True for c in configs)
            or loaded_enabled and (brake_config.get('enabled') is not True
                or loaded_config.get('receiving_exit_enabled',False) != brake_config.get('receiving_exit_enabled',False))):
        return dict(enabled=True,passed=False,issues=[dict(reason='configuration')])
    total_motor=loaded_config.get('total_motor_enabled',False) is True
    joint_motor=loaded_config.get('joint_target_enabled',False) is True
    if joint_motor and not total_motor:
        return dict(enabled=True,passed=False,issues=[dict(reason='joint motor configuration')])
    if total_motor and (not loaded_enabled or loaded_config.get('total_motor_margin_nm')!=.1):
        return dict(enabled=True,passed=False,issues=[dict(reason='total motor configuration')])
    period=result['control_trace_period_s']
    if not math.isfinite(period) or not 0<period<=.02:raise ValueError('Invalid physical clock')
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        report=replay((json.loads(line) for line in stream),period,
                      placement_qp=placement_enabled,loaded_recovery_qp=loaded_enabled,
                      receiving_brake=loaded_config.get('receiving_exit_enabled',False) is True,
                      total_motor_qp=total_motor,joint_motor_qp=joint_motor)
    report['passed'] &= report['observed_ticks']==result['control_ticks']
    report['enabled']=True
    return report
