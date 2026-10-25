"""Independently replay the opt-in loaded-braking entry gate from measurements."""
import json
import math
from pathlib import Path


def replay(rows,period, *, receiving_brake=False,attitude_hysteresis=False):
    if not math.isfinite(period) or not 0<period<=.02:
        raise ValueError('Loaded braking audit requires a positive physical control period')
    if not isinstance(attitude_hysteresis,bool):raise ValueError('Explicit hysteresis flag required')
    previous={};gates={};active={};retained_counts={};issues=[];count=0
    for row in rows:
        count+=1;tick=row['tick']
        if tick!=count or abs(row['sim_time']-tick*period)>1e-8 or not math.isfinite(row['sim_time']):
            if len(issues)<12:issues.append(dict(tick=tick,reason='control clock'))
        for name,state in row['robots'].items():
            gate=gates.setdefault(name,dict(active=False,stable=0.));active.setdefault(name,0)
            retained_counts.setdefault(name,0)
            arm=state.get('original_z1') or {};command=state['commanded_velocity']
            allowed_phase=state['phase']=='transporting' or receiving_brake and state['phase']=='leaving_receiving'
            requested=bool(state['recovering'] and allowed_phase
                and state['payload_held'] and arm.get('state')=='carried'
                and len(command)==3 and all(math.isfinite(v) and abs(v)<1e-9 for v in command))
            measured=previous.get(name);valid=measured is not None
            if valid:
                v=measured['root_velocity_w'];w=measured['root_angular_velocity_b']
                normals=measured.get('foot_contact_normal_n') or [];gravity=measured['gravity_z']
                valid=len(v)==len(w)==3 and len(normals)==4 and all(math.isfinite(x) for x in v+w+normals+[gravity])
            was_active=gate['active']
            threshold=-.97 if attitude_hysteresis and was_active else -.99
            geometry=bool(requested and valid and gravity<threshold and sum(x>1. for x in normals)>=3)
            eligible=bool(geometry and gravity<-.99 and math.hypot(*v[:2])<.14 and abs(v[2])<.20
                and math.sqrt(sum(x*x for x in w))<.50)
            if not geometry:gate.update(active=False,stable=0.)
            elif not gate['active']:
                gate['stable']=gate['stable']+period if eligible else 0.
                gate['active']=gate['stable']>=.04-1e-9
            balance=state.get('balance_recovery') or {};telemetry=balance.get('loaded_brake_support') or {}
            stable=telemetry.get('stable_time_s')
            matched=(telemetry.get('requested') is requested and telemetry.get('entry_eligible') is eligible
                and telemetry.get('active') is gate['active'] and isinstance(stable,(int,float))
                and math.isfinite(stable) and abs(stable-gate['stable'])<1e-8
                and (not gate['active'] or balance.get('active') is True))
            retained=bool(attitude_hysteresis and was_active and gate['active'] and gravity>=-.99)
            if attitude_hysteresis:
                matched &= telemetry.get('retained_after_entry') is retained
            retained_counts[name]+=int(retained)
            if not matched and len(issues)<12:issues.append(dict(tick=tick,robot=name,reason='loaded support scope or gate mismatch'))
            active[name]+=int(gate['active']);previous[name]=state
    return dict(passed=bool(count) and not issues,observed_ticks=count,issues=issues,
        independently_active_support_frames=active,
        attitude_hysteresis_enabled=attitude_hysteresis,
        independently_retained_support_frames=retained_counts,
        scope='Measured loaded-braking scope, entry and optional attitude retention; unused switch does not establish physical efficacy')


def audit_loaded_braking(folder,result):
    config=result.get('loaded_brake_support') or {}
    enabled=config.get('enabled',False) is True
    hysteresis=config.get('attitude_hysteresis_enabled',False)
    if (not isinstance(hysteresis,bool) or hysteresis and not enabled
            or 'retention_gravity_z_below' in config
                and config['retention_gravity_z_below']!=(-.97 if hysteresis else -.99)
            or hysteresis and
            (config.get('retention_gravity_z_below')!=-.97
             or config.get('gravity_z_below')!=-.99
             or config.get('planar_speed_below_mps')!=.14
             or config.get('vertical_speed_below_mps')!=.20
             or config.get('angular_speed_below_radps')!=.50
             or config.get('minimum_contacts')!=3 or config.get('contact_normal_above_n')!=1.
             or config.get('entry_stable_time_s')!=.04
             or (result.get('loaded_recovery_wrench_qp') or {}).get('joint_target_enabled') is not True
             or (result.get('loaded_recovery_wrench_qp') or {}).get('total_motor_enabled') is not True)):
        return dict(enabled=True,passed=False,issues=[dict(reason='hysteresis configuration')])
    if not enabled:return dict(enabled=False,passed=True,independently_active_support_frames={})
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        report=replay((json.loads(line) for line in stream),result['control_trace_period_s'],
            receiving_brake=config.get('receiving_exit_enabled',False) is True,
            attitude_hysteresis=hysteresis)
    report['enabled']=True
    report['passed'] &= report['observed_ticks']==result['control_ticks']
    return report
