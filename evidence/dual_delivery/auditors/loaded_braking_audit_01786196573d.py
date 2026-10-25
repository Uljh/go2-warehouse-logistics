"""Independently replay the opt-in loaded-braking entry gate from measurements."""
import json
import math
from pathlib import Path


def replay(rows,period):
    if not math.isfinite(period) or not 0<period<=.02:
        raise ValueError('Loaded braking audit requires a positive physical control period')
    previous={};gates={};active={};issues=[];count=0
    for row in rows:
        count+=1;tick=row['tick']
        if tick!=count or abs(row['sim_time']-tick*period)>1e-8 or not math.isfinite(row['sim_time']):
            if len(issues)<12:issues.append(dict(tick=tick,reason='control clock'))
        for name,state in row['robots'].items():
            gate=gates.setdefault(name,dict(active=False,stable=0.));active.setdefault(name,0)
            arm=state.get('original_z1') or {};command=state['commanded_velocity']
            requested=bool(state['recovering'] and state['phase']=='transporting'
                and state['payload_held'] and arm.get('state')=='carried'
                and len(command)==3 and all(math.isfinite(v) and abs(v)<1e-9 for v in command))
            measured=previous.get(name);valid=measured is not None
            if valid:
                v=measured['root_velocity_w'];w=measured['root_angular_velocity_b']
                normals=measured.get('foot_contact_normal_n') or [];gravity=measured['gravity_z']
                valid=len(v)==len(w)==3 and len(normals)==4 and all(math.isfinite(x) for x in v+w+normals+[gravity])
            geometry=bool(requested and valid and gravity<-.99 and sum(x>1. for x in normals)>=3)
            eligible=bool(geometry and math.hypot(*v[:2])<.14 and abs(v[2])<.20
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
            if not matched and len(issues)<12:issues.append(dict(tick=tick,robot=name,reason='loaded support scope or gate mismatch'))
            active[name]+=int(gate['active']);previous[name]=state
    return dict(passed=bool(count) and not issues,observed_ticks=count,issues=issues,
        independently_active_support_frames=active,
        scope='Measured loaded-braking scope and entry only; unused switch does not establish physical efficacy')


def audit_loaded_braking(folder,result):
    enabled=(result.get('loaded_brake_support') or {}).get('enabled',False) is True
    if not enabled:return dict(enabled=False,passed=True,independently_active_support_frames={})
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        report=replay((json.loads(line) for line in stream),result['control_trace_period_s'])
    report['enabled']=True
    report['passed'] &= report['observed_ticks']==result['control_ticks']
    return report
