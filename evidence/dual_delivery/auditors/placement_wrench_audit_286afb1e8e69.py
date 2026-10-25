"""Independently check scoped placement and loaded-recovery allocation."""
import json
import math
from pathlib import Path


def replay(rows,period, *, placement_qp=True, loaded_recovery_qp=False):
    count=0;issues=[];active={};loaded_active={};clipping={}
    for row in rows:
        count+=1
        if (row['tick']!=count or not math.isfinite(row['sim_time'])
                or abs(row['sim_time']-count*period)>1.e-8):
            if len(issues)<12:issues.append(dict(tick=row['tick'],reason='control clock'))
        for name,r in row['robots'].items():
            command=r['commanded_velocity'];balance=r.get('balance_recovery') or {}
            placement=bool(placement_qp and r['phase']=='placing' and r['payload_held'] and len(command)==3
                and all(math.isfinite(v) for v in command) and max(abs(v) for v in command)<.05)
            loaded=bool(loaded_recovery_qp and r.get('recovering') is True
                and r['phase']=='transporting' and r['payload_held']
                and (r.get('original_z1') or {}).get('state')=='carried' and len(command)==3
                and all(math.isfinite(v) and abs(v)<1e-9 for v in command))
            eligible=placement or loaded
            # Once the controller has faded out it does not run allocation.
            executed=balance.get('active') is True or balance.get('blend',0.)>0.
            active.setdefault(name,0);loaded_active.setdefault(name,0);clipping.setdefault(name,0.)
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
            active[name]+=int(placement)
            loaded_active[name]+=int(loaded)
    return dict(passed=bool(count) and not issues,observed_ticks=count,issues=issues,
        independently_scoped_qp_frames=active,maximum_observed_ff_clipping_nm=clipping,
        independently_scoped_loaded_recovery_qp_frames=loaded_active,
        scope='Allocator selection and requested feedforward only; no total-PD torque or delivery success claim')


def audit_placement_wrench(folder,result):
    config=result.get('placement_wrench_qp') or {}
    loaded_config=result.get('loaded_recovery_wrench_qp') or {}
    placement_enabled=config.get('enabled',False) is True
    loaded_enabled=loaded_config.get('enabled',False) is True
    if not (placement_enabled or loaded_enabled):return dict(enabled=False,passed=True)
    configs=[c for c in (config,loaded_config) if c.get('enabled') is True]
    if (any(c.get('base_allocator')!='sequential' or c.get('feedforward_limit_nm')!=18.
            or c.get('actuator_limits_unchanged') is not True for c in configs)
            or loaded_enabled and (result.get('loaded_brake_support') or {}).get('enabled') is not True):
        return dict(enabled=True,passed=False,issues=[dict(reason='configuration')])
    period=result['control_trace_period_s']
    if not math.isfinite(period) or not 0<period<=.02:raise ValueError('Invalid physical clock')
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        report=replay((json.loads(line) for line in stream),period,
                      placement_qp=placement_enabled,loaded_recovery_qp=loaded_enabled)
    report['passed'] &= report['observed_ticks']==result['control_ticks']
    report['enabled']=True
    return report
