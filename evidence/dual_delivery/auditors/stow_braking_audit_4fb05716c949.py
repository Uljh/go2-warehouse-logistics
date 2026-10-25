"""Check measured stopping before the optional post-departure empty fold.

Failed interventions remain valid failure evidence if no fold follows them.
An unused switch is reported as unused, never as a successful intervention.
"""
import json
import math
from pathlib import Path


def audit_stow_braking(folder, result):
    metadata = result.get('empty_stow_braking') or {}
    enabled = metadata.get('enabled', False) is True
    support_enabled = metadata.get('measured_support_entry_enabled', False) is True
    events = metadata.get('events', [])
    dt = result['control_trace_period_s']
    checks = dict(configuration=isinstance(events, list), event_clock=True,
                  measured_stop=True, fold_after_stop=True, reported_velocity=True,
                  measured_support_scope=True, measured_support_entry=True)
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        return dict(enabled=enabled, passed=False, checks=dict(control_clock=False))
    if not isinstance(events, list):
        events = []
    checks['configuration'] &= enabled or not events
    checks['configuration'] &= not support_enabled or enabled
    required = max(1, math.ceil(.2/dt-1e-9))
    prepared = []
    violations = []

    def reject(key, tick=None, robot=None):
        checks[key] = False
        if len(violations) < 12:
            violations.append(dict(check=key, tick=tick, robot=robot))

    for event in events:
        start, end = event.get('start_tick'), event.get('end_tick')
        completed = event.get('completed') is True
        valid = (event.get('robot') in result['arm_runtime_integrity']
            and isinstance(start, int) and 0 <= start <= result['control_ticks']
            and (not completed or isinstance(end, int) and start < end <= result['control_ticks']
                 and .3-1e-9 <= (end-start)*dt <= 3.+1e-9))
        if not valid:
            reject('event_clock', start, event.get('robot'))
            continue
        prepared.append(dict(event=event, stable=0, matched=False, completed=completed, last=None))
    previous = {}
    previous_measurements = {}
    support_states = {name:dict(active=False,stable_time_s=0.) for name in result['arm_runtime_integrity']}
    active_support_frames = {name:0 for name in support_states}
    folding_transitions = 0
    with (Path(folder)/'control_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            tick = row['tick']
            for name, state in row['robots'].items():
                arm = state.get('original_z1') or {}
                if support_enabled:
                    gate = support_states[name]
                    measured = previous_measurements.get(name)
                    in_event = any(item['event']['robot']==name
                        and item['event']['start_tick']<tick
                        and (not item['completed'] or tick<=item['event']['end_tick'])
                        for item in prepared)
                    requested = bool(in_event and state.get('recovering') is True
                        and state.get('phase')=='returning' and state.get('payload_held') is False)
                    valid = measured is not None
                    if valid:
                        linear=measured['root_velocity_w'];angular=measured['root_angular_velocity_b']
                        normals=measured.get('foot_contact_normal_n') or []
                        valid=(len(linear)==len(angular)==3 and len(normals)==4
                            and all(math.isfinite(v) for v in linear+angular+normals+[measured['gravity_z']]))
                    geometry=bool(requested and valid and measured['gravity_z']<-.99
                        and sum(v>1. for v in normals)>=3)
                    eligible=bool(geometry and math.hypot(*linear[:2])<.14 and abs(linear[2])<.20
                        and math.sqrt(sum(v*v for v in angular))<.50)
                    if not geometry:
                        gate.update(active=False,stable_time_s=0.)
                    elif not gate['active']:
                        gate['stable_time_s']=gate['stable_time_s']+dt if eligible else 0.
                        gate['active']=gate['stable_time_s']>=.04-1e-9
                    balance=state.get('balance_recovery') or {}
                    telemetry=balance.get('empty_stow_brake_support') or {}
                    if (telemetry.get('requested') is not requested
                            or telemetry.get('entry_eligible') is not eligible):
                        reject('measured_support_scope',tick,name)
                    reported_time=telemetry.get('stable_time_s')
                    if (telemetry.get('active') is not gate['active']
                            or not isinstance(reported_time,(int,float)) or not math.isfinite(reported_time)
                            or abs(reported_time-gate['stable_time_s'])>1e-8
                            or gate['active'] and balance.get('active') is not True):
                        reject('measured_support_entry',tick,name)
                    active_support_frames[name]+=int(gate['active'])
                folding = state.get('phase') == 'stowing_after_departure' and arm.get('state') == 'stow_empty_arm'
                for item in prepared:
                    event = item['event']
                    if event['robot'] != name or not item['completed']:
                        continue
                    if event['start_tick'] < tick <= event['end_tick']:
                        velocity, angular = state['root_velocity_w'], state['root_angular_velocity_b']
                        stopped = (state.get('phase') == 'returning' and state.get('recovering') is True
                            and state.get('payload_held') is False and state['height_m'] > .305
                            and state['gravity_z'] < -.990
                            and all(math.isfinite(v) for v in velocity+angular)
                            and math.sqrt(sum(v*v for v in velocity)) < .06
                            and math.sqrt(sum(v*v for v in angular)) < .15
                            and all(abs(v) < 1e-9 for v in state['commanded_velocity']))
                        item['stable'] = item['stable']+1 if stopped else 0
                        item['last'] = state
                    if tick == event['end_tick']:
                        if item['stable'] < required:
                            reject('measured_stop', tick, name)
                        for key in ('root_velocity_w', 'root_angular_velocity_b'):
                            reported = event.get(key)
                            measured = state[key]
                            if (not isinstance(reported, list) or len(reported) != 3
                                    or any(not math.isfinite(v) for v in reported)
                                    or any(abs(a-b) > 1e-7 for a,b in zip(reported, measured))):
                                reject('reported_velocity', tick, name)
                if enabled and folding and not previous.get(name, False):
                    folding_transitions += 1
                    matching = [item for item in prepared if item['completed']
                        and item['event']['robot'] == name and item['event']['end_tick'] == tick-1]
                    if len(matching) != 1 or matching[0]['matched']:
                        reject('fold_after_stop', tick, name)
                    else:
                        matching[0]['matched'] = True
                previous[name] = folding
                previous_measurements[name] = state
    for item in prepared:
        if item['completed'] and (item['last'] is None or not item['matched']):
            reject('fold_after_stop', item['event'].get('end_tick'), item['event']['robot'])
    return dict(enabled=enabled, passed=all(checks.values()), checks=checks,
        attempted_stops=len(events), completed_stops=sum(item['completed'] for item in prepared),
        observed_post_departure_folds=folding_transitions,
        independently_matched_folds=sum(item['matched'] for item in prepared), first_violations=violations,
        measured_support_entry_enabled=support_enabled,independently_active_support_frames=active_support_frames,
        scope='Measured stop and fold ordering only; zero interventions do not establish physical effectiveness')
