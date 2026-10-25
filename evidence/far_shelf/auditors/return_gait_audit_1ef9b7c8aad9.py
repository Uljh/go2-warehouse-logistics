"""Independently verify the optional return actor's identity and decision trace.

Eligibility is decided before the physics step. Compare its measured joints
with the previous post-step sample, and its phase/held flags with this sample.
This audits selection evidence, not whether the gait is physically successful.
"""
import hashlib
import json
import math
from pathlib import Path


HOME = (0., 1.2, -1.2, 0., 0., 0.)


def audit_return_gait(folder, result):
    folder = Path(folder)
    metadata = result.get('empty_return_gait_specialist') or {}
    enabled = metadata.get('enabled', False) is True
    empty_transit = metadata.get('empty_transit_enabled', False)
    resource_wait = metadata.get('resource_wait_enabled',False)
    repeat_pickup = metadata.get('repeat_pickup_enabled',False)
    idle_control = metadata.get('idle_control_enabled',False)
    names = tuple(result['arm_runtime_integrity'])
    stats = {name: dict(eligible_ticks=0, active_ticks=0, full_specialist_ticks=0) for name in names}
    checks = dict(checkpoint=True, training_manifest=True, configuration=True,
                  frame_coverage=True, eligibility=True, blend_ramp=True, counters=True,idle_control=True)
    idle_counts={name:0 for name in names}
    identity = metadata.get('checkpoint') or {}
    if enabled:
        path = folder/'return_policy_checkpoint.pt'
        checks['checkpoint'] = (identity.get('saved_checkpoint') == path.name and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == identity.get('sha256'))
        checks['configuration'] = (result.get('arm_model') == 'official_unitree_z1'
            and metadata.get('blend_seconds') == .5 and isinstance(empty_transit, bool)
            and isinstance(resource_wait,bool) and isinstance(repeat_pickup,bool)
            and isinstance(idle_control,bool) and (not idle_control or repeat_pickup is True))
        if identity.get('training_manifest_verified'):
            path = folder/'return_gait_training_manifest.json'
            manifest = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
            checks['training_manifest'] = (manifest.get('actual_training_completed') is True
                and manifest.get('trained_checkpoint_sha256') == identity.get('sha256')
                and manifest.get('source_checkpoint_sha256') == identity.get('source_checkpoint_sha256'))
    else:
        checks['configuration'] = (not identity and empty_transit is False and resource_wait is False
            and repeat_pickup is False and idle_control is False)
    dt = result['control_trace_period_s']
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        return dict(enabled=enabled, passed=False, checks=dict(invalid_control_period=False))
    previous_arm = {}
    weights = {name: 0. for name in names}
    frames = 0
    first_violations = []
    home_returns=metadata.get('home_return_events',[])
    returns_by_tick={}
    counts={name:0 for name in names}
    seen_codes=set()
    previous_event_tick=-1
    for event in home_returns:
        robot,tick,code=(event.get(k) for k in ('robot','tick','code'))
        valid=(robot in names and isinstance(tick,int) and not isinstance(tick,bool)
            and previous_event_tick<=tick<=result['control_ticks'] and tick>0 and code not in seen_codes)
        if valid:
            counts[robot]+=1
            valid=event.get('completed_returns')==counts[robot]
        if not valid:
            checks['configuration']=False
            continue
        previous_event_tick=tick;seen_codes.add(code)
        returns_by_tick.setdefault(tick,[]).append(event)
    completed_returns={name:0 for name in names}

    def violation(check, tick, name):
        checks[check] = False
        if len(first_violations) < 12:
            first_violations.append(dict(check=check, tick=tick, robot=name))

    with (folder/'control_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            frames += 1
            if (row['tick'] != frames or set(row['robots']) != set(names)
                    or abs(row['sim_time']-frames*dt) > 1e-8):
                violation('frame_coverage', row['tick'], None)
            for name in names:
                state = row['robots'].get(name)
                if state is None:
                    continue
                arm = state.get('original_z1') or {}
                telemetry = arm.get('return_gait_specialist')
                if enabled:
                    measured = previous_arm.get(name)
                    allowed = (state.get('phase') == 'returning' and arm.get('state') == 'stow_empty_arm')
                    if empty_transit is True:
                        allowed = (state.get('phase') in ('returning','approaching_receiving','staging_receiving',
                                    'withdrawing_receiving','waiting_receiving','parallel_navigation')
                                   and arm.get('state') in ('waiting','stow_empty_arm'))
                    if resource_wait is True:
                        allowed=allowed or (state.get('phase') in ('waiting_approach_corridor','waiting_west_spine',
                            'waiting_row_1','waiting_row_2','waiting_row_3','waiting_delivery_transit')
                            and arm.get('state')=='stow_empty_arm')
                    if repeat_pickup is True and completed_returns[name]>0:
                        allowed=allowed or (state.get('phase') in ('waiting_receiving','approaching_receiving',
                            'staging_receiving','withdrawing_receiving') and arm.get('state') in ('waiting','stow_empty_arm'))
                    eligible = (state.get('payload_held') is False and allowed and isinstance(measured, list)
                        and len(measured) == 6 and all(math.isfinite(value) for value in measured)
                        and max(abs(value-home) for value, home in zip(measured, HOME)) <= .06)
                    weight = (min(1., weights[name]+dt/.5) if eligible
                              else max(0., weights[name]-dt/.5))
                    if weight < 1e-9:
                        weight = 0.
                    if weight > 1.-1e-9:
                        weight = 1.
                    if not isinstance(telemetry, dict):
                        violation('frame_coverage', row['tick'], name)
                    else:
                        if telemetry.get('eligible') is not eligible:
                            violation('eligibility', row['tick'], name)
                        reported = telemetry.get('weight')
                        if (telemetry.get('blend_seconds') != .5 or not isinstance(reported, (int, float))
                                or not math.isfinite(reported) or abs(reported-weight) > 1e-8):
                            violation('blend_ramp', row['tick'], name)
                    weights[name] = weight
                    stats[name]['eligible_ticks'] += int(eligible)
                    stats[name]['active_ticks'] += int(weight > 0.)
                    stats[name]['full_specialist_ticks'] += int(weight == 1.)
                    if idle_control is True:
                        command=state.get('commanded_velocity')
                        expected_idle=(completed_returns[name]>0 and state.get('phase')=='waiting_receiving'
                            and state.get('payload_held') is False and arm.get('state') in ('waiting','stow_empty_arm')
                            and eligible and weight==1. and isinstance(command,list) and len(command)==3
                            and all(isinstance(v,(int,float)) and math.isfinite(v) and abs(v)<1e-9 for v in command))
                        balance=state.get('balance_recovery') or {}
                        reported_idle=balance.get('specialist_idle_control') or {}
                        if (reported_idle.get('requested') is not expected_idle
                                or expected_idle and (reported_idle.get('support_requested') is not False
                                    or balance.get('active') is not False or state.get('standing_pd') is not False)):
                            violation('idle_control',row['tick'],name)
                        idle_counts[name]+=int(expected_idle)
                elif telemetry is not None:
                    violation('configuration', row['tick'], name)
                previous_arm[name] = state.get('arm_position_rad')
            for event in returns_by_tick.get(row['tick'],[]):
                robot=event['robot'];state=row['robots'][robot]
                pos=state.get('root_position_w') or [];announced=event.get('root_xy') or []
                home=(0.,-1. if robot=='dog1' else 1.)
                delivery=any(d.get('robot')==robot and d.get('code')==event['code']
                    for d in result.get('delivery_evidence',[]))
                if (not delivery or len(pos)!=3 or len(announced)!=2
                        or not all(math.isfinite(v) for v in (*pos,*announced))
                        or math.dist(pos[:2],home)>.04 or math.dist(pos[:2],announced)>1e-5
                        or state.get('phase')!='returning' or state.get('payload_held') is not False):
                    violation('configuration',row['tick'],robot)
                completed_returns[robot]+=1
    checks['frame_coverage'] &= frames == result['control_ticks'] and frames > 0
    if enabled:
        checks['counters'] = metadata.get('stats') == stats
        if idle_control:checks['counters'] &= metadata.get('idle_control_frames')==idle_counts
    return dict(enabled=enabled, empty_transit_enabled=empty_transit, resource_wait_enabled=resource_wait,
        repeat_pickup_enabled=repeat_pickup,idle_control_enabled=idle_control,
        independently_checked_idle_frames=idle_counts,verified_home_returns=completed_returns,
        passed=all(checks.values()), checks=checks,
        audited_frames=frames, independently_recomputed_stats=stats, first_violations=first_violations,
        measurement_timing='previous post-step arm positions; current pre-step phase and held flags',
        scope='Actor file identity and measured eligibility/ramp only; physical task acceptance is separate')
