"""Measure pickup handoff and physical activity overlap from control records."""
import argparse
import json
import math
from pathlib import Path


def audit(result, frames):
    config=result.get('receiving_staging') or {}
    enabled=config.get('enabled') is True
    names=tuple(result['returned'])
    period=result['control_trace_period_s']
    valid_period=isinstance(period,(int,float)) and math.isfinite(period) and period>0
    coverage=(valid_period and bool(frames) and len(frames)==result['control_ticks']
        and all(row.get('tick')==i+1 and set(row.get('robots',{}))==set(names)
                for i,row in enumerate(frames)))
    owner=None
    pending={}
    acquired=None
    intervals=[]
    issues=[]
    previous=-1
    for event in config.get('reservation_events',[]):
        kind,robot,tick=(event.get(k) for k in ('event','robot','tick'))
        if (robot not in names or not isinstance(tick,int) or isinstance(tick,bool)
                or not previous<=tick<=result['control_ticks'] or event.get('resource')!='receiving'):
            issues.append(dict(reason='invalid reservation event',event=event))
            continue
        previous=tick
        if kind=='requested':
            if robot in pending or owner==robot:
                issues.append(dict(reason='duplicate receiving request',event=event))
            pending[robot]=tick
        elif kind=='acquired':
            if owner is not None or robot not in pending:
                issues.append(dict(reason='overlapping or unrequested receiving acquire',event=event))
            owner=robot
            acquired=tick
            pending.pop(robot,None)
        elif kind=='released':
            if owner!=robot:
                issues.append(dict(reason='receiving released by non-owner',event=event))
                continue
            intervals.append(dict(robot=robot,start_tick=acquired,end_tick=tick))
            owner=None
            acquired=None
        elif kind=='cancelled':
            if robot not in pending:
                issues.append(dict(reason='receiving cancellation without request',event=event))
            pending.pop(robot,None)
        else:
            issues.append(dict(reason='unknown receiving event',event=event))
    if result['state']=='complete' and (owner is not None or pending):
        issues.append(dict(reason='complete run retains receiving reservation or requests'))
    if owner is not None:
        intervals.append(dict(robot=owner,start_tick=acquired,end_tick=result['control_ticks'],unreleased=True))
    interval_source='reservation_events'
    if not config.get('reservation_events'):
        # Legacy runs did not log the slot events. Their control phases locate
        # the same handoff to one control period, without inventing ownership.
        interval_source='legacy_control_phase_inference'
        open_slots={}
        slot_phases=('approaching_receiving','waiting_visible_parcel','grasping','leaving_receiving')
        for frame in frames:
            for name,sample in frame['robots'].items():
                if sample.get('phase') in slot_phases:
                    open_slots.setdefault(name,frame['tick']-1)
                elif name in open_slots:
                    intervals.append(dict(robot=name,start_tick=open_slots.pop(name),end_tick=frame['tick']-1))
        for name,start in open_slots.items():
            intervals.append(dict(robot=name,start_tick=start,end_tick=result['control_ticks'],unreleased=True))
        intervals.sort(key=lambda entry:entry['start_tick'])
        if enabled:
            issues.append(dict(reason='enabled pipeline has no receiving reservation events'))

    def owner_at(tick):
        return next((i['robot'] for i in intervals if i['start_tick']<tick<=i['end_tick']),None)

    transitions=[]
    first_grasp=[]
    last={}
    first_carry={}
    overlap=both_loaded=both_moving=0
    violations=[]
    for frame in frames:
        tick=frame['tick']
        samples=frame['robots']
        for name,sample in samples.items():
            phase=sample.get('phase')
            code=sample.get('active_order_code') or (sample.get('original_z1') or {}).get('active_payload_code')
            state=(phase,code)
            if last.get(name)!=state:
                transitions.append(dict(robot=name,phase=phase,code=code,tick=tick,
                                        time_s=(tick-1)*period))
                if phase=='grasping':
                    first_grasp.append(dict(robot=name,code=code,tick=tick,time_s=(tick-1)*period))
                last[name]=state
            if sample.get('payload_held') and code is not None and phase in ('leaving_receiving','transporting'):
                first_carry.setdefault((name,code),tick)
            if enabled and phase in ('approaching_receiving','waiting_visible_parcel','grasping','leaving_receiving'):
                if owner_at(tick)!=name:
                    violations.append(dict(tick=tick,robot=name,phase=phase,reason='receiving slot not owned'))
        if len(samples)==2:
            a,b=(samples[n] for n in names)
            both_loaded+=int(a.get('payload_held') is True and b.get('payload_held') is True)
            overlap+=int((a.get('phase')=='grasping' and b.get('phase') in ('transporting','placing','returning'))
                or (b.get('phase')=='grasping' and a.get('phase') in ('transporting','placing','returning')))
            velocities=[samples[n].get('root_velocity_w') for n in names]
            both_moving+=int(all(isinstance(v,list) and len(v)==3 and math.hypot(*v[:2])>.05 for v in velocities))
    handoffs=[]
    for before,after in zip(intervals,intervals[1:]):
        if before['robot']==after['robot']:
            continue
        grasp=next((g for g in first_grasp if g['robot']==after['robot']
                    and after['start_tick']<g['tick']<=after['end_tick']),None)
        carry=next((tick for (name,_),tick in first_carry.items()
                    if name==before['robot'] and before['start_tick']<tick<=before['end_tick']),None)
        handoffs.append(dict(first_robot=before['robot'],next_robot=after['robot'],
            first_carry_s=None if carry is None else (carry-1)*period,
            released_s=before['end_tick']*period,acquired_s=after['start_tick']*period,
            next_grasp_s=None if grasp is None else grasp['time_s'],
            release_to_acquire_s=(after['start_tick']-before['end_tick'])*period,
            release_to_grasp_s=None if grasp is None else (grasp['tick']-1-before['end_tick'])*period,
            carry_to_next_grasp_s=None if grasp is None or carry is None else (grasp['tick']-carry)*period))
    for event in config.get('events',[]):
        if event.get('event')=='staging':
            start=event.get('start_tick')
            if (not isinstance(start,int) or event.get('trigger_phase') not in
                    ('waiting_visible_parcel','grasping','leaving_receiving')
                    or owner_at(start+1)!=event.get('blocking_robot')):
                issues.append(dict(reason='staging started without an eligible station owner',event=event))
        if result['state']=='complete' and event.get('completed') is not True:
            # Cancellation of a partially traversed staging route is legitimate
            # only when followed by a verified withdrawal callback.
            withdrawn=any(e.get('event')=='withdrawal' and e.get('robot')==event.get('robot')
                and e.get('completed') is True and e.get('start_tick',-1)>=event.get('end_tick',0)
                for e in config.get('events',[]))
            if not withdrawn:
                issues.append(dict(reason='unfinished staging without withdrawal',event=event))
    return dict(enabled=enabled,passed=(coverage and bool(intervals) and not issues and not violations) if enabled else None,
        complete_control_coverage=coverage,event_issues=issues,ownership_violation_count=len(violations),
        first_ownership_violations=violations[:10],receiving_intervals=intervals,pickup_handoffs=handoffs,
        receiving_interval_source=interval_source,
        grasp_and_peer_delivery_overlap_s=overlap*period,both_payloads_held_s=both_loaded*period,
        measured_both_moving_s=both_moving*period,phase_transitions=transitions,
        scope='Recorded slot ownership and activity timing only; no claim of contact success, collision freedom or complete delivery')


def analyze(folder, output=None):
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    with (folder/'control_trace.jsonl').open(encoding='utf-8') as stream:
        frames=[json.loads(line) for line in stream if line.strip()]
    report=audit(result,frames)
    destination=Path(output) if output is not None else folder/'receiving_pipeline_evidence.json'
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=analyze(args.folder,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='phase_transitions'},indent=2))
