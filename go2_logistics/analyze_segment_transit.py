"""Replay segment ownership and compare it with recorded physical control phases."""
import argparse
import json
import math
from pathlib import Path


def audit(result,frames,trace):
    config=result.get('segment_reservation') or {}
    if not config.get('enabled'):
        return dict(enabled=False,passed=None)
    names=set(result['returned'])
    short_spine=config.get('short_spine_enabled') is True
    short_geometry_valid=(not short_spine or (config.get('spine_x_m') == -7.4
        and config.get('station_release_clearance_m') == 1.45))
    events=config.get('events',[])
    owners={}
    pending={}
    starts={}
    intervals=[]
    issues=[]
    previous=-1
    if not short_geometry_valid:
        issues.append(dict(reason='short spine geometry differs from the reviewed candidate'))
    def row_resource(sample):
        from .sorting_layout import shelf_for_code
        code=sample.get('active_order_code') or (sample.get('original_z1') or {}).get('active_payload_code')
        try:
            return f'row_{shelf_for_code(code).row+1}'
        except (ValueError,TypeError):
            return None
    def clear_of_spine(sample):
        from .sorting_layout import shelf_for_code
        pos=sample.get('root_position_w')
        code=sample.get('active_order_code') or (sample.get('original_z1') or {}).get('active_payload_code')
        try:
            row_y=shelf_for_code(code).surface[1]
        except (ValueError,TypeError):
            return False
        return (short_geometry_valid and isinstance(pos,list) and len(pos)==3
            and all(isinstance(v,(int,float)) and math.isfinite(v) for v in pos)
            and pos[0]-config['spine_x_m']>=config['station_release_clearance_m']
            and abs(pos[1]-row_y)<=.65)
    frame_at={frame['tick']:frame for frame in frames} if short_spine else {}
    station_releases=0
    for event in events:
        kind,resource,robot,tick=(event.get(key) for key in ('event','resource','robot','tick'))
        if not isinstance(tick,int) or not previous<=tick<=result['control_ticks'] or robot not in names:
            issues.append(dict(reason='invalid event',event=event))
            continue
        previous=tick
        key=(robot,resource)
        if kind=='requested':
            if key in pending or owners.get(resource)==robot:
                issues.append(dict(reason='duplicate request',event=event))
            pending[key]=tick
        elif kind=='acquired':
            if resource in owners or key not in pending:
                issues.append(dict(reason='overlapping or unrequested acquire',event=event))
            owners[resource]=robot
            starts[resource]=(tick,pending.pop(key,tick))
        elif kind=='released':
            if owners.get(resource)!=robot:
                issues.append(dict(reason='release by non-owner',event=event))
                continue
            if short_spine and resource=='west_spine' and event.get('reason')=='inside_exclusive_row_at_dock':
                station_releases+=1
                sample=frame_at.get(tick,{}).get('robots',{}).get(robot,{})
                measured=sample.get('root_position_w') or []
                announced=event.get('root_xy') or []
                row=row_resource(sample)
                if (not clear_of_spine(sample) or row is None or owners.get(row)!=robot
                        or len(announced)!=2 or len(measured)!=3
                        or math.dist(announced,measured[:2])>1e-5):
                    issues.append(dict(reason='station spine release lacks measured row clearance',event=event))
            start,request=starts.pop(resource)
            intervals.append(dict(resource=resource,robot=robot,start_tick=start,end_tick=tick,
                                  requested_tick=request,wait_ticks=start-request))
            owners.pop(resource)
        else:
            issues.append(dict(reason='unknown event kind',event=event))
    if result['state']=='complete' and (owners or pending):
        issues.append(dict(reason='complete run retains reservations',owners=owners,pending=list(pending)))
    if short_spine and result['state']=='complete' and not station_releases:
        issues.append(dict(reason='short spine candidate completed without exercising dock release'))
    for resource,robot in owners.items():
        start,request=starts[resource]
        intervals.append(dict(resource=resource,robot=robot,start_tick=start,end_tick=result['control_ticks'],
                              requested_tick=request,wait_ticks=start-request,unreleased=True))
    def held_at(tick):
        return {row['resource']:row['robot'] for row in intervals
                if row['start_tick']<tick<=row['end_tick']}
    coverage=bool(frames) and [row['tick'] for row in frames]==list(range(1,result['control_ticks']+1))
    violations=[]
    inspected=0
    for frame in frames:
        held=held_at(frame['tick'])
        for robot,sample in frame['robots'].items():
            phase=sample['phase']
            row=row_resource(sample) if short_spine else None
            protected_dock=(short_spine and row is not None and held.get(row)==robot
                and clear_of_spine(sample))
            if phase in ('placing','stowing_after_departure'):
                inspected+=1
                if (held.get('west_spine')!=robot
                        and not (phase=='placing' and protected_dock)):
                    violations.append(dict(tick=frame['tick'],robot=robot,phase=phase,reason='spine not held'))
            if phase in ('transporting','returning'):
                inspected+=1
                if (robot not in (held.get('approach_corridor'),held.get('west_spine'))
                        and not (phase=='transporting' and protected_dock)):
                    violations.append(dict(tick=frame['tick'],robot=robot,phase=phase,reason='no transit segment held'))
            if short_spine and phase=='waiting_west_spine' and not protected_dock:
                # Outbound waits are allowed at the existing southern staging
                # point. Inbound waits after placing must retain their row and
                # stay east of the widened spine until ownership returns.
                pos=sample.get('root_position_w') or []
                outbound_staging=(len(pos)==3 and math.dist(pos[:2],(-4.7,-1.3))<=.30
                                  and row is not None and held.get(row)==robot)
                if not outbound_staging:
                    violations.append(dict(tick=frame['tick'],robot=robot,phase=phase,
                                           reason='waiting outside protected row dock or southern staging'))
            if short_spine and phase in ('placing','stowing_after_departure') and (row is None or held.get(row)!=robot):
                violations.append(dict(tick=frame['tick'],robot=robot,phase=phase,reason='shelf row not held'))
    snapshot_errors=[]
    for row in trace:
        measured={resource:owner for resource,owner in row.get('dispatcher',{}).get('resources',{}).items()
                  if resource=='approach_corridor' or resource=='west_spine' or resource.startswith('row_')}
        if measured!=held_at(row['tick']):
            snapshot_errors.append(row['tick'])
    waits={name:sum(row['wait_ticks'] for row in intervals if row['robot']==name)*result['control_trace_period_s']
           for name in names}
    return dict(enabled=True,passed=bool(events) and coverage and inspected>0 and not issues
        and not violations and bool(trace) and not snapshot_errors,
        complete_control_coverage=coverage,event_issues=issues,intervals=intervals,
        measured_station_spine_releases=station_releases,
        phase_ownership_violations=len(violations),first_violations=violations[:10],
        dispatcher_snapshot_mismatch_ticks=snapshot_errors[:10],reservation_wait_seconds=waits,
        note='Checks reservation consistency and exercised phase ownership; does not establish collision freedom.')


def analyze(folder):
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    def read(name):
        return [json.loads(line) for line in (folder/name).open(encoding='utf-8') if line.strip()]
    report=audit(result,read('control_trace.jsonl'),read('trace.jsonl'))
    (folder/'segment_transit_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    print(json.dumps(analyze(parser.parse_args().folder),indent=2))
