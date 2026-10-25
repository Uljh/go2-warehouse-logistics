"""Measure pickup handoff and physical activity overlap from control records."""
import argparse
import json
import math
from pathlib import Path


def _finite_vector(value, size):
    return (isinstance(value,list) and len(value)==size
        and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in value))


class PayloadActivity:
    """Separate a held flag during lifting from measured loaded locomotion."""
    def __init__(self, original_z1):
        self.original_z1=original_z1
        self.state_coverage=self.contact_coverage=self.velocity_coverage=True
        self.frames=self.carried=self.moving=self.guarded=0

    def observe(self, samples):
        self.frames+=1
        arms=[s.get('original_z1') or {} for s in samples]
        held=len(samples)==2 and all(s.get('payload_held') is True for s in samples)
        known_states=(self.original_z1 and len(arms)==2
            and all(isinstance(a.get('state'),str) and bool(a['state']) for a in arms))
        known_velocities=len(samples)==2 and all(_finite_vector(s.get('root_velocity_w'),3) for s in samples)
        known_contacts=(self.original_z1 and len(arms)==2
            and all(_finite_vector(a.get(k),2) and all(v>=0 for v in a[k])
                for a in arms for k in ('jaw_normal_n','jaw_total_n')))
        self.state_coverage &= known_states
        self.velocity_coverage &= known_velocities
        self.contact_coverage &= known_contacts
        carried=held and known_states and all(a['state']=='carried' for a in arms)
        self.carried+=int(carried)
        self.moving+=int(carried and known_velocities
            and all(math.hypot(*s['root_velocity_w'][:2])>.05 for s in samples))
        self.guarded+=int(held and known_contacts and all(min(a['jaw_normal_n'])>.05
            and max(a['jaw_normal_n']+a['jaw_total_n'])<=20. for a in arms))

    def report(self, period, coverage):
        state=bool(coverage and self.frames and self.state_coverage)
        velocity=bool(coverage and self.frames and self.velocity_coverage)
        contact=bool(coverage and self.frames and self.contact_coverage)
        return dict(carried_state_coverage=state,velocity_measurement_coverage=velocity,
            contact_measurement_coverage=contact,
            both_in_carried_state_s=self.carried*period if state else None,
            both_carried_and_moving_s=self.moving*period if state and velocity else None,
            both_guarded_contact_held_s=self.guarded*period if contact else None,
            movement_threshold_mps=.05,opposing_contact_threshold_n=.05,force_guard_n=20.,
            scope='Carried requires both original-Z1 arms in carried state; moving requires both measured planar speeds above threshold; contact is per-frame only, including lift, and does not certify a complete delivery. Missing measurements are null.')


def audit_compact_layout(result,frames):
    """Recompute the opt-in release geometry from measured control frames."""
    config=result.get('receiving_staging') or {}
    enabled=config.get('compact_layout_enabled') is True
    releases=[e for e in config.get('reservation_events',[]) if e.get('event')=='released']
    if not enabled:
        issues=[] if not any(e.get('compact_release') for e in releases) else ['compact release without its flag']
        return dict(enabled=False,passed=not issues,issues=issues)
    issues=[]
    if (config.get('enabled') is not True or (result.get('segment_reservation') or {}).get('enabled') is not True
            or config.get('approach_staged_only') is not True
            or config.get('approach_alignment_x_m')!=3.1
            or config.get('compact_release_clearance_m')!=1.45
            or config.get('approach_waypoints')!=[dict(x=3.1,y=.1,yaw=0.),dict(x=3.1,y=1.5,yaw=0.),dict(x=3.62,y=1.5,yaw=0.)]):
        issues.append(dict(reason='invalid coupled receiving layout configuration'))
    for event in config.get('events',[]):
        if event.get('event')!='staging':continue
        expected=[dict(x=0.,y=-.15,yaw=0.),dict(x=3.1,y=.1,yaw=0.)]
        if event.get('route')!=expected or event.get('target')!=expected[-1]:
            issues.append(dict(reason='standby route differs from coupled exit layout',event=event))
    period=result['control_trace_period_s']
    confirmation=math.ceil(.2/period-1e-9)
    for event in releases:
        robot,tick=event.get('robot'),event.get('tick')
        if not isinstance(tick,int) or not confirmation<=tick<=len(frames) or robot not in result['returned']:
            issues.append(dict(reason='compact release lacks measured control history',event=event));continue
        sample=frames[tick-1]['robots'][robot]
        position=sample.get('root_position_w') or []
        announced=event.get('root_xy') or []
        if (len(position)!=3 or not all(math.isfinite(v) for v in position)
                or len(announced)!=2 or not all(math.isfinite(v) for v in announced)
                or math.dist(announced,position[:2])>1e-5):
            issues.append(dict(reason='release position differs from recorded physics',event=event));continue
        x,y=position[:2]
        # Entire L-shaped incoming centerline: vertical segment then final dock.
        clearance=min(math.hypot(x-3.1,y-max(.1,min(1.5,y))),
                      math.hypot(x-max(3.1,min(3.62,x)),y-1.5))
        if (event.get('compact_release') is not True or abs(x-1.5)>.040001
                or abs(y-1.5)>.040001 or clearance<1.45
                or not isinstance(event.get('incoming_path_clearance_m'),(int,float))
                or not math.isfinite(event['incoming_path_clearance_m'])
                or abs(event['incoming_path_clearance_m']-clearance)>1e-5):
            issues.append(dict(reason='release lacks original dock precision or incoming-path clearance',event=event))
        for row in frames[tick-confirmation:tick]:
            state=row['robots'][robot]
            v=state.get('root_velocity_w') or [float('inf')]*3
            w=state.get('root_angular_velocity_b') or [float('inf')]*3
            command=state.get('commanded_velocity') or [float('inf')]*3
            if (state.get('phase')!='leaving_receiving' or state.get('payload_held') is not True
                    or not state.get('height_m',0.)>.305 or not state.get('gravity_z',1.)<-.990
                    or not math.sqrt(sum(a*a for a in v))<.06
                    or not math.sqrt(sum(a*a for a in w))<.15
                    or not all(math.isfinite(a) and abs(a)<1e-8 for a in command)):
                issues.append(dict(reason='early release lacks measured settled zero-command confirmation',tick=row['tick'],robot=robot));break
    if result['state']=='complete' and not releases:
        issues.append(dict(reason='complete compact run did not exercise its release'))
    return dict(enabled=True,passed=not issues,release_count=len(releases),issues=issues,
        scope='Coupled layout, measured release clearance and original 0.2 s settled confirmation; actual traffic safety and deliveries checked separately')


def audit_home_entry_priority(result,frames):
    """Replay corridor priority independently, including the release/acquire tick."""
    config=result.get('receiving_staging') or {}
    if config.get('home_entry_guard') is not True:
        return dict(enabled=False,passed=True)
    issues=[]
    events=(result.get('segment_reservation') or {}).get('events',[])
    def blocker(robot,tick):
        owner=None;pending={}
        for event in events:
            if event.get('resource')!='approach_corridor' or event.get('tick',-1)>tick:continue
            kind,peer=event.get('event'),event.get('robot')
            if kind=='requested':pending.setdefault(peer,event['tick'])
            elif kind=='acquired':owner=peer;pending.pop(peer,None)
            elif kind=='released':owner=None
            elif kind=='cancelled':pending.pop(peer,None)
        return owner if owner!=robot and owner is not None else next((peer for peer in pending if peer!=robot),None)
    def at_home(robot,sample):
        position=sample.get('root_position_w') or []
        home=(0.,-1. if robot=='dog1' else 1.)
        return len(position)==3 and all(math.isfinite(v) for v in position) and math.dist(position[:2],home)<=.21
    for event in config.get('reservation_events',[]):
        if event.get('event')!='acquired':continue
        robot,tick=event['robot'],event['tick']
        sample=frames[max(0,tick-1)]['robots'][robot]
        if at_home(robot,sample) and blocker(robot,tick) is not None:
            issues.append(dict(reason='home pickup jumped the corridor owner or an older queued request',event=event))
    checked=0
    for event in config.get('home_entry_wait_events',[]):
        robot,start=event.get('robot'),event.get('start_tick')
        end=event.get('end_tick',len(frames))
        if (robot not in result['returned'] or not isinstance(start,int) or not isinstance(end,int)
                or not 0<=start<=end<=len(frames) or event.get('resource')!='approach_corridor'
                or blocker(robot,start)!=event.get('blocking_robot')
                or result['state']=='complete' and event.get('completed') is not True):
            issues.append(dict(reason='invalid home entry wait event or blocker',event=event));continue
        for row in frames[start:end]:
            sample=row['robots'][robot];command=sample.get('commanded_velocity')
            if (sample.get('phase')!='waiting_receiving' or not at_home(robot,sample)
                    or not isinstance(command,list) or len(command)!=3
                    or not all(math.isfinite(v) and abs(v)<1e-8 for v in command)):
                issues.append(dict(reason='home waiter moved or disappeared from parked-peer planning',tick=row['tick'],robot=robot))
                break
            checked+=1
    return dict(enabled=True,passed=not issues,wait_events=len(config.get('home_entry_wait_events',[])),
        checked_wait_frames=checked,issues=issues,
        scope='Recorded reservation priority, home standby phase and zero commands; no claim of delivery success')


def audit_arrival_settling(result,frames):
    config=result.get('receiving_staging') or {}
    if config.get('arrival_settle_enabled') is not True:
        return dict(enabled=False,passed=True)
    issues=[];completed=0;period=result['control_trace_period_s']
    events=config.get('arrival_settle_events',[])
    confirmation=math.ceil(.2/period-1e-9)
    for event in events:
        if event.get('completed') is not True:
            if result['state']=='complete':issues.append(dict(reason='unfinished arrival braking',event=event))
            continue
        robot,start,end=event.get('robot'),event.get('start_tick'),event.get('end_tick')
        if (robot not in result['returned'] or not isinstance(start,int) or not isinstance(end,int)
                or not 0<=start<end<=len(frames) or not .3-1e-8<=(end-start)*period<=3.+1e-8
                or event.get('budget_s')!=3. or event.get('minimum_duration_s')!=.3
                or event.get('position_tolerance_m')!=.04 or end-start<confirmation):
            issues.append(dict(reason='invalid arrival braking clock, budget or original dock tolerance',event=event));continue
        rows=frames[start:end]
        if any(row['robots'][robot].get('phase')!='approaching_receiving'
                or max(map(abs,row['robots'][robot].get('commanded_velocity',[float('inf')]*3)))>1e-8 for row in rows):
            issues.append(dict(reason='arrival braking did not retain zero-command navigation',event=event))
        for row in rows[-confirmation:]:
            sample=row['robots'][robot]
            v=sample.get('root_velocity_w',[float('inf')]*3)
            w=sample.get('root_angular_velocity_b',[float('inf')]*3)
            if (not sample.get('height_m',0.)>.305 or not sample.get('gravity_z',1.)<-.990
                    or not math.sqrt(sum(x*x for x in v))<.06
                    or not math.sqrt(sum(x*x for x in w))<.15):
                issues.append(dict(reason='arrival exit lacks original 0.2 s measured settled confirmation',tick=row['tick'],robot=robot));break
        target=event.get('target') or {};position=rows[-1]['robots'][robot].get('root_position_w',[])
        if (len(position)!=3 or not all(k in target for k in ('x','y'))
                or math.dist(position[:2],(target['x'],target['y']))>.040001):
            issues.append(dict(reason='measured receiving dock error exceeds original tolerance',event=event))
        completed+=1
    for robot in result['returned']:
        last_phase=None
        for row in frames:
            phase=row['robots'][robot].get('phase')
            if phase=='waiting_visible_parcel' and last_phase!=phase:
                matching=[event for event in events if event.get('robot')==robot
                    and event.get('end_tick')==row['tick']-1 and event.get('completed') is True]
                if len(matching)!=1:
                    issues.append(dict(reason='stationary operation started without one completed arrival brake',tick=row['tick'],robot=robot))
            last_phase=phase
    return dict(enabled=True,passed=not issues,completed_events=completed,issues=issues,
        scope='Original arrival braking budget, measured settling and docking only; complete delivery is audited separately')


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
    payload_activity=PayloadActivity(result.get('arm_model')=='official_unitree_z1')
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
            payload_activity.observe([a,b])
            both_loaded+=int(a.get('payload_held') is True and b.get('payload_held') is True)
            overlap+=int((a.get('phase')=='grasping' and b.get('phase') in ('transporting','placing','returning'))
                or (b.get('phase')=='grasping' and a.get('phase') in ('transporting','placing','returning')))
            velocities=[samples[n].get('root_velocity_w') for n in names]
            both_moving+=int(all(_finite_vector(v,3) and math.hypot(*v[:2])>.05 for v in velocities))
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
    if config.get('approach_staged_only') is True:
        selections=config.get('approach_selection_events',[])
        acquisitions=[event for event in config.get('reservation_events',[]) if event.get('event')=='acquired']
        candidate=config.get('approach_waypoints',[])
        alignment=config.get('approach_alignment_x_m')
        valid_candidate=(isinstance(alignment,(int,float)) and math.isfinite(alignment)
            and 2.4<alignment<=3.1 and isinstance(candidate,list) and len(candidate) in (2,3)
            and all(isinstance(point,dict) and set(point)=={'x','y','yaw'}
                and all(isinstance(v,(int,float)) and math.isfinite(v) for v in point.values()) for point in candidate)
            and all(abs(point['x']-alignment)<1e-8 for point in candidate[:-1]))
        if not enabled or not valid_candidate or len(selections)!=len(acquisitions):
            issues.append(dict(reason='invalid staged-only approach configuration or selection coverage'))
        else:
            previous_acquire={}
            for acquired,selection in zip(acquisitions,selections):
                robot,tick=acquired['robot'],acquired['tick']
                staged=any(event.get('event')=='staging' and event.get('robot')==robot
                    and event.get('completed') is True
                    and previous_acquire.get(robot,-1)<event.get('start_tick',-1)<=event.get('end_tick',-1)<=tick
                    for event in config.get('events',[]))
                expected=[dict(point) for point in candidate]
                if not staged:
                    for point in expected[:-1]:
                        point['x']=2.4
                if (acquired.get('staged') is not staged or selection.get('staged') is not staged
                        or selection.get('robot')!=robot or selection.get('tick')!=tick
                        or selection.get('lateral_x_m')!=(alignment if staged else 2.4)
                        or selection.get('waypoints')!=expected):
                    issues.append(dict(reason='approach selection disagrees with measured staging event scope',event=selection))
                previous_acquire[robot]=tick
    home_entry=audit_home_entry_priority(result,frames) if coverage else dict(enabled=False,passed=False)
    if not home_entry['passed']:
        issues.append(dict(reason='home entry priority audit failed',detail=home_entry))
    arrival=audit_arrival_settling(result,frames) if coverage else dict(enabled=False,passed=False)
    if not arrival['passed']:
        issues.append(dict(reason='arrival settling audit failed',detail=arrival))
    compact=audit_compact_layout(result,frames) if coverage else dict(enabled=False,passed=False)
    if not compact['passed']:
        issues.append(dict(reason='compact receiving layout audit failed',detail=compact))
    return dict(enabled=enabled,passed=(coverage and bool(intervals) and not issues and not violations) if enabled else None,
        complete_control_coverage=coverage,event_issues=issues,ownership_violation_count=len(violations),
        first_ownership_violations=violations[:10],receiving_intervals=intervals,pickup_handoffs=handoffs,
        receiving_interval_source=interval_source,
        staged_only_approach_checked=config.get('approach_staged_only') is True,
        home_entry_priority=home_entry,
        compact_layout=compact,
        arrival_settling=arrival,
        grasp_and_peer_delivery_overlap_s=overlap*period,both_payloads_held_s=both_loaded*period,
        parallel_payload_activity=payload_activity.report(period,coverage),
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
