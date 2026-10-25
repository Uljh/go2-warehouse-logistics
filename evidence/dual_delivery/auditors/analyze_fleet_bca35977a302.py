"""Independent trace review for shared-world navigation/logistics runs."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def projected_half_extents(dimensions, quaternion_xyzw):
    """Axis-aligned half extents of the measured rotated cuboid; no Isaac dependency."""
    if len(dimensions)!=3 or len(quaternion_xyzw)!=4:
        raise ValueError('Expected three box dimensions and an xyzw quaternion')
    if not all(math.isfinite(v) and v>0 for v in dimensions):
        raise ValueError('Box dimensions must be finite and positive')
    norm=math.sqrt(sum(v*v for v in quaternion_xyzw))
    if not math.isfinite(norm) or abs(norm-1.)>.01:
        raise ValueError('Measured quaternion must be finite and normalized')
    x,y,z,w=(v/norm for v in quaternion_xyzw)
    rotation=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
              (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
              (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    return tuple(sum(abs(row[i])*dimensions[i]/2 for i in range(3)) for row in rotation)


def control_stability(folder, result, robots):
    """Use every recorded control frame; missing evidence cannot prove endurance."""
    path=folder/'control_trace.jsonl'
    if not path.exists():
        return dict(available=False,complete_coverage=False,robots={})
    frames=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    period=result.get('control_trace_period_s',0.)
    def finite_number(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    def valid_physics(sample):
        return finite_number(sample.get('height_m')) and finite_number(sample.get('gravity_z'))
    duration=result.get('sim_time_s')
    coverage=(finite_number(period) and period>0 and finite_number(duration)
        and abs(duration-len(frames)*period)<1e-6
        and len(frames)==result.get('control_ticks') and bool(frames)
        and all(row['tick']==i+1 and abs(row['sim_time']-(i+1)*period)<1e-6
                and set(row['robots'])==set(robots)
                and all(valid_physics(row['robots'][name]) for name in robots)
                for i,row in enumerate(frames)))
    summary={}
    for name in robots:
        samples=[row['robots'][name] for row in frames if name in row['robots']]
        if not samples:
            continue
        valid=[sample for sample in samples if valid_physics(sample)]
        longest=consecutive=0
        for sample in samples:
            consecutive=consecutive+1 if valid_physics(sample) and sample['height_m']<.285 else 0
            longest=max(longest,consecutive)
        by_command={}
        for sample in valid:
            velocity=sample.get('commanded_velocity')
            if not isinstance(velocity,list) or len(velocity)!=3 or not all(finite_number(v) for v in velocity):
                continue
            vx,vy,_=velocity
            mode=('recovery' if sample.get('recovering') else 'forward' if vx>.1
                  else 'reverse' if vx<-.1 else 'lateral' if abs(vy)>.1 else 'idle')
            by_command.setdefault(mode,[]).append(sample['height_m'])
        summary[name]=dict(min_height_m=min(row['height_m'] for row in valid) if valid else None,
            max_tilt_deg=max(math.degrees(math.acos(max(-1.,min(1.,-row['gravity_z'])))) for row in valid) if valid else None,
            invalid_physics_frames=len(samples)-len(valid),
            command_conditioned_height={mode:dict(frames=len(heights),duration_s=len(heights)*period,
                mean_height_m=sum(heights)/len(heights),
                below_recovery_height_time_s=sum(h<.285 for h in heights)*period)
                for mode,heights in by_command.items()},
            below_recovery_height_time_s=sum(row['height_m']<.285 for row in valid)*period,
            longest_below_recovery_height_s=longest*period,
            recovery_time_s=sum(row.get('recovering',False) for row in samples)*period,
            phase_observations_complete=all(isinstance(row.get('phase'),str) for row in samples),
            phase_time_s={phase:count*period for phase,count in Counter(
                row['phase'] for row in samples if isinstance(row.get('phase'),str)).items()},
            original_z1_phase_time_s={phase:count*period for phase,count in Counter(
                row['original_z1']['state'] for row in samples
                if isinstance(row.get('original_z1'),dict)
                and isinstance(row['original_z1'].get('state'),str)).items()},
            stationary_balance_support_time_s=sum(
                result.get('stationary_mode') == 'balance' and row.get('standing_pd', False)
                and (row.get('balance_recovery') or {}).get('active', False) for row in samples)*period,
            recovery_observations_complete=all(isinstance(row.get('recovering'),bool) for row in samples),
            hard_stop_frames=sum(row['height_m']<.22 or row['gravity_z']>-.85 for row in valid))
    from .fleet_drop_recovery import quarantine_check
    return dict(available=True,complete_coverage=coverage,frame_count=len(frames),period_s=period,
                robots=summary,drop_quarantine=quarantine_check(result,frames))


def streaming_feed_evidence(result):
    """A complete flag alone cannot establish multi-round physical delivery."""
    if not result.get('streaming_feed_enabled'):
        return dict(required=False,passed=True,reason='legacy finite batch')
    feed=result.get('streaming_feed')
    codes=result.get('configured_codes',[])
    if not isinstance(feed,dict):
        return dict(required=True,passed=False,reason='streaming ledger missing')
    required=('physical_queue_preloaded','visual_receiving_observation',
              'claimed_after_visual_observation','carrying','delivered_after_verified_placement')
    failures={}
    for code in codes:
        rows=[row for row in feed.get('events',[]) if row.get('code')==code]
        kinds=[row.get('event') for row in rows]
        ticks=[row.get('tick') for row in rows]
        ordered=(kinds==list(required) and all(isinstance(t,int) and t>=0 for t in ticks)
                 and ticks==sorted(ticks))
        visual=next((row for row in rows if row.get('event')=='visual_receiving_observation'),{})
        center=visual.get('center_world')
        visual_valid=(visual.get('source')=='rendered_rgb_depth' and isinstance(center,list)
            and len(center)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in center))
        bounds=feed.get('position_bounds',{}).get(code,{})
        moved=(bounds.get('samples',0)>=2 and isinstance(bounds.get('first'),list)
               and isinstance(bounds.get('last'),list)
               and math.dist(bounds['first'],bounds['last'])>=.08)
        if not (ordered and visual_valid and moved):
            failures[code]=dict(ordered_events=ordered,rendered_rgbd=visual_valid,measured_motion=moved)
    claims=feed.get('claim_counts',{})
    deliveries=feed.get('delivery_counts',{})
    states=feed.get('state',{})
    passed=(6<=len(codes)<=10 and len(codes)==len(set(codes))
        and feed.get('complete') is True and feed.get('all_delivered') is True
        and feed.get('source_exhausted') is True and set(feed.get('codes',[]))==set(codes)
        and set(feed.get('preloaded_codes',[]))==set(codes) and set(states)==set(codes)
        and all(states.get(code)=='delivered' and claims.get(code)==1 and deliveries.get(code)==1 for code in codes)
        and not feed.get('duplicate_events') and not failures
        and Counter(item.get('code') for item in result.get('delivery_evidence',[]))==Counter(codes)
        and set(result.get('dispatcher',{}).get('delivered',[]))==set(codes))
    return dict(required=True,passed=passed,parcel_count=len(codes),order_failures=failures,
        scope='bounded pre-created physical queue; runtime ledger cross-check, not infinite replenishment')


def analyze(folder, *, generate_plot=True, report_path=None):
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    records=[json.loads(line) for line in (folder/'trace.jsonl').read_text(encoding='utf-8').splitlines() if line]
    robots=sorted(result['returned'])
    feed_evidence=streaming_feed_evidence(result)
    controls=control_stability(folder,result,robots)
    stability={}
    for name in robots:
        heights=[row['roots'][name][2] for row in records]
        tilt=[math.degrees(math.acos(max(-1.,min(1.,-row['gravity'][name][2])))) for row in records]
        forces=[row.get('finger_contact_force_n',{}).get(name) for row in records if row['held'][name]]
        available=[force for force in forces if force is not None]
        stability[name]=dict(sampled_min_height_m=min(heights) if heights else None,
            sampled_max_tilt_deg=max(tilt) if tilt else None,
            sampled_carried_frames=sum(row['held'][name] for row in records),
            contact_frames_available=len(available),
            opposing_contact_frames=sum(min(force.values())>.05 for force in available))
    roots=result.get('final_roots',records[-1]['roots'] if records else {})
    home_errors={name:math.dist(roots[name][:2],(0.,-1. if i==0 else 1.)) for i,name in enumerate(robots) if name in roots}
    measured_overlap=0.
    traffic_wait={name:dict(sampled_veto_time_s=0.,longest_sampled_veto_s=0.) for name in robots}
    consecutive_wait={name:0. for name in robots}
    for before,after in zip(records,records[1:]):
        dt=after['sim_time']-before['sim_time']
        if dt<=0:
            continue
        for name in robots:
            vetoed=name in before.get('traffic_stopped',[])
            consecutive_wait[name]=consecutive_wait[name]+dt if vetoed else 0.
            traffic_wait[name]['sampled_veto_time_s']+=dt if vetoed else 0.
            traffic_wait[name]['longest_sampled_veto_s']=max(
                traffic_wait[name]['longest_sampled_veto_s'],consecutive_wait[name])
        if dt>0 and all(math.dist(before['roots'][name][:2],after['roots'][name][:2])/dt>.10
                       and math.hypot(*before.get('commanded_velocities',{}).get(name,[0.,0.])[:2])>.05
                       for name in robots):
            measured_overlap+=dt
    from .sorting_layout import shelf_for_code, SHELF_DECK_SIZE
    placement_checks={}
    for code,position in result.get('final_parcels',{}).items():
        shelf=shelf_for_code(code)
        planar=math.dist(position[:2],shelf.surface[:2])
        dimensions=result.get('parcel_dimensions_m',{}).get(code)
        quaternion=result.get('final_parcel_quaternions_xyzw',{}).get(code)
        geometry_available=dimensions is not None and quaternion is not None
        if geometry_available:
            extents=projected_half_extents(dimensions,quaternion)
            height=abs(position[2]-extents[2]-shelf.surface[2])
            footprint_fits=all(abs(position[i]-shelf.surface[i])+extents[i]<=SHELF_DECK_SIZE[i]/2 for i in (0,1))
        else:
            # Legacy recordings lack orientation: retain the upright assumption,
            # never infer a favorable orientation from center height alone.
            height=abs(position[2]-(shelf.surface[2]+.04))
            footprint_fits=None
        velocity=result.get('final_parcel_velocities',{}).get(code)
        speed=math.sqrt(sum(v*v for v in velocity)) if velocity is not None else None
        placement_checks[code]=dict(expected_shelf=shelf.number,planar_error_m=planar,
            height_error_m=height,final_speed_mps=speed,
            geometry_source='measured_cuboid_orientation' if geometry_available else 'legacy_upright_assumption',
            footprint_within_deck=footprint_fits,
            within_baseline_tolerance=planar<=.04 and height<=.015 and footprint_fits is not False
                and speed is not None and speed<.10)
    deliveries=result.get('delivery_evidence',[])
    delivery_identity_check=(Counter(item.get('code') for item in deliveries)==Counter(result['configured_codes'])
        and all(item.get('robot') in robots
                and item.get('shelf')==shelf_for_code(item['code']).number for item in deliveries))
    order_execution={}
    for code in result['configured_codes']:
        evidence=[item for item in deliveries if item.get('code')==code]
        owner=evidence[0]['robot'] if len(evidence)==1 else result.get('dispatcher',{}).get('orders',{}).get(code,{}).get('owner')
        selected=[row for row in records if owner and row.get('codes',{}).get(owner)==code]
        carried=[row for row in selected if row.get('held',{}).get(owner)]
        contact=[row.get('finger_contact_force_n',{}).get(owner) for row in carried]
        delivered=next((row['sim_time'] for row in records
                        if code in row.get('dispatcher',{}).get('delivered',[])),None)
        home=next((row['sim_time'] for row in records
                   if delivered is not None and row['sim_time']>=delivered and owner in robots
                   and row.get('codes',{}).get(owner) is None
                   and math.dist(row['roots'][owner][:2],(0.,-1. if robots.index(owner)==0 else 1.))<=.21),None)
        phase_time={}
        for before,after in zip(records,records[1:]):
            if owner and before.get('codes',{}).get(owner)==code:
                phase=before.get('phases',{}).get(owner,'unrecorded')
                phase_time[phase]=phase_time.get(phase,0.)+after['sim_time']-before['sim_time']
        order_execution[code]=dict(robot=owner,code_trace_available=bool(selected),
            first_identity_sample_s=selected[0]['sim_time'] if selected else None,
            post_departure_delivery_sample_s=delivered,first_home_after_delivery_sample_s=home,
            carried_samples=len(carried),opposing_contact_samples=sum(bool(f) and min(f.values())>.05 for f in contact),
            phase_duration_s=phase_time,
            note='Sampled at 5 Hz, not exact event times; delivery is recorded after departure/stow verification. Missing code trace cannot prove per-order contact.')
    per_order_contact_check=bool(order_execution) and all(
        item['code_trace_available'] and item['opposing_contact_samples']>0 for item in order_execution.values())
    from .fleet_drop_recovery import drop_recovery_check
    drops = drop_recovery_check(result)
    drops['traffic_quarantine'] = controls.get('drop_quarantine', {})
    drop_acceptance = ((not drops['detected_losses'] or drops['exercised_recovery_check'])
        and (not drops['injection_requested'] or
             drops['injection_exercised_check'] and drops['exercised_recovery_check'])
        and (not drops['detected_losses'] or controls.get('complete_coverage') and
             drops['traffic_quarantine'].get('peer_command_quarantine_check') is True))
    control_safety=(controls.get('complete_coverage') is True
        and set(controls['robots'])==set(robots)
        and all(item['hard_stop_frames']==0 for item in controls['robots'].values()))
    full_contact_check=(result['mode']=='shared_world_contact_logistics'
        and result.get('shared_world') is True and result.get('num_envs')==1
        and result.get('world_reset_between_orders') is False
        and control_safety and not result.get('failure')
        and feed_evidence['passed']
        and drop_acceptance
        and result['state']=='complete'
        and len(robots)==2 and all(result['returned'].values())
        and set(result['dispatcher']['delivered'])==set(result['configured_codes'])
        and set(placement_checks)==set(result['configured_codes'])
        and all(item['within_baseline_tolerance'] for item in placement_checks.values())
        and len(home_errors)==len(robots) and all(error<=.21 for error in home_errors.values())
        and delivery_identity_check
        and per_order_contact_check
        and all(not item['constraint_created'] and item['verified_lift_m']>.07 for item in result.get('delivery_evidence',[]))
        and len({item['robot'] for item in result.get('delivery_evidence',[])})==len(robots)
        and all(stability[name]['opposing_contact_frames']>0 for name in robots)
        and measured_overlap>0.)
    recovery_counts={name:result.get('navigation',{}).get(name,{}).get('stability_pauses') for name in robots}
    # This proves only repeated unloaded routes in this layout, with recovery
    # allowed. An uninterrupted gait claim requires no recovery intervention.
    repeated_route_check=(result['mode']=='shared_world_navigation_endurance'
        and result['state']=='complete' and len(robots)==2
        and result.get('shared_world') is True and result.get('num_envs')==1
        and result.get('world_reset_between_orders') is False
        and result.get('requested_endurance_cycles',0)>=2
        and all(result.get('completed_route_cycles',{}).get(name,0)>=result['requested_endurance_cycles'] for name in robots)
        and all(result['returned'].values()) and len(home_errors)==2
        and all(error<=.21 for error in home_errors.values()) and measured_overlap>0.
        and controls['complete_coverage'] and len(controls['robots'])==2
        and all(row['hard_stop_frames']==0 for row in controls['robots'].values()))
    endurance_check=repeated_route_check and result['sim_time_s']>=120.
    report=dict(state=result['state'],mode=result['mode'],failure=result['failure'],
        configured_codes=result['configured_codes'],delivered=result['dispatcher']['delivered'],
        recorded_sim_time_s=result['sim_time_s'],
        recorded_concurrent_navigation_time_s=result['concurrent_navigation_time_s'],
        sampled_concurrent_measured_motion_time_s=measured_overlap,
        sampled_both_carrying_frames=sum(all(row['held'][name] for name in robots) for row in records),
        recorded_minimum_trunk_separation_m=result['minimum_trunk_separation_m'],
        sampled_stability=stability,home_error_m=home_errors,
        sampled_traffic_wait=traffic_wait,
        navigation_recovery_counts=recovery_counts,
        control_frame_stability=controls,
        complete_control_safety_check=control_safety,
        unloaded_repeated_routes_with_recovery_check=repeated_route_check,
        unloaded_endurance_with_recovery_check=endurance_check,
        unloaded_endurance_without_recovery_check=endurance_check
            and all(count==0 for count in recovery_counts.values())
            and all(row['recovery_observations_complete'] and row['recovery_time_s']==0.
                    for row in controls['robots'].values()),
        observation_abi_checks=result.get('observation_abi_checks'),
        observation_abi_max_error=result.get('observation_abi_max_error'),
        delivery_evidence=result.get('delivery_evidence'),
        streaming_feed_evidence=feed_evidence,
        delivery_identity_check=delivery_identity_check,
        per_order_contact_check=per_order_contact_check,
        drop_recovery=drops,
        sampled_order_execution=order_execution,
        final_parcels=result.get('final_parcels'),
        independent_final_placement_checks=placement_checks,
        full_two_robot_contact_delivery_and_return_check=full_contact_check,
        note='Trace sampled every 10 control ticks; trunk separation is not whole-body collision proof. Navigation smoke never verifies deliveries. Missing final velocity cannot pass complete evidence. Older recordings without orientation retain an upright height assumption; stable side placement requires measured geometry.')
    Path(report_path or folder/'fleet_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if records and generate_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(1,2,figsize=(12,5))
        from .sorting_layout import numbered_shelves
        for shelf in numbered_shelves():
            axes[0].add_patch(plt.Rectangle((shelf.surface[0]-.25,shelf.surface[1]-.25),.5,.5,
                facecolor='lightgray',edgecolor='gray',alpha=.5))
            axes[0].text(*shelf.surface[:2],str(shelf.number),ha='center',va='center',fontsize=6)
        for name in robots:
            axes[0].plot([r['roots'][name][0] for r in records],[r['roots'][name][1] for r in records],label=name)
            axes[1].plot([r['sim_time'] for r in records],[r['roots'][name][2] for r in records],label=name)
        axes[0].set(xlabel='x (m)',ylabel='y (m)',title=f'Shared-world trajectories | {result["state"]}')
        axes[0].set_aspect('equal',adjustable='datalim')
        axes[1].axhline(.285,color='orange',linestyle='--',label='recovery threshold')
        axes[1].axhline(.22,color='red',linestyle='--',label='stop threshold')
        axes[1].set(xlabel='Simulation time (s)',ylabel='Trunk height (m)',title='Sampled stability')
        for axis in axes:
            axis.grid(alpha=.3)
            axis.legend()
        figure.tight_layout()
        figure.savefig(folder/'fleet_evidence.png',dpi=150)
        plt.close(figure)
        figure,axes=plt.subplots(3,len(robots),figsize=(6*len(robots),8),squeeze=False,sharex=True)
        times=[r['sim_time'] for r in records]
        for column,name in enumerate(robots):
            axes[0,column].plot(times,[r['roots'][name][2] for r in records])
            axes[0,column].axhline(.285,color='orange',linestyle='--')
            axes[0,column].axhline(.22,color='red',linestyle='--')
            axes[0,column].set(title=name,ylabel='Trunk height (m)')
            tilt=[math.degrees(math.acos(max(-1.,min(1.,-r['gravity'][name][2])))) for r in records]
            axes[1,column].plot(times,tilt)
            axes[1,column].axhline(math.degrees(math.acos(.985)),color='orange',linestyle='--')
            axes[1,column].axhline(math.degrees(math.acos(.85)),color='red',linestyle='--')
            axes[1,column].set(ylabel='Trunk tilt (deg)')
            for axis,label in enumerate(('forward','lateral')):
                axes[2,column].plot(times,[r.get('commanded_velocities',{}).get(name,[0.,0.])[axis]
                    for r in records],label=label)
            axes[2,column].set(xlabel='Simulation time (s)',ylabel='Policy command (m/s)')
            axes[2,column].legend()
        for row in axes:
            for axis in row:axis.grid(alpha=.3)
        figure.suptitle(f'{result["state"]} | trace sampled every 10 ticks; orange=recovery, red=hard stop')
        figure.tight_layout()
        figure.savefig(folder/'fleet_control_diagnostics.png',dpi=150)
        plt.close(figure)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path,nargs='?')
    args=parser.parse_args()
    folder=args.folder or Path(json.loads(Path('artifacts/fleet/latest_result.json').read_text(encoding='utf-8'))['run_dir'])
    print(json.dumps(analyze(folder),indent=2))


if __name__=='__main__':
    main()
