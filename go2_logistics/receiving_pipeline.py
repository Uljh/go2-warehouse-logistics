"""Receiving handoff on the existing single-thread simulation scheduler."""
import math

from .fleet import receiving_stage_route, receiving_staging_allowed, receiving_approach_route


def select_receiving_approach(pickup, *, original_z1, lateral_x, staged_only, staged):
    """Keep the initial arrival at baseline when testing only standby handoff."""
    if not isinstance(staged_only, bool) or not isinstance(staged, bool):
        raise ValueError('Receiving staging selection requires explicit booleans')
    # Validate the candidate even when this particular arrival uses baseline.
    candidate = receiving_approach_route(pickup, original_z1=original_z1, lateral_x=lateral_x)
    if staged_only and not staged:
        return receiving_approach_route(pickup, original_z1=original_z1)
    return candidate


def home_entry_blocker(resources, robot, position, home, *, tolerance=.21, reservation_events=()):
    """Keep a home standby visible to the loaded corridor owner's planner."""
    if (len(position)!=2 or len(home)!=2 or not math.isfinite(tolerance) or tolerance<=0
            or not all(math.isfinite(value) for value in (*position,*home))):
        raise ValueError('Home entry priority requires finite measured geometry')
    if math.dist(position,home)>tolerance:
        return None
    owner=resources.get('approach_corridor')
    if owner is not None:
        return owner if owner!=robot else None
    # On the home-return tick this coroutine releases the corridor before the
    # peer coroutine resumes. Its older queued request must retain priority.
    pending={}
    for event in reservation_events:
        if event.get('resource')!='approach_corridor':
            continue
        peer=event.get('robot')
        if event.get('event')=='requested':pending.setdefault(peer,event.get('tick',0))
        elif event.get('event') in ('acquired','cancelled'):pending.pop(peer,None)
    peers=[peer for peer in pending if peer!=robot]
    if peers:
        return min(peers,key=pending.get)
    return None


def settle_receiving_arrival(bridge,navigator,robot,target,*,tick,events):
    """Brake under the navigation policy before enabling stationary support."""
    event=dict(robot=robot,start_tick=tick(),target=vars(target),completed=False,
        position_tolerance_m=bridge.config.position_tolerance,
        controller='existing_zero_command_recovery',minimum_duration_s=.3,budget_s=3.)
    events.append(event)
    try:
        bridge._zero_command()
        navigator.recover_stance(bridge,settle_motion=True,minimum_duration_s=.3)
        pose=bridge.pose()
        error=math.hypot(pose.x-target.x,pose.y-target.y)
        event.update(end_pose=vars(pose),dock_error_m=error)
        if error>bridge.config.position_tolerance:
            raise RuntimeError(f'{robot}: receiving stop left the original dock tolerance ({error:.4f}m)')
        event['completed']=True
    finally:
        event['end_tick']=tick()


def wait_for_receiving(dispatcher, bridge, robot, *, codes, phases, pickup,
                       navigate, withdraw, phase, tick, staging_events,
                       reservation_events, enabled, entry_blocker=None, entry_wait_events=None,
                       compact_layout=False):
    """Preposition an unladen peer; only the slot owner may approach the bin.

    Callbacks can yield through CooperativeMissions, so readiness and ownership
    are checked again after each move. Orders are still claimed only from the
    camera at the bin. A surplus standby dog goes home without claiming work.
    """
    staged = False
    requested = False
    entry_wait = None
    while True:
        if all(code in dispatcher.orders for code in codes):
            if entry_wait is not None:
                entry_wait.update(completed=True,end_tick=tick(),outcome='no_unclaimed_orders')
            if requested:
                reservation_events.append(dict(event='cancelled', resource='receiving',
                    robot=robot, tick=tick(), reason='no_unclaimed_orders'))
            if staged:
                event = dict(robot=robot, event='withdrawal', start_tick=tick(), completed=False,
                             reason='no_unclaimed_orders')
                staging_events.append(event)
                phase('withdrawing_receiving')
                withdraw()
                event.update(completed=True, end_tick=tick())
            bridge._zero_command()
            return False
        if not requested:
            reservation_events.append(dict(event='requested', resource='receiving',
                robot=robot, tick=tick()))
            requested = True
        blocked_by=entry_blocker() if entry_blocker is not None and not staged else None
        if blocked_by is not None and entry_wait is None:
            entry_wait=dict(event='home_entry_wait',robot=robot,blocking_robot=blocked_by,
                resource='approach_corridor',start_tick=tick(),completed=False,start_pose=vars(bridge.pose()))
            if entry_wait_events is not None:
                entry_wait_events.append(entry_wait)
        if blocked_by is None and entry_wait is not None:
            entry_wait.update(completed=True,end_tick=tick(),outcome='corridor_available')
            entry_wait=None
        if blocked_by is None and dispatcher.acquire('receiving', robot):
            reservation_events.append(dict(event='acquired', resource='receiving',
                robot=robot, tick=tick(), staged=staged))
            return True
        receiver = dispatcher.resources.get('receiving')
        receiver_phase = phases.get(receiver)
        if enabled and not staged and receiver != robot and receiving_staging_allowed(receiver_phase):
            if compact_layout:
                from .compact_receiving import stage_route
                route = stage_route(pickup)
            else:
                route = receiving_stage_route(pickup)
            event = dict(event='staging', robot=robot, blocking_robot=receiver,
                trigger_phase=receiver_phase, start_tick=tick(), start_pose=vars(bridge.pose()),
                target=vars(route[-1]), route=[vars(p) for p in route], completed=False)
            staging_events.append(event)
            phase('staging_receiving')
            for point in route:
                navigate(point)
                staged = True
                if all(code in dispatcher.orders for code in codes):
                    break
            event.update(completed=point == route[-1], end_tick=tick(), end_pose=vars(bridge.pose()))
            phase('waiting_receiving')
            continue
        phase('waiting_receiving')
        bridge._zero_command()
        bridge._policy_step()
