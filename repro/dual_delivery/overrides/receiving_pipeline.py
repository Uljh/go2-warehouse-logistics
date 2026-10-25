"""Receiving handoff on the existing single-thread simulation scheduler."""
from .fleet import receiving_stage_route, receiving_staging_allowed


def wait_for_receiving(dispatcher, bridge, robot, *, codes, phases, pickup,
                       navigate, withdraw, phase, tick, staging_events,
                       reservation_events, enabled):
    """Preposition an unladen peer; only the slot owner may approach the bin.

    Callbacks can yield through CooperativeMissions, so readiness and ownership
    are checked again after each move. Orders are still claimed only from the
    camera at the bin. A surplus standby dog goes home without claiming work.
    """
    staged = False
    requested = False
    while True:
        if all(code in dispatcher.orders for code in codes):
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
        if dispatcher.acquire('receiving', robot):
            reservation_events.append(dict(event='acquired', resource='receiving',
                robot=robot, tick=tick()))
            return True
        receiver = dispatcher.resources.get('receiving')
        receiver_phase = phases.get(receiver)
        if enabled and not staged and receiver != robot and receiving_staging_allowed(receiver_phase):
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
