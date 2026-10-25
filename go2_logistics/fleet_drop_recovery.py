"""Fail-closed ownership and traffic quarantine for shared-world drop recovery.

Only transport losses are recoverable initially. A second simultaneous loss,
loss during manipulation, or an exhausted per-order budget stops the mission.
The sensor search/grasp is provided by DroppedParcelRecovery; this module never
uses parcel poses or advances physics.
"""
from .mission import PayloadLost
import math


class FleetDropRecovery:
    transport_phases = frozenset(('leaving_receiving', 'transporting',
                                  'waiting_delivery_transit',
                                  'waiting_approach_corridor',
                                  'waiting_west_spine', 'drop_injection',
                                  'waiting_row_1', 'waiting_row_2', 'waiting_row_3'))

    def __init__(self, dispatcher, *, max_losses_per_order=2):
        if max_losses_per_order < 1:
            raise ValueError('Drop recovery requires a positive per-order budget')
        self.dispatcher = dispatcher
        self.max_losses_per_order = max_losses_per_order
        self.owner = None
        self.loss_counts = {}
        self.events = []

    def begin_injection(self, robot, code, tick):
        order = self.dispatcher.orders[code]
        if self.owner is not None or order.owner != robot or order.state != 'carrying':
            raise RuntimeError('Drop injection requires exclusive carrying ownership')
        self.owner = robot
        self.events.append(dict(event='injection_started', robot=robot, code=code, tick=tick,
                                method='stationary_lower_then_open_fingers'))

    def detect(self, robot, code, phase, error, tick, loss):
        if not isinstance(error, PayloadLost):
            raise TypeError('Only a finger-contact loss can enter drop recovery')
        order = self.dispatcher.orders.get(code)
        if (order is None or order.owner != robot or order.state != 'carrying'
                or phase not in self.transport_phases):
            raise RuntimeError(f'Drop outside supported transport recovery: {robot}, {code}, {phase}')
        if self.owner not in (None, robot):
            raise RuntimeError('Concurrent payload losses exceed the initial recovery scope')
        count = self.loss_counts.get(code, 0) + 1
        self.loss_counts[code] = count
        if count > self.max_losses_per_order:
            raise RuntimeError(f'Drop recovery budget exhausted for {code}')
        self.owner = robot
        self.dispatcher.transition(code, robot, 'recovering')
        self.events.append(dict(event='contact_loss', robot=robot, code=code, tick=tick,
                                attempt=count, phase=phase, reason=str(error), loss=loss))

    def recover(self, robot, code, operation, arm, tick):
        order = self.dispatcher.orders[code]
        if self.owner != robot or order.owner != robot or order.state != 'recovering':
            raise RuntimeError('Recovery must retain the original robot and parcel identity')
        success = operation(code)
        success = bool(success and arm.held and arm.verified_lift > .07)
        self.events.append(dict(event='recovery_finished', robot=robot, code=code,
                                tick=tick(), success=success, verified_lift_m=arm.verified_lift))
        if not success:
            raise RuntimeError(f'Dropped parcel recovery failed: {arm.last_error}')
        self.dispatcher.transition(code, robot, 'carrying')
        self.owner = None

    def active_robots(self):
        return None if self.owner is None else {self.owner}


def wait_for_reservation(dispatcher, bridge, robot, resource, on_wait, on_loss=None, *, before_wait=None):
    """Recover a waiting payload without stealing the peer's reservation.

    The common physics loop detects the contact loss and quarantines peers;
    the bridge delivers its typed exception on resuming this waiting stack.
    """
    prepared = False
    while not dispatcher.acquire(resource, robot):
        bridge._zero_command()
        try:
            # Optional measured braking runs in the original navigation phase.
            # Only then may the caller enter its stationary reservation wait.
            if not prepared:
                if before_wait is not None:
                    before_wait()
                prepared = True
            on_wait()
            bridge._policy_step()
        except PayloadLost:
            if on_loss is None:
                raise
            on_loss()


def drop_recovery_check(result):
    """Audit the exercised recovery separately from final delivery acceptance."""
    config = result.get('drop_recovery', {})
    events = config.get('events', [])
    losses = [event for event in events if event['event'] == 'contact_loss']
    recovered = [event for event in events if event['event'] == 'recovery_finished' and event['success']]
    injected = [event for event in events if event['event'] == 'injection_started']
    searches = config.get('sensor_events', {})
    def ordered_recovery(loss, finish):
        if ((loss['robot'], loss['code']) != (finish['robot'], finish['code'])
                or not isinstance(loss.get('tick'), int) or not isinstance(finish.get('tick'), int)
                or finish['tick'] <= loss['tick'] or finish.get('verified_lift_m', 0.) <= .07):
            return False
        observations = searches.get(loss['robot'], [])
        sightings = [event for event in observations
            if event.get('event') == 'search' and event.get('object_id') == loss['code']
            and event.get('found') and (event.get('detection') or {}).get('parcel_code') == loss['code']
            and isinstance(event.get('tick'), int) and loss['tick'] < event['tick'] < finish['tick']]
        return any(event.get('event') == 'regrasp' and event.get('object_id') == loss['code']
            and event.get('success') and isinstance(event.get('tick'), int)
            and sighting['tick'] < event['tick'] <= finish['tick']
            for sighting in sightings for event in observations)
    recovery_ok = (bool(losses) and len(losses) == len(recovered)
        and all(ordered_recovery(loss, finish) and
                (index+1 == len(losses) or finish['tick'] < losses[index+1].get('tick', -1))
                for index, (loss, finish) in enumerate(zip(losses, recovered))))
    injection_ok = bool(injected) and all(
        isinstance(start.get('tick'), int) and any(
            loss['robot'] == start['robot'] and loss['code'] == start['code']
            and isinstance(loss.get('tick'), int) and any(
                event.get('event') == 'injection' and isinstance(event.get('tick'), int)
                and start['tick'] <= event['tick'] <= loss['tick']
                for event in searches.get(start['robot'], []))
            for loss in losses)
        for start in injected)
    return dict(enabled=bool(config.get('enabled')), injection_requested=bool(config.get('injection_requested')),
                injections=len(injected), detected_losses=len(losses), successful_recoveries=len(recovered),
                exercised_recovery_check=recovery_ok,
                injection_exercised_check=injection_ok if config.get('injection_requested') else None)


def quarantine_check(result, frames):
    """Check the peer's actual sent commands throughout every recovery interval."""
    intervals, owner, start, valid = [], None, None, True
    for event in result.get('drop_recovery', {}).get('events', []):
        if event['event'] in ('injection_started', 'contact_loss'):
            if owner is None:
                owner, start = event['robot'], event.get('tick')
            elif owner != event['robot']:
                valid = False
        elif event['event'] == 'recovery_finished' and event.get('success'):
            if owner != event['robot']:
                valid = False
            intervals.append((owner, start, event.get('tick')))
            owner, start = None, None
    if owner is not None:
        intervals.append((owner, start, result.get('control_ticks')))
    inspected, violations = 0, 0
    for owner, start, end in intervals:
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            valid = False
            continue
        selected = [row for row in frames if start < row['tick'] <= end]
        valid = valid and [row['tick'] for row in selected] == list(range(start+1, end+1))
        for row in selected:
            peers = [sample for name, sample in row['robots'].items() if name != owner]
            valid = valid and len(peers) == 1 and owner in row['robots']
            for sample in peers:
                command = sample.get('commanded_velocity')
                zero = (isinstance(command, list) and len(command) == 3 and all(
                    isinstance(value, (int, float)) and math.isfinite(value) and abs(value) <= 1e-9
                    for value in command))
                violations += int(sample.get('drop_quarantined') is not True or not zero)
            inspected += 1
    return dict(intervals=len(intervals), inspected_frames=inspected, violating_frames=violations,
                peer_command_quarantine_check=(bool(inspected) and valid and violations == 0) if intervals else None)
