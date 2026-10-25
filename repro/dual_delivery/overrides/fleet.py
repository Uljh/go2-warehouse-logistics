"""Single-thread fleet ownership and cooperative missions, independent of Isaac."""
from dataclasses import dataclass
import math


def navigation_step_budget(raw_steps, control_period_s, seconds=None):
    """Preserve the navigation loop budget across 50/200 Hz backends.

    Explicit raw-step recipes remain reproducible. A seconds-based recipe
    counts outer navigation iterations; nested stance recovery retains its
    separate original three-second guard and the mission's global tick cap.
    """
    if (isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps <= 0
            or not math.isfinite(control_period_s) or control_period_s <= 0):
        raise ValueError('Navigation budget requires positive steps and a finite control period')
    if seconds is None:
        return raw_steps
    seconds = float(seconds)
    if not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise ValueError('Navigation budget seconds must be finite in (0, 300]')
    return max(1, math.ceil(seconds/control_period_s-1e-9))


def limit_planar_speed(velocity, maximum):
    """Bound the combined carry command without changing direction or yaw."""
    if (len(velocity)!=3 or not all(math.isfinite(v) for v in velocity)
            or not math.isfinite(maximum) or maximum<=0.):
        raise ValueError('Planar speed limit requires a finite velocity and positive bound')
    scale=min(1.,maximum/max(math.hypot(*velocity[:2]),1e-12))
    return velocity[0]*scale,velocity[1]*scale,velocity[2]


def return_speed_limit(velocity, maximum, *, original_z1, payload_held, phase):
    """Opt-in empty return command cap, preserving traffic veto and yaw."""
    if not math.isfinite(maximum) or not 0 <= maximum <= .50:
        raise ValueError('Empty return speed limit must be finite in [0, .50]')
    if maximum and original_z1 and not payload_held and phase == 'returning':
        return limit_planar_speed(velocity, maximum)
    return tuple(velocity)


def slew_velocity(previous, requested, dt, linear_acceleration=.8, yaw_acceleration=2.):
    """Bound ordinary velocity-command changes; exact zero remains a safety stop.

    The traffic guard must run after this function and its veto must also be
    stored as the next previous command, so release cannot resume at full speed.
    This bounds policy inputs, not the robot's measured physical acceleration.
    """
    if dt<=0 or linear_acceleration<=0 or yaw_acceleration<=0:
        raise ValueError('Velocity slew requires positive timestep and acceleration')
    if all(abs(v)<1e-8 for v in requested):
        return (0.,0.,0.)
    dx,dy=requested[0]-previous[0],requested[1]-previous[1]
    distance=math.hypot(dx,dy)
    scale=min(1.,linear_acceleration*dt/max(distance,1e-12))
    yaw_delta=max(-yaw_acceleration*dt,min(yaw_acceleration*dt,requested[2]-previous[2]))
    return (previous[0]+dx*scale,previous[1]+dy*scale,previous[2]+yaw_delta)


def west_return_route(shelf, *, original_z1=False, spine_x=-6.6):
    """Keep the spine reservation until east of the waiting dog's approach.

    The former endpoint (-4.7, .65) left the homeward diagonal crossing the
    incoming diagonal from (-4.7, -1.3). Both could be safely vetoed forever.
    The final eastward segment clears that intersection before releasing.
    Static geometry is still scanned by SensorNavigator for each segment.
    """
    from .interfaces import Pose2D
    if original_z1:
        # Reverse the west-facing dock approach before turning south. A
        # lateral departure at the final short-arm dock crosses the posts.
        return (Pose2D(shelf.surface[0]-.65,shelf.surface[1]),
                Pose2D(shelf.surface[0]-.65,shelf.surface[1]-.85),
                Pose2D(spine_x,shelf.dock.y-.85),Pose2D(spine_x,.65),
                Pose2D(-4.7,.65),Pose2D(-2.5,.65))
    return (Pose2D(shelf.dock.x,shelf.dock.y-.85),
            Pose2D(spine_x,shelf.dock.y-.85),Pose2D(spine_x,.65),
            Pose2D(-4.7,.65),Pose2D(-2.5,.65))


def spine_release_clear(pose, *, spine_x, margin=1.45):
    """Require measured trunk clearance before parking without the spine lock.

    The margin retains 20 cm beyond the existing 1.25 m traffic guard. This is
    a layout eligibility gate, not a replacement for scans or physical tests.
    """
    if not all(math.isfinite(v) for v in (pose.x, pose.y, spine_x, margin)) or margin < 1.25:
        raise ValueError('Spine release requires finite geometry and the original traffic margin')
    return pose.x-spine_x >= margin


def placement_standoff(shelf):
    """Stop south of the deck before switching to stationary arm placement.

    A walking payload can descend onto the deck before the standard dock is
    reached. This offset keeps its measured planar footprint outside the deck
    while retaining the existing heading and rack height. Planar clearance
    alone does not establish arm reachability or collision-free arm motion.
    """
    from .interfaces import Pose2D
    return Pose2D(shelf.dock.x,shelf.dock.y-.45,shelf.dock.yaw)


def receiving_stage_pose(pickup):
    """Return the marked buffer pose used to overlap the next pickup approach.

    The second dog can travel here after the first has verified its grasp,
    while the first dog clears the single parcel slot.  The buffer is offset
    from the first dog's fixed exit leg, so the exclusive receiving
    reservation remains held until that leg is clear.  This is a warehouse
    layout waypoint; SensorNavigator still scans and plans the actual path.
    """
    from .interfaces import Pose2D
    if not math.isfinite(pickup.x):
        raise ValueError('Receiving staging requires a finite pickup coordinate')
    # warehouse_scene.py paints the temporary buffer around (4, -.15). Keep
    # its x aligned with the configured pickup fixture if that fixture shifts.
    return Pose2D(float(pickup.x), -.15)


def receiving_stage_route(pickup):
    """Approach the buffer along the marked southern lane, clear of the bin."""
    from .interfaces import Pose2D
    return (Pose2D(0., -.15), receiving_stage_pose(pickup))


def receiving_staging_allowed(receiver_phase):
    """Only start the handoff approach after the owner has reached the bin."""
    return receiver_phase in ('waiting_visible_parcel', 'grasping', 'leaving_receiving')


def handoff_owned_resource(dispatcher, robot, resource, brake, release, on_wait):
    """Brake under the old lease, then relinquish it before waiting elsewhere.

    Releasing first leaves a suspended navigation/braking stack unprotected.
    Keeping the lease while waiting for the next one can deadlock a returner.
    The measured braking callback must finish before the ownership handoff.
    """
    if dispatcher.resources.get(resource)!=robot:
        raise RuntimeError('Cannot hand off a resource the robot does not own')
    brake()
    if dispatcher.resources.get(resource)!=robot:
        raise RuntimeError('Resource ownership changed during braking')
    release()
    if dispatcher.resources.get(resource)==robot:
        raise RuntimeError('Previous resource was not released before waiting')
    on_wait()


def z1_placement_route(shelf):
    """Align clear of rack posts, then approach the original short arm's dock.

    These are station waypoints, still checked by the sensed occupancy map.
    Keeping the lateral leg 0.65 m west avoids crossing a front post's 0.35 m
    low-layer margin, unlike a lateral approach at the final 0.45 m dock.
    The final pose needs tighter navigation tolerance to preserve IK reserve.
    """
    from .interfaces import Pose2D
    x,y,_=shelf.surface
    return (Pose2D(x-.65,y-.85,shelf.dock.yaw),
            Pose2D(x-.65,y,shelf.dock.yaw),
            Pose2D(x-.45,y,shelf.dock.yaw))


def parked_home_peer_positions(robot, positions, phases, homes, tolerance=.21):
    """Plan around an idle dog at home; moving encounters retain the guard.

    A disc around a moving peer can suddenly consume the transporter's start
    and deadlock with the existing velocity veto. This limited candidate only
    supplies the already-returned waiting dog's measured XY to the planner.
    """
    return [tuple(position) for peer,position in positions.items()
            if peer!=robot and phases[peer] in ('waiting_receiving','complete')
            and math.dist(position,homes[peer])<=tolerance]


def clear_placement_station(arm, depart, *, before_stow=None):
    """Leave the shelf before a joint-space fold can sweep through its parcel."""
    if arm.held:
        raise RuntimeError('Cannot depart for an empty-arm fold with a held parcel')
    depart()
    if before_stow is not None:
        before_stow()
    arm.stow_empty(steps=150)
    if not arm._verify_placement():
        raise RuntimeError(arm.last_error)


@dataclass
class FleetOrder:
    code: str
    owner: str
    state: str = 'claimed'
    error: str = ''


def observed_grasp(arm, perception, code, *, attempts=3, emit=lambda event: None):
    """Bounded retries preserve the claimed ID and refresh observation after folding."""
    for attempt in range(1,attempts+1):
        if attempt>1:
            if arm.held:
                raise RuntimeError('Cannot retry a failed grasp while the payload is still held')
            arm.stow_empty()
        observed=perception.locate(code)
        if not observed:
            arm.last_error=f'Camera could not re-identify claimed parcel {code}'
        success=observed and arm.grasp(code)
        emit(dict(operation='grasp',code=code,attempt=attempt,observed=observed,
                  success=bool(success),error='' if success else arm.last_error))
        if not success and arm.held:
            raise RuntimeError(f'Grasp failed with payload still held: {arm.last_error}')
        if success:
            if not arm.held or arm.verified_lift<=.07:
                raise RuntimeError('Grasp reported success without verified physical lift')
            return True
    return False


def verified_release(arm, code, *, attempts=3, emit=lambda event: None):
    """Released parcels are rechecked by the controller; never reattached on retry."""
    for attempt in range(1,attempts+1):
        success=arm.release()
        emit(dict(operation='release',code=code,attempt=attempt,success=bool(success),
                  error='' if success else arm.last_error))
        if success:
            return True
    return False


class FleetDispatcher:
    """Observed IDs are claimed atomically; in-flight failures are quarantined.

    All calls occur on the main simulation thread. Pickup reservations outlive
    the grasp until the robot exits the station's protected area.
    """
    def __init__(self):
        self.orders = {}
        self.resources = {}
        self.events = []

    def acquire(self, resource, robot):
        owner = self.resources.get(resource)
        if owner not in (None, robot):
            return False
        self.resources[resource] = robot
        return True

    def release(self, resource, robot):
        if self.resources.get(resource) != robot:
            raise RuntimeError(f'{robot} does not own resource {resource}')
        del self.resources[resource]

    def claim(self, code, robot):
        from .sorting_layout import ParcelCode
        ParcelCode.parse(code)
        if code in self.orders:
            return False
        if any(order.owner == robot and order.state in ('claimed', 'carrying', 'recovering') for order in self.orders.values()):
            raise RuntimeError(f'{robot} already has an active order')
        self.orders[code] = FleetOrder(code, robot)
        self.events.append(dict(event='claimed', code=code, robot=robot))
        return True

    def transition(self, code, robot, state, error=''):
        order = self.orders[code]
        allowed = {'claimed': {'carrying', 'failed'},
                   'carrying': {'delivered', 'recovering', 'failed'},
                   'recovering': {'carrying', 'failed'}}
        if order.owner != robot or state not in allowed.get(order.state, set()):
            raise RuntimeError(f'Invalid order transition: {code}, {robot}, {order.state} -> {state}')
        order.state, order.error = state, error
        self.events.append(dict(event=state, code=code, robot=robot, error=error))

    @property
    def delivered(self):
        return sorted(code for code, order in self.orders.items() if order.state == 'delivered')

    def snapshot(self):
        return dict(orders={code:vars(order).copy() for code, order in self.orders.items()},
                    resources=dict(self.resources), delivered=self.delivered)


def closest_separation(a, va, b, vb, horizon):
    delta = (a[0]-b[0], a[1]-b[1])
    speed = (va[0]-vb[0], va[1]-vb[1])
    denominator = speed[0]**2 + speed[1]**2
    time = max(0., min(horizon, -(delta[0]*speed[0]+delta[1]*speed[1])/denominator)) if denominator else 0.
    return math.hypot(delta[0]+time*speed[0], delta[1]+time*speed[1])


def traffic_commands(positions, velocities, margin=1.25, horizon=1.5):
    """Conservative pairwise command veto; telemetry is centralized truth.

    Sensor navigation remains responsible for static geometry. This guard
    checks requested straight velocity over a finite horizon; it is not an
    optimal multi-agent planner or a formal guarantee under gait tracking error.
    """
    result = dict(velocities)
    stopped = set()
    names = sorted(positions)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if closest_separation(positions[first], result[first], positions[second], result[second], horizon) >= margin:
                continue
            # Give the first robot priority only if its path clears the stopped
            # peer. Stopping just one dog is unsafe when driving straight at it.
            if closest_separation(positions[first], result[first], positions[second], (0., 0.), horizon) >= margin:
                result[second] = (0., 0.)
                stopped.add(second)
            elif closest_separation(positions[first], (0., 0.), positions[second], result[second], horizon) >= margin:
                result[first] = (0., 0.)
                stopped.add(first)
            else:
                result[first] = result[second] = (0., 0.)
                stopped.update((first, second))
    # Later pair changes can invalidate an earlier priority decision. Recheck
    # with final velocities; stopping both gives a bounded conservative veto.
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if closest_separation(positions[first], result[first], positions[second], result[second], horizon) < margin:
                result[first] = result[second] = (0., 0.)
                stopped.update((first, second))
    return result, stopped


class CooperativeMissions:
    """Stackful coroutines preserve synchronous IK; only the parent ticks physics.

    greenlet switches within one OS thread. No worker calls Isaac or advances
    the world independently. Each task runs until yield_tick() once per frame.
    """
    def __init__(self):
        from greenlet import getcurrent
        self.parent = getcurrent()
        self.tasks = {}
        self.errors = {}

    def add(self, name, function):
        from greenlet import greenlet
        if name in self.tasks:
            raise ValueError(f'Duplicate coroutine: {name}')
        self.tasks[name] = greenlet(function, parent=self.parent)

    def yield_tick(self):
        from greenlet import getcurrent
        if getcurrent() is self.parent:
            raise RuntimeError('yield_tick must run inside a mission coroutine')
        self.parent.switch()

    def advance(self, active=None):
        for name, task in self.tasks.items():
            if task.dead or active is not None and name not in active:
                continue
            try:
                task.switch()
            except Exception as error:
                self.errors[name] = str(error)

    @property
    def finished(self):
        return all(task.dead for task in self.tasks.values())
