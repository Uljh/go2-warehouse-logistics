"""Optional forward cruise between measured, freely rotatable staging poses.

The existing A* map assumes yaw zero near fixtures. Cruise therefore uses a
separate circular swept footprint that covers every yaw of the trunk/arm/load,
and returns to docking yaw before handing back to that map. It never changes a
robot pose: outputs are body velocity commands for the published gait policy.
"""
import math


TURN_RADIUS = {0: .35 + .10, 1: math.hypot(.85, .40) + .10}


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def swept_turn_clear(start, end, returns, *, bounds, quantization=0., max_range=5.):
    """All-yaw capsule, including both end turns, within a fresh scan's range.

    The quantized persistent map needs another half cell diagonal of padding.
    Unknown geometry outside the scanner's range is never declared free. These
    remain ideal two-height range checks, not full 3-D collision certification.
    """
    sx, sy = start
    ex, ey = end
    length = math.dist(start, end)
    radius = max(TURN_RADIUS.values()) + quantization
    if length + radius > max_range - .10:
        return False
    a, b, c, d = bounds
    if any(not (a+radius <= x <= b-radius and c+radius <= y <= d-radius)
           for x, y in (start, end)):
        return False
    for item in returns:
        if len(item) < 3 or not all(math.isfinite(float(v)) for v in item[:3]):
            raise ValueError('Invalid travel-heading range return')
        x, y, layer = item[:3]
        if layer not in TURN_RADIUS:
            raise ValueError('Unknown travel-heading range layer')
        t = max(0., min(1., ((x-sx)*(ex-sx)+(y-sy)*(ey-sy)) / length**2)) if length else 0.
        if math.hypot(x-sx-t*(ex-sx), y-sy-t*(ey-sy)) <= TURN_RADIUS[layer]+quantization:
            return False
    return True


def select_cruise_endpoint(pose, target, fresh_returns, grid):
    """Choose a useful open prefix; keep close docking in the original frame."""
    start = (pose.x, pose.y)
    length = math.dist(start, target)
    if length < .70:
        return None
    heading = math.atan2(target[1]-pose.y, target[0]-pose.x)
    if abs(wrap_angle(heading-pose.yaw)) < .20:
        return None
    remembered = [(*grid.point((x, y)), layer) for x, y, layer in grid.observed]
    distance = min(length, 2.5)
    while distance >= .70:
        end = (pose.x+(target[0]-pose.x)*distance/length,
               pose.y+(target[1]-pose.y)*distance/length)
        if (swept_turn_clear(start, end, fresh_returns, bounds=grid.bounds)
                and swept_turn_clear(start, end, remembered, bounds=grid.bounds,
                                     quantization=grid.resolution/math.sqrt(2.))):
            return end
        distance -= .10
    return None


class ForwardCruise:
    """Align, move forward, restore docking yaw; fail closed on a blocked scan."""
    def __init__(self, pose, end, docking_yaw):
        self.end = tuple(end)
        self.heading = math.atan2(end[1]-pose.y, end[0]-pose.x)
        self.docking_yaw = docking_yaw
        self.phase = 'align'
        self.settled_steps = 0
        self.blocked = False
        self.event = dict(start=(pose.x, pose.y), end=self.end, heading_rad=self.heading,
                          docking_yaw_rad=docking_yaw, completed=False, blocked_scans=0,
                          phase_steps={}, max_heading_error_rad=0.)

    def check_scan(self, pose, returns, grid):
        start = (pose.x, pose.y)
        end = self.end if self.phase in ('align', 'forward') else start
        remembered = [(*grid.point((x, y)), layer) for x, y, layer in grid.observed]
        self.blocked = not (swept_turn_clear(start, end, returns, bounds=grid.bounds)
            and swept_turn_clear(start, end, remembered, bounds=grid.bounds,
                                 quantization=grid.resolution/math.sqrt(2.)))
        self.event['blocked_scans'] += int(self.blocked)

    def command(self, pose):
        steps = self.event['phase_steps']
        steps[self.phase] = steps.get(self.phase, 0) + 1
        if self.blocked:
            return (0., 0., 0.)
        distance = math.dist((pose.x, pose.y), self.end)
        if self.phase == 'forward' and distance <= .16:
            self.phase = 'restore'
        desired = (self.docking_yaw if self.phase in ('restore', 'settle') else
                   math.atan2(self.end[1]-pose.y, self.end[0]-pose.x))
        error = wrap_angle(desired-pose.yaw)
        self.event['max_heading_error_rad'] = max(self.event['max_heading_error_rad'], abs(error))
        yaw_rate = math.copysign(min(.60, max(.25, 1.2*abs(error))), error) if abs(error) > .06 else 0.
        if self.phase == 'align' and abs(error) <= .10:
            self.phase = 'forward'
        if self.phase == 'forward':
            if abs(error) > .30:
                self.phase = 'align'
            else:
                return (min(.45, max(.30, .8*distance)), 0., yaw_rate)
        if self.phase == 'restore' and abs(error) <= .08:
            self.phase = 'settle'
        if self.phase == 'settle':
            self.settled_steps = self.settled_steps+1 if abs(error) <= .10 else 0
            if self.settled_steps >= 10:
                self.event.update(completed=True, measured_end=(pose.x, pose.y), measured_yaw_rad=pose.yaw)
                self.phase = 'complete'
            return (0., 0., yaw_rate)
        return (0., 0., yaw_rate)
