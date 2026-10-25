"""Smooth moving waypoint changes only; preserve startup and immediate stops."""
import math
from .fleet import slew_velocity


class WaypointCommandRamp:
    linear_acceleration = .8
    yaw_acceleration = 2.
    zero_gap_s = .06

    def __init__(self):
        self.revision = None
        self.active = False
        self.since_motion_s = math.inf
        self.telemetry = {}

    def update(self, requested, previous, revision, dt, clearance):
        requested, previous = tuple(requested), tuple(previous)
        changed = self.revision is not None and revision != self.revision
        self.revision = revision
        moving = math.hypot(*previous[:2]) > .05
        self.since_motion_s = 0. if moving else self.since_motion_s+dt
        stopped = all(abs(v) < 1e-8 for v in requested)
        # Each navigate() emits an immediate zero before its successor starts.
        # A single zero frame does not imply the moving body has stopped.
        recent_motion = self.since_motion_s <= self.zero_gap_s+1e-9
        triggered = changed and recent_motion and not stopped
        self.active = (self.active or triggered) and not stopped
        applied = self.active
        candidate = (slew_velocity(previous, requested, dt, self.linear_acceleration, self.yaw_acceleration)
                     if applied else requested)
        # Map check concerns a nominal planar stopping segment, not measured
        # braking acceleration or a full 3-D swept-body certificate.
        veto = applied and not clearance(candidate)
        sent = (0., 0., 0.) if veto else candidate
        self.active = applied and (veto or math.dist(candidate, requested) > 1e-8)
        self.telemetry = dict(revision=revision, revision_changed=changed, triggered=triggered,
            recent_motion=recent_motion,since_motion_s=self.since_motion_s if math.isfinite(self.since_motion_s) else None,
            applied=applied, active_after=self.active, requested=list(requested), previous=list(previous),
            candidate=list(candidate), output=list(sent), map_veto=veto)
        return sent


def nominal_stop_segment(pose, command, dt):
    speed = math.hypot(*command[:2])
    horizon = speed/(2*WaypointCommandRamp.linear_acceleration)+dt
    vx, vy = command[:2]
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    start = (pose.x, pose.y)
    end = (pose.x+(c*vx-s*vy)*horizon, pose.y+(s*vx+c*vy)*horizon)
    return start, end
