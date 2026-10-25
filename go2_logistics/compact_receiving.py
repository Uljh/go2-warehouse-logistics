"""Coupled standby/exit layout; all waypoints still use sensed navigation."""
import math

from .interfaces import Pose2D


def _fixture(pickup):
    if not (math.isfinite(pickup.x) and math.isfinite(pickup.y)
            and abs(pickup.x-4.)<1e-8 and abs(pickup.y-1.5)<1e-8):
        raise ValueError('Compact receiving is scoped to the reviewed (4, 1.5) fixture')


def stage_route(pickup):
    """Stage on the incoming alignment, clear of the paired western exit."""
    _fixture(pickup)
    return (Pose2D(0.,-.15), Pose2D(3.1,.1))


def exit_route(pickup):
    """Release the bin after the first, precisely stopped western exit leg."""
    _fixture(pickup)
    return (Pose2D(1.5,pickup.y), Pose2D(1.5,-.3))


def release_clearance(pose,pickup):
    """Distance from the measured trunk to the whole incoming approach path."""
    _fixture(pickup)
    if not all(math.isfinite(v) for v in (pose.x,pose.y)):
        raise ValueError('Compact release requires finite measured trunk position')
    route=((3.1,.1),(3.1,1.5),(3.62,1.5))
    def segment(a,b):
        dx,dy=b[0]-a[0],b[1]-a[1]
        u=max(0.,min(1.,((pose.x-a[0])*dx+(pose.y-a[1])*dy)/(dx*dx+dy*dy)))
        return math.hypot(pose.x-a[0]-u*dx,pose.y-a[1]-u*dy)
    distance=min(segment(a,b) for a,b in zip(route,route[1:]))
    if (abs(pose.x-1.5)>.04+1e-8 or abs(pose.y-pickup.y)>.04+1e-8 or distance<1.45):
        raise RuntimeError('Compact receiving release lacks measured exit dock or 1.45 m incoming-path clearance')
    return distance
