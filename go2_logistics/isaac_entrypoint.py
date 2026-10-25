"""Integration seam for Isaac Lab / Isaac Sim.

Run the dependency-free demo first. In Isaac Sim, construct the three callable
adapters from your existing Go2 policy, camera detector, and cuRobo executor,
then pass them to ``build_mission`` from an Isaac extension or script.
"""

from __future__ import annotations

from .adapters import CallableArm, CallablePerception, CallableRobot
from .interfaces import Pose2D
from .mission import MissionController


def build_mission(
    *,
    pose_fn,
    navigate_fn,
    locate_fn,
    grasp_fn,
    release_fn,
    pickup: Pose2D = Pose2D(4.0, 2.0),
    dropoff: Pose2D = Pose2D(-3.0, 5.0),
    object_id: str = "box_red",
    home: Pose2D | None = None,
    state_fn=None,
) -> MissionController:
    """Create the same mission used by the local demo with real executors."""
    return MissionController(
        robot=CallableRobot(pose_fn, navigate_fn),
        perception=CallablePerception(locate_fn),
        arm=CallableArm(grasp_fn, release_fn),
        pickup=pickup,
        dropoff=dropoff,
        object_id=object_id,
        home=home,
        state_fn=state_fn,
    )
