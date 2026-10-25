"""Backend adapters used to connect the controller to Isaac Sim or ROS 2.

These classes intentionally accept callables. That keeps Isaac imports out of
the task logic and lets us connect either an Isaac Lab policy, Nav2 action
client, or a Unitree SDK bridge without changing the mission controller.
"""

from __future__ import annotations

from collections.abc import Callable

from .interfaces import Pose2D


class CallableRobot:
    def __init__(self, pose_fn: Callable[[], Pose2D], navigate_fn: Callable[[Pose2D], bool]):
        self._pose_fn = pose_fn
        self._navigate_fn = navigate_fn

    def pose(self) -> Pose2D:
        return self._pose_fn()

    def navigate(self, goal: Pose2D) -> bool:
        return self._navigate_fn(goal)


class CallablePerception:
    def __init__(self, locate_fn: Callable[[str], bool]):
        self._locate_fn = locate_fn

    def locate(self, object_id: str) -> bool:
        return self._locate_fn(object_id)


class CallableArm:
    def __init__(self, grasp_fn: Callable[[str], bool], release_fn: Callable[[], bool]):
        self._grasp_fn = grasp_fn
        self._release_fn = release_fn

    def grasp(self, object_id: str) -> bool:
        return self._grasp_fn(object_id)

    def release(self) -> bool:
        return self._release_fn()
