"""Small hardware boundary for Isaac Sim, ROS 2, and the local demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


class RobotInterface(Protocol):
    def pose(self) -> Pose2D: ...

    def navigate(self, goal: Pose2D) -> bool: ...


class PerceptionInterface(Protocol):
    def locate(self, object_id: str) -> bool: ...


class ArmInterface(Protocol):
    def grasp(self, object_id: str) -> bool: ...

    def release(self) -> bool: ...
