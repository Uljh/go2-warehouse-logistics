"""Deterministic local backend used for smoke tests and demo videos."""

from __future__ import annotations

import random

from .interfaces import Pose2D


class MockRobot:
    def __init__(self, rng: random.Random, failure_rate: float = 0.0):
        self.rng, self.failure_rate = rng, failure_rate
        self._pose = Pose2D(0.0, 0.0)

    def pose(self) -> Pose2D:
        return self._pose

    def navigate(self, goal: Pose2D) -> bool:
        if self.rng.random() < self.failure_rate:
            return False
        self._pose = goal
        return True


class MockPerception:
    def __init__(self, rng: random.Random, failure_rate: float = 0.0):
        self.rng, self.failure_rate = rng, failure_rate

    def locate(self, object_id: str) -> bool:
        return self.rng.random() >= self.failure_rate


class MockArm:
    def __init__(self, rng: random.Random, failure_rate: float = 0.0):
        self.rng, self.failure_rate, self.holding = rng, failure_rate, False

    def grasp(self, object_id: str) -> bool:
        self.holding = self.rng.random() >= self.failure_rate
        return self.holding

    def release(self) -> bool:
        released = self.holding
        self.holding = False
        return released
