"""Recoverable task-level controller independent of the simulator backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .interfaces import ArmInterface, PerceptionInterface, Pose2D, RobotInterface


class PayloadLost(RuntimeError):
    """A held parcel was lost; unrelated controller faults must not be caught."""


class MissionState(str, Enum):
    NAVIGATE_PICKUP = "navigate_pickup"
    LOCATE_OBJECT = "locate_object"
    GRASP = "grasp"
    NAVIGATE_DROPOFF = "navigate_dropoff"
    RECOVER_DROPPED = "recover_dropped"
    RELEASE = "release"
    RETURN_HOME = "return_home"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class MissionMetrics:
    transitions: int = 0
    recoveries: int = 0
    navigation_failures: int = 0
    perception_failures: int = 0
    grasp_failures: int = 0
    release_failures: int = 0
    payload_losses: int = 0
    drop_recoveries: int = 0
    drop_recovery_failures: int = 0
    elapsed_steps: int = 0


@dataclass
class MissionController:
    robot: RobotInterface
    perception: PerceptionInterface
    arm: ArmInterface
    pickup: Pose2D
    dropoff: Pose2D
    object_id: str = "box_red"
    max_retries: int = 2
    state: MissionState = MissionState.NAVIGATE_PICKUP
    metrics: MissionMetrics = field(default_factory=MissionMetrics)
    _retries: int = 0
    home: Pose2D | None = None
    state_fn: Callable[[MissionState], None] | None = None
    failure_reason: str = ""
    recover_drop_fn: Callable[[str], bool] | None = None
    max_drop_recoveries: int = 2

    def step(self) -> MissionState:
        if self.state in (MissionState.COMPLETE, MissionState.FAILED):
            return self.state
        self.metrics.elapsed_steps += 1
        previous = self.state
        if self.state_fn is not None:
            self.state_fn(self.state)
        if self.state is MissionState.NAVIGATE_PICKUP:
            ok = self.robot.navigate(self.pickup)
            self.state = MissionState.LOCATE_OBJECT if ok else self._recover("navigation", MissionState.NAVIGATE_PICKUP)
            self._reset_retries(ok)
        elif self.state is MissionState.LOCATE_OBJECT:
            ok = self.perception.locate(self.object_id)
            self.state = MissionState.GRASP if ok else self._recover("perception", MissionState.LOCATE_OBJECT)
            self._reset_retries(ok)
        elif self.state is MissionState.GRASP:
            ok = self.arm.grasp(self.object_id)
            self.state = self._recover("grasp", MissionState.GRASP) if not ok else MissionState.NAVIGATE_DROPOFF
            self._reset_retries(ok)
        elif self.state is MissionState.NAVIGATE_DROPOFF:
            try:
                ok = self.robot.navigate(self.dropoff)
            except PayloadLost as exc:
                self.metrics.payload_losses += 1
                if self.recover_drop_fn is None or self.metrics.payload_losses > self.max_drop_recoveries:
                    self.failure_reason = str(exc)
                    self.state = MissionState.FAILED
                else:
                    self._retries = 0
                    self.state = MissionState.RECOVER_DROPPED
            else:
                self.state = MissionState.RELEASE if ok else self._recover("navigation", MissionState.NAVIGATE_DROPOFF)
                self._reset_retries(ok)
        elif self.state is MissionState.RECOVER_DROPPED:
            ok = self.recover_drop_fn(self.object_id)
            if ok:
                self.metrics.drop_recoveries += 1
                self.metrics.recoveries += 1
                self.state = MissionState.NAVIGATE_DROPOFF
            else:
                self.state = self._recover("drop_recovery", MissionState.RECOVER_DROPPED)
            self._reset_retries(ok)
        elif self.state is MissionState.RELEASE:
            ok = self.arm.release()
            self.state = (MissionState.RETURN_HOME if self.home is not None else MissionState.COMPLETE) if ok else self._recover("release", MissionState.RELEASE)
            self._reset_retries(ok)
        elif self.state is MissionState.RETURN_HOME:
            ok = self.robot.navigate(self.home)
            self.state = MissionState.COMPLETE if ok else self._recover("navigation", MissionState.RETURN_HOME)
            self._reset_retries(ok)
        if self.state is not previous:
            self.metrics.transitions += 1
        return self.state

    def run(self, max_steps: int = 30) -> MissionState:
        for _ in range(max_steps):
            if self.step() in (MissionState.COMPLETE, MissionState.FAILED):
                break
        return self.state

    def _reset_retries(self, success: bool) -> None:
        if success:
            self._retries = 0

    def _recover(self, kind: str, resume_state: MissionState) -> MissionState:
        setattr(self.metrics, f"{kind}_failures", getattr(self.metrics, f"{kind}_failures") + 1)
        self._retries += 1
        if self._retries > self.max_retries:
            return MissionState.FAILED
        self.metrics.recoveries += 1
        return resume_state
