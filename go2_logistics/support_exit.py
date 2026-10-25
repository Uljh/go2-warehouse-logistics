"""Bound policy command acceleration after stationary motor support releases."""
from .fleet import slew_velocity


class SupportExitRamp:
    duration_s = .8
    linear_acceleration = .8
    yaw_acceleration = 2.

    def __init__(self):
        self.remaining_s = 0.
        self.telemetry = {}

    def update(self, requested, previous_sent, support_active, dt):
        if support_active:
            self.remaining_s = self.duration_s
        active = self.remaining_s > 1e-9
        command = (slew_velocity(previous_sent, requested, dt,
                    self.linear_acceleration, self.yaw_acceleration)
                   if active else tuple(requested))
        self.telemetry = dict(active=active, remaining_s=self.remaining_s,
                              previous_support_active=bool(support_active),
                              requested=list(requested), ramped=list(command))
        if not support_active:
            self.remaining_s = max(0., self.remaining_s-dt)
        return command
