"""Bounded dominant-axis persistence for the existing docking velocity floors."""
import math


class DockingAxisHold:
    """Keep a chosen axis for 200 ms, then require 25% error dominance to switch.

    An axis already inside the original half-tolerance deadband releases
    immediately. This chooses between existing command floors; it never changes
    the goal, arrival tolerance, speed limits or physical safety guards.
    """

    def __init__(self):
        self.axis = None
        self.age_s = 0.
        self.revision = None
        self.active_frames = 0
        self.switches = 0
        self.telemetry = None

    def reset(self):
        self.axis = None
        self.age_s = 0.
        self.revision = None
        self.telemetry = None

    def update(self, ex, ey, tolerance, dt, *, requested, revision):
        if (not isinstance(requested, bool) or not all(math.isfinite(v) for v in (ex, ey, tolerance, dt))
                or not 0. < tolerance or requested and tolerance > .04 or not 0. < dt <= .02):
            raise ValueError('Docking axis selection requires finite errors, original tolerance and control period')
        legacy = abs(ex) >= abs(ey)
        if not requested:
            self.reset()
            return legacy
        if self.revision != revision:
            self.reset()
            self.revision = revision
        errors = [abs(ex), abs(ey)]
        switched = False
        if self.axis is None:
            self.axis = 0 if legacy else 1
        else:
            current, other = errors[self.axis], errors[1-self.axis]
            deadband = min(.04, tolerance*.5)
            if other > deadband and (current <= deadband
                    or self.age_s >= .20-1.e-9 and other > 1.25*current):
                self.axis = 1-self.axis
                self.age_s = 0.
                self.switches += 1
                switched = True
        self.age_s += dt
        self.active_frames += 1
        self.telemetry = dict(active=True, axis='forward' if self.axis == 0 else 'lateral',
            selected_age_s=self.age_s, switched=switched, body_error_m=[ex, ey],
            tolerance_m=tolerance, target_revision=revision)
        return self.axis == 0
