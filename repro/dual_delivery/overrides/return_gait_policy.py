"""Opt-in empty-return gait specialist with measured arm-state gating."""
import math


def return_policy_eligible(*, original_z1, payload_held, phase, arm_state, measured_arm, home_arm):
    if not (original_z1 and not payload_held and phase=='returning' and arm_state=='stow_empty_arm'):
        return False
    if len(measured_arm)!=6 or len(home_arm)!=6:
        return False
    return (all(math.isfinite(v) for v in (*measured_arm,*home_arm))
        and max(abs(a-b) for a,b in zip(measured_arm,home_arm))<=.06)


class ReturnPolicyBlend:
    """Mix two leg actors over .5 seconds; never write measured state."""
    def __init__(self, blend_seconds=.5):
        if not math.isfinite(blend_seconds) or blend_seconds<=0:
            raise ValueError('Gait blending requires positive finite duration')
        self.seconds=blend_seconds
        self.weight=0.
        self.telemetry={}

    def update(self, eligible, dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Gait blending requires a positive control period')
        change=dt/self.seconds
        self.weight=min(1.,self.weight+change) if eligible else max(0.,self.weight-change)
        if self.weight<1e-9:self.weight=0.
        if self.weight>1.-1e-9:self.weight=1.
        self.telemetry=dict(eligible=bool(eligible),weight=self.weight,blend_seconds=self.seconds)
        return self.weight
