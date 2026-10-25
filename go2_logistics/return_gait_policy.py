"""Opt-in empty-return gait specialist with measured arm-state gating."""
import math


def specialist_idle_control_requested(*, enabled, completed_returns, phase, payload_held,
                                      arm_state, expert_eligible, expert_weight, command):
    """Let an already selected empty actor hold its own returned, parked dog.

    Keep the first approach and every manipulation unchanged. The existing
    balance controller must still fade out through its original exit slew.
    """
    return bool(enabled is True and completed_returns>0 and phase=='waiting_receiving'
        and payload_held is False and arm_state in ('waiting','stow_empty_arm')
        and expert_eligible is True and expert_weight==1. and len(command)==3
        and all(math.isfinite(v) and abs(v)<1e-9 for v in command))


def return_policy_eligible(*, original_z1, payload_held, phase, arm_state, measured_arm, home_arm,
                           empty_transit=False, resource_wait=False, repeat_pickup=False, completed_returns=0):
    if (not isinstance(empty_transit, bool) or not isinstance(resource_wait,bool)
            or not isinstance(repeat_pickup,bool)):
        raise ValueError('Empty transit specialist selection requires an explicit boolean')
    if not isinstance(completed_returns,int) or isinstance(completed_returns,bool) or completed_returns<0:
        raise ValueError('Completed home returns must be a nonnegative integer')
    allowed = phase=='returning' and arm_state=='stow_empty_arm'
    if empty_transit:
        allowed = (phase in ('returning','approaching_receiving','staging_receiving',
                            'withdrawing_receiving','waiting_receiving','parallel_navigation')
                   and arm_state in ('waiting','stow_empty_arm'))
    if resource_wait:
        allowed=allowed or (phase in ('waiting_approach_corridor','waiting_west_spine',
            'waiting_row_1','waiting_row_2','waiting_row_3','waiting_delivery_transit')
            and arm_state=='stow_empty_arm')
    if repeat_pickup and completed_returns>0:
        allowed=allowed or (phase in ('waiting_receiving','approaching_receiving','staging_receiving',
            'withdrawing_receiving') and arm_state in ('waiting','stow_empty_arm'))
    if not (original_z1 and not payload_held and allowed):
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
