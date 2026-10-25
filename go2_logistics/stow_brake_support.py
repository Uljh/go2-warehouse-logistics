"""Opt-in measured low-speed support entry for empty-arm departure braking.

Entry permits the existing motor controller to damp residual motion; this
never certifies that folding is safe. The original .06 m/s final stop,
three-second recovery budget and physical guards remain separate.
"""
import math


def loaded_brake_requested(recovering,phase,payload_held,arm_state,command, *, receiving_brake=False):
    """Allow measured support during loaded transport or opt-in receiving exit braking."""
    allowed_phase=phase=='transporting' or receiving_brake and phase=='leaving_receiving'
    return bool(recovering and allowed_phase and payload_held and arm_state=='carried'
        and len(command)==3 and all(math.isfinite(v) and abs(v)<1e-9 for v in command))


class StowBrakeSupport:
    def __init__(self, *, attitude_hysteresis=False):
        if not isinstance(attitude_hysteresis,bool):
            raise ValueError('Support attitude hysteresis requires an explicit boolean')
        self.attitude_hysteresis=attitude_hysteresis
        self.active=False
        self.stable_time_s=0.
        self.telemetry={}

    def update(self,requested,linear,angular,gravity_z,normals,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Brake support requires a positive control period')
        valid=(len(linear)==len(angular)==3 and len(normals)==4
            and all(math.isfinite(v) for v in (*linear,*angular,gravity_z,*normals)))
        was_active=self.active
        retention=-.97 if self.attitude_hysteresis and was_active else -.99
        geometry=bool(requested and valid and gravity_z<retention
            and sum(value>1. for value in normals)>=3)
        eligible=bool(geometry and gravity_z<-.99 and math.hypot(*linear[:2])<.14 and abs(linear[2])<.20
            and math.sqrt(sum(v*v for v in angular))<.50)
        if not geometry:
            self.active=False
            self.stable_time_s=0.
        elif not self.active:
            self.stable_time_s=self.stable_time_s+dt if eligible else 0.
            self.active=self.stable_time_s>=.04-1e-9
        self.telemetry=dict(requested=bool(requested),entry_eligible=eligible,
            active=self.active,stable_time_s=self.stable_time_s)
        if self.attitude_hysteresis:
            self.telemetry['retained_after_entry']=bool(was_active and self.active and gravity_z>=-.99)
        return self.active
