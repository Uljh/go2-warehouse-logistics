"""Opt-in measured low-speed support entry for empty-arm departure braking.

Entry permits the existing motor controller to damp residual motion; this
never certifies that folding is safe. The original .06 m/s final stop,
three-second recovery budget and physical guards remain separate.
"""
import math


def loaded_brake_requested(recovering,phase,payload_held,arm_state,command):
    """Allow damped support only during zero-command, loaded transport recovery."""
    return bool(recovering and phase=='transporting' and payload_held and arm_state=='carried'
        and len(command)==3 and all(math.isfinite(v) and abs(v)<1e-9 for v in command))


class StowBrakeSupport:
    def __init__(self):
        self.active=False
        self.stable_time_s=0.
        self.telemetry={}

    def update(self,requested,linear,angular,gravity_z,normals,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Brake support requires a positive control period')
        valid=(len(linear)==len(angular)==3 and len(normals)==4
            and all(math.isfinite(v) for v in (*linear,*angular,gravity_z,*normals)))
        geometry=bool(requested and valid and gravity_z<-.99
            and sum(value>1. for value in normals)>=3)
        eligible=bool(geometry and math.hypot(*linear[:2])<.14 and abs(linear[2])<.20
            and math.sqrt(sum(v*v for v in angular))<.50)
        if not geometry:
            self.active=False
            self.stable_time_s=0.
        elif not self.active:
            self.stable_time_s=self.stable_time_s+dt if eligible else 0.
            self.active=self.stable_time_s>=.04-1e-9
        self.telemetry=dict(requested=bool(requested),entry_eligible=eligible,
            active=self.active,stable_time_s=self.stable_time_s)
        return self.active
