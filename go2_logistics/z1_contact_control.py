"""Bounded fixed-jaw force admittance, acting only through arm motor targets.

The moving finger has its own force loop. This loop moves the wrist along
the fixed jaw's inward normal; it never reads or writes payload poses.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PairedGripReadiness:
    """Optional early close completion after half a second of measured squeeze.

    It only observes both filtered parcel contact pairs, never advances the
    nominal closing trajectory or substitutes for the independent force guard.
    One out-of-band sample restarts the continuous observation window.
    """
    target_n: float = 6.
    stable_s: float = 0.

    def update(self, normal_n, total_n, dt):
        if (len(normal_n)!=2 or len(total_n)!=2 or not 0 < dt <= .01
                or not math.isfinite(dt) or not 0 < self.target_n < 20
                or not all(math.isfinite(v) and v>=0 for v in [*normal_n,*total_n])):
            raise ValueError('Invalid measured paired-grip readiness sample')
        eligible=(all(abs(v-self.target_n)<=1.5 for v in normal_n)
                  and all(v<=10. for v in total_n))
        self.stable_s=self.stable_s+dt if eligible else 0.
        return self.stable_s>=.5-1e-9


def unload_moving_jaw_preload(reference, measured, observed_n, target_n, previous, dt,
                              open_rad=-1.45):
    """Remove stored closing error when measured force exceeds the target.

    Reducing an integral reference can still leave it ahead of the delayed
    physical finger. Track the measured joint instead in that case, while
    respecting the original motor position/rate bounds. This returns a motor
    request only; it neither changes joint state nor certifies a grasp.
    """
    if not (all(math.isfinite(v) for v in (reference,measured,observed_n,target_n,previous,dt,open_rad))
            and -math.pi/2 <= open_rad < 0 and 0 <= observed_n and 0 < target_n < 20
            and open_rad-1e-6 <= previous <= 1e-6 and 0 < dt <= .01):
        raise ValueError('Invalid moving-jaw tracking feedback')
    # Motor targets cross a float32 tensor boundary. Normalize only roundoff
    # at the configured endpoints; the returned request stays in range.
    previous=float(np.clip(previous,open_rad,0.))
    request=min(reference,measured) if observed_n>target_n else reference
    return float(np.clip(request,max(open_rad,previous-math.pi*dt),min(0.,previous+math.pi*dt)))


@dataclass
class GripperReleaseHandoff:
    """Open from the last force-loop target, inside the original joint range.

    The nominal trajectory assumes a fully closed finger at zero. A real
    force-limited grip stops earlier, so its nominal release would first
    squeeze the payload. Latch the actual command at this phase boundary.
    """
    open_rad: float
    duration_s: float = 2.
    start_rad: float | None = None
    elapsed_s: float = 0.

    def update(self, phase, previous_target, scheduled_target, dt):
        if not (math.isfinite(dt) and 0 < dt <= .01
                and math.isfinite(self.duration_s) and self.duration_s > 0
                and all(math.isfinite(value) and -math.pi/2 <= value <= 0
                        for value in (self.open_rad,previous_target,scheduled_target))):
            raise ValueError('Invalid original-range gripper release target or timing')
        if phase!='release':
            return scheduled_target
        if self.start_rad is None:
            if self.open_rad>previous_target:
                raise ValueError('Release target must open the original finger')
            self.start_rad=previous_target
        self.elapsed_s=min(self.duration_s,self.elapsed_s+dt)
        fraction=self.elapsed_s/self.duration_s
        smooth=fraction*fraction*(3.-2.*fraction)
        return self.start_rad+(self.open_rad-self.start_rad)*smooth


@dataclass
class JawNormalAdmittance:
    target_n: float
    gain_m_per_ns: float = .0005
    speed_limit_mps: float = .002
    offset_limit_m: float = .006
    filter_time_s: float = .03
    deadband_n: float = .3
    offset_m: float = 0.
    filtered_force_n: float | None = None
    latched: bool = False
    precontact_feedback: bool = False
    contact_threshold_n: float = .05
    precontact_active: bool = False
    precontact_limit_n: float | None = None

    def update(self, measured_normal_n, paired, phase, dt):
        preload_limit=self.target_n if self.precontact_limit_n is None else self.precontact_limit_n
        if not (math.isfinite(measured_normal_n) and measured_normal_n >= 0
                and math.isfinite(dt) and 0 < dt <= .01 and 0 < self.target_n < 20
                and self.contact_threshold_n < preload_limit <= self.target_n
                and 0 < self.contact_threshold_n < self.target_n):
            raise ValueError('Invalid normal-force measurement or control step')
        self.precontact_active=False
        if phase in ('close', 'lift', 'hold', 'carry', 'carry_settle', 'delivery_align', 'lower'):
            self.latched = self.latched or bool(paired)
            # A sensed single-jaw contact may regulate preload during closing.
            # It does not latch a grasp or replace the independent pair guard.
            # Losing that contact before pairing freezes this extra correction.
            self.precontact_active=bool(self.precontact_feedback and phase=='close'
                and not self.latched and measured_normal_n>self.contact_threshold_n)
            if self.latched or self.precontact_active:
                if self.filtered_force_n is None:
                    self.filtered_force_n = measured_normal_n
                alpha = dt / (self.filter_time_s + dt)
                self.filtered_force_n += alpha * (measured_normal_n-self.filtered_force_n)
                error = self.target_n-self.filtered_force_n
                if self.precontact_active:
                    # A single jaw cannot establish a controlled squeeze:
                    # chasing low force would push the parcel across its table.
                    # Relieve excess preload only, using current force as well
                    # as its filter so filtering cannot delay unloading a peak.
                    error=min(0.,preload_limit-max(measured_normal_n,self.filtered_force_n))
                error = math.copysign(max(0., abs(error)-self.deadband_n), error)
                velocity = np.clip(self.gain_m_per_ns*error, -self.speed_limit_mps, self.speed_limit_mps)
                self.offset_m = float(np.clip(self.offset_m+velocity*dt, -self.offset_limit_m, self.offset_limit_m))
        elif phase in ('retreat', 'settle'):
            # Keep the offset while releasing. Returning it abruptly while
            # the parcel is still between jaws would itself change contact.
            self.offset_m += float(np.clip(-self.offset_m, -self.speed_limit_mps*dt, self.speed_limit_mps*dt))
        return self.offset_m


def wrist_offset_target(kinematics, planned, previous, offset_m, dt):
    """Convert a <=6 mm jaw-normal offset to original-limit six-axis targets."""
    q = np.asarray(planned, dtype=float)
    prior = np.asarray(previous, dtype=float)
    if q.shape != (6,) or prior.shape != (6,) or not np.isfinite(np.r_[q, prior, offset_m, dt]).all():
        raise ValueError('Invalid six-axis wrist reference')
    if abs(offset_m) > .006000001 or not 0 < dt <= .01:
        raise ValueError('Wrist offset or time step exceeded diagnostic bounds')
    goal = kinematics.fk(q)
    normal = goal[:3, 2].copy()
    goal[:3, 3] += normal*offset_m
    corrected = q
    if abs(offset_m) > 1e-10:
        corrected, info = kinematics.inverse_kinematics(goal, prior, max_iters=60,
            position_tolerance=1e-5, orientation_tolerance=5e-4)
        if not info['success']:
            raise RuntimeError(f'Bounded wrist-normal IK failed: {info}')
    lower = np.maximum(kinematics.limits[:, 0], prior-math.pi*dt)
    upper = np.minimum(kinematics.limits[:, 1], prior+math.pi*dt)
    if np.any(lower > upper):
        raise RuntimeError('Inherited arm command is outside original position/velocity feasibility')
    command = np.clip(corrected, lower, upper)
    actual_goal = kinematics.fk(command)
    return command, dict(offset_m=offset_m, inward_normal_w=normal.tolist(),
        maximum_target_step_rad=float(abs(command-prior).max()),
        target_position_residual_m=float(np.linalg.norm(actual_goal[:3, 3]-goal[:3, 3])),
        velocity_or_position_limited=bool(np.max(abs(command-corrected)) > 1e-9))
