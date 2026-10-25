"""Shared command protocol and evidence helpers for mounted-Go2 adaptation.

Evaluation has no recovery controller and never clears a failure after a reset.
This is an isolated locomotion benchmark, not a delivery success criterion.
"""
from __future__ import annotations

import math
import json
from pathlib import Path


EVALUATION_SEGMENTS = (
    (0., 0., 0.), (.50, 0., 0.), (.50, .35, 0.), (0., .35, 0.),
    (-.50, 0., 0.), (-.50, -.35, 0.), (0., -.35, 0.), (0., 0., 0.),
)


def measured_z1_pose_bank(trace_path):
    """Read representative extended-arm targets from a measured contact run.

    These are static arm configurations for locomotion training, not expert
    leg actions, fabricated payloads or evidence of loaded-grasp success.
    """
    hold,moving=[],[]
    stopped=set()
    with Path(trace_path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if 'robots' not in row:
                if row.get('phase')=='hold':hold.append(row['target_rad'][:6])
                if row.get('phase')=='carry':moving.append(row['target_rad'][:6])
                continue
            for name,state in row['robots'].items():
                if name in stopped:continue
                height,gravity=state['height_m'],state['gravity_z']
                if not all(math.isfinite(v) for v in (height,gravity)) or height<.22 or gravity>-.85:
                    stopped.add(name)
                    continue
                z1=state.get('original_z1') or {}
                if not state.get('payload_held'):continue
                q=z1.get('arm_target_rad')
                if z1.get('state')=='lift':hold.append(q)
                elif z1.get('state')=='carried' and not state.get('recovering'):
                    moving.append(q)
    if not hold or not moving:
        raise ValueError('Measured Z1 pose source must contain hold and carry measurements')
    selected=[hold[-1]]+[moving[round(f*(len(moving)-1))] for f in (0.,.25,.5,.75,1.)]
    bank=[]
    for q in selected:
        if q is None or len(q)!=6 or not all(math.isfinite(value) for value in q):
            raise ValueError('Measured Z1 arm targets must contain six finite values')
        if not any(max(abs(a-b) for a,b in zip(q,existing))<.01 for existing in bank):
            bank.append(q)
    return bank


def training_layout(num_envs, pose_count, sequence_count):
    """Cross arm pose, replay/random commands and recording independently.

    Using parity for both pose and replay selection silently excludes carry
    poses from replay when there are two poses. One complete block here covers
    every combination. Pose indices retain their historical strided order.
    """
    if num_envs < 1 or pose_count < 1 or sequence_count < 0:
        raise ValueError('Invalid training group counts')
    return [dict(arm_pose_index=i % pose_count,
        replay_enabled=sequence_count > 0 and (i // pose_count) % 2 == 0,
        replay_sequence_index=(i // (2 * pose_count)) % sequence_count
            if sequence_count else None) for i in range(num_envs)]


def load_warehouse_commands(paths):
    """Read contiguous, finite commands before each dog's first unsafe frame.

    These are velocity requests, not expert leg actions or success labels.
    Keep 50 Hz timing, including braking and idle periods. A 200 Hz contact
    trace is reduced with four-tick block means; an unsafe/partial block is
    discarded rather than carrying post-failure requests into training.
    """
    sequences, sources = [], []
    for path in map(Path, paths):
        robots, stopped, buffers = {}, set(), {}
        previous_tick = 0
        input_period=None
        with path.open(encoding='utf-8') as stream:
          for line in stream:
            row = json.loads(line)
            if row['tick'] != previous_tick + 1:
                raise ValueError(f'Command replay must have contiguous control ticks: {path}')
            previous_tick = row['tick']
            if input_period is None:
                input_period=row['sim_time']
                if not any(abs(input_period-dt)<1e-8 for dt in (.005,.02)):
                    raise ValueError(f'Command replay requires a measured 50/200 Hz warehouse clock: {path}')
            if abs(row['sim_time'] - row['tick'] * input_period) > 1.e-6:
                raise ValueError(f'Command replay has an inconsistent warehouse clock: {path}')
            for name, state in row['robots'].items():
                if name in stopped:
                    continue
                command = state['commanded_velocity']
                finite = all(math.isfinite(x) for x in [*command, state['height_m'], state['gravity_z']])
                if (not finite or state['height_m'] < .22 or state['gravity_z'] > -.85):
                    stopped.add(name)
                    continue
                if len(command) != 3 or abs(command[0]) > .50001 or abs(command[1]) > .35001 or abs(command[2]) > 1.50001:
                    raise ValueError(f'Command replay exceeds warehouse limits: {path}, {name}')
                buffer=buffers.setdefault(name,[])
                buffer.append(command)
                if len(buffer)==round(.02/input_period):
                    robots.setdefault(name, []).append([sum(c[axis] for c in buffer)/len(buffer) for axis in range(3)])
                    buffer.clear()
        for name, commands in sorted(robots.items()):
            if len(commands) < 250:
                raise ValueError(f'Command replay needs at least five seconds: {path}, {name}')
            sequences.append(commands)
            sources.append(dict(path=str(path.resolve()), robot=name, frames=len(commands),
                                duration_s=len(commands)*.02, truncated_at_unsafe_frame=name in stopped,
                                source_control_period_s=input_period,training_control_period_s=.02,
                                resampling='four_tick_block_mean' if input_period==.005 else 'unchanged'))
    if not sequences:
        raise ValueError('Command replay has no usable recordings')
    return sequences, sources


def evaluation_command(elapsed_s, env_index, segment_s=5.):
    if not math.isfinite(elapsed_s) or elapsed_s < 0 or segment_s <= 0:
        raise ValueError('Invalid evaluation time')
    # Same sequence, offset across environments; avoids reporting only forward.
    index = (int(elapsed_s / segment_s) + env_index % len(EVALUATION_SEGMENTS)) % len(EVALUATION_SEGMENTS)
    return EVALUATION_SEGMENTS[index]


class SurvivalLedger:
    """Latch the first unsafe frame, including unexpected environment resets."""

    def __init__(self, count):
        self.first_failure_s = [None] * count
        self.reason = [None] * count

    def record(self, time_s, heights, gravities, dones):
        if not len(heights) == len(gravities) == len(dones) == len(self.reason):
            raise ValueError('Evaluation batch mismatch')
        for i, (height, gravity, done) in enumerate(zip(heights, gravities, dones)):
            if self.reason[i] is not None:
                continue
            reason = ('nonfinite_state' if not math.isfinite(height) or not math.isfinite(gravity)
                      else 'height_below_0.22m' if height < .22
                      else 'tilt_above_acos_0.85' if gravity > -.85
                      else 'unexpected_reset' if done else None)
            if reason:
                self.first_failure_s[i], self.reason[i] = float(time_s), reason

    def summary(self):
        return dict(total=len(self.reason), survived=sum(r is None for r in self.reason),
                    first_failure_s=self.first_failure_s, failure_reason=self.reason)
