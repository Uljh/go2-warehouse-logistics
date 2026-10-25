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
    rows=[json.loads(line) for line in Path(trace_path).read_text(encoding='utf-8').splitlines() if line]
    hold=[row for row in rows if row.get('phase')=='hold']
    moving=[row for row in rows if row.get('phase')=='carry']
    if not hold or not moving:
        raise ValueError('Measured Z1 pose source must contain hold and carry measurements')
    selected=[hold[-1]]+[moving[round(f*(len(moving)-1))] for f in (0.,.25,.5,.75,1.)]
    bank=[]
    for row in selected:
        q=row['target_rad'][:6]
        if len(q)!=6 or not all(math.isfinite(value) for value in q):
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
    Keep the original 50 Hz timing, including braking and idle periods.
    """
    sequences, sources = [], []
    for path in map(Path, paths):
        robots, stopped = {}, set()
        previous_tick = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['tick'] != previous_tick + 1:
                raise ValueError(f'Command replay must have contiguous control ticks: {path}')
            previous_tick = row['tick']
            if abs(row['sim_time'] - row['tick'] * .02) > 1.e-6:
                raise ValueError(f'Command replay requires the warehouse 50 Hz clock: {path}')
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
                robots.setdefault(name, []).append(command)
        for name, commands in sorted(robots.items()):
            if len(commands) < 250:
                raise ValueError(f'Command replay needs at least five seconds: {path}, {name}')
            sequences.append(commands)
            sources.append(dict(path=str(path.resolve()), robot=name, frames=len(commands),
                                duration_s=len(commands)*.02, truncated_at_unsafe_frame=name in stopped))
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
