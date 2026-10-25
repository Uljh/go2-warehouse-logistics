"""Independently verify the optional return actor's identity and decision trace.

Eligibility is decided before the physics step. Compare its measured joints
with the previous post-step sample, and its phase/held flags with this sample.
This audits selection evidence, not whether the gait is physically successful.
"""
import hashlib
import json
import math
from pathlib import Path


HOME = (0., 1.2, -1.2, 0., 0., 0.)


def audit_return_gait(folder, result):
    folder = Path(folder)
    metadata = result.get('empty_return_gait_specialist') or {}
    enabled = metadata.get('enabled', False) is True
    names = tuple(result['arm_runtime_integrity'])
    stats = {name: dict(eligible_ticks=0, active_ticks=0, full_specialist_ticks=0) for name in names}
    checks = dict(checkpoint=True, training_manifest=True, configuration=True,
                  frame_coverage=True, eligibility=True, blend_ramp=True, counters=True)
    identity = metadata.get('checkpoint') or {}
    if enabled:
        path = folder/'return_policy_checkpoint.pt'
        checks['checkpoint'] = (identity.get('saved_checkpoint') == path.name and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == identity.get('sha256'))
        checks['configuration'] = (result.get('arm_model') == 'official_unitree_z1'
            and metadata.get('blend_seconds') == .5)
        if identity.get('training_manifest_verified'):
            path = folder/'return_gait_training_manifest.json'
            manifest = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
            checks['training_manifest'] = (manifest.get('actual_training_completed') is True
                and manifest.get('trained_checkpoint_sha256') == identity.get('sha256')
                and manifest.get('source_checkpoint_sha256') == identity.get('source_checkpoint_sha256'))
    else:
        checks['configuration'] = not identity
    dt = result['control_trace_period_s']
    if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        return dict(enabled=enabled, passed=False, checks=dict(invalid_control_period=False))
    previous_arm = {}
    weights = {name: 0. for name in names}
    frames = 0
    first_violations = []

    def violation(check, tick, name):
        checks[check] = False
        if len(first_violations) < 12:
            first_violations.append(dict(check=check, tick=tick, robot=name))

    with (folder/'control_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            frames += 1
            if (row['tick'] != frames or set(row['robots']) != set(names)
                    or abs(row['sim_time']-frames*dt) > 1e-8):
                violation('frame_coverage', row['tick'], None)
            for name in names:
                state = row['robots'].get(name)
                if state is None:
                    continue
                arm = state.get('original_z1') or {}
                telemetry = arm.get('return_gait_specialist')
                if enabled:
                    measured = previous_arm.get(name)
                    eligible = (state.get('payload_held') is False and state.get('phase') == 'returning'
                        and arm.get('state') == 'stow_empty_arm' and isinstance(measured, list)
                        and len(measured) == 6 and all(math.isfinite(value) for value in measured)
                        and max(abs(value-home) for value, home in zip(measured, HOME)) <= .06)
                    weight = (min(1., weights[name]+dt/.5) if eligible
                              else max(0., weights[name]-dt/.5))
                    if weight < 1e-9:
                        weight = 0.
                    if weight > 1.-1e-9:
                        weight = 1.
                    if not isinstance(telemetry, dict):
                        violation('frame_coverage', row['tick'], name)
                    else:
                        if telemetry.get('eligible') is not eligible:
                            violation('eligibility', row['tick'], name)
                        reported = telemetry.get('weight')
                        if (telemetry.get('blend_seconds') != .5 or not isinstance(reported, (int, float))
                                or not math.isfinite(reported) or abs(reported-weight) > 1e-8):
                            violation('blend_ramp', row['tick'], name)
                    weights[name] = weight
                    stats[name]['eligible_ticks'] += int(eligible)
                    stats[name]['active_ticks'] += int(weight > 0.)
                    stats[name]['full_specialist_ticks'] += int(weight == 1.)
                elif telemetry is not None:
                    violation('configuration', row['tick'], name)
                previous_arm[name] = state.get('arm_position_rad')
    checks['frame_coverage'] &= frames == result['control_ticks'] and frames > 0
    if enabled:
        checks['counters'] = metadata.get('stats') == stats
    return dict(enabled=enabled, passed=all(checks.values()), checks=checks,
        audited_frames=frames, independently_recomputed_stats=stats, first_violations=first_violations,
        measurement_timing='previous post-step arm positions; current pre-step phase and held flags',
        scope='Actor file identity and measured eligibility/ramp only; physical task acceptance is separate')
