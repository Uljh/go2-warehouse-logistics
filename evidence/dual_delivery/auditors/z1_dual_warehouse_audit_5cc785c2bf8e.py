"""Independent two-Z1 warehouse check, scoped to one order per robot."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .analyze_fleet import analyze
from .analyze_segment_transit import analyze as analyze_segments
from .analyze_receiving_pipeline import analyze as analyze_pipeline
from .z1_fleet_diagnostics import summarize
from .z1_warehouse_audit import release_sequence, release_clearance_complete
from .return_gait_audit import audit_return_gait
from .stow_braking_audit import audit_stow_braking
from .loaded_braking_audit import audit_loaded_braking
from .placement_wrench_audit import audit_placement_wrench
from .close_completion_audit import audit_warehouse_close


def complete_release(states, counts, dt, config, jaws_clear):
    return (release_sequence(states) and jaws_clear
        and release_clearance_complete(states, counts, dt, config)
        and counts['retreat']*dt >= config['place_seconds']-dt
        and counts['settling']*dt >= config['settle_seconds']-dt)


def audit(folder):
    folder = Path(folder)
    result = json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    codes = result['configured_codes']
    if result.get('arm_model') != 'official_unitree_z1' or len(codes) != 2 or len(set(codes)) != 2:
        raise ValueError('This audit requires two distinct orders using original-mass Z1 arms')
    names = ('dog1', 'dog2')
    evidence, diagnostic = analyze(folder, generate_plot=False), summarize(folder)
    segments = analyze_segments(folder) if (result.get('segment_reservation') or {}).get('enabled') else None
    pipeline = analyze_pipeline(folder) if (result.get('receiving_staging') or {}).get('enabled') else None
    phases, counts = {n: [] for n in names}, {n: Counter() for n in names}
    clear = {n: False for n in names}
    for line in (folder/'control_trace.jsonl').open(encoding='utf-8'):
        for name, state in json.loads(line)['robots'].items():
            arm = state['original_z1']
            phase = arm['state']
            counts[name][phase] += 1
            if not phases[name] or phases[name][-1] != phase:
                phases[name].append(phase)
            if phase == 'released':
                clear[name] |= not state['payload_held'] and max(arm['jaw_normal_n']) < .05
    orders = result['dispatcher']['orders']
    ownership = {name: [code for code in codes if orders.get(code, {}).get('owner') == name] for name in names}
    one_each = all(len(ownership[name]) == 1 for name in names)
    robots = {}
    for name in names:
        config = result['manipulation_config'][name]
        measured = diagnostic['robots'][name]
        detection_file = folder/name/'detection.json'
        detection = json.loads(detection_file.read_text(encoding='utf-8')) if detection_file.is_file() else {}
        code = ownership[name][0] if len(ownership[name]) == 1 else None
        integrity = result['arm_runtime_integrity'][name]
        checks = dict(
            original_mass_free_root_no_bindings=integrity['passed']
                and abs(integrity['arm_mass_kg']-5.22096983) < 2e-5
                and integrity['fixed_base'] is False and integrity['payload_joints'] == []
                and integrity['payload_kinematic'] is False and integrity['runtime_state_writes'] == 0,
            unchanged_guards=config['grasp_mode'] == 'contact' and config['hard_contact_limit_n'] == 20.
                and 0 < config['placement_planar_tolerance'] <= .04
                and 0 < config['placement_height_tolerance'] <= .015
                and 0 < config['position_tolerance_m'] <= .012,
            image_identity_and_geometry=code is not None and detection.get('parcel_code') == code
                and detection.get('source') == 'rendered_rgb_depth'
                and detection.get('geometry', {}).get('source') in (
                    'rendered_depth_visible_rectangle_known_dimensions',
                    'rendered_depth_visible_rectangle_sloped_known_dimensions'),
            contact_lift_and_travel=measured['paired_before_lift_01s']
                and measured['measured_lift_m'] is not None and measured['measured_lift_m'] > .07
                and measured['minimum_carried_rise_m'] is not None and measured['minimum_carried_rise_m'] > .07
                and measured['maximum_loaded_root_displacement_m'] > .5,
            guarded_hold=measured['finite_forces'] and measured['maximum_measured_normal_or_total_force_n'] <= 20.
                and measured['guarded_hold_frames'] > 0 and measured['guarded_hold_paired_contact_fraction'] >= .90
                and measured['longest_guarded_paired_contact_loss_s'] < .25,
            complete_retreat_and_settling=complete_release(phases[name], counts[name],
                result['control_trace_period_s'], config, clear[name]),
        )
        robots[name] = dict(code=code, checks={k: bool(v) for k, v in checks.items()},
                            phases=phases[name], measurements=measured)
    source = (folder/'sources/z1_fleet_arm.py').read_text(encoding='utf-8')
    forbidden = [node.func.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and (node.func.attr.startswith('write_root_')
        or node.func.attr.startswith('write_joint_') or node.func.attr in
        ('set_world_pose', 'set_world_poses', 'set_local_pose'))]
    stow_braking = audit_stow_braking(folder, result)
    loaded_braking = audit_loaded_braking(folder,result)
    placement_wrench = audit_placement_wrench(folder,result)
    close_completion = audit_warehouse_close(folder,result)
    checks = dict(
        measured_close_completion=close_completion['passed'],
        placement_wrench_scope=placement_wrench['passed'],
        loaded_brake_support_scope=loaded_braking['passed'],
        empty_stow_measured_stop=stow_braking['passed'],
        generic_two_robot_delivery_and_return=evidence['full_two_robot_contact_delivery_and_return_check'],
        one_order_per_distinct_robot=one_each,
        complete_control_coverage=diagnostic['complete_control_coverage'] and evidence['complete_control_safety_check'],
        all_robot_checks=all(all(row['checks'].values()) for row in robots.values()),
        original_arm_source_has_no_state_writers=not forbidden,
        original_home_tolerance=set(result['final_home_error_m']) == set(names)
            and all(math.isfinite(v) and v <= .04 for v in result['final_home_error_m'].values()),
        segment_ownership=segments is None or segments['passed'] is True,
        receiving_handoff_ownership=pipeline is None or pipeline['passed'] is True,
    )
    video_file = folder/'video_audit.json'
    videos = json.loads(video_file.read_text(encoding='utf-8')) if video_file.is_file() else []
    videos = {v['file']: v for v in videos}
    expected = result['control_ticks']//result['video_timing']['sample_stride_ticks']
    return_gait = audit_return_gait(folder, result)
    records = dict(
        return_gait_identity_and_scope=return_gait['passed'],
        frozen_sources=bool(diagnostic['source_hashes']) and all(diagnostic['source_hashes'].values()),
        checkpoint=hashlib.sha256((folder/'policy_checkpoint.pt').read_bytes()).hexdigest() == result['gait_policy']['sha256'],
        full_videos=all(name in videos and videos[name]['complete']
            and videos[name]['decoded_frames'] == videos[name]['expected_frames'] == expected
            and abs(videos[name]['fps']-result['video_timing']['fps']) < 1e-6
            for name in ('fleet.mp4', 'fleet_detail.mp4', 'fleet_demo.mp4')),
    )
    report = dict(scope='Two original-mass Z1 dogs, one image-driven contact delivery each; no streaming or long-term throughput claim.',
        configured_codes=codes, fleet_result_sha256=hashlib.sha256((folder/'fleet_result.json').read_bytes()).hexdigest(),
        physical_passed=all(checks.values()), evidence_complete=all(records.values()),
        dual_order_passed=all(checks.values()) and all(records.values()),
        checks={k: bool(v) for k, v in checks.items()}, record_checks=records, robots=robots,
        measured_navigation_overlap_s=evidence['sampled_concurrent_measured_motion_time_s'],
        final_placement=evidence['independent_final_placement_checks'],
        final_home_error_m=result['final_home_error_m'], forbidden_calls=forbidden, return_gait_audit=return_gait,
        stow_braking_audit=stow_braking,loaded_braking_audit=loaded_braking,
        placement_wrench_audit=placement_wrench,close_completion_audit=close_completion,
        receiving_pipeline_audit=pipeline)
    frozen = folder/'independent_check'
    frozen.mkdir(exist_ok=True)
    report['auditor_sources'] = {}
    for name in ('z1_dual_warehouse_audit.py', 'z1_warehouse_audit.py', 'analyze_fleet.py',
                 'z1_fleet_diagnostics.py', 'analyze_segment_transit.py', 'analyze_receiving_pipeline.py', 'return_gait_audit.py', 'stow_braking_audit.py','loaded_braking_audit.py','placement_wrench_audit.py','close_completion_audit.py','z1_ground_pickup_audit.py'):
        data = Path(__file__).with_name(name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = frozen/f'{Path(name).stem}_{digest[:12]}.py'
        destination.write_bytes(data)
        report['auditor_sources'][name] = dict(file=str(destination.relative_to(folder)), sha256=digest)
    (folder/'dual_z1_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    result = audit(parser.parse_args().folder)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['dual_order_passed'] else 1)
