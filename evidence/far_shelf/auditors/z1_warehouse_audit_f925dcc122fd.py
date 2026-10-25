"""Independent single-order original-mass Z1 warehouse acceptance.

This deliberately does not replace the two-robot throughput acceptance. One
dog must complete the physical order while both dogs remain in the shared world.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .analyze_fleet import analyze
from .z1_fleet_diagnostics import summarize
from .return_gait_audit import audit_return_gait
from .stow_braking_audit import audit_stow_braking
from .loaded_braking_audit import audit_loaded_braking
from .placement_wrench_audit import audit_placement_wrench
from .close_completion_audit import audit_warehouse_close


RELEASE_PHASES=('place_preapproach','lowering','opening_fingers','retreat','settling','released')


def release_sequence(states):
    """Retries may resume a phase, but cannot skip unfinished physical steps."""
    index=0
    for state in states:
        if index<len(RELEASE_PHASES) and state==RELEASE_PHASES[index]:
            index+=1
    return index==len(RELEASE_PHASES)


def release_clearance_complete(states, counts, dt, config):
    """An enabled departure motion needs its own ordered duration evidence."""
    if not (config.get('release_clearance_backoff_m', 0.) or config.get('release_clearance_rise_m', 0.)):
        return True
    required = ('retreat', 'retreat_clearance', 'settling', 'released')
    index = 0
    for state in states:
        if index < len(required) and state == required[index]:
            index += 1
    seconds = config.get('release_clearance_seconds')
    return (index == len(required) and isinstance(seconds, (int, float))
        and math.isfinite(seconds) and seconds > 0
        and counts['retreat_clearance']*dt >= seconds-dt)


def audit(folder):
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    codes=result['configured_codes']
    if result.get('arm_model')!='official_unitree_z1' or len(codes)!=1:
        raise ValueError('This audit is scoped to a single order using the original Z1')
    code=codes[0]
    owner=result['dispatcher']['orders'][code]['owner']
    evidence=analyze(folder,generate_plot=False)
    diagnostics=summarize(folder)
    dog=diagnostics['robots'][owner]
    phases=[]; counts=Counter(); jaws_clear_after_retreat=False
    for line in (folder/'control_trace.jsonl').open(encoding='utf-8'):
        state=json.loads(line)['robots'][owner]
        arm=state['original_z1']
        phase=arm['state']
        counts[phase]+=1
        if not phases or phases[-1]!=phase:
            phases.append(phase)
        if phase=='released':
            jaws_clear_after_retreat |= not state['payload_held'] and max(arm['jaw_normal_n'])<.05
    dt=result['control_trace_period_s']
    config=result['manipulation_config'][owner]
    detection=json.loads((folder/owner/'detection.json').read_text(encoding='utf-8'))
    integrity=result['arm_runtime_integrity']
    source=(folder/'sources/z1_fleet_arm.py').read_text(encoding='utf-8')
    forbidden=[node.func.attr for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and (node.func.attr.startswith('write_root_')
        or node.func.attr.startswith('write_joint_') or node.func.attr in
        ('set_world_pose','set_world_poses','set_local_pose'))]
    stow_braking=audit_stow_braking(folder,result)
    loaded_braking=audit_loaded_braking(folder,result)
    placement_wrench=audit_placement_wrench(folder,result)
    close_completion=audit_warehouse_close(folder,result)
    physical={
        'measured_close_completion':close_completion['passed'],
        'placement_wrench_scope':placement_wrench['passed'],
        'loaded_brake_support_scope':loaded_braking['passed'],
        'empty_stow_measured_stop':stow_braking['passed'],
        'reported_complete_without_errors':result['state']=='complete' and not result['failure']
            and not result.get('coroutine_errors'),
        'one_shared_world_without_order_resets':result.get('shared_world') is True
            and result.get('num_envs')==1 and result.get('robot_count')==2
            and result.get('warehouse_asset') is True and result.get('world_reset_between_orders') is False,
        'original_mass_free_bases_no_payload_bindings':set(integrity)=={'dog1','dog2'} and all(
            row['passed'] is True and abs(row['arm_mass_kg']-5.22096983)<2e-5
            and row['fixed_base'] is False and row['payload_joints']==[]
            and row['payload_kinematic'] is False and row['runtime_state_writes']==0
            for row in integrity.values()) and not forbidden,
        'unchanged_force_and_placement_guards':result['grasp_model']=='contact'
            and config['grasp_mode']=='contact' and config['hard_contact_limit_n']==20.
            and 0<config['placement_planar_tolerance']<=.04
            and 0<config['placement_height_tolerance']<=.015
            and 0<config['position_tolerance_m']<=.012,
        'camera_identity_and_depth_geometry':detection.get('parcel_code')==code
            and detection.get('source')=='rendered_rgb_depth'
            and detection.get('geometry',{}).get('source') in (
                'rendered_depth_visible_rectangle_known_dimensions',
                'rendered_depth_visible_rectangle_sloped_known_dimensions'),
        'complete_safe_control_trace':evidence['complete_control_safety_check']
            and diagnostics['complete_control_coverage'],
        'paired_before_lift_and_guarded_hold':dog['paired_before_lift_01s']
            and dog['guarded_hold_frames']>0 and dog['guarded_hold_paired_contact_fraction']>=.90
            and dog['longest_guarded_paired_contact_loss_s']<.25
            and dog['finite_forces'] and dog['maximum_measured_normal_or_total_force_n']<=20.,
        'physical_lift_and_measured_loaded_travel':dog['measured_lift_m'] is not None
            and dog['measured_lift_m']>.07 and dog['minimum_carried_rise_m'] is not None
            and dog['minimum_carried_rise_m']>.07 and dog['maximum_loaded_root_displacement_m']>.5,
        'full_release_and_retreat_sequence':release_sequence(phases)
            and release_clearance_complete(phases,counts,dt,config)
            and counts['retreat']*dt>=config['place_seconds']-dt
            and counts['settling']*dt>=config['settle_seconds']-dt and jaws_clear_after_retreat,
        'correct_shelf_after_departure_and_return':evidence['delivery_identity_check']
            and evidence['per_order_contact_check']
            and set(result['dispatcher']['delivered'])==set(codes)
            and evidence['independent_final_placement_checks'][code]['within_baseline_tolerance']
            and len(result['delivery_evidence'])==1
            and not result['delivery_evidence'][0]['constraint_created'],
        'both_dogs_back_at_home':set(result['returned'])=={'dog1','dog2'}
            and all(result['returned'].values()) and set(result['final_home_error_m'])=={'dog1','dog2'}
            and all(math.isfinite(error) and error<=.04 for error in result['final_home_error_m'].values()),
    }
    videos=json.loads((folder/'video_audit.json').read_text(encoding='utf-8'))
    expected=result['control_ticks']//result['video_timing']['sample_stride_ticks']
    full_videos={row['file']:row for row in videos if row['file'] in
                 ('fleet.mp4','fleet_detail.mp4','fleet_demo.mp4')}
    return_gait=audit_return_gait(folder,result)
    records={
        'return_gait_identity_and_scope':return_gait['passed'],
        'frozen_runtime_sources_match':bool(diagnostics['source_hashes']) and all(diagnostics['source_hashes'].values()),
        'frozen_checkpoint_matches':hashlib.sha256((folder/'policy_checkpoint.pt').read_bytes()).hexdigest()
            ==result['gait_policy']['sha256'],
        'all_three_full_videos_decoded':len(full_videos)==3 and all(row['complete']
            and row['decoded_frames']==row['expected_frames']==expected
            and abs(row['fps']-result['video_timing']['fps'])<1e-6 for row in full_videos.values()),
    }
    report=dict(scope='Single original-mass Z1 order: image pickup, contact carry, shelf release/retreat, stow and return. Second dog present but no two-dog throughput claim.',
        configured_codes=codes,
        fleet_result_sha256=hashlib.sha256((folder/'fleet_result.json').read_bytes()).hexdigest(),
        physical_passed=all(physical.values()),evidence_complete=all(records.values()),
        single_order_passed=all(physical.values()) and all(records.values()),
        checks={k:bool(v) for k,v in physical.items()},record_checks=records,
        arm_phases=phases,phase_durations_s={key:value*dt for key,value in counts.items()},
        contact_measurements=dog,final_placement=evidence['independent_final_placement_checks'][code],
        final_home_error_m=result['final_home_error_m'],forbidden_arm_calls=forbidden,return_gait_audit=return_gait,
        stow_braking_audit=stow_braking,loaded_braking_audit=loaded_braking,
        placement_wrench_audit=placement_wrench,close_completion_audit=close_completion)
    frozen=folder/'independent_check'
    frozen.mkdir(exist_ok=True)
    report['auditor_sources']={}
    for name in ('z1_warehouse_audit.py','analyze_fleet.py','z1_fleet_diagnostics.py','return_gait_audit.py','stow_braking_audit.py','loaded_braking_audit.py','placement_wrench_audit.py','close_completion_audit.py','z1_ground_pickup_audit.py'):
        source=Path(__file__).with_name(name).read_bytes()
        digest=hashlib.sha256(source).hexdigest()
        saved=frozen/(Path(name).stem+'_'+digest[:12]+'.py')
        saved.write_bytes(source)
        report['auditor_sources'][name]=dict(file=str(saved.relative_to(folder)),sha256=digest)
    (folder/'single_z1_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    report=audit(args.folder)
    print(json.dumps(report,indent=2))
    if not report['single_order_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
