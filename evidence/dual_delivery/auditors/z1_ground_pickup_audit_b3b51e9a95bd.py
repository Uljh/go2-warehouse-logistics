"""Accept only local conveyor/bin contact pickup and hold, never warehouse delivery."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def audit_close_completion(rows,result):
    """Recompute optional early completion from recorded forces, not the gate."""
    cfg=result['config']
    if cfg.get('close_on_stable_contact',False) is not True:
        return dict(enabled=False,passed=True)
    close=[row for row in rows if row['robots']['dog1']['state']=='closing_fingers']
    continuous=0;first_ready=None;target=cfg['force_target_n']
    for index,row in enumerate(close,1):
        robot=row['robots']['dog1'];normal=robot['jaw_normal_n'];total=robot['jaw_total_n']
        valid=(len(normal)==len(total)==2 and all(math.isfinite(v) and v>=0 for v in normal+total)
               and all(abs(v-target)<=1.5 for v in normal) and all(v<=10. for v in total))
        continuous=continuous+1 if valid else 0
        if continuous>=100 and first_ready is None:first_ready=index
    completed=result.get('close_completion') or {};budget=cfg['close_seconds']
    clock=bool(close) and all(b['tick']==a['tick']+1 for a,b in zip(close,close[1:]))
    reason=('measured_paired_force_stable_0.5s' if first_ready is not None else 'trajectory_duration')
    required=first_ready if first_ready is not None else math.ceil(budget/.005)
    match=(clock and len(close)==required and completed.get('reason')==reason
        and completed.get('steps')==len(close) and completed.get('budget_s')==budget
        and completed.get('final_normal_n')==close[-1]['robots']['dog1']['jaw_normal_n']
        and completed.get('final_total_n')==close[-1]['robots']['dog1']['jaw_total_n'])
    for key,expected in (('elapsed_s',len(close)*.005),('stable_force_s',continuous*.005)):
        value=completed.get(key)
        match=match and isinstance(value,(int,float)) and math.isfinite(value) and abs(value-expected)<1e-8
    return dict(enabled=True,passed=bool(match),first_independently_ready_step=first_ready,
        measured_close_steps=len(close),measured_close_s=len(close)*.005,
        time_saved_against_budget_s=budget-len(close)*.005,
        scope='Local closing timing only; lift, hold and warehouse success remain separate')


def audit(folder):
    folder=Path(folder)
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if result.get('scenario')!='ground_pickup_only':
        raise ValueError('Ground pickup audit requires its explicit local scenario')
    rows=[json.loads(line) for line in (folder/'trace.jsonl').open(encoding='utf-8')]
    if not rows:raise ValueError('No measured local pickup frames')
    cfg=result['config'];integrity=result['runtime_integrity']
    detection=json.loads((folder/'dog1/detection.json').read_text(encoding='utf-8'))
    close=[row for row in rows if row['robots']['dog1']['state']=='closing_fingers']
    held=[row for row in rows if row['robots']['dog1']['held']]
    # The loss guard clears held in the exact failure frame. Include that
    # recorded sample when reporting the terminal contact-loss interval.
    monitored=[row for row in rows if row['robots']['dog1']['held']
        or row.get('guard_failure','').startswith('opposing jaw contact absent')]
    carried=[row for row in held if row['robots']['dog1']['state']=='carried']
    initial=(close[0]['parcel_position_w'][2] if close else None)
    forces=[value for row in rows for name in ('dog1','dog2')
        for key in ('jaw_normal_n','jaw_total_n') for value in row['robots'][name][key]]
    loss=longest=0
    for row in monitored:
        loss=0 if min(row['robots']['dog1']['jaw_normal_n'])>.05 else loss+1
        longest=max(longest,loss)
    paired=sum(min(row['robots']['dog1']['jaw_normal_n'])>.05 for row in held)/len(held) if held else 0.
    minimum_rise=min((row['parcel_position_w'][2]-initial for row in carried),default=0.) if initial is not None else 0.
    code=detection.get('parcel_code');orders=result['dispatcher']['orders']
    phases=[row['robots']['dog1']['state'] for row in rows]
    sleep=[(row['tick'],row['robots']['dog1'].get('payload_sleep')) for row in rows
           if row['robots']['dog1'].get('payload_sleep') is not None]
    checks=dict(local_runtime_completed=result.get('local_pickup_completed') is True and not result['failure'],
        no_warehouse_success_claim=result.get('passed') is False and not result['dispatcher'].get('delivered'),
        clock_complete=len(rows)==result['ticks'] and all(row['tick']==i+1 and row['physics_steps']==i+1
            and math.isfinite(row['time_s']) and abs(row['time_s']-(i+1)*.005)<1e-8 for i,row in enumerate(rows)),
        original_mass_free_bases=set(integrity['real_arm_masses_kg'])=={'dog1','dog2'}
            and all(abs(v-5.22096983)<1e-4 for v in integrity['real_arm_masses_kg'].values())
            and set(integrity['fixed_bases'])=={'dog1','dog2'} and not any(integrity['fixed_bases'].values()),
        no_runtime_resets_or_state_writes=all(integrity.get(k)==0 for k in
            ('runtime_resets','runtime_robot_state_writes','runtime_payload_state_writes')),
        camera_code_and_depth_geometry=detection.get('source')=='rendered_rgb_depth'
            and (detection.get('geometry') or {}).get('source') in (
                'rendered_depth_visible_rectangle_known_dimensions','rendered_depth_visible_rectangle_sloped_known_dimensions')
            and set(orders)=={code} and orders[code]['owner']=='dog1',
        dispatcher_carrying_at_end=set(orders)=={code} and orders[code]['state']=='carrying',
        original_force_guard=cfg['hard_contact_limit_n']==20. and cfg['grasp_mode']=='contact'
            and bool(forces) and np.isfinite(forces).all() and min(forces)>=0. and max(forces)<=20.,
        paired_before_lift=len(close)>=20 and all(min(row['robots']['dog1']['jaw_normal_n'])>.05 for row in close[-20:]),
        guarded_contact_hold=paired>=.90 and longest*.005<.25,
        physical_lift_and_three_second_hold=len(carried)>=600 and minimum_rise>.07,
        both_dogs_upright=all(row['robots'][name]['height_m']>=.28 and row['robots'][name]['gravity_z']<=-.94
            for row in rows for name in ('dog1','dog2')),
        pickup_only_phase_boundary=phases[-1]=='carried' and not set(phases)&{
            'place_preapproach','lowering','opening_fingers','retreat','released'},
        held_at_end=rows[-1]['robots']['dog1']['held'] is True)
    manifest=json.loads((folder/'source_manifest.json').read_text(encoding='utf-8'))
    hashes={name:hashlib.sha256((folder/'sources'/name).read_bytes()).hexdigest()==value for name,value in manifest.items()}
    source=(folder/'sources/z1_fleet_arm.py').read_text(encoding='utf-8')
    forbidden=[node.func.attr for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and (node.func.attr.startswith(('write_root_','write_joint_'))
            or node.func.attr in ('set_world_pose','set_world_poses','set_local_pose'))]
    checks['no_arm_body_state_writers']=not forbidden
    initial_pose=result.get('initial_parcel_pose')
    if initial_pose:
        q=rows[0]['parcel_quaternion_xyzw'];authored=initial_pose.get('quaternion_xyzw') or []
        expected_yaw=initial_pose.get('yaw_deg')
        initial_ok=(len(q)==len(authored)==4 and isinstance(expected_yaw,(int,float))
            and math.isfinite(expected_yaw) and -10.<=expected_yaw<=10.
            and initial_pose.get('authored_before_initial_reset') is True
            and all(math.isfinite(v) for v in q+authored))
        if initial_ok:
            expected=[0.,0.,math.sin(math.radians(expected_yaw)/2),math.cos(math.radians(expected_yaw)/2)]
            initial_ok=(np.allclose(authored,expected,atol=1.e-9)
                and abs(np.dot(q,authored))>math.cos(math.radians(.1)/2))
        checks['initial_parcel_yaw_verified']=bool(initial_ok)
    close_report=audit_close_completion(rows,result)
    checks['measured_close_completion']=close_report['passed']
    import cv2
    cap=cv2.VideoCapture(str(folder/'integration.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    video=dict(declared=declared,decoded=decoded,expected=len(rows)//20,fps=fps)
    complete=checks['clock_complete'] and bool(hashes) and all(hashes.values()) and decoded==declared==len(rows)//20 and decoded>0 and abs(fps-10.)<1e-6
    report=dict(local_pickup_physical_passed=all(bool(v) for v in checks.values()),evidence_complete=complete,
        checks={k:bool(v) for k,v in checks.items()},metrics=dict(sim_seconds=rows[-1]['time_s'],
            minimum_carried_rise_m=minimum_rise,paired_contact_fraction=paired,longest_pair_loss_s=longest*.005,
            carried_hold_s=len(carried)*.005,maximum_force_n=max(forces)),
        sleep_diagnostic=dict(observed_frames=len(sleep),sleeping_frames=sum(s['sleeping'] for _,s in sleep),
            first_sleeping_tick=next((tick for tick,s in sleep if s['sleeping']),None)),
        video=video,source_hashes=hashes,scope='Local conveyor/bin camera pickup and three-second hold only; no walking, release or warehouse completion')
    report['measured_close_completion']=close_report
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    args=parser.parse_args();report=audit(args.folder)
    data=Path(__file__).read_bytes();digest=hashlib.sha256(data).hexdigest()
    check=args.folder/'independent_check';check.mkdir(exist_ok=True)
    frozen=check/f'z1_ground_pickup_audit_{digest[:12]}.py';frozen.write_bytes(data)
    report['auditor_source']=dict(file=str(frozen.relative_to(args.folder)),sha256=digest)
    (args.folder/'ground_pickup_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['local_pickup_physical_passed'] and report['evidence_complete'] else 1)
