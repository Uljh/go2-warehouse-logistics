"""Recompute partial Z1 pickup/contact measurements, never full delivery success.

Measurements stay scoped to one parcel per robot. A two-robot experiment must
have distinct, explicit owners; multiple parcels for the same robot still
require per-order evidence and cannot share one initial height.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(folder):
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    codes=result['configured_codes']
    if result.get('arm_model')!='official_unitree_z1' or not codes or len(set(codes))!=len(codes):
        raise ValueError('Contact diagnostic requires distinct parcels and the original Z1 backend')
    owners={}
    if len(codes)>1:
        orders=result.get('dispatcher',{}).get('orders',{})
        for code in codes:
            owner=orders.get(code,{}).get('owner')
            if owner is None:
                continue
            if not owner or owner in owners:
                raise ValueError('Multi-parcel contact measurements require one explicit distinct owner per parcel')
            owners[owner]=code
    stats={}
    frames=0
    clock=True
    period=result['control_trace_period_s']
    with (folder/'control_trace.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            frames+=1
            clock &= row['tick']==frames and abs(row['sim_time']-frames*period)<1e-8
            for name,state in row['robots'].items():
                z=state['original_z1']
                if len(codes)>1 and name not in owners and (state['payload_held']
                        or z['state'] not in ('waiting','stow_empty_arm')):
                    raise ValueError(f'Multi-parcel contact measurements require a distinct owner for {name}')
                code=owners.get(name) if len(codes)>1 else codes[0]
                active_code=z.get('active_payload_code',code)
                idle_unassigned=(active_code is None and not state['payload_held']
                    and z['state'] in ('waiting','stow_empty_arm'))
                if active_code!=code and not idle_unassigned:
                    raise ValueError(f'Contact trace changed the active parcel for {name}')
                if active_code is not None and 'active_payload_path' in z:
                    index=codes.index(active_code)
                    expected_path='/World/envs/env_0/Parcel'+(str(index) if index else '')
                    if z['active_payload_path']!=expected_path:
                        raise ValueError(f'Contact trace payload path disagrees with parcel identity for {name}')
                sample=stats.setdefault(name,dict(initial=None,lift_end=None,carry_origin=None,
                    paired_before_lift=[],held_frames=0,held_paired_frames=0,longest_loss_frames=0,
                    loss_frames=0,maximum_force_n=0.,minimum_carried_rise_m=None,
                    guarded_frames=0,guarded_paired_frames=0,guarded_loss_frames=0,longest_guarded_loss_frames=0,
                    maximum_loaded_root_displacement_m=0.,finite_forces=True))
                forces=z['jaw_normal_n']+z['jaw_total_n']
                sample['finite_forces'] &= all(math.isfinite(v) and v>=0 for v in forces)
                sample['maximum_force_n']=max(sample['maximum_force_n'],*forces)
                paired=min(z['jaw_normal_n'])>.05
                if z['state']=='closing_fingers':
                    if sample['initial'] is None:sample['initial']=z['active_payload_position_w']
                    sample['paired_before_lift'].append(paired)
                    sample['paired_before_lift']=sample['paired_before_lift'][-math.ceil(.1/period):]
                if z['state']=='lift':sample['lift_end']=z['active_payload_position_w']
                if state['payload_held']:
                    sample['held_frames']+=1
                    sample['held_paired_frames']+=paired
                    sample['loss_frames']=0 if paired else sample['loss_frames']+1
                    sample['longest_loss_frames']=max(sample['longest_loss_frames'],sample['loss_frames'])
                # Opening the jaws on the target shelf deliberately removes
                # paired contact. Report that separately from guarded holding.
                guarded=state['payload_held'] and z['state'] not in (
                    'opening_fingers','retreat','retreat_clearance','settling','released')
                if guarded:
                    sample['guarded_frames']+=1
                    sample['guarded_paired_frames']+=paired
                    sample['guarded_loss_frames']=0 if paired else sample['guarded_loss_frames']+1
                    sample['longest_guarded_loss_frames']=max(sample['longest_guarded_loss_frames'],sample['guarded_loss_frames'])
                else:
                    sample['guarded_loss_frames']=0
                if z['state']=='carried' and sample['initial'] is not None:
                    if sample['carry_origin'] is None:sample['carry_origin']=state['root_position_w']
                    displacement=math.dist(state['root_position_w'][:2],sample['carry_origin'][:2])
                    sample['maximum_loaded_root_displacement_m']=max(sample['maximum_loaded_root_displacement_m'],displacement)
                    rise=z['active_payload_position_w'][2]-sample['initial'][2]
                    sample['minimum_carried_rise_m']=min(sample['minimum_carried_rise_m'],rise) if sample['minimum_carried_rise_m'] is not None else rise
    manifest=json.loads((folder/'source_manifest.json').read_text(encoding='utf-8'))
    hashes={name:hashlib.sha256((folder/'sources'/name).read_bytes()).hexdigest()==digest
            for name,digest in manifest.items()}
    output={}
    for name,sample in stats.items():
        measured=sample['initial'] is not None and sample['lift_end'] is not None
        output[name]=dict(parcel_code=owners.get(name) if len(codes)>1 else codes[0],
            measured_lift_m=sample['lift_end'][2]-sample['initial'][2] if measured else None,
            paired_before_lift_01s=len(sample['paired_before_lift'])==math.ceil(.1/period) and all(sample['paired_before_lift']),
            held_frames=sample['held_frames'],held_paired_contact_fraction=sample['held_paired_frames']/sample['held_frames'] if sample['held_frames'] else None,
            longest_paired_contact_loss_s=sample['longest_loss_frames']*period,
            held_metric_scope='Includes intentional opening while the runtime held flag is still true',
            guarded_hold_frames=sample['guarded_frames'],
            guarded_hold_paired_contact_fraction=(sample['guarded_paired_frames']/sample['guarded_frames']
                if sample['guarded_frames'] else None),
            longest_guarded_paired_contact_loss_s=sample['longest_guarded_loss_frames']*period,
            maximum_measured_normal_or_total_force_n=sample['maximum_force_n'],finite_forces=sample['finite_forces'],
            minimum_carried_rise_m=sample['minimum_carried_rise_m'],
            maximum_loaded_root_displacement_m=sample['maximum_loaded_root_displacement_m'])
    return dict(scope='Partial contact measurements for one parcel per robot. Does not accept warehouse delivery.',
        mission_state=result['state'],reported_failure=result['failure'],
        complete_control_coverage=clock and frames==result['control_ticks'],robots=output,source_hashes=hashes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    report=summarize(args.folder)
    source=Path(__file__).read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    audit_dir=args.folder/'independent_check'
    audit_dir.mkdir(exist_ok=True)
    filename='z1_fleet_diagnostics_'+digest[:12]+'.py'
    (audit_dir/filename).write_bytes(source)
    report['diagnostic_source']=dict(path='independent_check/'+filename,sha256=digest)
    (args.folder/'contact_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
