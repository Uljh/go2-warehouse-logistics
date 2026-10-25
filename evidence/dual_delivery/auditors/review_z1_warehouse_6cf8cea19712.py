"""Save a review of a completed Z1 experiment, including failed experiments."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from .analyze_fleet import analyze
from .analyze_receiving_pipeline import analyze as analyze_pipeline
from .compose_fleet_video import compose
from .z1_fleet_diagnostics import summarize
from .z1_warehouse_audit import audit as audit_single
from .z1_dual_warehouse_audit import audit as audit_dual
from .z1_batch_warehouse_audit import audit as audit_batch
from .z1_order_diagnostics import summarize_orders


def verify_videos(folder, result):
    """Decode every frame and compare with the simulation's recorded clock."""
    import cv2
    folder=Path(folder)
    timing=result['video_timing']
    expected=result['control_ticks']//timing['sample_stride_ticks']
    expected_fps=1./(result['control_trace_period_s']*timing['sample_stride_ticks'])
    reports=[]
    for name in ('fleet.mp4','fleet_detail.mp4','fleet_demo.mp4'):
        capture=cv2.VideoCapture(str(folder/name))
        declared=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=capture.get(cv2.CAP_PROP_FPS)
        frames=0
        try:
            while capture.read()[0]:
                frames+=1
        finally:
            capture.release()
        reports.append(dict(file=name,expected_frames=expected,declared_frames=declared,
            decoded_frames=frames,fps=fps,expected_fps=expected_fps,
            duration_s=frames/fps if fps else None,
            sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest() if (folder/name).is_file() else None,
            complete=frames==declared==expected and frames>0
                and math.isfinite(fps) and abs(fps-expected_fps)<1e-6
                and abs(fps-timing['fps'])<1e-6))
    (folder/'video_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return reports


def montage(folder):
    """Keep the original camera view and annotate the visible robot focus."""
    import cv2
    import numpy as np
    records=[json.loads(line) for line in (folder/'trace.jsonl').open(encoding='utf-8')]
    selected=[]
    for robot in ('dog1','dog2'):
        for phase in ('grasping','transporting','placing'):
            matches=[i for i,row in enumerate(records) if row['phases'][robot]==phase]
            if matches:
                selected.append((matches[len(matches)//2],robot,phase))
    selected.append((len(records)-1,'both','final recorded frame'))
    captures={name:cv2.VideoCapture(str(folder/name)) for name in ('fleet.mp4','fleet_detail.mp4')}
    thumbs=[]
    try:
        for index,robot,phase in selected:
            row=records[index]
            closeup=(row.get('detail_camera') or {}).get('focus')==robot and phase!='transporting'
            capture=captures['fleet_detail.mp4' if closeup else 'fleet.mp4']
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=capture.read()
            if not ok:
                raise RuntimeError(f'Review frame {index} could not be decoded')
            thumb=cv2.resize(frame,(480,300))
            cv2.rectangle(thumb,(0,0),(480,28),(15,15,15),-1)
            cv2.putText(thumb,f'{row["sim_time"]:.1f}s | {robot}: {phase}',(8,20),
                cv2.FONT_HERSHEY_SIMPLEX,.46,(245,245,245),1)
            thumbs.append(thumb)
        while len(thumbs)%3:
            thumbs.append(np.full_like(thumbs[0],240))
        image=np.concatenate([np.concatenate(thumbs[i:i+3],axis=1) for i in range(0,len(thumbs),3)])
        if not cv2.imwrite(str(folder/'stage_montage.jpg'),image):
            raise RuntimeError('Could not save stage montage')
    finally:
        for capture in captures.values():
            capture.release()


def finalize(folder):
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    order_count=len(result['configured_codes'])
    if result.get('arm_model')!='official_unitree_z1' or not 1<=order_count<=10:
        raise ValueError('Review supports one to ten distinct original-Z1 orders')
    evidence=analyze(folder)
    pipeline=analyze_pipeline(folder)
    diagnostic=summarize(folder) if order_count<=2 else summarize_orders(folder)
    (folder/'contact_diagnostics.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    compose(folder,accelerated=False)
    verify_videos(folder,result)
    audit=audit_single if order_count==1 else audit_dual if order_count==2 else audit_batch
    report=audit(folder)
    passed_key='single_order_passed' if order_count==1 else 'dual_order_passed' if order_count==2 else 'batch_passed'
    if report[passed_key]:
        # The final PASS label may only come from the full independent audit.
        compose(folder,accelerated=False)
        verify_videos(folder,result)
        report=audit(folder)
    montage(folder)
    audit_name='single_z1_audit.json' if order_count==1 else 'dual_z1_audit.json' if order_count==2 else 'batch_z1_audit.json'
    status='PASSED' if report[passed_key] else 'NOT PASSED'
    failed=[key for key,value in report['checks'].items() if not value]
    rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(value))}</td></tr>'
        for name,value in dict(runtime_state=result['state'],
            sim_seconds=result['control_ticks']*result['control_trace_period_s'],
            configured_codes=result['configured_codes'],failure=result['failure'],
            independent_physical_passed=report['physical_passed'],
            evidence_complete=report['evidence_complete'],failed_checks=failed,
            measured_navigation_overlap_s=evidence['sampled_concurrent_measured_motion_time_s'],
            pickup_handoffs=pipeline['pickup_handoffs'],
            grasp_and_peer_delivery_overlap_s=pipeline['grasp_and_peer_delivery_overlap_s'],
            both_payloads_held_s=pipeline['both_payloads_held_s']).items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Z1 warehouse: {status}</title>
<style>body{{font:16px/1.6 system-ui;max-width:1120px;margin:24px auto;padding:0 16px;color:#222}}
video,img{{display:block;max-width:100%}}table{{border-collapse:collapse;width:100%;margin:20px 0}}
td{{padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}}a{{color:#006b5d}}</style>
<h1>Original-mass Z1 warehouse: {status}</h1><p>{html.escape(folder.name)}</p>
<p>{html.escape(report['scope'])}</p><video controls preload="metadata" src="fleet_demo.mp4"></video>
<p>Full recording at 1x. <a href="fleet.mp4">Original overview</a> | <a href="fleet_detail.mp4">Original detail</a>
 | <a href="{audit_name}">Independent acceptance</a> | <a href="video_audit.json">Full video decode</a>
 | <a href="contact_diagnostics.json">Per-robot contact measurements</a>
 | <a href="receiving_pipeline_evidence.json">Pickup handoff and overlap measurements</a></p><table>{rows}</table>
<img src="stage_montage.jpg" alt="Recorded stages with robot identity and simulation time">
<p>Range scans and localization use simulator measurements. This recording does not establish onboard SLAM,
dynamic-obstacle robustness, continuous replenishment or long-term throughput.</p></html>'''
    (folder/'z1_warehouse_review.html').write_text(page,encoding='utf-8')
    data=Path(__file__).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    frozen=folder/'independent_check'/f'review_z1_warehouse_{digest[:12]}.py'
    frozen.write_bytes(data)
    summary=dict(passed=report[passed_key],physical_passed=report['physical_passed'],
        evidence_complete=report['evidence_complete'],failed_checks=failed,
        review='z1_warehouse_review.html',finalizer_sha256=digest,
        final_video_redecoded_and_reaudited=report[passed_key])
    (folder/'finalized_review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    report=finalize(parser.parse_args().folder)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
