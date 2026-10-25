"""Independently review completed dynamic-cart navigation, including failures."""
import argparse
import hashlib
import html
import json
import math
from itertools import zip_longest
from pathlib import Path


def audit(folder):
    from .analyze_fleet import analyze
    from .compose_fleet_video import compose
    from .review_z1_warehouse import verify_videos
    from .range_log_audit import audit_range_log
    folder=Path(folder)
    result=json.loads((folder/'fleet_result.json').read_text(encoding='utf-8'))
    names=('dog1','dog2');dt=result['control_trace_period_s']
    samples=0;contiguous=True;matched=True;guarded=True;finite=True
    peak_contact=0.;minimum_height={name:float('inf') for name in names}
    positions=[];observed_scans={name:set() for name in names}
    with (folder/'dynamic_obstacle_trace.jsonl').open(encoding='utf-8') as ds, (folder/'control_trace.jsonl').open(encoding='utf-8') as cs:
        for dynamic_line,control_line in zip_longest(ds,cs):
            if dynamic_line is None or control_line is None:
                contiguous=False
                continue
            d,c=json.loads(dynamic_line),json.loads(control_line);samples+=1
            contiguous &= d['tick']==c['tick']==samples and abs(d['sim_time_s']-samples*dt)<1.e-7
            peak_contact=max(peak_contact,d['fixture_contact_n'])
            finite &= math.isfinite(d['fixture_contact_n']) and all(math.isfinite(x) for x in d['fixture_position_w'])
            positions.append(d['fixture_position_w'])
            for name in names:
                state=c['robots'][name];observation=d['robots'][name]
                matched &= math.dist(state['root_position_w'],observation['root_position_w'])<1.e-6
                finite &= math.isfinite(state['height_m']) and math.isfinite(state['gravity_z'])
                guarded &= state['height_m']>=.22 and state['gravity_z']<=-.85
                minimum_height[name]=min(minimum_height[name],state['height_m'])
                if observation['nearest_fixture_ray_hits']>0:observed_scans[name].add(observation['scan'])
    raw_scans={name:set() for name in names}
    range_audits={}
    for name in names:
        profile=(result['navigation'][name].get('range_measurement') or {}).get('configuration')
        range_audits[name]=audit_range_log(folder/name/'range_measurements.jsonl',profile)
        raw_scans[name]=set(range_audits[name]['raw_fixture_scans'])
    source_manifest=json.loads((folder/'source_manifest.json').read_text(encoding='utf-8'))
    # Runtime uses a dedicated frozen source directory, never current code.
    candidates=[folder/'source',folder/'sources']
    source_dir=next((p for p in candidates if (p/'dynamic_obstacle.py').is_file()),None)
    sources_ok=source_dir is not None and all((source_dir/name).is_file() and
        hashlib.sha256((source_dir/name).read_bytes()).hexdigest()==digest for name,digest in source_manifest.items())
    generic=analyze(folder)
    compose(folder,accelerated=False)
    videos=verify_videos(folder,result)
    integrity=result.get('arm_runtime_integrity',{})
    mass_ok=result.get('arm_model')=='official_unitree_z1' and all(
        integrity.get(name,{}).get('passed') is True and
        abs(integrity[name]['arm_mass_kg']-5.22096983)<.001 and
        integrity[name]['fixed_base'] is False and integrity[name]['runtime_state_writes']==0
        for name in names)
    checks=dict(frozen_sources=sources_ok,all_physical_frames=contiguous and samples==result['control_ticks'] and samples>0,
        dynamic_and_control_roots_match=matched,finite=finite,original_body_guards=guarded,
        original_mass_free_roots=mass_ok,no_fixture_collision=peak_contact<=1.,
        fixture_moved=bool(positions) and max(math.dist(p,positions[0]) for p in positions)>.5,
        both_saw_real_nearest_fixture_rays=all(raw_scans[name] and observed_scans[name] and
            observed_scans[name]<=raw_scans[name] for name in names),
        both_received_usable_fixture_ranges=all(range_audits[name]['usable_fixture_scans'] for name in names),
        recorded_range_measurements_valid=all(range_audits[name]['passed'] for name in names),
        both_routes_and_home=result['state']=='complete' and all(
            result['completed_route_cycles'][name]>=1 and result['final_home_error_m'][name]<=.21 for name in names),
        generic_control_safety=generic['complete_control_safety_check'],
        no_world_reset=result['world_reset_between_orders'] is False,
        full_video_decode=all(v['complete'] for v in videos))
    physical_keys=('original_body_guards','original_mass_free_roots','no_fixture_collision','fixture_moved',
        'both_saw_real_nearest_fixture_rays','both_received_usable_fixture_ranges',
        'both_routes_and_home','generic_control_safety','no_world_reset')
    report=dict(scope='Original-mass Z1 two-dog unloaded dynamic-cart navigation; scripted kinematic obstacle, PhysX ranging, simulation localization. Not cargo delivery, SLAM or learned obstacle motion.',
        passed=all(checks.values()),physical_passed=all(checks[k] for k in physical_keys),
        evidence_complete=all(checks[k] for k in ('frozen_sources','all_physical_frames','dynamic_and_control_roots_match','finite','full_video_decode','recorded_range_measurements_valid')),
        checks=checks,control_frames=samples,minimum_height_m=minimum_height,
        maximum_fixture_contact_n=peak_contact,fixture_observed_scans={n:len(s) for n,s in observed_scans.items()},
        raw_fixture_observed_scans={n:len(s) for n,s in raw_scans.items()},
        range_measurement_audit=range_audits,
        final_home_error_m=result['final_home_error_m'],videos=videos,
        evidence_hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ('fleet_result.json','control_trace.jsonl','dynamic_obstacle_trace.jsonl','dynamic_obstacle.json','source_manifest.json')})
    audit_sources={}
    independent=folder/'independent_check'
    independent.mkdir(exist_ok=True)
    for source in (Path(__file__),Path(__file__).with_name('range_log_audit.py')):
        data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
        saved=independent/f'{source.stem}_{digest[:12]}.py'
        saved.write_bytes(data)
        audit_sources[source.name]=dict(file=str(saved.relative_to(folder)),sha256=digest)
    report['auditor_sources']=audit_sources
    (folder/'dynamic_navigation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    status='PASSED' if report['passed'] else 'NOT PASSED'
    table=''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k,v in checks.items())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双狗动态障碍：{status}</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto;padding:20px}}video,img{{width:100%}}td{{padding:6px 16px;border-bottom:1px solid #ccc}}</style>
<h1>双狗动态障碍：{status}</h1><p>{html.escape(report['scope'])}</p><video controls preload="metadata" src="fleet_demo.mp4"></video>
<p>原速录像，保留完整结果及失败。最大障碍接触力：{peak_contact:.6f}N；原臂质量各约5.221kg。</p>
<table>{table}</table><p><a href="dynamic_navigation_audit.json">逐项审计与哈希</a> · <a href="dynamic_obstacle_trace.jsonl">每物理步动态轨迹</a> · <a href="fleet.mp4">原始全景</a> · <a href="fleet_detail.mp4">原始跟随</a></p></html>'''
    (folder/'dynamic_navigation_review.html').write_text(page,encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    report=audit(parser.parse_args().folder)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
