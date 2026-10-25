"""Verify published sources, measurements and optional video/model payloads."""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',action='store_true')
    parser.add_argument('--media',action='store_true')
    parser.add_argument('--models',action='store_true')
    args=parser.parse_args();catalog=read(ROOT/'demos/catalog.json');reports={}
    for key,summary in catalog.items():
        manifest=read(ROOT/'repro'/key/'source_manifest.json')
        for name,digest in manifest.items():
            overlay=ROOT/'repro'/key/'overrides'/name
            source=overlay if overlay.exists() else ROOT/'go2_logistics'/name
            if sha(source)!=digest:raise RuntimeError('Recorded source mismatch '+key+'/'+name)
        acceptance=read(ROOT/'evidence'/key/'acceptance.json')['report']
        assert acceptance['physical_passed'] and acceptance['evidence_complete']
        projection=ROOT/'evidence'/key/'measurements_200hz.jsonl.gz'
        proof=read(ROOT/'evidence'/key/'projection_manifest.json')
        if sha(projection)!=proof['projected_trace_sha256']:raise RuntimeError('Published measurements changed')
        full=ROOT/'docs/media'/key/'full_run_1x.mp4'
        if sha(full)!=summary['original_video_sha256']:raise RuntimeError('Complete recording changed')
        reports[key]=dict(frozen_sources=len(manifest),physical_acceptance_recorded=True,
                          measurement_archive_hash_verified=True,original_full_video_hash_verified=True)
        if args.measurements:
            n=0
            with gzip.open(projection,'rt',encoding='utf-8') as f:
                for line in f:
                    frame=json.loads(line);n+=1
                    if frame['tick']!=n or abs(frame['sim_time_s']-n*.005)>1e-8:raise RuntimeError('Projected clock gap')
                    for name,s in frame['robots'].items():
                        if s['height_m']<.22 or s['gravity_z']>-.85:raise RuntimeError('Original body guard violated')
                        if max(s['arm']['jaw_normal_n']+s['arm']['jaw_total_n'])>20.:raise RuntimeError('Original force guard violated')
            if n!=proof['original_frames']:raise RuntimeError('Projection frame count differs')
            reports[key]['all200hz_frames_rechecked']=n
    syntax=0
    for folder in [ROOT/'go2_logistics',ROOT/'scripts',ROOT/'repro',ROOT/'training']:
        for p in folder.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'));syntax+=1
    for name in ['carry','empty_return']:
        checkpoint=ROOT/'weights'/name/'mounted_gait.pt';m=read(checkpoint.parent/'manifest.json')
        if sha(checkpoint)!=m['trained_checkpoint_sha256']:raise RuntimeError('Trained weight identity differs')
        captured=read(ROOT/'training'/name/'source_manifest.json')
        if any(sha(ROOT/'training'/name/n)!=h for n,h in captured.items()):raise RuntimeError('Captured training source differs')
    if args.media:
        import cv2
        videos={}
        for p in (ROOT/'docs/media').rglob('*.mp4'):
            cap=cv2.VideoCapture(str(p));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);n=0
            while cap.read()[0]:n+=1
            cap.release()
            if n!=declared or fps!=5.:raise RuntimeError('Incomplete video '+str(p))
            videos[str(p.relative_to(ROOT))]=n
        reports['full_video_decodes']=videos
    if args.models:
        import torch
        values={}
        for name in ['carry','empty_return']:
            checkpoint=torch.load(ROOT/'weights'/name/'mounted_gait.pt',map_location='cpu',weights_only=True)
            state=checkpoint['actor_state_dict']
            expected={'mlp.0.weight':(128,48),'mlp.2.weight':(128,128),
                      'mlp.4.weight':(128,128),'mlp.6.weight':(12,128),'distribution.std_param':(12,)}
            if any(tuple(state[k].shape)!=v for k,v in expected.items()):raise RuntimeError('Actor tensor ABI differs')
            if not all(bool(torch.isfinite(v).all()) for v in state.values()):raise RuntimeError('Nonfinite actor tensor')
            values[name]=dict(format='RSL-RL checkpoint/state_dict',input_dimension=48,output_dimension=12,
                hidden_dimensions=[128,128,128],all_actor_tensors_finite=True,
                scope='Tensor schema and identity only; no inference or physical rollout.')
        reports['model_tensor_schema_only']=values
    reports.update(python_sources_parsed=syntax,started_simulation=False,
        scope='Saved payload identity/coverage/guards and optional media/model interface only; not a new physical run or success-rate estimate.')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
