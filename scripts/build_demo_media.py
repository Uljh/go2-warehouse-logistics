"""Create annotated portfolio videos from recorded frames, without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS={
    'dual_delivery':'z1_dual_handoff_supportqp_20261023_133835',
    'far_shelf':'z1_far_single_retained_support_20261024_0057',
    'dynamic_navigation':'z1_dynamic_cart_20261024_e_perturbed_range',
}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).is_file():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def writer(path):
    return imageio.get_writer(str(path),fps=5,codec='libx264',quality=None,
        macro_block_size=1,ffmpeg_log_level='error',
        ffmpeg_params=['-crf','22','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart'])


def frame(path,index):
    c=cv2.VideoCapture(str(path));c.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,value=c.read();c.release()
    if not ok:raise RuntimeError('Missing recorded frame '+str(path)+' '+str(index))
    return cv2.cvtColor(value,cv2.COLOR_BGR2RGB)


def banded(image,title,subtitle):
    h,w=image.shape[:2]
    canvas=Image.new('RGB',(w,h+76),'#111c26');canvas.paste(Image.fromarray(image),(0,76))
    d=ImageDraw.Draw(canvas);d.text((20,12),title,font=font(25),fill='#f3fafc')
    d.text((20,46),subtitle,font=font(16),fill='#9dd6d1')
    return np.asarray(canvas)


def transcode(source,target):
    c=cv2.VideoCapture(str(source));n=0
    if abs(c.get(cv2.CAP_PROP_FPS)-5)>1e-6:raise RuntimeError('Unexpected source sampling rate')
    with writer(target) as out:
        while True:
            ok,img=c.read()
            if not ok:break
            out.append_data(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));n+=1
    expected=int(c.get(cv2.CAP_PROP_FRAME_COUNT));c.release()
    if n!=expected:raise RuntimeError('Source decode incomplete')
    return n


def verify(path,expected):
    c=cv2.VideoCapture(str(path));n=0
    while c.read()[0]:n+=1
    fps=c.get(cv2.CAP_PROP_FPS);c.release()
    if n!=expected or abs(fps-5)>1e-6:raise RuntimeError('Generated media decode mismatch')
    return dict(file=path.name,frames=n,fps=fps,duration_s=n/fps,sha256=digest(path),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings-root',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    a=parser.parse_args();root=a.repository;media=root/'docs/media';records={}
    for key in ['dual_delivery','far_shelf']:
        run=a.recordings_root/RUNS[key];dst=media/key;dst.mkdir(parents=True,exist_ok=True)
        source=run/('pipeline_closeup_highlights_1x.mp4' if key=='dual_delivery' else 'far_shelf_release_excerpt_1x.mp4')
        audit=run/('pipeline_closeup_highlights_1x_audit.json' if key=='dual_delivery' else 'far_shelf_release_excerpt_audit.json')
        expected=json.loads(audit.read_text(encoding='utf-8'))['decoded_frames']
        result=transcode(source,dst/'demo_1x.mp4');assert result==expected
        records[key]=dict(source_sha256=digest(source),original_edit=json.loads(audit.read_text()),
            output=verify(dst/'demo_1x.mp4',expected),
            note='Recorded source frames; re-encoded only.1x inside each segment; gaps are explicit in original timeline.')
    key='dynamic_navigation';run=a.recordings_root/RUNS[key];dst=media/key;dst.mkdir(parents=True,exist_ok=True)
    overview=cv2.VideoCapture(str(run/'fleet.mp4'));detail=cv2.VideoCapture(str(run/'fleet_detail.mp4'));n=0
    with writer(dst/'demo_1x.mp4') as out:
        while True:
            ok,left=overview.read();ok2,right=detail.read()
            if ok!=ok2:raise RuntimeError('Native camera streams are not synchronized')
            if not ok:break
            both=np.concatenate([cv2.resize(left,(640,480)),cv2.resize(right,(640,480))],axis=1)
            rgb=cv2.cvtColor(both,cv2.COLOR_BGR2RGB)
            out.append_data(banded(rgb,'DYNAMIC OBSTACLE YIELD | two Go2 + original-mass Z1',
                f'1x simulation time | measured t={(n+1)/5:.1f}s | 0 N cart contact | synthetic range errors; truth localization'))
            n+=1
    overview.release();detail.release();assert n==185
    records[key]=dict(sources={x:digest(run/x) for x in ['fleet.mp4','fleet_detail.mp4']},
        output=verify(dst/'demo_1x.mp4',185),
        note='Both cameras are the same physical run/frame; no interpolation.37.170s physical endpoint, last recorded image37.0s.')
    titles={'dual_delivery':('Two dogs, two deliveries','364.070s | shelves1/23 | both robots return'),
            'far_shelf':('Contact delivery to shelf23','247.155s | final placement error5.42mm | return'),
            'dynamic_navigation':('Range-based dynamic obstacle yield','37.170s | measured cart contact0N | both return')}
    for key,times in [('dual_delivery',(31.,39.)),('far_shelf',(8.,16.)),('dynamic_navigation',(6.,14.))]:
        dst=media/key;source=dst/'demo_1x.mp4';values=[]
        for i in range(round(times[0]*5),round(times[1]*5)):
            rgb=frame(source,i);h,w=rgb.shape[:2]
            values.append(Image.fromarray(rgb).resize((640,round(h*640/w)),Image.Resampling.LANCZOS).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        values[0].save(dst/'preview.gif',save_all=True,append_images=values[1:],duration=200,loop=0,optimize=True,disposal=2)
        rgb=frame(source,round(sum(times)/2*5));poster=banded(rgb,*titles[key]);Image.fromarray(poster).save(dst/'poster.jpg',quality=91)
        records[key]['preview']=dict(source_demo_frames=[round(times[0]*5),round(times[1]*5)-1],
            duration_s=8.,speed='1x',sha256=digest(dst/'preview.gif'))
    # A compact edited tour. Native per-segment times remain printed in frames.
    selections=[('dual_delivery',0.,10.),('dual_delivery',31.,41.),('far_shelf',5.,17.),
                ('dynamic_navigation',6.,18.),('dual_delivery',93.,101.2)]
    overview_path=media/'showreel_1x.mp4';n=0;segments=[]
    with writer(overview_path) as out:
        for key,start,end in selections:
            first=round(start*5);last=round(end*5);source=media/key/'demo_1x.mp4'
            for i in range(first,last):
                image=frame(source,i)
                canvas=np.zeros((720,1280,3),dtype=np.uint8)
                h,w=image.shape[:2];scale=min(1280/w,640/h);resized=cv2.resize(image,(round(w*scale),round(h*scale)))
                rh,rw=resized.shape[:2];canvas[76:76+rh,(1280-rw)//2:(1280-rw)//2+rw]=resized
                p=Image.fromarray(canvas);d=ImageDraw.Draw(p)
                d.text((24,12),'GO2 WAREHOUSE LOGISTICS | '+titles[key][0],font=font(25),fill='#ecf6ff')
                d.text((24,48),'RECORDED SIMULATION | edited excerpts at1x | full runs and exact scope in README',font=font(17),fill='#91d8d0')
                out.append_data(np.asarray(p));n+=1
            segments.append(dict(demo=key,demo_start_s=start,demo_end_s=end,frames=last-first))
    records['showreel']=dict(output=verify(overview_path,n),segments=segments,
        note='Edited recorded excerpts,1x inside segments, no claim of one continuous task or simultaneous cargo transport.')
    (media/'media_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({k:v.get('output') for k,v in records.items()}))


if __name__=='__main__':main()
