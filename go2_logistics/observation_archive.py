"""Immutable per-parcel RGB-D snapshots; no synthetic camera observations."""
import hashlib
import json
from pathlib import Path

from .sorting_layout import ParcelCode


FILES=('pickup_rgb.png','pickup_depth.npy','pickup_red_mask.png','observation.json','detection.json')


def archive_observation(folder, detection, index):
    folder=Path(folder)
    code=str(ParcelCode.parse(detection['parcel_code']))
    if detection.get('source')!='rendered_rgb_depth' or isinstance(index,bool) or not isinstance(index,int) or index<1:
        raise ValueError('Archive requires a rendered RGB-D detection and positive observation index')
    contents={name:(folder/name).read_bytes() for name in FILES}
    if json.loads(contents['detection.json'])!=detection:
        raise ValueError('Persisted detection does not match this camera observation')
    destination=folder/'observations'/code/f'{index:04d}'
    destination.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name,data in contents.items():
        (destination/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    manifest=dict(parcel_code=code,observation_index=index,files=hashes,
        scope='Rendered RGB-D and saved calibration. External camera, not onboard perception or SLAM.')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return dict(manifest=str((destination/'manifest.json').relative_to(folder)),
        sha256=hashlib.sha256((destination/'manifest.json').read_bytes()).hexdigest(),parcel_code=code)


def verify_observation(folder, reference):
    folder=Path(folder).resolve()
    manifest_path=(folder/reference['manifest']).resolve()
    if not manifest_path.is_relative_to(folder/'observations'):
        raise ValueError('Observation reference must stay inside the observation archive')
    try:
        manifest_bytes=manifest_path.read_bytes()
        manifest=json.loads(manifest_bytes)
        hashes=manifest['files']
        return (hashlib.sha256(manifest_bytes).hexdigest()==reference['sha256']
            and manifest['parcel_code']==reference['parcel_code']
            and set(hashes)==set(FILES)
            and all(hashlib.sha256((manifest_path.parent/name).read_bytes()).hexdigest()==hashes[name] for name in FILES)
            and json.loads((manifest_path.parent/'detection.json').read_bytes())['parcel_code']==reference['parcel_code'])
    except (OSError,ValueError,KeyError,TypeError):
        return False
