"""Cache unchanged official Isaac6 assets, retaining upstream-relative paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('entry',choices=['ground','sky','warehouse','go2'])
    p.add_argument('--proxy')
    p.add_argument('--powershell',default=shutil.which('pwsh'))
    a=p.parse_args()
    from go2_logistics import prepare_warehouse_asset as cache
    entries=dict(ground='Isaac/Environments/Grid/default_environment.usd',
        sky='Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr',
        warehouse='Isaac/Environments/Simple_Warehouse/warehouse.usd',
        go2='Isaac/Robots/Unitree/Go2/go2.usd')
    entry=entries[a.entry];pending={cache.ASSET_ROOT+entry};seen=set();records=[];builtin=set()
    while pending:
        current=sorted(pending-seen)
        if not current:break
        seen.update(current);pending=set()
        for url in current:
            _,path=cache.fetch(url,proxy=a.proxy)
            data=path.read_bytes();records.append(dict(url=url,path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            for dep in cache.dependencies(path):
                if dep in ('OmniPBR.mdl','OmniGlass.mdl','UsdPreviewSurface'):builtin.add(dep);continue
                absolute=urljoin(url,dep)
                if not absolute.startswith(cache.ASSET_ROOT):raise RuntimeError('Inspect upstream external dependency: '+absolute)
                pending.add(absolute)
        print('Cached',len(records),'files; queued',len(pending-seen),flush=True)
    manifest=dict(source=cache.ASSET_ROOT+entry,entry=str(cache.local_path(cache.ASSET_ROOT+entry)),
        files=records,builtin_shaders=sorted(builtin),
        license_url='https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/license.html',
        license_note='Original vendor assets, downloaded separately under upstream terms; no mass/geometry overrides.')
    (cache.DESTINATION/(a.entry+'.source.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if a.entry=='go2':
        from run_demo import directory_link
        destination=ROOT/'artifacts/go2_asset'
        if not destination.exists():directory_link(Path(manifest['entry']).parent,destination,a.powershell)
        if not (destination/'go2.usd').is_file():raise RuntimeError('Go2 local entry not available')
    print(json.dumps(dict(entry=a.entry,files=len(records),source=manifest['source'])))


if __name__=='__main__':main()
