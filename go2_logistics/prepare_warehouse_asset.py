"""Mirror NVIDIA's warehouse USD dependency tree for repeatable local loading.

Run with the Isaac Python environment (pxr required; no simulator startup).
Original layers/textures are retained; license remains NVIDIA's upstream terms.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from functools import partial
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import hashlib
import json
import re
import urllib.request

ASSET_ROOT = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/'
ENTRY = 'Isaac/Environments/Simple_Warehouse/warehouse.usd'
DESTINATION = Path(__file__).resolve().parents[1] / 'artifacts/isaac_assets'


def local_path(url):
    if not url.startswith(ASSET_ROOT):
        raise ValueError(f'Asset outside approved upstream root: {url}')
    path = (DESTINATION / unquote(url[len(ASSET_ROOT):])).resolve()
    if not path.is_relative_to(DESTINATION.resolve()):
        raise ValueError(f'Asset path escapes cache: {url!r}, {str(path)!r}, root={str(DESTINATION.resolve())!r}')
    return path


def fetch(url, proxy=None):
    path = local_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + '.part')
        last_error = None
        for _ in range(3):
            try:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
                ) if proxy else urllib.request.build_opener()
                with opener.open(url, timeout=40) as response:
                    temporary.write_bytes(response.read())
                temporary.replace(path)
                break
            except Exception as error:
                last_error = error
        else:
            raise RuntimeError(f'Could not fetch {url}: {last_error}')
    return url, path


def dependencies(path):
    from pxr import Sdf
    if path.suffix.lower() in ('.usd', '.usda', '.usdc'):
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise RuntimeError(f'Invalid USD layer: {path}')
        # Include reference, payload, texture and shader assets; @@ is empty.
        return set(re.findall(r'@([^@\r\n]*)@', layer.ExportToString())) - {''}
    if path.suffix.lower() == '.mdl':
        source=path.read_text(encoding='utf-8')
        assets=set(re.findall(r'"([^"\r\n]+\.(?:png|jpg|jpeg|exr|dds|hdr))"', source))
        # MDL's relative module imports have no @asset@ or quoted filename.
        # Standard ::math / ::state modules are supplied by the renderer.
        for module in re.findall(r'(?:using|import)\s+\.::([\w:]+)',source):
            assets.add(module.rstrip(':').replace('::','/')+'.mdl')
        return assets
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', choices=('warehouse', 'ground', 'sky'), default='warehouse')
    parser.add_argument('--proxy', help='Explicit download proxy, e.g. http://127.0.0.1:7897')
    args = parser.parse_args()
    entry = dict(warehouse=ENTRY, ground='Isaac/Environments/Grid/default_environment.usd',
                 sky='Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr')[args.entry]
    pending = {ASSET_ROOT + entry}
    seen, records, builtin = set(), [], set()
    while pending:
        current = sorted(pending - seen)
        if not current:
            break
        seen.update(current)
        pending = set()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for url, path in pool.map(partial(fetch, proxy=args.proxy), current):
                data = path.read_bytes()
                records.append(dict(url=url, path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                for dep in dependencies(path):
                    if dep in ('OmniPBR.mdl', 'OmniGlass.mdl', 'UsdPreviewSurface'):
                        builtin.add(dep)
                        continue
                    absolute = urljoin(url, dep)
                    if absolute.startswith(ASSET_ROOT):
                        pending.add(absolute)
                    else:
                        raise RuntimeError(f'Unmirrored external dependency: {dep} in {url}')
        print(f'WAREHOUSE_CACHE files={len(records)} bytes={sum(r["bytes"] for r in records)} pending={len(pending-seen)}', flush=True)
    manifest = dict(source=ASSET_ROOT+entry, entry=str(local_path(ASSET_ROOT+entry)), files=records,
                    builtin_shaders=sorted(builtin), license_url='https://docs.isaacsim.omniverse.nvidia.com/latest/common/license.html',
                    license_note='NVIDIA Isaac Sim published assets; retain upstream license terms. Not a scanned real warehouse.')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION/f'{args.entry}.source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'WAREHOUSE_CACHE complete entry={manifest["entry"]}', flush=True)


if __name__ == '__main__':
    main()
