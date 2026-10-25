"""Select an unchanged, fully cached official Grid plane before simulator launch."""
import hashlib
import json
from pathlib import Path

from .prepare_warehouse_asset import ASSET_ROOT, DESTINATION

GROUND_ENTRY = 'Isaac/Environments/Grid/default_environment.usd'
SKY_ENTRY = 'Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'


def create_cached_terrain(cfg):
    # InteractiveScene calls class_type directly. Defer simulator imports until
    # that call, after launch_simulation has initialized Isaac extensions.
    from .local_ground_importer import CachedGroundTerrainImporter
    return CachedGroundTerrainImporter(cfg)


def configure_local_ground(env_cfg, *, allow_remote=False, cache=DESTINATION):
    terrain = env_cfg.scene.terrain
    if terrain.terrain_type != 'plane':
        return False
    expected = verified_cached_asset(cache, 'ground', GROUND_ENTRY, allow_remote=allow_remote)
    if expected is None:
        return False
    terrain.usd_path = str(expected)
    terrain.class_type = create_cached_terrain
    print(f'LOCAL_GROUND entry={expected}', flush=True)
    return True


def configure_local_sky(env_cfg, *, allow_remote=False, cache=DESTINATION):
    sky = getattr(env_cfg.scene, 'sky_light', None)
    if sky is None or sky.spawn.texture_file != ASSET_ROOT + SKY_ENTRY:
        return False
    expected = verified_cached_asset(cache, 'sky', SKY_ENTRY, allow_remote=allow_remote)
    if expected is None:
        return False
    sky.spawn.texture_file = str(expected)
    print(f'LOCAL_SKY entry={expected}', flush=True)
    return True


def verified_cached_asset(cache, name, entry, *, allow_remote):
    cache = Path(cache).resolve()
    manifest_path = cache / f'{name}.source.json'
    if not manifest_path.is_file():
        if allow_remote:
            return None
        raise FileNotFoundError(
            f'Official {name} cache missing. Run: python -m '
            f'go2_logistics.prepare_warehouse_asset --entry {name} '
            '[--proxy http://127.0.0.1:7897]'
        )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = cache / entry
    if (manifest['source'] != ASSET_ROOT + entry
            or Path(manifest['entry']).resolve() != expected
            or not manifest['files']):
        raise ValueError(f'{name} manifest does not identify the official asset')
    validated = set()
    for record in manifest['files']:
        path = Path(record['path']).resolve()
        if not path.is_relative_to(cache):
            raise ValueError(f'{name} dependency outside cache: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'{name} dependency hash mismatch: {path}')
        validated.add(path)
    if expected not in validated:
        raise ValueError(f'{name} entry missing from verified dependencies')
    return expected
