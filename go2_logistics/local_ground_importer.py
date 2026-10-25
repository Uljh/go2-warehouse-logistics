"""Project-local plane importer retaining the official ground spawner semantics."""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


class CachedGroundTerrainImporter(TerrainImporter):
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        # Mirror TerrainImporter.import_ground_plane; only the USD source differs.
        prim_path = self.cfg.prim_path + f'/{name}'
        if prim_path in self.terrain_prim_paths:
            raise ValueError(f'Terrain already exists: {prim_path}')
        asset = Path(self.cfg.usd_path)
        if not asset.is_file():
            raise FileNotFoundError(asset)
        color = (0., 0., 0.)
        if self.cfg.visual_material is not None:
            color = self.cfg.visual_material.to_dict().get('diffuse_color', color)
        ground_cfg = sim_utils.GroundPlaneCfg(
            usd_path=str(asset), physics_material=self.cfg.physics_material,
            size=size, color=color,
        )
        ground_cfg.func(prim_path, ground_cfg)
        self.terrain_prim_paths.append(prim_path)
