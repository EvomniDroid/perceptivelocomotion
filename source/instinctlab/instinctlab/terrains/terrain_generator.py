from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):
        print(f"[InstinctLab FiledTerrainGenerator] __init__ called! cfg type: {type(cfg).__name__} id={id(self)}")
        print(f"[InstinctLab FiledTerrainGenerator] __init__ cfg.id: {id(cfg)}")
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        self._orig_sub_terrains = {}
        if hasattr(cfg, 'sub_terrains') and isinstance(cfg.sub_terrains, dict):
            for patch_key, patch_cfg in cfg.sub_terrains.items():
                try:
                    patch_cfg.name = patch_key
                except Exception:
                    object.__setattr__(patch_cfg, "name", patch_key)
                self._orig_sub_terrains[patch_key] = patch_cfg
        self._terrain_layout_names = list(cfg.terrain_layout) if cfg.terrain_layout else []
        super().__init__(cfg, device)
        print(f"[InstinctLab FiledTerrainGenerator] after super().__init__, subterrain_specific_cfgs len={len(self._subterrain_specific_cfgs)}")
        print(f"[FiledTerrainGenerator] terrain_layout_names = {self._terrain_layout_names}")
        print(f"[FiledTerrainGenerator] num_rows={cfg.num_rows}, num_cols={cfg.num_cols}, total={cfg.num_rows * cfg.num_cols}")

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        return mesh, origin

    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, sub_terrain_cfg: SubTerrainBaseCfg
    ):
        cfg_params = (
            getattr(sub_terrain_cfg, 'proportion', None),
            getattr(sub_terrain_cfg, 'platform_width', None),
            getattr(sub_terrain_cfg, 'border_width', None),
            type(sub_terrain_cfg).__name__,
        )
        original_name = None
        for patch_key, patch_cfg in self._orig_sub_terrains.items():
            patch_params = (
                getattr(patch_cfg, 'proportion', None),
                getattr(patch_cfg, 'platform_width', None),
                getattr(patch_cfg, 'border_width', None),
                type(patch_cfg).__name__,
            )
            if cfg_params == patch_params:
                original_name = patch_key
                break
        if original_name is not None:
            try:
                sub_terrain_cfg.name = original_name
            except Exception:
                object.__setattr__(sub_terrain_cfg, "name", original_name)
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)

    def _generate_subterrain(self, name, cfg, *args, **kwargs):
        self._current_subterrain_name = name
        try:
            cfg.name = name
        except Exception:
            object.__setattr__(cfg, "name", name)
        print(f"[FiledTerrainGenerator] _generate_subterrain: 强制cfg.name={cfg.name}")
        self._current_subterrain_name = None
        return None

    def _generate_random_terrains(self):
        """按顺序分配terrain，而不是随机分配（覆盖父类方法）"""
        sub_terrains_list = list(self.cfg.sub_terrains.values())
        num_terrains = len(sub_terrains_list)

        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            terrain_idx = index % num_terrains
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_list[terrain_idx])
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_list[terrain_idx])

    @property
    def subterrain_specific_cfgs(self) -> list[SubTerrainBaseCfg]:
        """Get the specific configurations for all subterrains."""
        return self._subterrain_specific_cfgs.copy()  # Return a copy to avoid external modification.

    def get_subterrain_cfg(
        self, row_ids: int | "torch.Tensor", col_ids: int | "torch.Tensor"
    ) -> list[SubTerrainBaseCfg] | SubTerrainBaseCfg | None:
        """Get the specific configuration for a subterrain by its row and column index, with debug info."""
        num_cols = self.cfg.num_cols
        idx = row_ids * num_cols + col_ids
        import sys
        print(f"[DEBUG][get_subterrain_cfg] row_ids={row_ids}, col_ids={col_ids}, num_cols={num_cols}, idx={idx}", file=sys.stderr)
        if hasattr(idx, 'cpu'):
            idx_list = idx.cpu().numpy().tolist()
            result = []
            for i in idx_list:
                if 0 <= i < len(self._subterrain_specific_cfgs):
                    cfg = self._subterrain_specific_cfgs[i]
                    print(f"[DEBUG][get_subterrain_cfg] idx={i}, cfg.name={getattr(cfg, 'name', None)}, cfg type={type(cfg)}", file=sys.stderr)
                    result.append(cfg)
                else:
                    print(f"[DEBUG][get_subterrain_cfg] idx={i} out of range", file=sys.stderr)
                    result.append(None)
            return result
        if isinstance(idx, int):
            if 0 <= idx < len(self._subterrain_specific_cfgs):
                cfg = self._subterrain_specific_cfgs[idx]
                print(f"[DEBUG][get_subterrain_cfg] idx={idx}, cfg.name={getattr(cfg, 'name', None)}, cfg type={type(cfg)}", file=sys.stderr)
                return cfg
            else:
                print(f"[DEBUG][get_subterrain_cfg] idx={idx} out of range", file=sys.stderr)
                return None
