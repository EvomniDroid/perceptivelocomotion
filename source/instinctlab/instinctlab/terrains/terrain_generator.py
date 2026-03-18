from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGenerator

if TYPE_CHECKING:
    from .terrain_generator_cfg import FiledTerrainGeneratorCfg


class FiledTerrainGenerator(TerrainGenerator):
    """A terrain generator that uses the filed generator."""

    def __init__(self, cfg: FiledTerrainGeneratorCfg, device: str = "cpu"):
        print(f"[InstinctLab FiledTerrainGenerator] __init__ called! cfg type: {type(cfg)} id={id(self)}")
        self._subterrain_specific_cfgs: list[SubTerrainBaseCfg] = []
        super().__init__(cfg, device)
        print(f"[InstinctLab FiledTerrainGenerator] after super().__init__, subterrain_specific_cfgs len={len(self._subterrain_specific_cfgs)}")
        # 一次到位：patch阶段分配地形名并append到subterrain_specific_cfgs，训练阶段只用patch_cfgs
        if hasattr(cfg, 'sub_terrains') and isinstance(cfg.sub_terrains, dict):
            print("[DEBUG] 自动生成patch，遍历sub_terrains:")
            for patch_key, patch_cfg in cfg.sub_terrains.items():
                # 强制 patch_cfg.name = patch_key
                try:
                    patch_cfg.name = patch_key
                except Exception:
                    object.__setattr__(patch_cfg, "name", patch_key)
                print(f"[DEBUG][PATCH] patch_key={patch_key}, patch_cfg.name={getattr(patch_cfg, 'name', None)}, patch_cfg type={type(patch_cfg)}")
                self._generate_subterrain(patch_key, patch_cfg)
                self._subterrain_specific_cfgs.append(patch_cfg)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # 优先用 cfg.name，再用 _current_subterrain_name，最后兜底
        name = getattr(cfg, "name", None)
        if name is None or (isinstance(name, str) and name.strip() == ""):
            name = getattr(self, "_current_subterrain_name", None)
        if name is None or (isinstance(name, str) and name.strip() == ""):
            name = "unknown"
        if name == "unknown":
            name = f"auto_{getattr(cfg, 'proportion', 'p')}_{getattr(cfg, 'platform_width', 'pw')}_{getattr(cfg, 'border_width', 'bw')}"
            print(f"[DEBUG] name==unknown, 兜底生成: {name}")
        try:
            cfg.name = name
        except Exception:
            object.__setattr__(cfg, "name", name)
        print(f"[FiledTerrainGenerator] cfg.name={cfg.name}, 来源: _current_subterrain_name={getattr(self, '_current_subterrain_name', None)}, cfg.name={getattr(cfg, 'name', None)}")
        # 不再 append 到 subterrain_specific_cfgs，保证只保存 patch阶段的 cfg
        return mesh, origin

    def _generate_subterrain(self, name, cfg, *args, **kwargs):
        # 记录当前subterrain的名字，供_get_terrain_mesh用
        self._current_subterrain_name = name
        # 强制给cfg加上name字段
        try:
            cfg.name = name
        except Exception:
            object.__setattr__(cfg, "name", name)
        print(f"[FiledTerrainGenerator] _generate_subterrain: 强制cfg.name={cfg.name}")
        self._current_subterrain_name = None
        return None

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
