"""
Visualization script for Instinct-Parkour terrain.

Usage:
    cd /home/zh/isaac/instinctlab
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 0
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 5
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 10

=======================================================================
                        地形配置说明
=======================================================================

地形配置位于 instinctlab/terrains/shared_terrain_cfg.py 中。
vis.py 从该文件导入地形配置，不再在本地定义地形。

修改地形请编辑: instinctlab/terrains/shared_terrain_cfg.py

=======================================================================
"""

import argparse

parser = argparse.ArgumentParser(description="可视化地形生成")
parser.add_argument("--terrain_level", type=int, default=0, help="地形难度等级 (0-10)")
args_cli = parser.parse_args()

app_launcher = None
simulation_app = None

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from instinctlab.terrains.terrain_importer import TerrainImporter
from instinctlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinctlab.terrains.shared_terrain_cfg import MY_TERRAIN_CFG
from instinctlab.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg


# ========================================================================
#                        地形导入器配置
# ========================================================================
#
# max_init_terrain_level: 初始最大地形难度 (0-10)
#   - 0: 只有最简单的地形 (perlin_rough)
#   - 5: 中等难度
#   - 10: 所有地形包括最难的地形
#
# virtual_obstacles: 虚拟障碍物配置（用于导航）
#   - edges: 边缘检测障碍物
#
# ========================================================================

MY_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    class_type=TerrainImporter,
    prim_path="/World/ground",
    terrain_type="hacked_generator",
    terrain_generator=MY_TERRAIN_CFG,
    max_init_terrain_level=10,
    collision_group=-1,
    virtual_obstacles={
        "edges": GreedyconcatEdgeCylinderCfg(
            cylinder_radius=0.05,
            min_points=2,
        ),
    },
)


def main():
    """生成并显示地形"""
    sim = SimulationContext(SimulationCfg())
    sim.set_camera_view(eye=(5.0, 8.0, 8.0), target=(5.0, 0.0, -2.0))

    cfg = sim_utils.DistantLightCfg(intensity=1200.0, exposure=5.0, color=(1.0, 1.0, 1.0))

    cfg.func("/World/Light", cfg)

    print(f"[INFO] Generating terrain at level {args_cli.terrain_level}...")
    terrain_importer = TerrainImporter(cfg=MY_TERRAIN_IMPORTER_CFG)

    print(f"[INFO] Terrain visualization ready.")
    print(f"[INFO] Terrain levels available: 0 to ~10")
    print(f"[INFO] Press Ctrl+C to exit or close the Isaac Sim window.")

    try:
        while True:
            sim.step()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")


if __name__ == "__main__":
    main()
