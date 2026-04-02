"""
Visualization script for Instinct-Parkour terrain.

Usage:
    cd /home/zh/isaac/instinctlab
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 0
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 5
    python source/instinctlab/instinctlab/tasks/parkour/scripts/vis.py --terrain_level 10
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Instinct-Parkour terrain")
parser.add_argument("--terrain_level", type=int, default=0, help="Terrain difficulty level (0-10)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils

sys.path.insert(0, "source/instinctlab")
from instinctlab.tasks.parkour.config.parkour_env_cfg import ParkourEnvCfg
from instinctlab.terrains.terrain_importer import TerrainImporter


def main():
    """Generate and visualize terrain at specified difficulty level."""
    sim = SimulationContext(SimulationCfg())
    sim.set_camera_view(eye=(5.0, 8.0, 8.0), target=(5.0, 0.0, -2.0))

    cfg = sim_utils.DistantLightCfg(intensity=3000.0)
    cfg.func("/World/Light", cfg)

    env_cfg = ParkourEnvCfg()
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level

    print(f"[INFO] Generating terrain at level {args_cli.terrain_level}...")
    terrain_importer = TerrainImporter(cfg=env_cfg.scene.terrain)

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
