from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sub_terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the sub terrain.

    If the actor moves too close to the edge of the sub terrain, the termination is activated. The distance
    to the edge of the sub terrain is calculated based on the size of the sub terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 0] - env.scene.terrain.env_origins[:, 0])
            > 0.5 * grid_width - distance_buffer
        )
        y_out_of_bounds = (
            torch.abs(asset.data.root_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1])
            > 0.5 * grid_length - distance_buffer
        )
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def root_height_below_env_origin_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_base_height = torch.clamp(env.scene.env_origins[:, 2], max=0.0)
    return asset.data.root_pos_w[:, 2] - terrain_base_height < minimum_height


def bad_pitch(
    env: ManagerBasedRLEnv,
    max_pitch: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base pitch (绕 y 轴) exceeds the limit.

    不同于 `bad_orientation`（3D 总姿态角），这里**只检查 pitch 后仰**：
      - pitch > 0 → 后仰
      - max_pitch 默认 0.5 rad ≈ 28°

    这样能在机器人刚开始"坐"的时候立刻重置，避免它收敛到 spider-pose。

    Args:
        max_pitch: 后仰阈值（弧度），超过则终止。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat_w)
    return pitch > max_pitch

