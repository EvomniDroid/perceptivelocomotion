from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    reward_group_name: str = "rewards",
    success_ratio: float = 0.8,
    lin_x_step: float = 0.1,
    lin_y_step: float = 0.1,
    yaw_step: float = 0.1,
) -> torch.Tensor:
    """Expand command ranges after the policy masters the current range."""
    command = env.command_manager.get_term("base_velocity")
    ranges = command.cfg.ranges
    limits = command.cfg.limit_ranges

    # CurriculumManager can call this for many reset subsets in one step. Only
    # assess it once per nominal episode so the global command range cannot
    # jump several levels from the same batch of returns.
    if env.common_step_counter - command._last_curriculum_update_step < env.max_episode_length:
        return torch.tensor(ranges.lin_vel_x[1], device=env.device)
    command._last_curriculum_update_step = env.common_step_counter

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name, group_name=reward_group_name)
    reward_key = f"{reward_group_name}_{reward_term_name}"
    normalized_reward = torch.mean(env.reward_manager._episode_sums[reward_key][env_ids]) / env.max_episode_length_s
    if normalized_reward > reward_cfg.weight * success_ratio:
        ranges.lin_vel_x = _expand_symmetric_range(ranges.lin_vel_x, limits.lin_vel_x, lin_x_step)
        ranges.lin_vel_y = _expand_symmetric_range(ranges.lin_vel_y, limits.lin_vel_y, lin_y_step)
        ranges.ang_vel_z = _expand_symmetric_range(ranges.ang_vel_z, limits.ang_vel_z, yaw_step)

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def _expand_symmetric_range(
    current: tuple[float, float] | list[float],
    limit: tuple[float, float] | list[float],
    step: float,
) -> tuple[float, float]:
    return (
        max(float(limit[0]), float(current[0]) - step),
        min(float(limit[1]), float(current[1]) + step),
    )
