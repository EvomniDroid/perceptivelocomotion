from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCamera
from isaaclab.assets import Articulation

from instinctlab.sensors import NoisyGroupedRayCasterCamera, NoisyRayCasterCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_without_stand(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges. No pushing when standing still."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    add_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    lin_vel = torch.norm(env.command_manager.get_command(command_name)[env_ids, :2], dim=1) > 0.15
    ang_vel = torch.abs(env.command_manager.get_command(command_name)[env_ids, 2]) > 0.15
    should_push = torch.logical_or(lin_vel, ang_vel).float().unsqueeze(-1)

    vel_w += add_vel * should_push
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_joints_to_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_targets: dict[str, float],
    joint_vel_target: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset selected joints to exact target positions.

    This is useful for keeping passive arm joints in a fixed folded pose,
    independent of articulation default joint offsets.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()

    for joint_name, target_pos in joint_pos_targets.items():
        joint_ids, _ = asset.find_joints(joint_name)
        if len(joint_ids) == 0:
            continue
        joint_id = joint_ids[0]
        joint_pos[:, joint_id] = target_pos
        joint_vel[:, joint_id] = joint_vel_target

    # Clamp to soft limits for safety.
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
