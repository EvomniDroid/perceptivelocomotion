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


def randomize_joints_near_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_targets: dict[str, float],
    joint_pos_range: float | dict[str, tuple[float, float]] = 0.0,
    joint_vel_target: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set selected joints near target positions with per-joint random offsets.

    This is intended for interval-time perturbations on the arm while preserving the
    lower-body locomotion setup. The randomization is applied directly in joint state
    space and clamped to the articulation soft limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()

    for joint_name, target_pos in joint_pos_targets.items():
        joint_ids, _ = asset.find_joints(joint_name)
        if len(joint_ids) == 0:
            continue
        joint_id = joint_ids[0]

        if isinstance(joint_pos_range, dict):
            low, high = joint_pos_range.get(joint_name, (0.0, 0.0))
        else:
            low, high = (-float(joint_pos_range), float(joint_pos_range))

        offset = math_utils.sample_uniform(
            torch.full((len(env_ids),), low, device=asset.device),
            torch.full((len(env_ids),), high, device=asset.device),
            (len(env_ids),),
            device=asset.device,
        )
        joint_pos[:, joint_id] = target_pos + offset
        joint_vel[:, joint_id] = joint_vel_target

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_arm_safe_carry_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_library: list[dict[str, float]],
    joint_pos_range: float | dict[str, tuple[float, float]] = 0.0,
    joint_vel_target: float = 0.0,
    target_attr_name: str = "_b2rm_arm_carry_joint_pos_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample one static arm carry pose per env from a curated safe pose library.

    The sampled pose is written both to the simulated joint state and to an
    articulation attribute consumed by DynamicTargetJointPositionAction. This
    makes the arm hold the selected carry pose while preserving the policy's
    historical action dimensions.
    """
    if len(pose_library) == 0 or len(env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()

    if not hasattr(asset, target_attr_name):
        setattr(asset, target_attr_name, asset.data.joint_pos.clone())
    dynamic_target = getattr(asset, target_attr_name)

    library_ids = torch.randint(len(pose_library), (len(env_ids),), device=asset.device)
    joint_names = list(pose_library[0].keys())

    for joint_name in joint_names:
        joint_ids, _ = asset.find_joints(joint_name)
        if len(joint_ids) == 0:
            continue
        joint_id = joint_ids[0]

        base_values = torch.tensor(
            [pose_library[int(library_id.item())][joint_name] for library_id in library_ids],
            device=asset.device,
            dtype=joint_pos.dtype,
        )

        if isinstance(joint_pos_range, dict):
            low, high = joint_pos_range.get(joint_name, (0.0, 0.0))
        else:
            low, high = (-float(joint_pos_range), float(joint_pos_range))
        noise = math_utils.sample_uniform(
            torch.full((len(env_ids),), low, device=asset.device),
            torch.full((len(env_ids),), high, device=asset.device),
            (len(env_ids),),
            device=asset.device,
        )

        joint_pos[:, joint_id] = base_values + noise
        joint_vel[:, joint_id] = joint_vel_target

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    dynamic_target[env_ids] = joint_pos
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _sphere_to_cart(sphere: torch.Tensor) -> torch.Tensor:
    radius = sphere[:, 0]
    pitch = sphere[:, 1]
    yaw = sphere[:, 2]
    xy = radius * torch.cos(pitch)
    return torch.stack((xy * torch.cos(yaw), xy * torch.sin(yaw), radius * torch.sin(pitch)), dim=-1)


def _inside_any_aabb(points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    if lower.numel() == 0:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    inside = (points[:, None, :] > lower[None, :, :]) & (points[:, None, :] < upper[None, :, :])
    return torch.any(torch.all(inside, dim=-1), dim=1)


def randomize_arm_workspace_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_range: tuple[float, float],
    sphere_center_offset_b: tuple[float, float, float] = (-0.19836152, 0.0, 0.0),
    ground_clearance: float = 0.50,
    collision_lower_limits: list[tuple[float, float, float]] | None = None,
    collision_upper_limits: list[tuple[float, float, float]] | None = None,
    corridor_collision_lower_limits: tuple[float, float, float] | None = None,
    corridor_collision_upper_limits: tuple[float, float, float] | None = None,
    corridor_collision_num_samples: int = 5,
    max_resample_attempts: int = 12,
    target_pos_attr_name: str = "_b2rm_arm_workspace_target_pos_w",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample a safe end-effector target in the rear workspace for arm IK tracking."""
    if len(env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    count = len(env_ids)
    if not hasattr(asset, target_pos_attr_name):
        setattr(asset, target_pos_attr_name, asset.data.body_pos_w[:, 0].clone())
    target_pos_w = getattr(asset, target_pos_attr_name)

    center_offset_b = torch.tensor(sphere_center_offset_b, device=device, dtype=asset.data.root_pos_w.dtype)
    root_pos_w = asset.data.root_pos_w[env_ids]
    root_quat_w = asset.data.root_quat_w[env_ids]
    center_w = root_pos_w + math_utils.quat_apply_yaw(root_quat_w, center_offset_b.unsqueeze(0).expand(count, -1))

    lower = torch.tensor(collision_lower_limits or [], device=device, dtype=root_pos_w.dtype)
    upper = torch.tensor(collision_upper_limits or [], device=device, dtype=root_pos_w.dtype)
    corridor_lower = (
        torch.tensor(corridor_collision_lower_limits, device=device, dtype=root_pos_w.dtype)
        if corridor_collision_lower_limits is not None
        else None
    )
    corridor_upper = (
        torch.tensor(corridor_collision_upper_limits, device=device, dtype=root_pos_w.dtype)
        if corridor_collision_upper_limits is not None
        else None
    )
    corridor_t = torch.linspace(0.0, 1.0, max(2, corridor_collision_num_samples), device=device, dtype=root_pos_w.dtype)

    best_local = torch.zeros(count, 3, device=device, dtype=root_pos_w.dtype)
    unresolved = torch.ones(count, dtype=torch.bool, device=device)

    for _ in range(max(1, max_resample_attempts)):
        active_ids = unresolved.nonzero(as_tuple=False).flatten()
        if active_ids.numel() == 0:
            break
        sample_shape = (active_ids.numel(),)
        sphere = torch.stack(
            (
                math_utils.sample_uniform(radius_range[0], radius_range[1], sample_shape, device=device),
                math_utils.sample_uniform(pitch_range[0], pitch_range[1], sample_shape, device=device),
                math_utils.sample_uniform(yaw_range[0], yaw_range[1], sample_shape, device=device),
            ),
            dim=-1,
        )
        candidate_local = _sphere_to_cart(sphere)

        invalid = _inside_any_aabb(candidate_local, lower, upper)
        if corridor_lower is not None and corridor_upper is not None:
            corridor_points = candidate_local[:, None, :] * corridor_t[None, :, None]
            corridor_inside = (corridor_points > corridor_lower[None, None, :]) & (
                corridor_points < corridor_upper[None, None, :]
            )
            invalid = invalid | torch.any(torch.all(corridor_inside, dim=-1), dim=1)

        valid = ~invalid
        if torch.any(valid):
            resolved_ids = active_ids[valid]
            best_local[resolved_ids] = candidate_local[valid]
            unresolved[resolved_ids] = False

    if torch.any(unresolved):
        fallback = torch.tensor([-0.65, 0.0, 0.55], device=device, dtype=root_pos_w.dtype)
        best_local[unresolved] = fallback

    sampled_target_w = center_w + math_utils.quat_apply_yaw(root_quat_w, best_local)
    sampled_target_w[:, 2] = torch.maximum(
        sampled_target_w[:, 2],
        torch.full((count,), ground_clearance, device=device, dtype=sampled_target_w.dtype),
    )
    target_pos_w[env_ids] = sampled_target_w
