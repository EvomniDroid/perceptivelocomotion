from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励双足机器人脚部腾空时间，鼓励迈大步。
    
    如果在指令很小（即机器人不应该迈步）的情况下，奖励为零。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 计算奖励：获取指定刚体（双脚）当前的腾空时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # 获取指定刚体当前的触地时间
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 兼容四足：奖励摆动腿的腾空时间，避免双足任务中的“单脚支撑”假设导致奖励几乎恒为0。
    in_contact = contact_time > 0.0
    swing_air_time = torch.where(in_contact, torch.zeros_like(air_time), air_time)
    num_contact = torch.sum(in_contact.int(), dim=1)
    has_swing = torch.logical_and(num_contact > 0, num_contact < in_contact.shape[1])
    reward = torch.mean(swing_air_time, dim=1) * has_swing.float()
    # 针对零指令的情况不给奖励
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """惩罚在没有速度指令时仍然移动的行为。"""
    # 提取使用的资产对象
    asset = env.scene[asset_cfg.name]
    # 计算所有关节当前位置与默认位置偏差的绝对值总和
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    # 当 xy 方向和偏航方向的指令均小于阈值时进行惩罚
    return (
        (dof_error - offset)
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
    )


def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """当双脚在 y 轴距离过近时进行惩罚，防止互相绊倒。"""
    # 提取机器人实体对象
    asset = env.scene[asset_cfg.name]

    # 获取脚的世界坐标位置 (假设前两个是左脚和右脚)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    # 获取机器人在世界坐标系下的偏航角
    heading_w = asset.data.heading_w

    # 根据偏航角转换到机器人本体坐标系
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # 旋转左脚到本体坐标系
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    # 旋转右脚到本体坐标系
    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    # 计算双脚在机器人本体 y 轴的绝对距离
    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # 返回连续惩罚，使用指数衰减
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """计算机器人当前航向和目标航向的误差。"""
    # 计算偏航指令的绝对值
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当有向前的速度指令时，惩罚机器人的静止行为。"""
    # 获取机器人本体对象
    asset: RigidObject = env.scene[asset_cfg.name]
    # 获取 x 轴 前进指令
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    # 获取机器人机身 x 轴 的实际前进速度
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    # 如果指令 > 0.2，而实际速度过低则惩罚
    return (lin_vel_cmd_x > 0.2) * ((lin_vel_x < 0.2).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励脚在接触地面时保持垂直状态（平踩）。"""
    # 提取实体资产
    asset: RigidObject = env.scene[asset_cfg.name]
    # 左脚四元数及重力投影
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    # 右脚四元数及重力投影
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    
    # 获取接触传感器及接触力
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    # 累加并返回惩罚 (重力投影在xy平面的平方和)
    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """奖励脚部处于扫描到的地面平面以上的一定高度。"""
    # 提取实体资产
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    # 判断接触力是否大于 1
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    
    # 左脚高度扫描射线数据
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    
    # 右脚高度扫描射线数据
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    
    # 左右脚实际高度
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    # 计算与地面的高度差，并截断
    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    # 求和
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """使用 L2 平方核惩罚连杆（如躯干）的非水平姿态。"""
    # 取实体资产
    asset: RigidObject = env.scene[asset_cfg.name]
    # 取四元数，获取身体朝向
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    # 映射偏航重力投影
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    # 惩罚项计算
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def base_pitch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚 base_link 的 pitch 后仰角（绕 y 轴），前倾不惩罚。

    IsaacLab 约定：x=前进, y=左, z=上。绕 y 轴旋转得到 pitch：
      - pitch > 0 → 后仰（nose 抬起）
      - pitch < 0 → 前倾（nose 低下）

    返回的 L2 惩罚只对 pitch > 0 部分生效，前倾给 0 奖励避免上坡时被误伤。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    _, pitch, _ = euler_xyz_from_quat(quat_w)
    # 只惩罚后仰：clip 到 [0, +inf)
    pitch_pos = torch.clamp(pitch, min=0.0)
    return torch.square(pitch_pos)


