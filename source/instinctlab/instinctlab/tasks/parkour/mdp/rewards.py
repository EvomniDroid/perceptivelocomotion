from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_mul, normalize

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
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
    # ===== 新增 4 足均衡约束 =====
    # 单脚最大腾空时间不能显著大于平均（防止某只脚长期悬空）
    max_swing = torch.max(air_time, dim=1).values
    mean_swing = torch.mean(air_time, dim=1)
    asymmetry = torch.clamp(max_swing - 1.5 * (mean_swing + 0.1), min=0.0) ** 2
    reward = reward - 0.3 * asymmetry
    # ===== 结束 =====
    # 针对零指令的情况不给奖励
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def foot_contact_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    max_air_time: float = 0.5,
) -> torch.Tensor:
    """强制 4 足都参与：每只脚在 max_air_time 秒内必须接触过至少 1 次。

    惩罚：任何一只脚腾空时间超过 max_air_time 的平方和。
    用法：weight=-2.0，强制策略让所有 4 脚都参与支撑 / 摆动周期。

    与 feet_air_time 的 4 足均衡约束互补：
    - feet_air_time 的 asymmetry 是"软约束"（max > 1.5*mean 扣分）
    - foot_contact_balance 是"硬约束"（单脚 > max_air_time 直接扣分）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (B, 4)
    # 单脚腾空超过阈值 → 扣分（平方惩罚）
    penalty = torch.sum(torch.clamp(air_time - max_air_time, min=0.0) ** 2, dim=1)
    return penalty


def feet_air_time_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    vel_threshold: float = 0.15,
) -> torch.Tensor:
    """惩罚两组对角腿腾空时间不均衡。

    B2RM 足端顺序按 [FL, FR, RL, RR] 处理：
    - FL/RR 是一组对角腿
    - FR/RL 是一组对角腿

    直走时约束最强，转向时放轻，避免把转向需要的非对称步态压掉。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    fl_rr_air_time = 0.5 * (air_time[:, 0] + air_time[:, 3])
    fr_rl_air_time = 0.5 * (air_time[:, 1] + air_time[:, 2])
    error = torch.square(fl_rr_air_time - fr_rl_air_time)

    cmd_vel = env.command_manager.get_command(command_name)
    lin_cmd_mag = torch.norm(cmd_vel[:, :2], dim=1)
    forward_cmd = lin_cmd_mag > vel_threshold
    yaw_cmd = torch.abs(cmd_vel[:, 2]) > vel_threshold
    gate = torch.where(
        forward_cmd & ~yaw_cmd,
        torch.ones_like(lin_cmd_mag),
        torch.where(
            yaw_cmd & ~forward_cmd,
            torch.full_like(lin_cmd_mag, 0.2),
            torch.full_like(lin_cmd_mag, 0.5),
        ),
    )
    has_cmd = torch.logical_or(forward_cmd, yaw_cmd)
    return error * gate * has_cmd.float()


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


def heading_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机器人在 yaw 方向上的实际角速度与指令之间的误差。
    
    当命令 ω_z=0 时，机器人实际还在转就要扣分。
    当命令 ω_z≠0 时，惩罚跟踪误差。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    actual_wz = asset.data.root_ang_vel_b[:, 2]
    error = torch.abs(actual_wz - cmd[:, 2])
    return error


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


def must_turn(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_threshold: float = 0.2,
    min_turn_rate: float = 0.15,
    target_ratio: float = 0.0,
) -> torch.Tensor:
    """当存在明确 yaw 指令时，惩罚机器人没有朝正确方向开始转动。"""
    asset: RigidObject = env.scene[asset_cfg.name]
    yaw_cmd = env.command_manager.get_command(command_name)[:, 2]
    actual_wz = asset.data.root_ang_vel_b[:, 2]

    signed_turn_rate = torch.sign(yaw_cmd) * actual_wz
    target_turn_rate = torch.full_like(yaw_cmd, min_turn_rate)
    if target_ratio > 0.0:
        target_turn_rate = torch.maximum(target_turn_rate, target_ratio * torch.abs(yaw_cmd))
    penalty = torch.clamp(target_turn_rate - signed_turn_rate, min=0.0) / torch.clamp(
        target_turn_rate, min=1.0e-6
    )
    return (torch.abs(yaw_cmd) > cmd_threshold).float() * penalty


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


def roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚 roll 角度（绕 x 轴），使用绝对值惩罚。

    对应 HYT 的 roll = -2.0: rew = |roll|。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    roll, _, _ = euler_xyz_from_quat(quat_w)
    return torch.abs(roll)


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    target_height: float = 0.3,
    vel_threshold: float = 0.15,
) -> torch.Tensor:
    """奖励摆动脚抬到目标高度。

    对应 HYT 的 feet_height = +1.0, feet_height_target = 0.3m。
    只在脚离地（摆动相）时计算，目标高度 0.3m，使用高斯型奖励。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0

    # 脚的世界 z 坐标 - 地面高度 (用 base z 做近似)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # 地面高度近似 = base_z - base_height_target
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    foot_height_rel = foot_z - (base_z - 0.4)  # 0.4 ≈ 地面到基座的粗略偏移

    # 摆动相不触地的脚，高度接近 target 就给奖励
    swing_height = torch.where(in_contact, torch.zeros_like(foot_height_rel), foot_height_rel)
    reward = torch.exp(-torch.square(swing_height - target_height) / 0.04)  # sigma≈0.2
    reward = torch.mean(reward, dim=1)

    # 零指令时不奖励
    has_cmd = torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward * has_cmd.float()


def feet_height_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    max_height: float = 0.36,
    base_to_ground_height: float = 0.4,
) -> torch.Tensor:
    """惩罚对角摆动腿高度不对称，以及单脚抬得过高。

    B2RM 足端顺序按 [FL, FR, RL, RR] 处理：
    - FL/RR 是一组对角腿
    - FR/RL 是一组对角腿

    只在两只对角腿同时处于摆动相时比较高度。直走时约束最强，转向时放轻，
    避免压坏转向策略。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    swing = contact_time <= 0.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    foot_height_rel = foot_z - (base_z - base_to_ground_height)

    diagonal_pairs = ((0, 3), (1, 2))
    pair_errors = []
    for left_id, right_id in diagonal_pairs:
        pair_swing = swing[:, left_id] & swing[:, right_id]
        height_diff = foot_height_rel[:, left_id] - foot_height_rel[:, right_id]
        pair_errors.append(torch.square(height_diff) * pair_swing.float())
    symmetry_error = torch.stack(pair_errors, dim=1).sum(dim=1)

    over_height = torch.clamp(foot_height_rel - max_height, min=0.0)
    over_height_error = torch.sum(torch.square(over_height) * swing.float(), dim=1)

    cmd_vel = env.command_manager.get_command(command_name)
    lin_cmd_mag = torch.norm(cmd_vel[:, :2], dim=1)
    forward_cmd = lin_cmd_mag > 0.1
    yaw_cmd = torch.abs(cmd_vel[:, 2]) > 0.1
    gate = torch.where(
        forward_cmd & ~yaw_cmd,
        torch.ones_like(lin_cmd_mag),
        torch.where(
            yaw_cmd & ~forward_cmd,
            torch.full_like(lin_cmd_mag, 0.2),
            torch.full_like(lin_cmd_mag, 0.5),
        ),
    )
    has_cmd = torch.logical_or(forward_cmd, yaw_cmd)
    return (symmetry_error + over_height_error) * gate * has_cmd.float()


def work_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚净做功（机械功率）的绝对值。|sum(tau * w)|。

    对应 HYT 的 work = -0.003。
    跳跃时做功大 → 重罚，步行时做功小 → 轻罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(torch.sum(
        asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids], dim=1
    ))
    return power


def delta_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚力矩变化平方和（力矩抖振）。

    对应 HYT 的 delta_torques = -1.0e-7。
    需要缓存上一帧的力矩。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 无状态，使用近似：当前步的力矩平方乘以归一化系数
    # 更好的实现需要缓存，但为了简单先这样
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torques), dim=1) * 1.0


class delta_torques(ManagerTermBase):
    """力矩变化平方和（delta_torques），带缓存。

    对应 HYT 的 delta_torques = -1.0e-7。
    rew = sum((tau_t - tau_{t-1})^2)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self._last_torques = torch.zeros_like(self.asset.data.applied_torque[:, asset_cfg.joint_ids])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        curr_torques = self.asset.data.applied_torque[:, asset_cfg.joint_ids]
        delta = torch.sum(torch.square(curr_torques - self._last_torques), dim=1)
        self._last_torques[:] = curr_torques
        return delta

    def reset(self, env_ids):
        self._last_torques[env_ids] = 0.0


def feet_jerk_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """惩罚足端接触力变化率（接触力抖动）。

    对应 HYT 的 feet_jerk = -0.0002。
    由于需要两帧力数据，我们用力平方近似。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    curr_forces = forces[:, -1, sensor_cfg.body_ids]
    # 用力的平方近似变化率（无状态版本）
    return torch.sum(torch.norm(curr_forces, dim=-1), dim=1)


class feet_jerk(ManagerTermBase):
    """足端接触力变化率惩罚，带缓存。

    对应 HYT 的 feet_jerk = -0.0002。
    rew = sum(|F_t - F_{t-1}|)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("contact_forces", body_names=".*_foot"))
        self.sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._last_forces = torch.zeros(
            env.num_envs, len(sensor_cfg.body_ids), 3, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    ) -> torch.Tensor:
        forces = self.sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
        jerk = torch.sum(torch.norm(forces - self._last_forces, dim=-1), dim=1)
        self._last_forces[:] = forces
        return jerk

    def reset(self, env_ids):
        self._last_forces[env_ids] = 0.0


def contact_forces_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """惩罚超过阈值大小的足端接触力。

    对应 HYT 的 feet_contact_forces = -0.001, max_contact_force=120N(B2RM)。
    rew = sum(max(0, |F| - threshold))
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    contact_forces_norm = torch.norm(forces, dim=-1)
    penalty = torch.clamp(contact_forces_norm - threshold, min=0.0)
    return torch.sum(penalty, dim=1)


def tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    sigma: float = 0.5,
    kappa: float = 0.07,
) -> torch.Tensor:
    """Diagonal Trot 步态约束：惩罚对角接触力不对称 + 极端同步。

    对应 HYT 的 tracking_contacts_shaped_force = -2.0。

    原 bug：用 episode 时间硬跑 sin² 相位，期望"phase=0 全踩 / phase=0.5 全腾"，
    4 足 Trot 模式下持续扣分 → 策略被逼学 bounding 跳。

    修复后的语义：
    - 惩罚 **非对角腿同时强接触**（anti-pacing / anti-bound）
    - 鼓励 **"总接触力 ≈ 一半"**（避免全离地跳 + 避免四足同时重踩）
    - kappa 用作 phase EMA 平滑（防止 episode reset 突变）

    期望 contact 信号：
    - 对角步时，非对角腿同步接触应尽量少
    - 直行时约束更强，转向时适度放松
    期望 force 均值：0.5（B2RM 满力 120N → 期望 60N 单腿均值）
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    contact_norm = torch.norm(contact_forces, dim=-1)  # (batch_size, 4)

    # 归一化到 [0, 1]
    max_force = 120.0
    contact_norm = torch.clamp(contact_norm / max_force, 0.0, 1.0)

    # 1) Anti-pacing：惩罚非对角腿同时强接触。
    #    B2RM body_ids 顺序通常是 [FL, FR, RL, RR]。
    #    理想对角步中，非对角配对同时强接触应尽量少。
    symmetry_error = (
        contact_norm[:, 0] * contact_norm[:, 1]  # FL-FR
        + contact_norm[:, 0] * contact_norm[:, 2]  # FL-RL
        + contact_norm[:, 1] * contact_norm[:, 3]  # FR-RR
        + contact_norm[:, 2] * contact_norm[:, 3]  # RL-RR
    )

    # 2) 总接触力 ≈ 0.5（防 bounding 跳 + 防全 4 脚 stand）
    #    行走时 2 脚触地，期望 mean force ≈ 0.5
    #    sigma 当作容忍带：|mean-0.5| <= sigma 不扣分
    mean_force = torch.mean(contact_norm, dim=1)
    mean_dev = torch.abs(mean_force - 0.5) - sigma
    mean_error = torch.clamp(mean_dev, min=0.0) ** 2

    # kappa 用作 phase EMA 平滑（这里用 contact mean 当 phase proxy，避免 episode 突变）
    # 当前步 phase 偏离上一步越多 → 越要抑制（防止策略乱跳）
    if not hasattr(env, "_last_contact_mean"):
        env._last_contact_mean = torch.zeros_like(mean_force)
    phase_jitter = (mean_force - env._last_contact_mean) ** 2
    env._last_contact_mean = (1.0 - kappa) * env._last_contact_mean + kappa * mean_force

    # 总误差
    error = symmetry_error + mean_error + phase_jitter

    # 没速度命令时（应该 stand）不约束步态相位。
    # 直行时最强调对角步；转向时适度放松，避免把已学到的转向能力压坏。
    cmd_vel = env.command_manager.get_command(command_name)
    lin_cmd_mag = torch.norm(cmd_vel[:, :2], dim=1)
    forward_cmd = lin_cmd_mag > 0.1
    yaw_cmd = torch.abs(cmd_vel[:, 2]) > 0.1
    gate = torch.where(
        forward_cmd & ~yaw_cmd,
        torch.ones_like(lin_cmd_mag),
        torch.where(
            yaw_cmd & ~forward_cmd,
            torch.full_like(lin_cmd_mag, 0.3),
            torch.full_like(lin_cmd_mag, 0.7),
        ),
    )
    has_cmd = torch.logical_or(forward_cmd, yaw_cmd)
    return error * gate * has_cmd.float()


def tracking_contacts_shaped_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """步态相位约束（足端速度形状）。

    对应 HYT 的 tracking_contacts_shaped_vel = -2.0。

    期望：摆动相足端速度快（抬腿），支撑相足端速度慢（接地不动）。
    足端速度大的脚应该是摆动相，速度小的脚应该是支撑相。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    is_contact = torch.norm(contact_forces, dim=-1) > 1.0  # (batch_size, 4)

    # 足端速度
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (batch_size, 4, 3)
    foot_speed = torch.norm(foot_vel, dim=-1)  # (batch_size, 4)

    # 期望：接触的脚速度=0，不接触的脚速度≥sigma
    # 惩罚：接触的脚在滑动，或不接触的脚不动（没抬腿）
    swing_speed = torch.where(is_contact, torch.zeros_like(foot_speed), foot_speed - sigma)
    swing_penalty = torch.clamp(-swing_speed, min=0.0)  # 摆动腿速度低于sigma → 惩罚
    stance_speed = torch.where(is_contact, foot_speed, torch.zeros_like(foot_speed))
    stance_penalty = stance_speed  # 接触腿还在滑动 → 惩罚

    penalty = torch.sum(swing_penalty + stance_penalty, dim=1)

    # 没速度命令时（应该 stand）不惩罚
    cmd_vel = env.command_manager.get_command(command_name)
    has_cmd = torch.logical_or(
        torch.norm(cmd_vel[:, :2], dim=1) > 0.1,
        torch.abs(cmd_vel[:, 2]) > 0.1,
    )
    penalty = penalty * has_cmd.float()
    return penalty


def walking_dof(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_threshold: float = 0.15,
    sigma: float = 0.05,
) -> torch.Tensor:
    """有行走命令时，鼓励关节保持 default 姿态。

    对应 HYT 的 walking_dof = +1.5。
    rew = exp(-0.05 * sum(|q - q_default|))
    只在有速度指令时激活。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ), dim=1)
    reward = torch.exp(-sigma * dof_error)

    # 只在有行走命令时激活
    has_cmd = torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward * has_cmd.float()
