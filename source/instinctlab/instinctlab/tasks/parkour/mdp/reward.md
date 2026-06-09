# B2RM Parkour 完整奖励项清单

**出处**: `instinctlab/tasks/parkour/config/b2rm/b2rm_parkour_cfg.py` (lines 195-430) + `instinctlab/tasks/parkour/mdp/rewards.py`

> **"+"** = 正向奖励 (鼓励), **"-"** = 负向惩罚 (抑制)
> **Weight 注释**: 改动过的项标了 [HYT 改] / [已加] / [原值保留]

---

## 1. 任务主奖励（让机器人"做对事"）

| Reward 项 | Weight | 函数 | 含义 |
|---|---|---|---|
| `track_lin_vel_xy_exp` | +0.5 | `track_lin_vel_xy_exp` | 跟踪 xy 线速度指令（高斯核，越接近指令越大） |
| `track_ang_vel_z_exp` | +0.5 | `track_ang_vel_z_exp` | 跟踪 yaw 角速度指令（高斯核） |
| `heading_error` | -1.5 | `heading_error` | `|ω_z_actual - ω_z_cmd|`：有指令时压跟踪误差，零指令时抑制乱转 |
| `dont_wait` | -2.0 | `dont_wait` | 收到前进指令 (cmd_x>0.2) 但机器人不动 / 后退 / 走太慢 → 重罚 |
| `must_turn` | -2.0 | `must_turn` | 收到明确 yaw 指令 (|cmd_wz|>0.05) 但没朝正确方向开始转 → 重罚（target_ratio=0.6） |
| `is_alive` | +2.0 | `is_alive` | 每步存活奖励 = 1（鼓励"别倒"） |
| `stand_still` | -1.0 | `stand_still` | 零速度指令时仍动 → 扣分（dof 偏差 - 0.0 超 0.05 阈值才扣） |

---

## 2. 步态/足端奖励（让 4 条腿"对的事"）

| Reward 项 | Weight | 函数 | 含义 |
|---|---|---|---|
| `volume_points_penetration` | -2.8 | `volume_points_penetration` | 腿部足端 "体积点云" 穿透到地形内部 → 重罚（防"插地"） |
| `feet_air_time` | +0.5 | `feet_air_time` | 摆动相腾空时间平均奖励；**内置 4 足均衡**: `max > 1.5*mean` 平方扣分（防单脚长期悬空） |
| `foot_contact_balance` | -2.0 | `foot_contact_balance` | **新增** [HYT 改] 单脚腾空 > 1.0s → 平方扣分（硬约束，防止"三条腿走路"局部最优） |
| `feet_air_time_balance` | -1.0 | `feet_air_time_balance` | **新增** 对角腿腾空时间差：`[(FL+RR) - (FR+RL)]²`，直走 gate=1.0 / 转向 gate=0.2~0.5 |
| `feet_slide` | -0.5 | `contact_slide` | 接触状态下足端滑动速度 > 1.0 m/s → 扣分（防滑步） |
| `feet_close_xy_gauss` | 0.0 | `feet_close_xy_gauss` | **当前权重 0.0**（未启用）：左右脚 y 距离过近（< 0.25）时高斯扣分（防绊倒） |
| `feet_height` | +1.0 | `feet_height` | **新增** [HYT 改] 摆动相抬腿高度接近 0.3m → 高斯奖励（`exp(-(h-0.3)²/0.04)`） |
| `feet_height_balance` | -4.0 | `feet_height_balance` | **新增** [HYT 改] 对角摆动腿高度差平方和 + 单脚 > 0.36m 超高扣分 |
| `tracking_contacts_shaped_force` | -2.0 | `tracking_contacts_shaped_force` | **新增** [HYT 改] 步态相位力约束：(a) 非对角腿同时强接触扣分 (b) 4 脚均力偏离 0.5 扣分 (c) phase EMA 抖振扣分 |
| `tracking_contacts_shaped_vel` | -2.0 | `tracking_contacts_shaped_vel` | **新增** [HYT 改] 步态相位速度约束：摆动腿速度 < 0.5 扣分 / 支撑腿速度 > 0 扣分（滑步） |
| `walking_dof` | +0.5 | `walking_dof` | **新增** [HYT 改] 有速度指令时，鼓励关节保持 default 姿态（`exp(-0.05*|q-q_default|)`） |

---

## 3. 姿态/平衡惩罚（让机器人"不倒"）

| Reward 项 | Weight | 函数 | 含义 |
|---|---|---|---|
| `ang_vel_xy_l2` | -0.2 | `ang_vel_xy_l2` | 机身 roll/pitch 角速度平方和（抑制侧向晃） |
| `lin_vel_z_l2` | -1.5 | `lin_vel_z_l2` | 机身 z 方向线速度平方（抑制上下蹦跳） |
| `roll_l2` | -2.0 | `roll_l2` | **HYT 改** `|roll|`（绝对值，绕 x 轴侧倾） |
| `flat_orientation_l2` | -2.5 | `flat_orientation_l2` | 机身姿态偏离水平方向 (roll+pitch) 的 L2 |
| `base_pitch_l2` | -2.0 | `base_pitch_l2` | **HYT 改** 后仰 (pitch>0) 平方惩罚，**前倾不惩罚**（避免上坡误伤） |
| `base_height` | -4.0 | `base_height_l2` | 机身高度偏离目标 0.55m 的 L2（防蹲/防跛脚垫高） |

---

## 4. 关节/力矩正则（让动作"平滑"）

| Reward 项 | Weight | 函数 | 含义 |
|---|---|---|---|
| `joint_deviation_arm` | -1.5 | `joint_deviation_square` | 机械臂 6 个 joint 偏离 default 的平方（防乱摆） |
| `joint_deviation_legs` | -0.3 | `joint_deviation_l1` | 12 条腿 joint 偏离 default 的 L1 |
| `dof_torques_l2` | -2.5e-5 | `joint_torques_l2` | 12 腿关节力矩平方 |
| `dof_acc_l2` | -7.5e-7 | `joint_acc_l2` | 12 腿关节加速度平方 |
| `dof_vel_l2` | -0.0001 | `joint_vel_l2` | 12 腿关节速度平方 |
| `action_rate_l2` | -0.015 | `action_rate_l2` | 连续动作差平方（动作平滑） |
| `dof_pos_limits` | -1.0 | `joint_pos_limits` | 关节位置靠近极限的平方惩罚 |
| `work_l2` | -0.003 | `work_l2` | **新增** [HYT 改] 净做功 `|Σ(τ·ω)|`（防高频抖振能耗） |
| `delta_torques` | -1.0e-7 | `delta_torques` | **新增** [HYT 改] 力矩变化 `Σ(τ_t - τ_{t-1})²`（防力矩抖振，带缓存） |
| `feet_jerk` | -0.0002 | `feet_jerk` | **新增** [HYT 改] 足端力变化 `Σ|F_t - F_{t-1}|`（防接触力抖，带缓存） |
| `contact_forces_penalty` | -0.001 | `contact_forces_penalty` | **新增** [HYT 改] 足端力 > 120N 的部分平方和（防硬踩） |

---

## 📊 总览统计

- **奖励项总数**: 28 项
- **正奖励 (+)**: 5 项 — track_lin_vel_xy_exp, track_ang_vel_z_exp, is_alive, feet_air_time, feet_height, walking_dof
- **负惩罚 (-)**: 22 项
- **权重为 0 (未启用)**: 1 项 — `feet_close_xy_gauss`
- **新增 [HYT 改]**: 9 项 — foot_contact_balance, feet_air_time_balance, feet_height, feet_height_balance, tracking_contacts_shaped_force, tracking_contacts_shaped_vel, walking_dof, work_l2, delta_torques, feet_jerk, contact_forces_penalty
- **已修改**: roll_l2 (改 |roll|), base_pitch_l2 (改后仰专项), must_turn (target_ratio)

## 🎯 设计目标（按 HYT 系列）

1. **不倒**: roll/pitch/base_height/orientation
2. **不滑/不插地**: feet_slide, volume_points_penetration
3. **4 足都参与**: feet_air_time (软) + foot_contact_balance (硬) + feet_air_time_balance (对角均衡)
4. **对角 Trot 步态**: tracking_contacts_shaped_force (anti-pacing) + tracking_contacts_shaped_vel (相位)
5. **抬腿高度合理**: feet_height (目标 0.3m) + feet_height_balance (对角对称, ≤0.36m)
6. **跟指令**: track_vel (主) + heading_error + dont_wait + must_turn
7. **动作平滑**: action_rate + work + delta_torques + feet_jerk
8. **能效**: dof_torques/dof_vel/dof_acc 三件套
