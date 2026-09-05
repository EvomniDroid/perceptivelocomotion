# InstinctLab 导航框架说明

## 概述

本项目实现了一个基于强化学习和视觉伺服的机器人导航系统，集成了摔倒率评估和前沿点探索功能。

## 目录结构

```
/home/zh/isaac/
├── instinctlab/
│   └── source/instinctlab/instinctlab/tasks/parkour/
│       └── scripts/
│           ├── deploy.py              # 主部署脚本
│           └── fusion_cost.py         # 融合成本计算器
└── liveratemodel/
    ├── local_planner.py               # 本地规划器（摔倒率地图 + 前沿点检测）
    ├── qwen_vl_detector.py            # Qwen VL 目标检测器
    └── test_qwen_api.py               # Qwen API 测试脚本
```

## 核心模块

### 1. deploy.py - 主部署脚本

**功能**：整合所有模块，运行仿真循环

**主要流程**：

```
初始化 → 仿真循环 → 前沿点检测 → 目标选择 → 策略执行 → 数据保存
```

**关键参数**：

| 参数                       | 说明                 | 默认值     |
| -------------------------- | -------------------- | ---------- |
| `--preset_fall_rate_map` | 使用预设摔倒率地图   | False      |
| `--auto_frontier_nav`    | 自动选择前沿点导航   | False      |
| `--fall_rate_penalty`    | 摔倒率惩罚系数 (0-1) | 1.0        |
| `--urgency_ref_dist`     | 紧迫度参考距离(米)   | 3.0        |
| `--qwen_target_color`    | Qwen检测目标颜色     | "红色方块" |
| `--scan_angvel`          | 扫描模式角速度       | 0.15 rad/s |

**数据保存**：

- `frontier/` - 前沿点可视化数据 (JSON)
- `terminal/` - 终端日志 (terminal.log)
- `fushi_depth_images/` - 俯视深度图
- `images/` - RGB图像

### 2. fusion_cost.py - 融合成本计算器

**功能**：计算前沿点的综合得分，平衡目标紧迫度和摔倒风险

**核心方法**：

- `calculate_urgency()` - 计算目标紧迫度
- `predict_danger_ahead()` - 预测前方是否有高风险区
- `calculate_direction_penalty()` - 计算绕路方向惩罚因子

**得分公式**：

```
score = conf * direction_score^2 * fall_factor * dist_score * direction_penalty
```

### 3. local_planner.py - 本地规划器

**功能**：

- 摔倒率地图管理与更新
- 前沿点检测（基于视锥和已探索区域边界）

**关键类**：

- `LocalFallRateMap` - 摔倒率地图

  - `update_map()` - 更新地图
  - `get_fall_rate_at()` - 获取某点摔倒率
  - `_init_preset_fall_rate_map()` - 初始化预设地图
- `FrontierDetector` - 前沿点检测器

  - `detect_frontiers_by_layer()` - 按深度分层检测前沿点
  - `detect_frontiers_in_fov()` - 在视锥内检测前沿点

**预设危险区域**（当前配置）：

```
x ∈ [3.5, 4.5], y ∈ [1.5, 2.5]  →  fall_rate = 0.8
其他区域                              →  fall_rate = 0.1
```

### 4. qwen_vl_detector.py - Qwen VL 目标检测器

**功能**：使用 Qwen VL 模型检测目标物体世界坐标

**关键方法**：

- `detect_from_image()` - 检测目标并返回世界坐标
- `pixel_to_world()` - 像素坐标转世界坐标

**参数**：

- `target_color` - 目标颜色描述（如"红色方块"、"蓝色方块"）

## 运行命令

### 标准运行

```bash



cd /home/zh/isaac/instinctlab

python source/instinctlab/instinctlab/tasks/parkour/scripts/deploy.py \
  --task=Instinct-Parkour-Target-Amp-G1-v0 \
  --load_run=/home/zh/isaac/instinctlab/logs/instinct_rl/g1_parkour/20260326_142216/ \
  --checkpoint=model_40000.pt \
  --num_envs=1 \
  --use_frontier_test_terrain \
  --preset_fall_rate_map \
  --spawn_pos=0.0,0.0 \
  --auto_frontier_nav \
  --termination_mode=full \
  --keyboard_control \
  --frontier_save_interval=100 \
  --qwen_detect_interval=10 \
  --qwen_init_interval=10 \
  --scan_angvel=0.15 \
  --qwen_target_color "红色方块" \
  --fall_rate_penalty=1.0 \
  --urgency_ref_dist=3.0

python source/instinctlab/instinctlab/tasks/parkour/scripts/deploy.py     --task=Instinct-Parkour-Target-Amp-G1-v0  --load_run=/home/zh/isaac/instinctlab/logs/instinct_rl/g1_parkour/20260326_142216/     --checkpoint=model_40000.pt     --num_envs=1     --use_frontier_test_terrain   --preset_fall_rate_map     --spawn_pos=0.0,0.0      --vel_debug     --frontier_debug     --auto_frontier_nav     --termination_mode=full    --keyboard_control  --frontier_save_interval=100      --debug_ray   --save_rgb_interval=10    --save_fushi_depth_interval=10          --qwen_detect_interval=10   --qwen_init_interval=10  --scan_angvel=0.15  --qwen_target_color "红色方块"   --fall_rate_penalty=0.5  --urgency_ref_dist=3.0
```

### 带调试输出运行

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/deploy.py \
  ...（同上）\
  --vel_debug \
  --frontier_debug \
  --debug_ray \
  --save_rgb_interval=10 \
  --save_fushi_depth_interval=10
```

## 坐标系说明

### 世界坐标系

- 原点：地图左下角 (0, 0)
- X正方向：向右
- Y正方向：向上

### 相机坐标系

- Z轴（深度）：从相机指向前方
- X轴（水平）：相机右侧
- Y轴（垂直）：相机下方

### 坐标转换

```python
# 像素 → 相机3D
camera_x = (u - cx) * depth / fx
camera_y = (v - cy) * depth / fy
camera_z = depth

# 相机3D → 世界坐标
world_x = robot_x + camera_z * cos(yaw) - camera_x * sin(yaw)
world_y = robot_y + camera_z * sin(yaw) + camera_x * cos(yaw)
```

## 当前状态

### 已完成

- [X] 目标世界坐标检测（Qwen VL + 深度图）
- [X] 视觉伺服控制（保持目标在视野中央）
- [X] 摔倒率地图管理（预设 + 实时更新）
- [X] 前沿点检测（视锥 + 深度分层）
- [X] 融合成本计算（紧迫度 + 摔倒率 + 方向惩罚）
- [X] 终端日志实时记录
- [X] 数据可视化保存（前沿点、深度图、RGB图）
- [X] 视觉运控实现

### 待完成

- [ ] 深度模型调优（摔倒率输出部分）
- [ ] 绕路逻辑未实现（前沿点生成 + 方向选择，现在用预设的摔倒率不能实现绕路）

### 已知问题

1. 前沿点主要生成在机器人侧后方，而非正前方目标方向
2. 当危险区域位于正前方时，绕路逻辑未能有效选择侧向路径

## 调参建议

### 摔倒率相关

- `--fall_rate_penalty`：增大 → 更倾向绕开危险区域
- `--urgency_ref_dist`：减小 → 更关注近距离目标

### 前沿点检测

- `max_depth`：增加 → 视野范围更大
- `min_frontier_size`：减小 → 更敏感的前沿检测

### 视觉伺服

- `SERVO_GAIN`：减小 → 更平滑的跟踪

## 数据分析

### 查看前沿点数据

```python
import json
with open('frontier/run_XXXXXXXX_XXXXX/frontier_vis_t300.json') as f:
    data = json.load(f)
# data[0] 包含 timestep, robot_pos, frontiers 列表
```

### 分析终端日志

```bash
tail -f terminal/run_XXXXXXXX_XXXXX/terminal.log
```
