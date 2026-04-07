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

地形配置位于文件底部的 MY_TERRAIN_CFG.sub_terrains 字典中。
每种地形的参数说明：

1. perlin_rough - Perlin噪声粗糙地面
   - noise_scale: 噪声幅度范围，如 [0.0, 0.1]
   - noise_frequency: 噪声频率，越高越密集
   - fractal_octaves: 分形层数，越多细节越丰富
   - proportion: 该地形在地图中的比例

2. square_gaps - 方形坑洞地形
   - gap_distance_range: 坑洞间距范围
   - gap_depth: 坑洞深度范围
   - platform_width: 平台宽度

3. pyramid_stairs - 金字塔楼梯
   - step_height_range: 台阶高度范围
   - step_width: 台阶宽度
   - platform_width: 平台宽度

4. pyramid_stairs_inv - 反向金字塔楼梯（凹陷）
   - 同上，但地形向下凹陷

5. boxes - 离散障碍物地形
   - obstacle_height_range: 障碍物高度范围
   - obstacle_width_range: 障碍物宽度范围
   - num_obstacles: 障碍物数量

6. my_wave_terrain - 自定义波浪地形
   - amplitude_range: 波浪振幅范围
   - num_waves: 波浪数量

7. my_stepping_stones - 踏脚石地形
   - stone_width_range: 石头宽度范围
   - stone_distance_range: 石头间距范围
   - stone_height_max: 石头最大高度

通用参数：
   - wall_prob: [左, 右, 前, 后] 墙壁出现概率
   - wall_height: 墙的高度
   - wall_thickness: 墙的厚度
   - border_width: 边界宽度
   - flat_patch_sampling: 平坦区域采样配置
       - num_patches: 采样点数量（所有地形需保持一致！）
       - patch_radius: 采样半径范围

=======================================================================
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
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from instinctlab.terrains.terrain_importer import TerrainImporter
from instinctlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains import GreedyconcatEdgeCylinderCfg

from instinctlab.terrains import (
    PerlinPlaneTerrainCfg,
    PerlinSquareGapTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinPyramidSlopedTerrainCfg,
    PerlinInvertedPyramidSlopedTerrainCfg,
    PerlinWaveTerrainCfg,
    PerlinSteppingStonesTerrainCfg,
    PerlinParapetTerrainCfg,
    PerlinGutterTerrainCfg,
    PerlinStairsUpDownTerrainCfg,
    PerlinStairsDownUpTerrainCfg,
    PerlinTiltTerrainCfg,
    PerlinTiltedRampTerrainCfg,
    PerlinSlopeTerrainCfg,
    PerlinCrossStoneTerrainCfg,
)


def _inject_name_to_cfgs(sub_terrains):
    """为每个子地形配置注入 name 属性（key 值）"""
    for k, v in sub_terrains.items():
        try:
            setattr(v, "name", k)
        except Exception:
            object.__setattr__(v, "name", k)
    return sub_terrains


# ========================================================================
#                        地形生成器配置
# ========================================================================
#
# 修改这里来自定义你的地形！
#
# MY_TERRAIN_CFG 控制整体地形生成参数：
#   - size: 每个子地形的尺寸 (宽, 长)
#   - num_rows, num_cols: 地形的行列数量 (总共 num_rows x num_cols 个子地形)
#   - horizontal_scale: 水平方向分辨率（越小越精细）
#   - vertical_scale: 垂直方向分辨率（越小越精细）
#   - curriculum: 是否启用课程学习（True = 难度渐进）
#
# sub_terrains 字典定义所有可用的子地形类型：
#   - key: 地形名称（会显示在日志中）
#   - value: 地形配置对象
#
# ========================================================================

MY_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=0,
    size=(8.0, 8.0),          # 每个子地形的大小（米）
    border_width=0.2,            # 边界宽度
    num_rows=1,              # 地形行数
    num_cols=15,              # 地形列数（等于地形数量）
    horizontal_scale=0.05,     # 水平分辨率（越小越精细）
    vertical_scale=0.005,      # 垂直分辨率（越小越精细）
    slope_threshold=1.0,       # 斜坡阈值
    use_cache=False,           # 是否使用缓存
    curriculum=False,          # 关闭课程学习，让每个地形都显示
    sub_terrains=_inject_name_to_cfgs({

        # ==================================================================
        #  基础 Perlin 噪声地形（8种）
        # ==================================================================

        # ------------------------------------------------------------------
        # 地形1: Perlin噪声粗糙地面 - 简单的随机粗糙地面
        # # ------------------------------------------------------------------
        # "perlin_rough": PerlinPlaneTerrainCfg(
        #     proportion=1.0,
        #     noise_scale=[0.0, 0.1],
        #     noise_frequency=20,
        #     fractal_octaves=2,
        #     fractal_lacunarity=2.0,
        #     fractal_gain=0.25,
        #     centering=True,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # ------------------------------------------------------------------
        # 地形 2: 方形坑洞 - 中间有方形缺口的地形
        # 结构：平地上随机分布方形凹陷坑洞
        # 用途：训练机器人跨越坑洞或识别可通行区域
        # ------------------------------------------------------------------
        "square_gaps": PerlinSquareGapTerrainCfg(
            proportion=1.0,
            gap_distance_range=(0.1, 0.7),   # 坑洞间距范围（米）
            gap_depth=(0.8, 1.2),            # 坑洞深度范围（米）- 加深让坑洞更明显
            platform_width=1.0,              # 中间平台宽度（减小以显示更多坑洞）
            border_width=1.0,                # 边界宽度
            wall_prob=[0.0, 0.0, 0.0, 0.0],  # 围墙概率 [左，右，前，后]
            wall_height=5.0,                 # 围墙高度
            wall_thickness=0.05,             # 围墙厚度
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(1.0, 3.0),
                    y_range=(-1.0, 1.0),
                ),
            },
        ),

        # # ------------------------------------------------------------------
        # # 地形3: 金字塔楼梯 - 向上走的楼梯
        # # ------------------------------------------------------------------
        # "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
        #     proportion=1.0,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=2.5,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.05,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],
        #             max_height_diff=0.05,
        #             x_range=(1.0, 3.0),
        #             y_range=(-1.0, 1.0),
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形4: 反向金字塔楼梯 - 向下走的楼梯（凹陷地形）
        # # ------------------------------------------------------------------
        # "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
        #     proportion=1.0,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=2.5,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.05,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],
        #             max_height_diff=0.05,
        #             x_range=(1.0, 3.0),
        #             y_range=(-1.0, 1.0),
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形5: 离散障碍物 - 随机分布的方形障碍物
        # # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 地形 5: 离散障碍物 - 随机分布的方形障碍物
        # 结构：平地上随机放置凸起的方形障碍物块
        # 用途：训练机器人跨越或绕过随机障碍
        # ------------------------------------------------------------------
        # "boxes": PerlinDiscreteObstaclesTerrainCfg(
        #     proportion=1.0,
        #     obstacle_height_range=(0.05, 0.40),  # 障碍物高度范围（米）
        #     obstacle_width_range=(0.1, 0.4),      # 障碍物宽度范围（米）
        #     num_obstacles=50,                     # 障碍物数量（越多越杂乱）
        #     platform_width=2.5,                   # 中间平台宽度
        #     border_width=1.0,                      # 边界宽度
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],     # 围墙概率 [左，右，前，后]
        #     wall_height=5.0,                      # 围墙高度
        #     wall_thickness=0.05,                  # 围墙厚度
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,           # Perlin 噪声缩放
        #         noise_frequency=20,         # 噪声频率
        #         fractal_octaves=2,          # 分频层数
        #         fractal_lacunarity=2.0,     # 分频 lacunarity
        #         fractal_gain=0.25,          # 分频增益
        #         centering=True,             # 是否居中
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,         # 目标平坦区域数量
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],  # 平坦区域半径选项
        #             max_height_diff=0.05    # 最大高度差（米）
        #         ),
        #     },
        # ),

        # ------------------------------------------------------------------
        # 地形 6: 波浪地形 - 波纹状起伏地形
        # 结构：类似正弦波的起伏地形，有波峰和波谷
        # 用途：训练机器人在不平整地面行走
        # ------------------------------------------------------------------
        # "wave_terrain": PerlinWaveTerrainCfg(
        #     proportion=1.0,
        #     amplitude_range=(0.1, 0.4),      # 波幅范围（米）- 起伏的高度
        #     num_waves=3,                    # 波浪数量- 波峰/波谷的数量
        #     border_width=1.0,               # 边界宽度
        #     wall_prob=[0.0, 0.0, 0.0, 0.0], # 围墙概率 [左，右，前，后]
        #     wall_height=5.0,                # 围墙高度
        #     wall_thickness=0.05,            # 围墙厚度
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,           # Perlin 噪声缩放
        #         noise_frequency=20,         # 噪声频率
        #         fractal_octaves=2,          # 分频层数
        #         fractal_lacunarity=2.0,     # 分频 lacunarity
        #         fractal_gain=0.25,          # 分频增益
        #         centering=True,             # 是否居中
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,         # 目标平坦区域数量
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],  # 平坦区域半径选项
        #             max_height_diff=0.05    # 最大高度差（米）
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形7: 踏脚石 - 需要跳跃的分离石块
        # # ------------------------------------------------------------------
        # "stepping_stones": PerlinSteppingStonesTerrainCfg(
        #     proportion=1.0,
        #     stone_width_range=(0.2, 0.6),
        #     stone_distance_range=(0.3, 0.8),
        #     stone_height_max=0.15,
        #     platform_width=0.8,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.01,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形8: 金字塔斜坡 - 向上倾斜的地形
        # # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 地形 8: 金字塔斜坡 - 向上倾斜的地形（四面往中心倾斜）
        # 结构：四面从边缘向中心向上倾斜，形成截头金字塔
        # 用途：训练机器人侧向行走和斜坡平衡
        # ------------------------------------------------------------------
        # "pyramid_sloped": PerlinPyramidSlopedTerrainCfg(
        #     proportion=1.0,
        #     slope_range=(0.1, 0.5),
        #     platform_width=2.5,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.05,           # Perlin 噪声缩放
        #         noise_frequency=20,         # 噪声频率
        #         fractal_octaves=2,          # 分频层数
        #         fractal_lacunarity=2.0,     # 分频 lacunarity
        #         fractal_gain=0.25,          # 分频增益
        #         centering=True,             # 是否居中
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,         # 目标平坦区域数量
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],  # 平坦区域半径选项
        #             max_height_diff=0.05    # 最大高度差（米）
        #         ),
        #     },
        # ),

        # # ==================================================================
        # #  instinctlab 新增地形（9种）
        # # ==================================================================

        # # ------------------------------------------------------------------
        # # 地形9: 矮墙/栏杆 - 用于跳跃和跨栏
        # # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # 地形 9: 矮墙/栏杆 - 用于跳跃和跨栏
        # 结构：平地上随机放置凸起的矮墙障碍物
        # 用途：训练机器人跨过矮墙或跳跃障碍
        # ------------------------------------------------------------------
        # "parapet": PerlinParapetTerrainCfg(
        #     proportion=1.0,
        #     parapet_height=(0.5),      # 矮墙高度范围（米）- 0.1-0.3米
        #     parapet_length=(0.1,0.3),       # 矮墙长度范围（米）- 0.1-0.3米
        #     parapet_width=None,              # 矮墙宽度（None=用地形宽度）
        #     curved_top_rate=None,            # 顶部曲线比率（None=直角）
        #     border_width=1.0,               # 边界宽度
        #     wall_prob=[0.0, 0.0, 0.0, 0.0], # 围墙概率 [左，右，前，后]
        #     wall_height=5.0,                # 围墙高度
        #     wall_thickness=0.05,            # 围墙厚度
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,           # Perlin 噪声缩放
        #         noise_frequency=20,         # 噪声频率
        #         fractal_octaves=2,          # 分频层数
        #         fractal_lacunarity=2.0,     # 分频 lacunarity
        #         fractal_gain=0.25,          # 分频增益
        #         centering=True,             # 是否居中
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,         # 目标平坦区域数量
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],  # 平坦区域半径选项
        #             max_height_diff=0.05    # 最大高度差（米）
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形10: 沟渠/排水沟 - 两侧有沟的地形
        # # ------------------------------------------------------------------
        # "gutter": PerlinGutterTerrainCfg(
        #     proportion=1.0,
        #     gutter_length=(0.5, 1.5),
        #     gutter_depth=(0.1, 0.3),
        #     gutter_width=None,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形 11: 楼梯先上后下 - 先上坡再下坡
        # # 结构：地面 → 上楼梯 → 中间平台 → 下楼梯 → 地面
        # # 用途：训练机器人上下楼梯的连续过渡能力
        # # ------------------------------------------------------------------
        # "stairs_up_down": PerlinStairsUpDownTerrainCfg(
        #     proportion=1.0,
        #     per_step_height=(0.05, 0.15),   # 每级台阶高度范围（米）
        #     per_step_width=None,            # 台阶宽度（None=用地形宽度）
        #     per_step_length=(0.1, 0.2),      # 每级台阶深度范围（米）- 改小以放入更多台阶
        #     num_steps=(3, 16),               # 台阶数量范围（上/下坡各这么多级）
        #     platform_length=0.8,             # 中间平台长度（米）- 稍微减小
        #     border_width=1.0,               # 边界宽度
        #     wall_prob=[0.0, 0.0, 0.0, 0.0], # 围墙概率 [左，右，前，后]
        #     wall_height=5.0,                # 围墙高度
        #     wall_thickness=0.05,            # 围墙厚度
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,           # Perlin 噪声缩放
        #         noise_frequency=20,         # 噪声频率
        #         fractal_octaves=2,          # 分频层数
        #         fractal_lacunarity=2.0,     # 分频 lacunarity
        #         fractal_gain=0.25,          # 分频增益
        #         centering=True,             # 是否居中
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,         # 目标平坦区域数量
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],  # 平坦区域半径选项
        #             max_height_diff=0.05    # 最大高度差（米）
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形12: 楼梯先下后上 - 先下坡再上坡
        # # ------------------------------------------------------------------
        # "stairs_down_up": PerlinStairsDownUpTerrainCfg(
        #     proportion=1.0,
        #     per_step_height=(0.05, 0.15),
        #     per_step_width=None,
        #     per_step_length=(0.3, 0.5),
        #     num_steps=(3, 6),
        #     platform_length=1.0,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形13: 倾斜墙壁 - 有倾斜墙壁需要绕过的地形
        # # ------------------------------------------------------------------
        # "tilt": PerlinTiltTerrainCfg(
        #     proportion=1.0,
        #     wall_height=(0.2, 0.5),
        #     wall_width=None,
        #     wall_length=(0.3, 0.6),
        #     wall_opening_angle=(20.0, 45.0),
        #     wall_opening_width=(0.5, 1.0),
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # # ------------------------------------------------------------------
        # # 地形14: 倾斜斜面 - 可切换方向的斜面
        # # ------------------------------------------------------------------
        # "tilted_ramp": PerlinTiltedRampTerrainCfg(
        #     proportion=1.0,
        #     tilt_angle=(10.0, 20.0),    # 更小的角度 = 更长的坡
        #     tilt_height=(0.3, 0.6),     # 高度
        #     tilt_width=(1.5, 2.5),      # 宽度
        #     tilt_length=(4.0, 4.8),     # 超长坡（接近地形总长 5 米）
        #     switch_spacing=(2.5, 3.5),  # 更大的切换间距
        #     spacing_curriculum=True,
        #     overlap_size=0.5,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # ------------------------------------------------------------------
        # 地形15: 坡道地形 - 上下坡中间有平地
        # 注意：此地形参数较复杂，暂时注释掉
        # ------------------------------------------------------------------
        # "slope": PerlinSlopeTerrainCfg(
        #     proportion=1.0,
        #     slope_angle=(0.2, 0.5),
        #     per_slope_length=(1.0, 2.0),
        #     platform_length=1.0,
        #     slope_width=2.0,
        #     up_down=True,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),

        # ------------------------------------------------------------------
        # 地形16: 十字石 - 十字形排列的石块
        # ------------------------------------------------------------------
        # "cross_stone": PerlinCrossStoneTerrainCfg(
        #     proportion=1.0,
        #     stone_size=(0.2, 0.5),
        #     stone_height=(0.1, 0.3),
        #     stone_spacing=(0.3, 0.6),
        #     ground_depth=-0.5,
        #     platform_width=1.5,
        #     xy_random_ratio=0.2,
        #     border_width=1.0,
        #     wall_prob=[0.0, 0.0, 0.0, 0.0],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=PerlinPlaneTerrainCfg(
        #         noise_scale=0.02,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05
        #         ),
        #     },
        # ),
        
        
    }),
)


# ========================================================================
#                        地形导入器配置
# ========================================================================
#
# max_init_terrain_level: 初始最大地形难度 (0-10)
#   - 0: 只有最简单的地形 (perlin_rough, perlin_rough_stand)
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
    max_init_terrain_level=10,       # 最大初始难度 (0-10)
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
