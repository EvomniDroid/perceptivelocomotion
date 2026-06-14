"""
共享地形配置文件
所有地形配置集中在这里定义，vis.py 和 parkour_env_cfg.py 都引用此文件
修改地形只需改这一处
"""

from isaaclab.terrains import FlatPatchSamplingCfg
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains import (
    AtecDPitAndPlatformTerrainCfg,
    PerlinBowlPitTerrainCfg,
    PerlinPitTerrainCfg,
    PerlinPlaneTerrainCfg,
    PerlinSquareGapTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinMeshRandomMultiBoxTerrainCfg,
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
    PerlinCircleTrackTerrainCfg,
)


def _inject_name_to_cfgs(sub_terrains):
    """为每个子地形配置注入 name 属性（key 值），但不覆盖 terrain_type"""
    for k, v in sub_terrains.items():
        if not hasattr(v, 'terrain_type') or v.terrain_type is None:
            try:
                setattr(v, "name", k)
            except Exception:
                object.__setattr__(v, "name", k)
        else:
            try:
                setattr(v, "name", k)
            except Exception:
                object.__setattr__(v, "name", k)
    return sub_terrains


def _terrain_with_sampling(name, enabled=True, num_patches=3):
    """
    创建带或不带 flat_patch_sampling 的地形配置

    Args:
        name: 地形名称（在 SHARED_SUB_TERRAINS 中的 key）
        enabled: 是否启用 flat_patch_sampling 采样
        num_patches: 采样数量（仅在 enabled=True 时使用）

    Returns:
        地形配置对象
    """
    import copy
    cfg = copy.deepcopy(SHARED_SUB_TERRAINS[name])
    return cfg


def _terrain_layout_to_ordered_dict(terrain_layout, expected_count=None):
    """
    根据名字列表生成有序的 sub_terrains 字典，支持重复地形名

    Args:
        terrain_layout: 地形名字列表，如 ["perlin_rough", "pyramid_stairs", ...]
                       每个元素对应一个格子，重复的名字表示重复该地形
        expected_count: 期望的元素数量（num_rows * num_cols），用于验证

    Returns:
        OrderedDict，key 是 terrain_0, terrain_1, ...（按顺序），value 是对应配置

    Raises:
        ValueError: 如果 terrain_layout 长度与 expected_count 不匹配
    """
    from collections import OrderedDict

    actual_count = len(terrain_layout)
    if expected_count is not None and actual_count != expected_count:
        raise ValueError(
            f"terrain_layout 元素数量不匹配！"
            f"提供了 {actual_count} 个，但需要 {expected_count} 个 (num_rows * num_cols)。"
            f"当前 num_rows * num_cols = {expected_count}，请确保列表长度一致。"
        )

    result = OrderedDict()
    for idx, name in enumerate(terrain_layout):
        if name not in SHARED_SUB_TERRAINS:
            raise ValueError(f"未知的地形名称: '{name}'。可用的地形: {list(SHARED_SUB_TERRAINS.keys())}")
        key = f"terrain_{idx}"
        result[key] = _terrain_with_sampling(name, enabled=False)
    return result


# ====================================================================
# 共享的 sub_terrains 字典 - 所有地形类型定义
# ====================================================================to 1 or 2 rather than 0), together with this flag, to improve the accuracy of velocity updates.
SHARED_SUB_TERRAINS = _inject_name_to_cfgs({

    # 地形1: Perlin噪声粗糙地面
    # 特点：基于Perlin噪声生成的起伏地面，表面相对平坦
    "perlin_rough": PerlinPlaneTerrainCfg(
        proportion=1.0,                     # 该地形占子地块的比例
        noise_scale=[0.0, 0.02],            # 噪声缩放系数（低频，高频）- 改小使地面更平坦
        noise_frequency=20,                  # 噪声频率，越大地形细节越密
        fractal_octaves=2,                   # 分形噪声的八度数，越多细节越丰富
        fractal_lacunarity=2.0,               # 分形间隙度，控制噪声层间频率比
        fractal_gain=0.25,                   # 分形增益，控制每层噪声的贡献权重
        centering=True,                       # 是否将噪声中心化（使地面在0高度附近）
        wall_prob=[0.0, 0.0, 0.0, 0.0],     # 四面墙出现的概率[前，后，左，右]
        wall_height=5.0,                     # 墙的高度
        wall_thickness=0.05,                 # 墙的厚度
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,              # 采样平坦区域的数量
                patch_radius=[0.05, 0.10, 0.15, 0.20],  # 采样半径范围
                max_height_diff=0.15         # 最大高度差阈值（判断是否平坦）- 放宽
            ),
        },
    ),

    # 地形2: 方形坑洞
    # 特点：地面上有随机分布的方形深坑，机器人需要避开
    "square_gaps": PerlinSquareGapTerrainCfg(
        proportion=1.0,
        gap_distance_range=(0.1, 0.7),       # 相邻坑洞间的距离范围（米）
        gap_depth=(0.8, 1.2),                # 坑洞深度范围（米）
        platform_width=1.5,                   # 坑洞间平台宽度（米）
        border_width=0.3,                    # 坑洞边缘宽度（米）
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(-0.5, 0.5),
                y_range=(-0.5, 0.5),
            ),
        },
    ),

    # 地形 3: 金字塔楼梯
    "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.15, 0.35),
        step_width=0.25,
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=3,
                patch_radius=[0.05],
                max_height_diff=0.5,
                x_range=(-0.5, 0.5),
                y_range=(-0.5, 0.5),
            ),
        },
    ),

    # 地形 4: 反向金字塔楼梯
    "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.15, 0.35),
        step_width=0.25,
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.5,
                x_range=(-0.5, 0.5),
                y_range=(-0.5, 0.5),
            ),
        },
    ),

    # 地形 5: 离散障碍物
    "discrete_obstacles": PerlinDiscreteObstaclesTerrainCfg(
        proportion=1.0,
        num_obstacles=20,
        obstacle_height_range=(0.05, 0.45),
        obstacle_width_range=(0.1, 0.4),
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 6: 金字塔斜坡
    "pyramid_slope": PerlinPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(0.1, 0.3),
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 7: 反向金字塔斜坡
    "pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(0.1, 0.25),
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 8: 波浪地形
    "wave": PerlinWaveTerrainCfg(
        proportion=1.0,
        amplitude_range=(0.1, 0.3),
        num_waves=3,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 9: 踏脚石
    "stepping_stones": PerlinSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_width_range=(0.1, 0.5),
        stone_height_max=0.1,
        stone_distance_range=(0.15, 0.35),
        platform_width=0.0,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        holes_depth=-0.3,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 10: 矮墙/栏杆
    "parapet": PerlinParapetTerrainCfg(
        proportion=1.0,
        parapet_height=(0.1, 0.3),
        parapet_length=(0.1, 0.3),
        parapet_width=None,
        curved_top_rate=None,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 11: 排水沟/檐沟
    "gutter": PerlinGutterTerrainCfg(
        proportion=1.0,
        gutter_length=(0.5, 1.5),
        gutter_depth=(0.1, 0.3),
        gutter_width=0.3,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 12: 上下楼梯
    "stairs_up_down": PerlinStairsUpDownTerrainCfg(
        proportion=1.0,
        per_step_height=(0.05, 0.15),
        per_step_width=0.25,
        per_step_length=(0.15, 0.15),
        num_steps=(8, 10),
        platform_length=0.3,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 13: 下上楼梯
    "stairs_down_up": PerlinStairsDownUpTerrainCfg(
        proportion=1.0,
        per_step_height=(0.05, 0.15),
        per_step_width=0.25,
        per_step_length=(0.15, 0.15),
        num_steps=(8, 10),
        platform_length=0.3,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 14: 倾斜地面（墙壁开口）
    "tilt": PerlinTiltTerrainCfg(
        proportion=1.0,
        wall_height=(0.1, 0.3),
        wall_width=0.2,
        wall_length=(0.5, 1.0),
        wall_opening_angle=(10, 30),
        wall_opening_width=(0.3, 0.6),
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.3,
            ),
        },
    ),

    # 地形 16: 倾斜坡道
    "tilted_ramp": PerlinTiltedRampTerrainCfg(
        proportion=1.0,
        tilt_angle=(5, 15),
        tilt_height=(0.1, 0.3),
        tilt_width=(0.5, 1.0),
        tilt_length=(1.0, 1.5),
        switch_spacing=(0.5, 1.0),
        spacing_curriculum=False,
        overlap_size=0.2,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.3,
            ),
        },
    ),

    # 地形 12: 十字石
    "cross_stone": PerlinCrossStoneTerrainCfg(
        proportion=1.0,
        stone_size=(0.2, 0.4),
        stone_height=(0.1, 0.2),
        stone_spacing=(0.15, 0.3),
        platform_width=0.6,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        xy_random_ratio=0.2,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.5,
            ),
        },
    ),

    "atec_d": AtecDPitAndPlatformTerrainCfg(
        proportion=0.10,
        border_width=1.0,
        pit_depth=1.0,
        pit_width_range=(1.3, 1.4),
        box_size=(0.8, 1.0, 0.6),
        box_pos=(3.0, 5.6),
    ),
})


# ====================================================================
# FLAT_TRAINING_SUB_TERRAINS - 仅平地的训练地形（阶段1）
# 用于两阶段训练：先让机器人在平地上学会走路，再引入复杂地形
# ====================================================================

FLAT_TRAINING_SUB_TERRAINS = _inject_name_to_cfgs({
    "perlin_rough": PerlinPlaneTerrainCfg(
        proportion=0.50,
        noise_scale=[0.0, 0.02],
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "perlin_rough_stand": PerlinPlaneTerrainCfg(
        proportion=0.50,
        noise_scale=[0.0, 0.02],
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
})


# ====================================================================
# vis.py 用的地形配置 - 用于可视化
# ====================================================================
# MY_TERRAIN_CFG 用于 vis.py，显示 11 种有效地形
# 使用 terrain_layout 指定每个格子的地形类型（按名字）
# num_rows=6, num_cols=4 -> 24 个格子，terrain_layout 需要 24 个元素
# 布局设计：前两行展示所有 11 种有效地形，后面用重复地形填充

MY_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=0,
    size=(2.0, 2.0),
    border_width=0.05,
    num_rows=6,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=[
        # 第 1 行 (y=10~12): 终点区域
        "perlin_rough",        # 1 (x=0~2)
        "pyramid_slope",       # 2 (x=2~4)
        "pyramid_stairs",      # 3 (x=4~6)
        "perlin_rough",        # 4 (x=6~8)
        # 第 2 行 (y=8~10)
        "pyramid_stairs",      # 5 (x=0~2)
        "pyramid_slope",       # 6 (x=2~4)
        "discrete_obstacles",  # 7 (x=4~6)
        "wave",                # 8 (x=6~8)
        # 第 3 行 (y=6~8): 障碍区域
        "discrete_obstacles",  # 9 (x=0~2)
        "stepping_stones",     # 10 (x=2~4)
        "pyramid_stairs",      # 11 (x=4~6)
        "cross_stone",         # 12 (x=6~8)
        # 第 4 行 (y=4~6)
        "pyramid_stairs_inv",  # 13 (x=0~2)
        "pyramid_slope",       # 14 (x=2~4)
        "discrete_obstacles",  # 15 (x=4~6)
        "wave",                # 16 (x=6~8)
        # 第 5 行 (y=2~4)
        "pyramid_stairs",      # 17 (x=0~2)
        "pyramid_slope",       # 18 (x=2~4)
        "wave",                # 19 (x=4~6)
        "perlin_rough",        # 20 (x=6~8)
        # 第 6 行 (y=0~2): 起点区域 - 出生点 (1,1)
        "perlin_rough",        # 21 (x=0~2)
        "discrete_obstacles",  # 22 (x=2~4)
        "pyramid_stairs",      # 23 (x=4~6)
        "pyramid_slope",       # 24 (x=6~8)
    ],
    sub_terrains=_terrain_layout_to_ordered_dict([
        "perlin_rough", "pyramid_slope", "pyramid_stairs", "perlin_rough",
        "pyramid_stairs", "pyramid_slope", "discrete_obstacles", "wave",
        "discrete_obstacles", "stepping_stones", "pyramid_stairs", "cross_stone",
        "pyramid_stairs_inv", "pyramid_slope", "discrete_obstacles", "wave",
        "pyramid_stairs", "pyramid_slope", "wave", "perlin_rough",
        "perlin_rough", "discrete_obstacles", "pyramid_stairs", "pyramid_slope",
    ], expected_count=6*4),
)

# 恢复 flat_patch_sampling 但用更宽松的参数
for name, cfg in MY_TERRAIN_CFG.sub_terrains.items():
    if hasattr(cfg, 'flat_patch_sampling'):
        cfg.flat_patch_sampling = {
            "target": FlatPatchSamplingCfg(
                num_patches=3,
                patch_radius=[0.05],
                max_height_diff=1.0,  # 增加高度差容忍度
            ),
        }


# ====================================================================
# FRONTIER_TEST_TERRAIN_CFG - 用于测试前沿点导航的简单地形
# 特点：
#   - 简单地形（平地+简单斜坡）
#   - 摔倒率预设在 LocalFallRateMap 中手动指定
#   - 地图布局：6行4列，每格 2m x 2m，总共 12m x 8m
#   - 摔倒率预设：起点安全(0.0)，随距离和位置增加
# ====================================================================

FRONTIER_TEST_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=42,
    size=(2.0, 2.0),
    border_width=0.05,
    num_rows=6,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=[
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
    ],
    sub_terrains=_terrain_layout_to_ordered_dict([
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
    ], expected_count=6*4),
)

# 恢复 flat_patch_sampling 但用更宽松的参数
for name, cfg in FRONTIER_TEST_TERRAIN_CFG.sub_terrains.items():
    if hasattr(cfg, 'flat_patch_sampling'):
        cfg.flat_patch_sampling = {
            "target": FlatPatchSamplingCfg(
                num_patches=3,
                patch_radius=[0.05],
                max_height_diff=1.0,  # 增加高度差容忍度
            ),
        }


# ====================================================================
# ROUGH_TERRAINS_CFG 用的子地形 - 用于训练
# ====================================================================
# 训练时使用的地形子集，包含课程学习比例设置

TRAINING_SUB_TERRAINS = _inject_name_to_cfgs({
    "perlin_rough": PerlinPlaneTerrainCfg(
        proportion=0.05,
        noise_scale=[0.0, 0.1],
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "perlin_rough_stand": PerlinPlaneTerrainCfg(
        proportion=0.05,
        noise_scale=[0.0, 0.1],
        noise_frequency=20,
        fractal_octaves=2,
        fractal_lacunarity=2.0,
        fractal_gain=0.25,
        centering=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "square_gaps": PerlinSquareGapTerrainCfg(
        proportion=0.08,
        gap_distance_range=(0.1, 0.7),
        gap_depth=(0.4, 0.6),
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
        proportion=0.12,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pyramid_stairs_high": PerlinPyramidStairsTerrainCfg(
        proportion=0.03,
        step_height_range=(0.05, 0.32),
        step_width=1.5,
        platform_width=4.0,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
        proportion=0.12,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pyramid_stairs_inv_high": PerlinInvertedPyramidStairsTerrainCfg(
        proportion=0.03,
        step_height_range=(0.05, 0.32),
        step_width=1.5,
        platform_width=4.0,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "boxes": PerlinDiscreteObstaclesTerrainCfg(
        proportion=0.09,
        num_obstacles=20,
        obstacle_height_mode="fixed",
        obstacle_width_range=(0.8, 1.5),
        obstacle_height_range=(0.05, 0.45),
        platform_width=1.5,
        border_width=0.1,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.05,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "mesh_boxes": PerlinMeshRandomMultiBoxTerrainCfg(
        proportion=0.14,
        box_height_mean=(0.1, 0.4),
        box_height_range=0.05,
        box_length_mean=0.4,
        box_length_range=0.1,
        box_width_mean=0.4,
        box_width_range=0.1,
        platform_width=1.5,
        generation_ratio=0.3,
        no_perlin_at_obstacle=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(num_patches=10, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.1),
        },
    ),
    "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
        proportion=0.06,
        slope_range=(0.0, 0.7),
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.00,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "raised_mound": PerlinBowlPitTerrainCfg(
        proportion=0.18,
        pit_depth=(0.05, 1.0),
        pit_radius=(1.0, 1.0),
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pit_crater": PerlinPitTerrainCfg(
        proportion=0.18,
        pit_depth=(0.05, 1.0),
        pit_radius=(1.0, 1.0),
        raise_surrounding_ground=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "raised_mound": PerlinBowlPitTerrainCfg(
        proportion=0.18,
        pit_depth=(0.05, 1.0),
        pit_radius=(1.0, 1.0),
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "pit_crater": PerlinPitTerrainCfg(
        proportion=0.18,
        pit_depth=(0.05, 1.0),
        pit_radius=(1.0, 1.0),
        raise_surrounding_ground=True,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(
            noise_scale=0.02,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
        ),
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
                x_range=(3.7, 3.7),
                y_range=(-0.0, 0.0),
            ),
        },
    ),
    "wave": PerlinWaveTerrainCfg(
        proportion=0.06,
        amplitude_range=(0.1, 0.3),
        num_waves=3,
        border_width=0.0,
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.1,
            ),
        },
    ),
})


# ====================================================================
# ATEC_D_TERRAIN_CFG - 用于测试 ATEC D 赛题地形 (坑+平台)
# 特点：
#   - 仅 1 格地形, 使用 atec_d 坑+平台地形
#   - 用于部署测试，验证机器人是否能跳过坑或走上高平台
# ====================================================================

ATEC_D_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=42,
    size=(12.0, 8.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=["atec_d"],
    sub_terrains=_terrain_layout_to_ordered_dict(["atec_d"], expected_count=1),
)

for name, cfg in ATEC_D_TERRAIN_CFG.sub_terrains.items():
    cfg.flat_patch_sampling = {
        "target": FlatPatchSamplingCfg(
            num_patches=1,
            patch_radius=[0.05],
            max_height_diff=10.0,
            x_range=(0.0, 3.0),
            y_range=(3.0, 5.0),
        ),
    }
