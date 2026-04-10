"""
共享地形配置文件
所有地形配置集中在这里定义，vis.py 和 parkour_env_cfg.py 都引用此文件
修改地形只需改这一处
"""

from isaaclab.terrains import FlatPatchSamplingCfg
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains import (
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


def _terrain_with_sampling(name, enabled=True, num_patches=10):
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
        platform_width=0.8,
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
                max_height_diff=0.3,
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
        platform_width=0.8,
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
                max_height_diff=0.3,
                x_range=(-0.5, 0.5),
                y_range=(-0.5, 0.5),
            ),
        },
    ),

    # 地形5: 离散障碍物
    "discrete_obstacles": PerlinDiscreteObstaclesTerrainCfg(
        proportion=1.0,
        num_obstacles=20,
        obstacle_height_range=(0.05, 0.45),
        obstacle_width_range=(0.1, 0.4),
        platform_width=1.5,
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
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 6: 金字塔斜坡
    "pyramid_slope": PerlinPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(5, 10),
        platform_width=1.0,
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
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 7: 反向金字塔斜坡
    "pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(5, 15),
        platform_width=1.0,
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
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形8: 波浪地形
    "wave": PerlinWaveTerrainCfg(
        proportion=1.0,
        amplitude_range=(0.1, 0.3),
        num_waves=3,
        border_width=0.3,
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

    # 地形9: 踏脚石
    "stepping_stones": PerlinSteppingStonesTerrainCfg(
        proportion=1.0,
        stone_width_range=(0.1, 0.5),
        stone_height_max=0.1,
        stone_distance_range=(0.15, 0.35),
        platform_width=1.5,
        border_width=0.3,
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

    # 地形10: 矮墙/栏杆
    "parapet": PerlinParapetTerrainCfg(
        proportion=1.0,
        parapet_height=(0.1, 0.3),
        parapet_length=(0.1, 0.3),
        parapet_width=None,
        curved_top_rate=None,
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
        border_width=0.3,
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
        wall_prob=[0.8, 0.8, 0.8, 0.8],
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

    # 地形 15: 倾斜坡道（参数不适合 2x2 terrain，暂用 perlin_rough）
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
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形 16: 十字石
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
                max_height_diff=0.1,
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
        # 第 1 行：4 种地形
        "perlin_rough",        # 粗糙地面
        "square_gaps",         # 方形坑洞
        "pyramid_stairs",      # 金字塔楼梯（上）
        "pyramid_stairs_inv",  # 金字塔楼梯（下）
        # 第 2 行：4 种地形
        "discrete_obstacles",  # 离散障碍物
        "pyramid_slope",       # 金字塔斜坡
        "pyramid_slope_inv",   # 反向金字塔斜坡
        "wave",                # 波浪地形
        # 第 3 行：4 种地形
        "stepping_stones",     # 踏脚石
        "parapet",             # 矮墙
        "gutter",              # 排水沟
        "cross_stone",         # 十字石
        # 第 4 行：4 种地形
        "stairs_up_down",      # 上下楼梯
        "stairs_down_up",      # 下上楼梯
        "tilt",                # 倾斜墙壁
        "tilted_ramp",         # 倾斜坡道
        # 第 5-6 行：用粗糙地面填充
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
    ],
    sub_terrains=_terrain_layout_to_ordered_dict([
        "perlin_rough", "square_gaps", "pyramid_stairs", "pyramid_stairs_inv", "discrete_obstacles", "pyramid_slope",
        "pyramid_slope_inv", "wave", "stepping_stones", "parapet", "gutter", "cross_stone",
        "stairs_up_down", "stairs_down_up", "tilt", "tilted_ramp",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
        "perlin_rough", "perlin_rough", "perlin_rough", "perlin_rough",
    ], expected_count=6*4),
)


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
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        wall_prob=[0.3, 0.3, 0.3, 0.3],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(
                num_patches=10, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.1
            ),
        },
    ),
    "square_gaps": PerlinSquareGapTerrainCfg(
        proportion=0.10,
        gap_distance_range=(0.1, 0.7),
        gap_depth=(0.4, 0.6),
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.15,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.10,
        step_height_range=(0.05, 0.45),
        step_width=1.5,
        platform_width=4.0,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.15,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.10,
        step_height_range=(0.05, 0.45),
        step_width=1.5,
        platform_width=4.0,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.10,
        num_obstacles=20,
        obstacle_height_mode="fixed",
        obstacle_width_range=(0.8, 1.5),
        obstacle_height_range=(0.05, 0.45),
        platform_width=1.5,
        border_width=0.1,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
        proportion=0.10,
        box_height_mean=(0.1, 0.4),
        box_height_range=0.05,
        box_length_mean=0.4,
        box_length_range=0.1,
        box_width_mean=0.4,
        box_width_range=0.1,
        platform_width=1.5,
        generation_ratio=0.3,
        no_perlin_at_obstacle=True,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
        wall_height=5.0,
        wall_thickness=0.05,
        flat_patch_sampling={
            "target": FlatPatchSamplingCfg(num_patches=10, patch_radius=[0.05, 0.10, 0.15], max_height_diff=0.1),
        },
    ),
    "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
        proportion=0.10,
        slope_range=(0.0, 0.7),
        platform_width=1.5,
        border_width=0.3,
        wall_prob=[0.3, 0.3, 0.3, 0.3],
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
})
