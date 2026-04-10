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
    """为每个子地形配置注入 name 属性（key 值）"""
    for k, v in sub_terrains.items():
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
        地形配置对象，如果 enabled=False 则 flat_patch_sampling=None
    """
    import copy
    cfg = copy.deepcopy(SHARED_SUB_TERRAINS[name])

    if enabled:
        cfg.flat_patch_sampling = {
            "target": FlatPatchSamplingCfg(
                num_patches=num_patches,
                patch_radius=[0.05, 0.10, 0.15, 0.20],
                max_height_diff=0.15,
            ),
        }
    else:
        cfg.flat_patch_sampling = None

    return cfg


# ====================================================================
# 共享的 sub_terrains 字典 - 所有地形类型定义
# ====================================================================
# 这个字典包含所有可用的子地形类型
# vis.py 和 parkour_env_cfg.py 都引用此字典

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
                x_range=(1.0, 3.0),          # 采样区域x范围
                y_range=(-1.0, 1.0),          # 采样区域y范围
            ),
        },
    ),

    # 地形3: 金字塔楼梯
    # 特点：金字塔形状的楼梯，从外向内逐级上升/下降
    "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.23),      # 每级台阶高度范围（米）
        step_width=0.3,                       # 台阶宽度（米）
        platform_width=1.5,                   # 金字塔顶部平台宽度（米）
        border_width=0.3,                    # 边缘宽度（米）
        wall_prob=[0.0, 0.0, 0.0, 0.0],
        wall_height=5.0,
        wall_thickness=0.05,
        perlin_cfg=PerlinPlaneTerrainCfg(    # 楼梯表面叠加的Perlin噪声
            noise_scale=0.05,                 # 噪声缩放（表面粗糙度）
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
                x_range=(1.0, 3.0),
                y_range=(-1.0, 1.0),
            ),
        },
    ),

    # 地形4: 反向金字塔楼梯
    "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
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
                x_range=(1.0, 3.0),
                y_range=(-1.0, 1.0),
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

    # 地形6: 金字塔斜坡
    "pyramid_slope": PerlinPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(20, 30),
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

    # 地形7: 反向金字塔斜坡
    "pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(20, 30),
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

    # 地形11: 排水沟/檐沟
    "gutter": PerlinGutterTerrainCfg(
        proportion=1.0,
        gutter_length=(0.5, 1.5),
        gutter_depth=(0.0, 0.0),
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

    # 地形12: 上下楼梯
    "stairs_up_down": PerlinStairsUpDownTerrainCfg(
        proportion=1.0,
        per_step_height=(0.05, 0.15),
        per_step_width=0.25,
        per_step_length=(0.25, 0.25),
        num_steps=(14, 18),
        platform_length=1.5,
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

    # 地形13: 下上楼梯
    "stairs_down_up": PerlinStairsDownUpTerrainCfg(
        proportion=1.0,
        per_step_height=(0.05, 0.15),
        per_step_width=0.25,
        per_step_length=(0.25, 0.25),
        num_steps=(14, 18),
        platform_length=1.5,
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

    # 地形14: 倾斜地面
    "tilt": PerlinTiltTerrainCfg(
        proportion=1.0,
        wall_height=(0.1, 0.3),
        wall_width=0.2,
        wall_length=(1.0, 3.0),
        wall_opening_angle=(30, 60),
        wall_opening_width=(0.5, 1.5),
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
                max_height_diff=0.1,
            ),
        },
    ),

    # 地形15: 倾斜坡道
    "tilted_ramp": PerlinTiltedRampTerrainCfg(
        proportion=1.0,
        tilt_angle=(15, 25),
        tilt_height=(0.5, 1.0),
        tilt_width=(1.0, 2.0),
        tilt_length=(3.0, 5.0),
        switch_spacing=(2.0, 4.0),
        spacing_curriculum=True,
        overlap_size=0.5,
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

    # 地形16: 交叉石
    "cross_stone": PerlinCrossStoneTerrainCfg(
        proportion=1.0,
        stone_size=(0.3, 0.6),
        stone_height=(0.1, 0.2),
        stone_spacing=(0.2, 0.4),
        platform_width=1.5,
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
# MY_TERRAIN_CFG 用于 vis.py，显示选择的6种地形

MY_TERRAIN_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=0,
    size=(2.0, 4.0),
    border_width=0.05,
    num_rows=4,
    num_cols=3,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "perlin_rough": _terrain_with_sampling("perlin_rough", enabled=True, num_patches=10),
        "square_gaps": _terrain_with_sampling("square_gaps", enabled=False),
        "pyramid_stairs": _terrain_with_sampling("pyramid_stairs", enabled=False),
        "pyramid_stairs_inv": _terrain_with_sampling("pyramid_stairs_inv", enabled=False),
        "discrete_obstacles": _terrain_with_sampling("discrete_obstacles", enabled=False),
        "pyramid_slope": _terrain_with_sampling("pyramid_slope", enabled=False),
        "pyramid_slope_inv": _terrain_with_sampling("pyramid_slope_inv", enabled=False),
        "wave": _terrain_with_sampling("wave", enabled=False),
        "stepping_stones": _terrain_with_sampling("stepping_stones", enabled=False),
        "parapet": _terrain_with_sampling("parapet", enabled=False),
        "gutter": _terrain_with_sampling("gutter", enabled=False),
    },
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
