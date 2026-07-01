"""Physical terrain presets and a geometry-only terrain catalog.

This file separates two concerns:

1. Geometry catalog:
   Reuses every base terrain defined once in ``shared_terrain_cfg.py`` so we can
   inspect all geometric terrain types without mixing in training-only variants.
2. Physical presets:
   Stores reusable ground material presets (friction / restitution) that can be
   plugged into a TerrainImporterCfg at the task level.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils

from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg

from .shared_terrain_cfg import SHARED_SUB_TERRAINS, TRAINING_SUB_TERRAINS


# Keep the base geometric terrain vocabulary separate from training-only variants.
ALL_SHARED_GEOMETRIC_TERRAIN_NAMES = list(SHARED_SUB_TERRAINS.keys())

# A smaller terrain family focused on studying contact/material effects rather
# than broad geometric coverage. These terrains keep contact patterns diverse
# while avoiding overly task-specific layouts such as atec_d.
PHYSICAL_STUDY_TERRAIN_NAMES = [
    "perlin_rough",
    "wave",
    "pyramid_slope",
    "pyramid_slope_inv",
    "discrete_obstacles",
    "boxes",
    "mesh_boxes",
    "mesh_boxes_dense",
    "stepping_stones",
    "tilt",
    "gutter",
]

PHYSICAL_COMPARISON_MATERIAL_NAMES = [
    "default",
    "low_friction",
    "springy",
    "high_grip",
    "slippery_bouncy",
    "damped_soft_like",
]

PHYSICAL_DYNAMIC_ARENA_NAMES = [
    "stability_dynamic_support",
    "looseness_dynamic_rubble",
    "looseness_dense_small_rubble",
]

PHYSICAL_DISPLAY_TERRAIN_NAMES = PHYSICAL_STUDY_TERRAIN_NAMES + PHYSICAL_DYNAMIC_ARENA_NAMES


def _make_visualization_sub_terrains(names: list[str]) -> dict:
    """Copy terrain cfgs and disable flat patch sampling for pure geometry inspection."""
    sub_terrains = {}
    for idx, name in enumerate(names):
        if name in TRAINING_SUB_TERRAINS:
            source_cfg = TRAINING_SUB_TERRAINS[name]
        elif name in SHARED_SUB_TERRAINS:
            source_cfg = SHARED_SUB_TERRAINS[name]
        else:
            available_names = sorted(set(SHARED_SUB_TERRAINS.keys()) | set(TRAINING_SUB_TERRAINS.keys()))
            raise ValueError(f"Unknown terrain name '{name}'. Available terrains: {available_names}")

        copied_cfg = copy.deepcopy(source_cfg)
        if hasattr(copied_cfg, "flat_patch_sampling"):
            copied_cfg.flat_patch_sampling = None
        sub_terrains[f"terrain_{idx}"] = copied_cfg
    return sub_terrains


# We use a 17 x 1 layout so every base terrain appears exactly once.
GEOMETRIC_ALL_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=7,
    size=(2.0, 2.0),
    border_width=0.05,
    num_rows=len(ALL_SHARED_GEOMETRIC_TERRAIN_NAMES),
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=ALL_SHARED_GEOMETRIC_TERRAIN_NAMES,
    sub_terrains=_make_visualization_sub_terrains(ALL_SHARED_GEOMETRIC_TERRAIN_NAMES),
)


PHYSICAL_STUDY_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=17,
    size=(2.5, 2.5),
    border_width=0.05,
    num_rows=len(PHYSICAL_STUDY_TERRAIN_NAMES),
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=PHYSICAL_STUDY_TERRAIN_NAMES,
    sub_terrains=_make_visualization_sub_terrains(PHYSICAL_STUDY_TERRAIN_NAMES),
)


# Curriculum-style layout for physical-property studies:
# columns = terrain families, rows = difficulty levels from easy to hard.
PHYSICAL_STUDY_CURRICULUM_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=19,
    size=(2.5, 2.5),
    border_width=0.05,
    num_rows=8,
    num_cols=len(PHYSICAL_STUDY_TERRAIN_NAMES),
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    deterministic_curriculum_rows=True,
    terrain_layout=PHYSICAL_STUDY_TERRAIN_NAMES,
    sub_terrains=_make_visualization_sub_terrains(PHYSICAL_STUDY_TERRAIN_NAMES),
)


# These are not attached directly to the terrain geometry here because the
# terrain mesh and the ground material are combined later inside TerrainImporterCfg.
PHYSICAL_MATERIAL_PRESETS = {
    "default": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    "low_friction": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.15,
        dynamic_friction=0.12,
        restitution=0.0,
    ),
    "springy": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.9,
        dynamic_friction=0.8,
        restitution=0.35,
    ),
    "high_grip": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.6,
        dynamic_friction=1.4,
        restitution=0.0,
    ),
    "slippery_bouncy": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.18,
        dynamic_friction=0.12,
        restitution=0.30,
    ),
    "damped_soft_like": sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.75,
        dynamic_friction=0.65,
        restitution=0.08,
    ),
}


# This is a lightweight catalog rather than a fully bound TerrainImporterCfg.
# The actual importer chooses one geometry cfg and one material preset at runtime.
PHYSICAL_TERRAIN_COLLECTIONS = {
    "shared_all": {
        "terrain_cfg": GEOMETRIC_ALL_TERRAINS_CFG,
        "default_material": "default",
        "description": "All shared geometric terrains once each, for broad inspection.",
    },
    "physical": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "default",
        "description": "Contact/material study set with moderate geometric diversity.",
    },
    "physical_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "default",
        "description": "Physical study set with rows increasing in terrain difficulty.",
    },
    "physical_low_friction": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "low_friction",
        "description": "Same physical study set, tuned for slip-prone contact.",
    },
    "physical_low_friction_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "low_friction",
        "description": "Low-friction physical study set with rows increasing in terrain difficulty.",
    },
    "physical_springy": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "springy",
        "description": "Same physical study set, tuned for higher rebound.",
    },
    "physical_springy_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "springy",
        "description": "Springy physical study set with rows increasing in terrain difficulty.",
    },
    "physical_high_grip": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "high_grip",
        "description": "Same physical study set, tuned for aggressive traction.",
    },
    "physical_high_grip_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "high_grip",
        "description": "High-grip physical study set with rows increasing in terrain difficulty.",
    },
    "physical_slippery_bouncy": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "slippery_bouncy",
        "description": "Same physical study set, combining low friction with rebound.",
    },
    "physical_slippery_bouncy_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "slippery_bouncy",
        "description": "Slippery-and-bouncy physical study set with rows increasing in terrain difficulty.",
    },
    "physical_damped_soft_like": {
        "terrain_cfg": PHYSICAL_STUDY_TERRAINS_CFG,
        "default_material": "damped_soft_like",
        "description": "Same physical study set, approximating softer and more damped contact.",
    },
    "physical_damped_soft_like_curriculum": {
        "terrain_cfg": PHYSICAL_STUDY_CURRICULUM_CFG,
        "default_material": "damped_soft_like",
        "description": "Damped-soft-like physical study set with rows increasing in terrain difficulty.",
    },
}
