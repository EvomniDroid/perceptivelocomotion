from __future__ import annotations

import random

import isaaclab.sim as sim_utils
import omni.usd
from pxr import Gf, UsdGeom

from .physical_terrain_cfg import PHYSICAL_MATERIAL_PRESETS, PHYSICAL_DYNAMIC_ARENA_NAMES

PHYSICAL_DYNAMIC_ARENA_TILE_SIZE = (4.0, 4.0)
PHYSICAL_DYNAMIC_ARENA_GAP_Y = 1.0


def set_root_translation(prim_path: str, translation: tuple[float, float, float]) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim path not found for translation: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    translate_attr = prim.GetAttribute("xformOp:translate")
    if translate_attr.IsValid():
        UsdGeom.XformOp(translate_attr).Set(Gf.Vec3d(*translation))
    else:
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))


def spawn_primitive(prim_path: str, cfg, translation: tuple[float, float, float]) -> None:
    cfg.func(prim_path, cfg)
    set_root_translation(prim_path, translation)


def make_static_cuboid_cfg(size: tuple[float, float, float], material_name: str, color: tuple[float, float, float]):
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=PHYSICAL_MATERIAL_PRESETS[material_name],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
    )


def make_dynamic_cuboid_cfg(
    size: tuple[float, float, float],
    material_name: str,
    color: tuple[float, float, float],
    mass: float,
):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=PHYSICAL_MATERIAL_PRESETS[material_name],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85),
    )


def make_dynamic_cylinder_cfg(
    radius: float,
    height: float,
    material_name: str,
    color: tuple[float, float, float],
    mass: float,
):
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis="Y",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=PHYSICAL_MATERIAL_PRESETS[material_name],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )


PHYSICAL_DYNAMIC_ARENA_TILE_SIZE = (4.0, 4.0)


def default_row_centers_x(num_rows: int, tile_length_x: float) -> list[float]:
    return [row_idx * tile_length_x for row_idx in range(num_rows)]


def dynamic_arena_center_y(arena_index: int, num_static_cols: int, static_spacing_y: float) -> float:
    static_y_max = ((num_static_cols - 1) / 2) * static_spacing_y
    arena_half_width = 0.5 * PHYSICAL_DYNAMIC_ARENA_TILE_SIZE[1]
    return static_y_max + PHYSICAL_DYNAMIC_ARENA_GAP_Y + arena_half_width + arena_index * (
        PHYSICAL_DYNAMIC_ARENA_TILE_SIZE[1] + PHYSICAL_DYNAMIC_ARENA_GAP_Y
    )


def _resolve_row_entries(row_centers_x: list[float], row_indices: list[int] | None) -> list[tuple[int, float]]:
    if row_indices is None:
        return list(enumerate(row_centers_x))
    return [(row_idx, row_centers_x[row_idx]) for row_idx in row_indices]


def spawn_looseness_dynamic_rubble_column(
    root_path: str,
    material_name: str,
    y_center: float,
    row_centers_x: list[float],
    row_indices: list[int] | None = None,
) -> None:
    floor_color = (0.62, 0.62, 0.62)
    wall_color = (0.46, 0.46, 0.46)
    rubble_color = (0.56, 0.56, 0.56)
    floor_thickness = 0.08
    wall_thickness = 0.12
    wall_height = 1.50
    row_entries = _resolve_row_entries(row_centers_x, row_indices)
    for row_idx, x_center in row_entries:
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(len(row_centers_x) - 1, 1)
        floor_cfg = make_static_cuboid_cfg((4.00, 4.00, floor_thickness), material_name, floor_color)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.5 * floor_thickness))
        wall_cfg = make_static_cuboid_cfg((4.00, wall_thickness, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_front", wall_cfg, (x_center, y_center + 2.00, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_back", wall_cfg, (x_center, y_center - 2.00, 0.5 * wall_height))
        wall_side_cfg = make_static_cuboid_cfg((wall_thickness, 4.00, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_left", wall_side_cfg, (x_center - 2.00, y_center, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_right", wall_side_cfg, (x_center + 2.00, y_center, 0.5 * wall_height))

        rng = random.Random(1000 + row_idx * 37 + len(material_name))
        object_count = 540 + int(round(330 * difficulty))
        for obj_idx in range(object_count):
            px = x_center + rng.uniform(-1.55, 1.55)
            py = y_center + rng.uniform(-1.55, 1.55)
            if obj_idx % 2 == 0:
                radius = rng.uniform(0.0525, 0.0975)
                height = rng.uniform(0.075, 0.15)
                cfg = make_dynamic_cylinder_cfg(radius, height, material_name, rubble_color, mass=0.04)
                spawn_primitive(f"{tile_root}/rubble_cyl_{obj_idx}", cfg, (px, py, radius + 0.01))
            else:
                sx = rng.uniform(0.0675, 0.135)
                sy = rng.uniform(0.0675, 0.135)
                sz = rng.uniform(0.0375, 0.0825)
                cfg = make_dynamic_cuboid_cfg((sx, sy, sz), material_name, rubble_color, mass=0.025)
                spawn_primitive(f"{tile_root}/rubble_box_{obj_idx}", cfg, (px, py, 0.5 * sz + 0.01))


def spawn_looseness_dense_small_rubble_column(
    root_path: str,
    material_name: str,
    y_center: float,
    row_centers_x: list[float],
    row_indices: list[int] | None = None,
) -> None:
    floor_color = (0.60, 0.60, 0.60)
    wall_color = (0.40, 0.40, 0.40)
    rubble_color = (0.54, 0.54, 0.54)
    floor_thickness = 0.08
    wall_thickness = 0.18
    wall_height = 1.50
    row_entries = _resolve_row_entries(row_centers_x, row_indices)
    for row_idx, x_center in row_entries:
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(len(row_centers_x) - 1, 1)
        floor_cfg = make_static_cuboid_cfg((4.00, 4.00, floor_thickness), material_name, floor_color)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.5 * floor_thickness))
        wall_cfg = make_static_cuboid_cfg((4.00, wall_thickness, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_front", wall_cfg, (x_center, y_center + 2.00, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_back", wall_cfg, (x_center, y_center - 2.00, 0.5 * wall_height))
        wall_side_cfg = make_static_cuboid_cfg((wall_thickness, 4.00, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_left", wall_side_cfg, (x_center - 2.00, y_center, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_right", wall_side_cfg, (x_center + 2.00, y_center, 0.5 * wall_height))

        rng = random.Random(3000 + row_idx * 53 + len(material_name))
        object_count = 84 + int(round(48 * difficulty))
        for obj_idx in range(object_count):
            px = x_center + rng.uniform(-1.60, 1.60)
            py = y_center + rng.uniform(-1.60, 1.60)
            if obj_idx % 3 == 0:
                radius = rng.uniform(0.035, 0.065)
                height = rng.uniform(0.08, 0.16)
                cfg = make_dynamic_cylinder_cfg(radius, height, material_name, rubble_color, mass=0.08)
                spawn_primitive(f"{tile_root}/dense_cyl_{obj_idx}", cfg, (px, py, radius + 0.01))
            else:
                sx = rng.uniform(0.05, 0.10)
                sy = rng.uniform(0.05, 0.10)
                sz = rng.uniform(0.05, 0.09)
                cfg = make_dynamic_cuboid_cfg((sx, sy, sz), material_name, rubble_color, mass=0.05)
                spawn_primitive(f"{tile_root}/dense_box_{obj_idx}", cfg, (px, py, 0.5 * sz + 0.01))


def spawn_stability_dynamic_support_column(
    root_path: str,
    material_name: str,
    y_center: float,
    row_centers_x: list[float],
    row_indices: list[int] | None = None,
) -> None:
    floor_color = (0.62, 0.62, 0.62)
    rail_color = (0.44, 0.44, 0.44)
    support_color = (0.54, 0.54, 0.54)
    floor_cfg = make_static_cuboid_cfg((4.00, 4.00, 0.08), material_name, floor_color)
    rail_cfg = make_static_cuboid_cfg((3.20, 0.08, 0.18), material_name, rail_color)
    row_entries = _resolve_row_entries(row_centers_x, row_indices)
    for row_idx, x_center in row_entries:
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(len(row_centers_x) - 1, 1)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.04))
        spawn_primitive(f"{tile_root}/rail_l", rail_cfg, (x_center, y_center - 1.45, 0.09))
        spawn_primitive(f"{tile_root}/rail_r", rail_cfg, (x_center, y_center + 1.45, 0.09))

        roller_count = 2 + row_idx // 2
        roller_length = 1.55 - 0.75 * difficulty
        roller_radius = 0.10 + 0.06 * difficulty
        x_span = 1.8
        if roller_count == 1:
            local_x_positions = [0.0]
        else:
            local_x_positions = [
                -x_span / 2 + i * (x_span / (roller_count - 1)) for i in range(roller_count)
            ]
        for roller_idx, local_x in enumerate(local_x_positions):
            cfg = make_dynamic_cylinder_cfg(
                radius=roller_radius,
                height=roller_length,
                material_name=material_name,
                color=support_color,
                mass=0.55 + 0.12 * difficulty,
            )
            spawn_primitive(f"{tile_root}/roller_{roller_idx}", cfg, (x_center + local_x, y_center, roller_radius + 0.01))


def spawn_dynamic_arena_column(
    arena_name: str,
    root_path: str,
    material_name: str,
    y_center: float,
    row_centers_x: list[float] | None = None,
    row_indices: list[int] | None = None,
) -> None:
    if row_centers_x is None:
        row_centers_x = default_row_centers_x(8, PHYSICAL_DYNAMIC_ARENA_TILE_SIZE[0])
    if arena_name == "stability_dynamic_support":
        spawn_stability_dynamic_support_column(root_path, material_name, y_center, row_centers_x, row_indices)
    elif arena_name == "looseness_dynamic_rubble":
        spawn_looseness_dynamic_rubble_column(root_path, material_name, y_center, row_centers_x, row_indices)
    elif arena_name == "looseness_dense_small_rubble":
        spawn_looseness_dense_small_rubble_column(root_path, material_name, y_center, row_centers_x, row_indices)
    else:
        raise ValueError(f"Unsupported dynamic arena name: {arena_name}")
