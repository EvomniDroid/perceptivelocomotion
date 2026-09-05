"""Visualize terrain only, without spawning a robot.

Examples:
    cd /home/zh/isaac/instinctlab
    python source/instinctlab/instinctlab/tasks/parkour/scripts/visualize_terrain_only.py
    python source/instinctlab/instinctlab/tasks/parkour/scripts/visualize_terrain_only.py --terrain-set my
    python source/instinctlab/instinctlab/tasks/parkour/scripts/visualize_terrain_only.py --terrain-set training
    python source/instinctlab/instinctlab/tasks/parkour/scripts/visualize_terrain_only.py --material low_friction
"""

from __future__ import annotations

import argparse
import ast
import copy
import random
from pathlib import Path

parser = argparse.ArgumentParser(description="Visualize terrain only, without spawning a robot.")
parser.add_argument(
    "--terrain-set",
    type=str,
    default="shared_all",
    choices=[
        "shared_all",
        "physical",
        "physical_curriculum",
        "physical_low_friction",
        "physical_low_friction_curriculum",
        "physical_springy",
        "physical_springy_curriculum",
        "physical_high_grip",
        "physical_high_grip_curriculum",
        "physical_slippery_bouncy",
        "physical_slippery_bouncy_curriculum",
        "physical_damped_soft_like",
        "physical_damped_soft_like_curriculum",
        "my",
        "training",
        "frontier",
        "atec_d",
    ],
    help="Which terrain generator preset to visualize.",
)
parser.add_argument(
    "--terrain-names",
    type=str,
    default=None,
    help="Comma-separated terrain names to visualize in order. Overrides --terrain-set.",
)
parser.add_argument(
    "--material",
    type=str,
    default="default",
    choices=["default", "low_friction", "springy", "high_grip", "slippery_bouncy", "damped_soft_like"],
    help="Ground physics material preset.",
)
parser.add_argument(
    "--camera",
    type=str,
    default="overview",
    choices=["overview", "top", "angled"],
    help="Camera view preset.",
)
parser.add_argument(
    "--material-grid",
    action="store_true",
    help="Replicate the selected terrain set four times side-by-side using default/low_friction/springy/high_grip materials.",
)
parser.add_argument(
    "--compare-terrain-materials",
    type=str,
    default=None,
    help="Visualize one terrain as multiple material columns with difficulty increasing by row, e.g. --compare-terrain-materials tilt.",
)
parser.add_argument("--list", action="store_true", help="Print available terrain presets and terrain names, then exit.")
args_cli = parser.parse_args()

SCRIPT_FILE = Path(__file__).resolve()
INSTINCTLAB_PKG_DIR = SCRIPT_FILE.parents[3]
TERRAINS_DIR = INSTINCTLAB_PKG_DIR / "terrains"
SHARED_TERRAIN_CFG_FILE = TERRAINS_DIR / "shared_terrain_cfg.py"


def _extract_dict_keys_from_cfg(var_name: str) -> list[str]:
    text = SHARED_TERRAIN_CFG_FILE.read_text()
    module = ast.parse(text)
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != var_name:
                continue
            if not isinstance(node.value, ast.Call) or not node.value.args:
                continue
            dict_node = node.value.args[0]
            if not isinstance(dict_node, ast.Dict):
                continue
            keys: list[str] = []
            for key in dict_node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
            return keys
    return []


SHARED_TERRAIN_NAMES = _extract_dict_keys_from_cfg("SHARED_SUB_TERRAINS")
TRAINING_TERRAIN_NAMES = _extract_dict_keys_from_cfg("TRAINING_SUB_TERRAINS")


def print_available_options() -> None:
    print("Available terrain-set presets:")
    for name in [
        "shared_all",
        "physical",
        "physical_curriculum",
        "physical_low_friction",
        "physical_low_friction_curriculum",
        "physical_springy",
        "physical_springy_curriculum",
        "physical_high_grip",
        "physical_high_grip_curriculum",
        "physical_slippery_bouncy",
        "physical_slippery_bouncy_curriculum",
        "physical_damped_soft_like",
        "physical_damped_soft_like_curriculum",
        "my",
        "training",
        "frontier",
        "atec_d",
    ]:
        print(f"  - {name}")
    print("")
    print("Available shared/base terrain names:")
    for name in SHARED_TERRAIN_NAMES:
        print(f"  - {name}")
    print("")
    print("Available training-specific terrain names:")
    for name in TRAINING_TERRAIN_NAMES:
        print(f"  - {name}")
    print("")
    print("Example custom selection:")
    print("  --terrain-names perlin_rough,pyramid_stairs,wave,mesh_boxes")
    print("")
    print("Example four-material comparison:")
    print("  --terrain-set physical_curriculum --material-grid")
    print("")
    print("Example single-terrain material comparison:")
    print("  --compare-terrain-materials tilt")


if args_cli.list:
    print_available_options()
    raise SystemExit(0)

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from pxr import Gf, UsdGeom

from instinctlab.terrains.physical_terrain_cfg import (
    PHYSICAL_COMPARISON_MATERIAL_NAMES,
    PHYSICAL_DISPLAY_TERRAIN_NAMES,
    PHYSICAL_DYNAMIC_ARENA_NAMES,
    GEOMETRIC_ALL_TERRAINS_CFG,
    PHYSICAL_MATERIAL_PRESETS,
    PHYSICAL_TERRAIN_COLLECTIONS,
    PHYSICAL_STUDY_TERRAIN_NAMES,
)
from instinctlab.terrains.shared_terrain_cfg import (
    ATEC_D_TERRAIN_CFG,
    FRONTIER_TEST_TERRAIN_CFG,
    MY_TERRAIN_CFG,
    SHARED_SUB_TERRAINS,
    TRAINING_SUB_TERRAINS,
)
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_importer import TerrainImporter
from instinctlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinctlab.terrains.virtual_obstacle.edge_cylinder_cfg import GreedyconcatEdgeCylinderCfg


TRAINING_ONLY_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=11,
    size=(2.0, 2.0),
    border_width=0.05,
    num_rows=len(TRAINING_SUB_TERRAINS),
    num_cols=1,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=False,
    terrain_layout=list(TRAINING_SUB_TERRAINS.keys()),
    sub_terrains=copy.deepcopy(TRAINING_SUB_TERRAINS),
)

for _, cfg in TRAINING_ONLY_TERRAINS_CFG.sub_terrains.items():
    if hasattr(cfg, "flat_patch_sampling"):
        cfg.flat_patch_sampling = None


def build_custom_terrain_cfg(terrain_names: list[str]) -> FiledTerrainGeneratorCfg:
    unknown = [name for name in terrain_names if name not in SHARED_SUB_TERRAINS and name not in TRAINING_SUB_TERRAINS]
    if unknown:
        raise ValueError(f"Unknown terrain names: {unknown}")
    # Prefer training-specific variants when names overlap there.
    sub_terrains = {}
    for idx, name in enumerate(terrain_names):
        if name in TRAINING_SUB_TERRAINS:
            cfg = copy.deepcopy(TRAINING_SUB_TERRAINS[name])
            if hasattr(cfg, "flat_patch_sampling"):
                cfg.flat_patch_sampling = None
            sub_terrains[f"terrain_{idx}"] = cfg
        else:
            sub_terrains[f"terrain_{idx}"] = copy.deepcopy(SHARED_SUB_TERRAINS[name])
    return FiledTerrainGeneratorCfg(
        class_type=FiledTerrainGenerator,
        seed=13,
        size=(2.0, 2.0),
        border_width=0.05,
        num_rows=len(terrain_names),
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        use_cache=False,
        curriculum=False,
        terrain_layout=terrain_names,
        sub_terrains=sub_terrains,
    )


def build_single_terrain_curriculum_cfg(terrain_name: str) -> FiledTerrainGeneratorCfg:
    if terrain_name in TRAINING_SUB_TERRAINS:
        cfg = copy.deepcopy(TRAINING_SUB_TERRAINS[terrain_name])
    elif terrain_name in SHARED_SUB_TERRAINS:
        cfg = copy.deepcopy(SHARED_SUB_TERRAINS[terrain_name])
    else:
        raise ValueError(f"Unknown terrain name for material comparison: {terrain_name}")
    if hasattr(cfg, "flat_patch_sampling"):
        cfg.flat_patch_sampling = None
    return FiledTerrainGeneratorCfg(
        class_type=FiledTerrainGenerator,
        seed=23,
        size=(2.5, 2.5),
        border_width=0.05,
        num_rows=8,
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=1.0,
        use_cache=False,
        curriculum=True,
        deterministic_curriculum_rows=True,
        terrain_layout=[terrain_name],
        sub_terrains={"terrain_0": cfg},
    )


def resolve_terrain_generator():
    if args_cli.compare_terrain_materials:
        return build_single_terrain_curriculum_cfg(args_cli.compare_terrain_materials)
    if args_cli.terrain_names:
        terrain_names = [name.strip() for name in args_cli.terrain_names.split(",") if name.strip()]
        if not terrain_names:
            raise ValueError("--terrain-names was provided but no valid names were parsed.")
        return build_custom_terrain_cfg(terrain_names)
    if args_cli.terrain_set in PHYSICAL_TERRAIN_COLLECTIONS:
        return PHYSICAL_TERRAIN_COLLECTIONS[args_cli.terrain_set]["terrain_cfg"]
    if args_cli.terrain_set == "shared_all":
        return GEOMETRIC_ALL_TERRAINS_CFG
    if args_cli.terrain_set == "my":
        return MY_TERRAIN_CFG
    if args_cli.terrain_set == "training":
        return TRAINING_ONLY_TERRAINS_CFG
    if args_cli.terrain_set == "frontier":
        return FRONTIER_TEST_TERRAIN_CFG
    if args_cli.terrain_set == "atec_d":
        return ATEC_D_TERRAIN_CFG
    raise ValueError(f"Unsupported terrain set: {args_cli.terrain_set}")


def use_grouped_physical_material_layout() -> bool:
    return (
        args_cli.compare_terrain_materials is None
        and not args_cli.material_grid
        and args_cli.terrain_set
        in {
            "physical_curriculum",
            "physical_low_friction_curriculum",
            "physical_springy_curriculum",
            "physical_high_grip_curriculum",
            "physical_slippery_bouncy_curriculum",
            "physical_damped_soft_like_curriculum",
        }
    )


def set_camera(sim: SimulationContext) -> None:
    if args_cli.camera == "top":
        sim.set_camera_view(eye=(0.0, 0.0, 35.0), target=(0.0, 0.0, 0.0))
    elif args_cli.camera == "angled":
        sim.set_camera_view(eye=(8.0, 14.0, 9.0), target=(8.0, 0.0, -1.5))
    else:
        sim.set_camera_view(eye=(6.0, 12.0, 10.0), target=(6.0, 0.0, -2.0))


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


def make_terrain_importer_cfg(
    terrain_generator,
    material_name: str,
    prim_path: str,
) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        class_type=TerrainImporter,
        prim_path=prim_path,
        terrain_type="hacked_generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=PHYSICAL_MATERIAL_PRESETS[material_name],
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        virtual_obstacles={
            "edges": GreedyconcatEdgeCylinderCfg(
                cylinder_radius=0.05,
                min_points=2,
            ),
        },
    )


def terrain_root_prim_path_from_importer(importer: TerrainImporter) -> str:
    if not getattr(importer, "terrain_prim_paths", None):
        raise RuntimeError("TerrainImporter did not report any terrain prim paths.")
    return importer.terrain_prim_paths[0]


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


def row_centers_x(num_rows: int, tile_length_x: float) -> list[float]:
    # The curriculum terrain importer lays out rows starting from x ~= 0.
    # Shift dynamic arenas by half a tile so their footprint aligns visually.
    return [row_idx * tile_length_x for row_idx in range(num_rows)]


def spawn_looseness_dynamic_rubble_column(
    root_path: str,
    material_name: str,
    y_center: float,
    num_rows: int = 8,
    tile_size: tuple[float, float] = (2.5, 2.5),
) -> None:
    tile_length_x, _ = tile_size
    floor_color = (0.62, 0.62, 0.62)
    wall_color = (0.46, 0.46, 0.46)
    rubble_color = (0.56, 0.56, 0.56)
    floor_thickness = 0.08
    wall_thickness = 0.12
    wall_height = 0.34
    for row_idx, x_center in enumerate(row_centers_x(num_rows, tile_length_x)):
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(num_rows - 1, 1)
        floor_cfg = make_static_cuboid_cfg((2.50, 2.50, floor_thickness), material_name, floor_color)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.5 * floor_thickness))
        wall_cfg = make_static_cuboid_cfg((2.50, wall_thickness, wall_height), material_name, wall_color)
        spawn_primitive(
            f"{tile_root}/wall_front",
            wall_cfg,
            (x_center + 0.0, y_center + 1.25, 0.5 * wall_height),
        )
        spawn_primitive(
            f"{tile_root}/wall_back",
            wall_cfg,
            (x_center + 0.0, y_center - 1.25, 0.5 * wall_height),
        )
        wall_side_cfg = make_static_cuboid_cfg((wall_thickness, 2.50, wall_height), material_name, wall_color)
        spawn_primitive(
            f"{tile_root}/wall_left",
            wall_side_cfg,
            (x_center - 1.25, y_center, 0.5 * wall_height),
        )
        spawn_primitive(
            f"{tile_root}/wall_right",
            wall_side_cfg,
            (x_center + 1.25, y_center, 0.5 * wall_height),
        )

        rng = random.Random(1000 + row_idx * 37 + len(material_name))
        object_count = 90 + int(round(60 * difficulty))
        for obj_idx in range(object_count):
            px = x_center + rng.uniform(-0.95, 0.95)
            py = y_center + rng.uniform(-0.95, 0.95)
            if obj_idx % 2 == 0:
                radius = rng.uniform(0.06, 0.11)
                height = rng.uniform(0.14, 0.24)
                cfg = make_dynamic_cylinder_cfg(radius, height, material_name, rubble_color, mass=0.24)
                spawn_primitive(
                    f"{tile_root}/rubble_cyl_{obj_idx}",
                    cfg,
                    (px, py, radius + 0.01),
                )
            else:
                sx = rng.uniform(0.09, 0.18)
                sy = rng.uniform(0.09, 0.18)
                sz = rng.uniform(0.09, 0.16)
                cfg = make_dynamic_cuboid_cfg((sx, sy, sz), material_name, rubble_color, mass=0.18)
                spawn_primitive(
                    f"{tile_root}/rubble_box_{obj_idx}",
                    cfg,
                    (px, py, 0.5 * sz + 0.01),
                )


def spawn_looseness_dense_small_rubble_column(
    root_path: str,
    material_name: str,
    y_center: float,
    num_rows: int = 8,
    tile_size: tuple[float, float] = (2.5, 2.5),
) -> None:
    tile_length_x, _ = tile_size
    floor_color = (0.60, 0.60, 0.60)
    wall_color = (0.40, 0.40, 0.40)
    rubble_color = (0.54, 0.54, 0.54)
    floor_thickness = 0.08
    wall_thickness = 0.18
    wall_height = 0.60
    for row_idx, x_center in enumerate(row_centers_x(num_rows, tile_length_x)):
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(num_rows - 1, 1)
        floor_cfg = make_static_cuboid_cfg((2.50, 2.50, floor_thickness), material_name, floor_color)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.5 * floor_thickness))
        wall_cfg = make_static_cuboid_cfg((2.50, wall_thickness, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_front", wall_cfg, (x_center, y_center + 1.25, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_back", wall_cfg, (x_center, y_center - 1.25, 0.5 * wall_height))
        wall_side_cfg = make_static_cuboid_cfg((wall_thickness, 2.50, wall_height), material_name, wall_color)
        spawn_primitive(f"{tile_root}/wall_left", wall_side_cfg, (x_center - 1.25, y_center, 0.5 * wall_height))
        spawn_primitive(f"{tile_root}/wall_right", wall_side_cfg, (x_center + 1.25, y_center, 0.5 * wall_height))

        rng = random.Random(3000 + row_idx * 53 + len(material_name))
        object_count = 160 + int(round(90 * difficulty))
        for obj_idx in range(object_count):
            px = x_center + rng.uniform(-0.96, 0.96)
            py = y_center + rng.uniform(-0.96, 0.96)
            if obj_idx % 3 == 0:
                radius = rng.uniform(0.035, 0.065)
                height = rng.uniform(0.08, 0.16)
                cfg = make_dynamic_cylinder_cfg(radius, height, material_name, rubble_color, mass=0.08)
                spawn_primitive(
                    f"{tile_root}/dense_cyl_{obj_idx}",
                    cfg,
                    (px, py, radius + 0.01),
                )
            else:
                sx = rng.uniform(0.05, 0.10)
                sy = rng.uniform(0.05, 0.10)
                sz = rng.uniform(0.05, 0.09)
                cfg = make_dynamic_cuboid_cfg((sx, sy, sz), material_name, rubble_color, mass=0.05)
                spawn_primitive(
                    f"{tile_root}/dense_box_{obj_idx}",
                    cfg,
                    (px, py, 0.5 * sz + 0.01),
                )


def spawn_stability_dynamic_support_column(
    root_path: str,
    material_name: str,
    y_center: float,
    num_rows: int = 8,
    tile_size: tuple[float, float] = (2.5, 2.5),
) -> None:
    tile_length_x, _ = tile_size
    floor_color = (0.62, 0.62, 0.62)
    rail_color = (0.44, 0.44, 0.44)
    support_color = (0.54, 0.54, 0.54)
    floor_cfg = make_static_cuboid_cfg((2.50, 2.50, 0.08), material_name, floor_color)
    rail_cfg = make_static_cuboid_cfg((2.0, 0.08, 0.18), material_name, rail_color)
    for row_idx, x_center in enumerate(row_centers_x(num_rows, tile_length_x)):
        tile_root = f"{root_path}/row_{row_idx}"
        difficulty = row_idx / max(num_rows - 1, 1)
        spawn_primitive(f"{tile_root}/floor", floor_cfg, (x_center, y_center, -0.04))
        spawn_primitive(f"{tile_root}/rail_l", rail_cfg, (x_center, y_center - 0.92, 0.09))
        spawn_primitive(f"{tile_root}/rail_r", rail_cfg, (x_center, y_center + 0.92, 0.09))

        roller_count = 2 + row_idx // 2
        roller_length = 1.55 - 0.75 * difficulty
        roller_radius = 0.10 + 0.06 * difficulty
        x_span = 0.9
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
            spawn_primitive(
                f"{tile_root}/roller_{roller_idx}",
                cfg,
                (x_center + local_x, y_center, roller_radius + 0.01),
            )


def spawn_dynamic_arena_column(
    arena_name: str,
    root_path: str,
    material_name: str,
    y_center: float,
) -> None:
    if arena_name == "stability_dynamic_support":
        spawn_stability_dynamic_support_column(root_path, material_name, y_center)
    elif arena_name == "looseness_dynamic_rubble":
        spawn_looseness_dynamic_rubble_column(root_path, material_name, y_center)
    elif arena_name == "looseness_dense_small_rubble":
        spawn_looseness_dense_small_rubble_column(root_path, material_name, y_center)
    else:
        raise ValueError(f"Unsupported dynamic arena name: {arena_name}")


def main():
    sim = SimulationContext(SimulationCfg())

    light_cfg = sim_utils.DomeLightCfg(
        intensity=430.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )
    light_cfg.func("/World/skyLight", light_cfg)

    terrain_generator = resolve_terrain_generator()
    material_name = args_cli.material
    if args_cli.terrain_set in PHYSICAL_TERRAIN_COLLECTIONS and args_cli.material == "default":
        material_name = PHYSICAL_TERRAIN_COLLECTIONS[args_cli.terrain_set]["default_material"]

    if args_cli.terrain_names:
        print(f"[INFO] terrain_names = {args_cli.terrain_names}")
    elif args_cli.compare_terrain_materials:
        print(f"[INFO] compare_terrain_materials = {args_cli.compare_terrain_materials}")
    else:
        print(f"[INFO] terrain_set = {args_cli.terrain_set}")
    if use_grouped_physical_material_layout():
        print("[INFO] grouped_material_layout = enabled")
    elif args_cli.material_grid:
        print(f"[INFO] material_grid = {', '.join(PHYSICAL_COMPARISON_MATERIAL_NAMES)}")
    else:
        print(f"[INFO] material = {material_name}")
    if args_cli.compare_terrain_materials:
        print("[INFO] Single-terrain curriculum enabled: row 0 is easiest, last row is hardest.")
    elif not args_cli.terrain_names and "curriculum" in args_cli.terrain_set:
        print("[INFO] Curriculum rows enabled: row 0 is easiest, last row is hardest.")
    if use_grouped_physical_material_layout():
        print(f"[INFO] terrain_layout = {PHYSICAL_DISPLAY_TERRAIN_NAMES}")
    elif hasattr(terrain_generator, "terrain_layout") and terrain_generator.terrain_layout is not None:
        print(f"[INFO] terrain_layout = {list(terrain_generator.terrain_layout)}")
    print("[INFO] Generating terrain only scene...")
    if use_grouped_physical_material_layout():
        grid_materials = PHYSICAL_COMPARISON_MATERIAL_NAMES
        tile_width_y = 2.5
        intra_group_gap_y = 0.0
        inter_group_gap_y = 0.0
        base_y_spacing = tile_width_y + intra_group_gap_y
        terrain_y_spacing = base_y_spacing * len(grid_materials) + inter_group_gap_y
        total_width_y = (
            len(PHYSICAL_DISPLAY_TERRAIN_NAMES) * len(grid_materials) * tile_width_y
            + len(PHYSICAL_DISPLAY_TERRAIN_NAMES) * (len(grid_materials) - 1) * intra_group_gap_y
            + (len(PHYSICAL_DISPLAY_TERRAIN_NAMES) - 1) * inter_group_gap_y
        )
        y_center_offset = 0.5 * total_width_y
        print("[INFO] Grouped material columns by terrain:")
        for terrain_idx, terrain_name in enumerate(PHYSICAL_DISPLAY_TERRAIN_NAMES):
            print(f"  [terrain {terrain_idx}] {terrain_name}")
            for material_idx, grid_material in enumerate(grid_materials):
                prim_path = f"/World/{terrain_name}_{grid_material}"
                y_offset = (
                    terrain_idx * terrain_y_spacing
                    + material_idx * base_y_spacing
                    + 0.5 * tile_width_y
                    - y_center_offset
                )
                if terrain_name in PHYSICAL_DYNAMIC_ARENA_NAMES:
                    spawn_dynamic_arena_column(terrain_name, prim_path, grid_material, y_offset)
                    terrain_prim_path = prim_path
                else:
                    terrain_cfg_copy = build_single_terrain_curriculum_cfg(terrain_name)
                    importer_cfg = make_terrain_importer_cfg(terrain_cfg_copy, grid_material, prim_path)
                    importer = TerrainImporter(cfg=importer_cfg)
                    terrain_prim_path = terrain_root_prim_path_from_importer(importer)
                    set_root_translation(terrain_prim_path, (0.0, y_offset, 0.0))
                print(
                    f"    - material_col={material_idx} material={grid_material} "
                    f"prim={terrain_prim_path} y_offset={y_offset:.2f}"
                )
        print(
            f"[INFO] Grouped grid summary: terrains={len(PHYSICAL_DISPLAY_TERRAIN_NAMES)}, "
            f"materials_per_terrain={len(grid_materials)}, rows=8, single_column_tile_size=(2.5, 2.5)"
        )
        if args_cli.camera == "top":
            sim.set_camera_view(
                eye=(10.0, 0.0, 55.0),
                target=(10.0, 0.0, 0.0),
            )
        elif args_cli.camera == "angled":
            sim.set_camera_view(
                eye=(11.0, total_width_y * 0.55 + 8.0, 18.0),
                target=(10.0, 0.0, -1.5),
            )
        else:
            sim.set_camera_view(
                eye=(10.0, total_width_y * 0.55 + 6.0, 16.0),
                target=(10.0, 0.0, -2.0),
            )
    elif args_cli.material_grid or args_cli.compare_terrain_materials:
        grid_materials = PHYSICAL_COMPARISON_MATERIAL_NAMES
        extent_y = terrain_generator.num_cols * terrain_generator.size[1]
        spacing_y = extent_y + 2.0
        total_width_y = spacing_y * (len(grid_materials) - 1)
        y_center_offset = 0.5 * total_width_y
        print("[INFO] Material grid columns:")
        for idx, grid_material in enumerate(grid_materials):
            prim_path = f"/World/ground_{grid_material}"
            terrain_cfg_copy = copy.deepcopy(terrain_generator)
            importer_cfg = make_terrain_importer_cfg(terrain_cfg_copy, grid_material, prim_path)
            importer = TerrainImporter(cfg=importer_cfg)
            y_offset = idx * spacing_y - y_center_offset
            terrain_prim_path = terrain_root_prim_path_from_importer(importer)
            set_root_translation(terrain_prim_path, (0.0, y_offset, 0.0))
            print(f"  - col={idx} material={grid_material} prim={terrain_prim_path} y_offset={y_offset:.2f}")
        print(
            f"[INFO] Grid summary: num_rows={terrain_generator.num_rows}, num_cols={terrain_generator.num_cols}, "
            f"tile_size={terrain_generator.size}, column_spacing_y={spacing_y:.2f}"
        )
        if args_cli.camera == "top":
            sim.set_camera_view(eye=(terrain_generator.num_rows * terrain_generator.size[0] * 0.6, 0.0, 55.0), target=(terrain_generator.num_rows * terrain_generator.size[0] * 0.5, 0.0, 0.0))
        elif args_cli.camera == "angled":
            sim.set_camera_view(
                eye=(terrain_generator.num_rows * terrain_generator.size[0] * 0.55, total_width_y * 0.55 + 10.0, 18.0),
                target=(terrain_generator.num_rows * terrain_generator.size[0] * 0.5, 0.0, -1.5),
            )
        else:
            sim.set_camera_view(
                eye=(terrain_generator.num_rows * terrain_generator.size[0] * 0.55, total_width_y * 0.55 + 8.0, 16.0),
                target=(terrain_generator.num_rows * terrain_generator.size[0] * 0.5, 0.0, -2.0),
            )
    else:
        set_camera(sim)
        terrain_importer_cfg = make_terrain_importer_cfg(terrain_generator, material_name, "/World/ground")
        TerrainImporter(cfg=terrain_importer_cfg)

    print("[INFO] Ready. Close the Isaac Sim window or press Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            sim.step()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting terrain-only viewer...")


if __name__ == "__main__":
    main()
