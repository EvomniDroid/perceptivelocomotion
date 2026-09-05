"""Parallel evaluation/data collection for B2RM parkour fall-rate datasets.

This script is intentionally separate from play.py:
- play.py is for one/few robots and visual inspection.
- this script is for many parallel envs, fixed checkpoints, depth recording,
  and per terrain-row fall-rate statistics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


DEFAULT_EVAL_TERRAIN_KEYS = [
    "perlin_rough",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "boxes",
    "mesh_boxes",
    "hf_pyramid_slope_inv",
    "raised_mound",
    "pit_crater",
    "wave",
]


parser = argparse.ArgumentParser(description="Evaluate B2RM fall rates and collect first-person depth images.")
parser.add_argument("--num_envs", type=int, default=20, help="Number of parallel envs to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Task name. Prefer the B2RM Play task for grid evaluation.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--debug", action="store_true", default=False, help="Wait for debugger attach.")
parser.add_argument("--free_view", action="store_true", default=True, help="Use a world-fixed viewer camera instead of following the robot.")
parser.add_argument("--follow_view", action="store_false", dest="free_view", help="Follow the robot with the viewer.")
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load env cfg from checkpoint log dir if supported.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load agent cfg from checkpoint log dir.")
parser.add_argument("--eval_dir", type=str, default=None, help="Output directory. Defaults to <run>/record/<timestamp>.")
parser.add_argument("--episodes_per_cell", type=int, default=20, help="Target completed episodes per terrain-row cell.")
parser.add_argument("--max_steps", type=int, default=200000, help="Safety cap on simulator steps.")
parser.add_argument("--save_depth_interval", type=int, default=20, help="Save depth every N sim-policy steps; 0 disables images.")
parser.add_argument("--save_raw_depth", action="store_true", default=False, help="Also save raw depth as .npy.")
parser.add_argument("--save_depth_u16", action="store_true", default=False, help="Also save metric depth as 16-bit PNG in millimeters.")
parser.add_argument(
    "--depth_vis_mode",
    type=str,
    default="adaptive",
    choices=("adaptive", "fixed"),
    help="8-bit preview PNG normalization: adaptive uses each frame range; fixed uses --depth_min/--depth_max.",
)
parser.add_argument("--depth_min", type=float, default=0.0, help="Min depth for fixed preview and uint16 clipping.")
parser.add_argument("--depth_max", type=float, default=4.0, help="Max depth for fixed preview and uint16 clipping.")
parser.add_argument("--depth_valid_min", type=float, default=0.05, help="Ignore smaller depth values for adaptive preview min/max.")
parser.add_argument(
    "--record_rgbd",
    action="store_true",
    default=False,
    help="Also record rendered first-person RGB and rendered depth. Expensive; use small num_envs.",
)
parser.add_argument("--eval_lin_vel_x", type=float, default=0.4, help="Commanded forward velocity injected into policy obs.")
parser.add_argument("--eval_lin_vel_y", type=float, default=0.0, help="Commanded lateral velocity injected into policy obs.")
parser.add_argument("--eval_ang_vel_z", type=float, default=0.0, help="Commanded yaw velocity injected into policy obs.")
parser.add_argument(
    "--eval_terrain_keys",
    type=str,
    default=",".join(DEFAULT_EVAL_TERRAIN_KEYS),
    help="Comma-separated terrain keys to evaluate, one column per key by default.",
)
parser.add_argument(
    "--disable_arm_disturbance",
    action="store_true",
    default=False,
    help="Disable arm payload/workspace randomization during evaluation.",
)
parser.add_argument(
    "--relaxed_termination",
    action="store_true",
    default=False,
    help="Use visual-fall style termination: ignore thigh/calf link contacts, keep root height and orientation.",
)
parser.add_argument(
    "--relaxed_root_height",
    type=float,
    default=0.18,
    help="Root-height threshold used by --relaxed_termination.",
)
parser.add_argument(
    "--visual_failure_checks",
    action="store_true",
    default=False,
    help="Count visual failures in eval: sustained base contact or stuck/no forward progress.",
)
parser.add_argument("--stuck_warmup_s", type=float, default=1.0, help="Warmup seconds before stuck detection starts.")
parser.add_argument("--stuck_time_s", type=float, default=2.0, help="Seconds of low forward velocity before stuck failure.")
parser.add_argument("--stuck_lin_vel_x", type=float, default=0.05, help="Body-frame x velocity threshold for stuck failure.")
parser.add_argument("--base_contact_force", type=float, default=30.0, help="Base/head contact force threshold for visual failure.")
parser.add_argument("--base_contact_time_s", type=float, default=0.3, help="Sustained base/head contact duration before failure.")
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=None,
    help="Override evaluation terrain rows before env creation. Default uses task cfg.",
)
parser.add_argument(
    "--terrain_cols",
    type=int,
    default=None,
    help="Override evaluation terrain columns before env creation. Default uses task cfg.",
)

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Raycaster depth scales to many envs. Rendered RGB/RGBD cameras do not, so keep
# them opt-in for small visual dataset runs.
if args_cli.record_rgbd:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.io import load_yaml

import instinctlab.tasks  # noqa: F401
from instinctlab.terrains.shared_terrain_cfg import TRAINING_SUB_TERRAINS
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg


@dataclass
class CellStats:
    terrain: str
    row: int
    col: int
    episodes: int = 0
    falls: int = 0
    timeouts: int = 0
    frames: int = 0

    @property
    def fall_rate(self) -> float:
        return float(self.falls / self.episodes) if self.episodes > 0 else 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _disable_arm_disturbance_events(env_cfg) -> None:
    if not hasattr(env_cfg, "events") or env_cfg.events is None:
        return
    for event_name in (
        "arm_tip_payload",
        "arm_safe_carry_pose",
        "arm_workspace_target_reset",
        "arm_workspace_target_interval",
        "arm_pose_interval",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)


def _apply_relaxed_termination(env_cfg) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return

    if hasattr(terminations, "root_height") and terminations.root_height is not None:
        terminations.root_height.params["minimum_height"] = args_cli.relaxed_root_height

    # These are useful during training, but too strict for visual fall-rate eval:
    # stepping over boxes/pits often brushes calf/thigh links without an actual fall.
    for name in ("base_contact", "leg_link_contact", "calf_link_contact"):
        if hasattr(terminations, name):
            setattr(terminations, name, None)


def _parse_eval_terrain_keys() -> list[str]:
    keys = [key.strip() for key in args_cli.eval_terrain_keys.split(",") if key.strip()]
    if not keys:
        raise RuntimeError("--eval_terrain_keys is empty.")
    missing = [key for key in keys if key not in TRAINING_SUB_TERRAINS]
    if missing:
        raise RuntimeError(
            f"Unknown eval terrain keys: {missing}. Available: {list(TRAINING_SUB_TERRAINS.keys())}"
        )
    return list(dict.fromkeys(keys))


def _apply_eval_terrain_columns(env_cfg, terrain_keys: list[str]) -> None:
    generator = env_cfg.scene.terrain.terrain_generator
    sub_terrains = OrderedDict()
    for key in terrain_keys:
        cfg = copy.deepcopy(TRAINING_SUB_TERRAINS[key])
        cfg.proportion = 1.0
        try:
            cfg.name = key
        except Exception:
            object.__setattr__(cfg, "name", key)
        sub_terrains[key] = cfg
    generator.sub_terrains = sub_terrains
    if args_cli.terrain_cols is None:
        generator.num_cols = len(terrain_keys)
    print(f"[EVAL] terrain columns ({len(terrain_keys)}): {terrain_keys}")


def _resolve_checkpoint(agent_cfg: InstinctRlOnPolicyRunnerCfg) -> tuple[str, str]:
    log_root_path = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
    agent_cfg.load_run = args_cli.load_run.rstrip("/\\") if args_cli.load_run is not None else None
    if agent_cfg.load_run is None:
        raise RuntimeError("Please provide --load_run and --checkpoint for eval.")
    if os.path.isabs(agent_cfg.load_run):
        resume_path = get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
        )
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return resume_path, os.path.dirname(resume_path)


def _terrain_names_2d(terrain) -> np.ndarray:
    if hasattr(terrain, "terrain_names"):
        return terrain.terrain_names
    num_rows = int(terrain.cfg.terrain_generator.num_rows)
    num_cols = int(terrain.cfg.terrain_generator.num_cols)
    names = np.full((num_rows, num_cols), "", dtype=object)
    col_names = getattr(terrain, "terrain_type_names", None) or []
    for col in range(num_cols):
        names[:, col] = col_names[col] if col < len(col_names) else f"col{col}"
    return names


def _cell_physical_params(terrain, row: int, col: int) -> dict[str, Any]:
    generator = getattr(terrain, "terrain_generator", None)
    cfg = None
    if generator is not None and hasattr(generator, "get_subterrain_cfg"):
        try:
            cfg = generator.get_subterrain_cfg(row, col)
        except Exception:
            cfg = None
    if cfg is None:
        return {}

    params = {}
    for key, value in vars(cfg).items():
        if key.startswith("_"):
            continue
        if callable(value):
            continue
        params[key] = _jsonable(value)
    return params


def _write_grid_manifest(output_dir: str, terrain) -> list[tuple[int, int, str]]:
    names = _terrain_names_2d(terrain)
    combos: list[tuple[int, int, str]] = []
    grid_manifest = []
    for row in range(names.shape[0]):
        for col in range(names.shape[1]):
            terrain_name = str(names[row, col] or f"col{col}")
            combos.append((row, col, terrain_name))
            grid_manifest.append(
                {
                    "row": row,
                    "col": col,
                    "terrain": terrain_name,
                    "origin": terrain.terrain_origins[row, col].detach().cpu().tolist()
                    if hasattr(terrain, "terrain_origins") and terrain.terrain_origins is not None
                    else None,
                    "physical_params": _cell_physical_params(terrain, row, col),
                }
            )
    with open(os.path.join(output_dir, "terrain_grid_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(grid_manifest, f, indent=2, ensure_ascii=False)
    return combos


def _set_env_cells(raw_env, assignments: dict[int, tuple[int, int, str]]) -> None:
    terrain = raw_env.scene.terrain
    env_ids = sorted(assignments.keys())
    if not env_ids:
        return

    env_id_tensor = torch.tensor(env_ids, device=raw_env.device, dtype=torch.long)
    for env_id in env_ids:
        row, col, _terrain_name = assignments[env_id]
        new_origin = terrain.terrain_origins[row, col].clone()
        terrain.env_origins[env_id] = new_origin
        if hasattr(raw_env.scene, "env_origins") and raw_env.scene.env_origins is not None:
            raw_env.scene.env_origins[env_id] = new_origin
        if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
            terrain.terrain_levels[env_id] = row
        if hasattr(terrain, "terrain_types") and terrain.terrain_types is not None:
            terrain.terrain_types[env_id] = col
    raw_env._reset_idx(env_id_tensor)


def _termination_reason(raw_env, env_id: int, infos: dict) -> tuple[str, bool]:
    termination_manager = getattr(raw_env, "termination_manager", None)
    if termination_manager is not None:
        term_names = getattr(termination_manager, "_term_names", [])
        term_cfgs = getattr(termination_manager, "_term_cfgs", [])
        term_dones = getattr(termination_manager, "_term_dones", None)
        if term_dones is not None:
            try:
                active_indices = torch.nonzero(term_dones[env_id], as_tuple=False).flatten().tolist()
            except Exception:
                active_indices = []
            if active_indices:
                active_names = [term_names[i] for i in active_indices if i < len(term_names)]
                is_timeout = any(
                    bool(getattr(term_cfgs[i], "time_out", False))
                    for i in active_indices
                    if i < len(term_cfgs)
                )
                return "+".join(active_names), is_timeout

    time_outs = infos.get("time_outs", None)
    if time_outs is not None:
        try:
            if bool(time_outs[env_id].item()):
                return "timeout", True
        except Exception:
            pass

    episode_info = infos.get("episode", {})
    if not episode_info:
        episode_info = infos.get("log", {})
    active = []
    for key, value in episode_info.items():
        if "Episode_Termination" not in key:
            continue
        try:
            reason_value = float(value[env_id].item())
        except Exception:
            continue
        if reason_value > 0.5:
            active.append(key.replace("Episode_Termination/", ""))
    return ("+".join(active) if active else "unknown", False)


def _write_label_file(output_dir: str, image_rel: str, metadata: dict[str, Any]) -> str:
    label_rel = os.path.splitext(image_rel)[0] + ".txt"
    label_abs = os.path.join(output_dir, label_rel)
    with open(label_abs, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {_jsonable(value)}\n")
    return label_rel


def _save_depth_frame(
    output_dir: str,
    depth: np.ndarray,
    modality: str,
    timestep: int,
    env_id: int,
    episode_id: int,
    row: int,
    col: int,
    terrain_name: str,
) -> tuple[str, str | None, str | None, float, float]:
    depth = np.squeeze(depth)
    depth = np.nan_to_num(depth, nan=0.0, posinf=args_cli.depth_max, neginf=0.0)
    depth = np.clip(depth, args_cli.depth_min, args_cli.depth_max)
    d_min = float(depth.min()) if depth.size > 0 else 0.0
    d_max = float(depth.max()) if depth.size > 0 else 0.0
    if args_cli.depth_vis_mode == "adaptive":
        valid = depth[depth > args_cli.depth_valid_min]
        vis_min = float(valid.min()) if valid.size > 0 else d_min
        vis_max = float(valid.max()) if valid.size > 0 else d_max
    else:
        vis_min = args_cli.depth_min
        vis_max = args_cli.depth_max
    if vis_max - vis_min > 1e-6:
        normalized = ((depth - vis_min) / (vis_max - vis_min) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth, dtype=np.uint8)

    rel_dir = os.path.join(modality, terrain_name, f"row{row:02d}_col{col:02d}", f"env{env_id:03d}")
    abs_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    stem = f"step{timestep:08d}_ep{episode_id:05d}_env{env_id:03d}"
    png_rel = os.path.join(rel_dir, f"{stem}.png")
    Image.fromarray(normalized).save(os.path.join(output_dir, png_rel))

    npy_rel = None
    if args_cli.save_raw_depth:
        npy_rel = os.path.join(rel_dir, f"{stem}.npy")
        np.save(os.path.join(output_dir, npy_rel), depth.astype(np.float32))

    u16_rel = None
    if args_cli.save_depth_u16:
        u16_rel = os.path.join(rel_dir, f"{stem}_u16mm.png")
        depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(os.path.join(output_dir, u16_rel))

    return png_rel, npy_rel, u16_rel, d_min, d_max


def _save_rgb_frame(
    output_dir: str,
    rgb: np.ndarray,
    timestep: int,
    env_id: int,
    episode_id: int,
    row: int,
    col: int,
    terrain_name: str,
) -> tuple[str, float, float]:
    rgb = np.squeeze(rgb)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    rgb_min = float(rgb.min()) if rgb.size > 0 else 0.0
    rgb_max = float(rgb.max()) if rgb.size > 0 else 0.0

    rel_dir = os.path.join("rgb", terrain_name, f"row{row:02d}_col{col:02d}", f"env{env_id:03d}")
    abs_dir = os.path.join(output_dir, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    stem = f"step{timestep:08d}_ep{episode_id:05d}_env{env_id:03d}"
    png_rel = os.path.join(rel_dir, f"{stem}.png")
    Image.fromarray(rgb).save(os.path.join(output_dir, png_rel))
    return png_rel, rgb_min, rgb_max


def _write_stats(output_dir: str, stats: dict[tuple[int, int], CellStats], reason_counts: Counter) -> None:
    stats_path = os.path.join(output_dir, "fall_rate_by_cell.csv")
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["terrain", "row", "col", "episodes", "falls", "timeouts", "frames", "fall_rate"],
        )
        writer.writeheader()
        for cell in sorted(stats.values(), key=lambda s: (s.col, s.row)):
            row = asdict(cell)
            row["fall_rate"] = cell.fall_rate
            writer.writerow(row)
    with open(os.path.join(output_dir, "termination_reason_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(reason_counts), f, indent=2, ensure_ascii=False)


def _next_combo(combos: list[tuple[int, int, str]], stats: dict[tuple[int, int], CellStats], cursor: int) -> tuple[int | None, int]:
    for offset in range(len(combos)):
        idx = (cursor + offset) % len(combos)
        row, col, terrain_name = combos[idx]
        if stats[(row, col)].episodes < args_cli.episodes_per_cell:
            return idx, idx + 1
    return None, cursor


def _get_base_contact_force(raw_env) -> torch.Tensor | None:
    try:
        contact_sensor = raw_env.scene.sensors["contact_forces"]
    except Exception:
        try:
            contact_sensor = raw_env.scene["contact_forces"]
        except Exception:
            return None

    try:
        base_ids, _ = contact_sensor.find_bodies("base_link")
    except Exception:
        return None
    if not base_ids:
        return None

    forces = contact_sensor.data.net_forces_w_history[:, :, base_ids, :]
    return torch.norm(forces, dim=-1).amax(dim=(1, 2))


def _compute_visual_failures(
    raw_env,
    episode_steps: torch.Tensor,
    stuck_counts: torch.Tensor,
    base_contact_counts: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, list[str]]:
    if not args_cli.visual_failure_checks:
        return torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.bool), ["" for _ in range(raw_env.num_envs)]

    reasons = ["" for _ in range(raw_env.num_envs)]
    robot = raw_env.scene["robot"]

    stuck_warmup_steps = max(1, int(args_cli.stuck_warmup_s / step_dt))
    stuck_steps = max(1, int(args_cli.stuck_time_s / step_dt))
    base_contact_steps = max(1, int(args_cli.base_contact_time_s / step_dt))

    if abs(args_cli.eval_lin_vel_x) > 1e-4:
        forward_vel = robot.data.root_lin_vel_b[:, 0]
        low_forward = forward_vel < args_cli.stuck_lin_vel_x
        low_forward &= episode_steps > stuck_warmup_steps
        stuck_counts[:] = torch.where(low_forward, stuck_counts + 1, torch.zeros_like(stuck_counts))
    else:
        stuck_counts.zero_()
    stuck_failed = stuck_counts >= stuck_steps

    base_force = _get_base_contact_force(raw_env)
    if base_force is not None:
        base_contact = base_force > args_cli.base_contact_force
        base_contact_counts[:] = torch.where(
            base_contact, base_contact_counts + 1, torch.zeros_like(base_contact_counts)
        )
        base_failed = base_contact_counts >= base_contact_steps
    else:
        base_failed = torch.zeros(raw_env.num_envs, device=raw_env.device, dtype=torch.bool)

    failed = stuck_failed | base_failed
    failed_ids = torch.nonzero(failed, as_tuple=False).flatten().tolist()
    for env_id in failed_ids:
        active = []
        if bool(stuck_failed[env_id].item()):
            active.append("visual_stuck")
        if bool(base_failed[env_id].item()):
            active.append("visual_base_contact")
        reasons[env_id] = "+".join(active)
    return failed, reasons


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    resume_path, log_dir = _resolve_checkpoint(agent_cfg)

    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if args_cli.disable_arm_disturbance:
        _disable_arm_disturbance_events(env_cfg)
        print("[EVAL] disabled arm disturbance events.")
    if args_cli.relaxed_termination:
        _apply_relaxed_termination(env_cfg)
        print(
            "[EVAL] --relaxed_termination: disabled base/thigh/calf contact termination, "
            f"root_height.minimum_height={args_cli.relaxed_root_height}."
        )

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.free_view and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (8.0, 8.0, 5.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        print("[EVAL] --free_view: viewer set to world-fixed camera.")
    if args_cli.record_rgbd:
        if not hasattr(env_cfg.scene, "rgb_camera") or env_cfg.scene.rgb_camera is None:
            print("[EVAL][WARN] scene.rgb_camera is not configured; only raycaster depth will be recorded.")
    elif hasattr(env_cfg.scene, "rgb_camera"):
        env_cfg.scene.rgb_camera = None
        print("[EVAL] --record_rgbd not set: disabled rendered rgb_camera; raycaster depth remains enabled.")
    if hasattr(env_cfg.scene, "camera_rgb_record"):
        env_cfg.scene.camera_rgb_record = None
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    if getattr(env_cfg.scene, "terrain", None) is not None and env_cfg.scene.terrain.terrain_generator is not None:
        terrain_keys = _parse_eval_terrain_keys()
        _apply_eval_terrain_columns(env_cfg, terrain_keys)
        generator = env_cfg.scene.terrain.terrain_generator
        if args_cli.terrain_rows is not None:
            generator.num_rows = args_cli.terrain_rows
        if args_cli.terrain_cols is not None:
            generator.num_cols = args_cli.terrain_cols
        generator.curriculum = True
        generator.deterministic_curriculum_rows = True
        env_cfg.scene.terrain.max_init_terrain_level = 0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args_cli.eval_dir or os.path.join(log_dir, "record", run_id)
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    with open(os.path.join(output_dir, "eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "num_envs": args_cli.num_envs,
                "episodes_per_cell": args_cli.episodes_per_cell,
                "save_depth_interval": args_cli.save_depth_interval,
                "save_raw_depth": args_cli.save_raw_depth,
                "save_depth_u16": args_cli.save_depth_u16,
                "depth_vis_mode": args_cli.depth_vis_mode,
                "depth_min": args_cli.depth_min,
                "depth_max": args_cli.depth_max,
                "depth_valid_min": args_cli.depth_valid_min,
                "eval_terrain_keys": _parse_eval_terrain_keys(),
                "relaxed_termination": args_cli.relaxed_termination,
                "relaxed_root_height": args_cli.relaxed_root_height,
                "visual_failure_checks": args_cli.visual_failure_checks,
                "stuck_warmup_s": args_cli.stuck_warmup_s,
                "stuck_time_s": args_cli.stuck_time_s,
                "stuck_lin_vel_x": args_cli.stuck_lin_vel_x,
                "base_contact_force": args_cli.base_contact_force,
                "base_contact_time_s": args_cli.base_contact_time_s,
                "command": [args_cli.eval_lin_vel_x, args_cli.eval_lin_vel_y, args_cli.eval_ang_vel_z],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw_env = env.unwrapped

    terrain = raw_env.scene.terrain
    combos = _write_grid_manifest(output_dir, terrain)
    stats = {
        (row, col): CellStats(terrain=terrain_name, row=row, col=col)
        for row, col, terrain_name in combos
    }
    reason_counts: Counter = Counter()

    env = InstinctRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    print(f"[EVAL] loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw_env.device)

    obs_segments = env.get_obs_segments()
    command_obs_slice = get_obs_slice(obs_segments, "velocity_commands")
    command = torch.tensor(
        [args_cli.eval_lin_vel_x, args_cli.eval_lin_vel_y, args_cli.eval_ang_vel_z],
        device=env.device,
        dtype=torch.float32,
    )

    current_assignment: dict[int, tuple[int, int, str]] = {}
    current_episode_id = [0 for _ in range(args_cli.num_envs)]
    current_frames: list[list[dict[str, Any]]] = [[] for _ in range(args_cli.num_envs)]
    inactive_envs: set[int] = set()
    cursor = 0
    episode_steps = torch.zeros(args_cli.num_envs, device=raw_env.device, dtype=torch.long)
    stuck_counts = torch.zeros_like(episode_steps)
    base_contact_counts = torch.zeros_like(episode_steps)

    initial_assignments = {}
    for env_id in range(args_cli.num_envs):
        combo_idx, cursor = _next_combo(combos, stats, cursor)
        if combo_idx is None:
            inactive_envs.add(env_id)
            continue
        assignment = combos[combo_idx]
        current_assignment[env_id] = assignment
        initial_assignments[env_id] = assignment
    _set_env_cells(raw_env, initial_assignments)
    obs, _ = env.get_observations()

    print(
        f"[EVAL] output={output_dir} | cells={len(combos)} | "
        f"target episodes/cell={args_cli.episodes_per_cell} | num_envs={args_cli.num_envs}"
    )
    print("[EVAL] fall definition: done && !timeout. Timeout is counted as success.")

    timestep = 0
    with open(metadata_path, "a", encoding="utf-8") as metadata_file:
        while simulation_app.is_running() and timestep < args_cli.max_steps:
            with torch.inference_mode():
                if command_obs_slice is not None:
                    obs[:, command_obs_slice[0]] = command.repeat(
                        obs.shape[0], command_obs_slice[1][0] // 3
                    )
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                episode_steps += 1

                visual_failed, visual_reasons = _compute_visual_failures(
                    raw_env,
                    episode_steps,
                    stuck_counts,
                    base_contact_counts,
                    raw_env.step_dt,
                )

                if args_cli.save_depth_interval > 0 and timestep % args_cli.save_depth_interval == 0:
                    ray_depth_data = raw_env.scene["camera"].data.output.get("distance_to_image_plane", None)
                    rgb_data = None
                    rgb_depth_data = None
                    if args_cli.record_rgbd:
                        try:
                            rgb_outputs = raw_env.scene["rgb_camera"].data.output
                            rgb_data = rgb_outputs.get("rgb", None)
                            rgb_depth_data = rgb_outputs.get("distance_to_image_plane", None)
                        except Exception:
                            rgb_data = None
                            rgb_depth_data = None

                    if ray_depth_data is not None or rgb_data is not None or rgb_depth_data is not None:
                        for env_id in range(args_cli.num_envs):
                            if env_id in inactive_envs or env_id not in current_assignment:
                                continue
                            row, col, terrain_name = current_assignment[env_id]
                            base_meta = {
                                "step": timestep,
                                "env_id": env_id,
                                "episode_id": current_episode_id[env_id],
                                "terrain": terrain_name,
                                "row": row,
                                "col": col,
                                "command": [args_cli.eval_lin_vel_x, args_cli.eval_lin_vel_y, args_cli.eval_ang_vel_z],
                            }

                            if rgb_data is not None:
                                png_rel, rgb_min, rgb_max = _save_rgb_frame(
                                    output_dir,
                                    rgb_data[env_id].detach().cpu().numpy(),
                                    timestep,
                                    env_id,
                                    current_episode_id[env_id],
                                    row,
                                    col,
                                    terrain_name,
                                )
                                current_frames[env_id].append(
                                    {
                                        **base_meta,
                                        "modality": "rgb",
                                        "png": png_rel,
                                        "label_txt": os.path.splitext(png_rel)[0] + ".txt",
                                        "rgb_min": rgb_min,
                                        "rgb_max": rgb_max,
                                    }
                                )

                            if rgb_depth_data is not None:
                                png_rel, npy_rel, u16_rel, d_min, d_max = _save_depth_frame(
                                    output_dir,
                                    rgb_depth_data[env_id].detach().cpu().numpy(),
                                    "rgb_depth",
                                    timestep,
                                    env_id,
                                    current_episode_id[env_id],
                                    row,
                                    col,
                                    terrain_name,
                                )
                                current_frames[env_id].append(
                                    {
                                        **base_meta,
                                        "modality": "rgb_depth",
                                        "png": png_rel,
                                        "npy": npy_rel,
                                        "u16_png": u16_rel,
                                        "label_txt": os.path.splitext(png_rel)[0] + ".txt",
                                        "depth_min": d_min,
                                        "depth_max": d_max,
                                        "depth_vis_mode": args_cli.depth_vis_mode,
                                    }
                                )

                            if ray_depth_data is not None:
                                png_rel, npy_rel, u16_rel, d_min, d_max = _save_depth_frame(
                                    output_dir,
                                    ray_depth_data[env_id].detach().cpu().numpy(),
                                    "raycaster_depth",
                                    timestep,
                                    env_id,
                                    current_episode_id[env_id],
                                    row,
                                    col,
                                    terrain_name,
                                )
                                current_frames[env_id].append(
                                    {
                                        **base_meta,
                                        "modality": "raycaster_depth",
                                        "png": png_rel,
                                        "npy": npy_rel,
                                        "u16_png": u16_rel,
                                        "label_txt": os.path.splitext(png_rel)[0] + ".txt",
                                        "depth_min": d_min,
                                        "depth_max": d_max,
                                        "depth_vis_mode": args_cli.depth_vis_mode,
                                    }
                                )

                            stats[(row, col)].frames += 1

                done_mask = dones.to(dtype=torch.bool) | visual_failed
                done_env_ids = [env_id for env_id in range(args_cli.num_envs) if bool(done_mask[env_id].item())]
                reassignments = {}
                for env_id in done_env_ids:
                    if env_id in inactive_envs or env_id not in current_assignment:
                        continue
                    row, col, terrain_name = current_assignment[env_id]
                    if bool(visual_failed[env_id].item()) and not bool(dones[env_id].item()):
                        reason = visual_reasons[env_id] or "visual_failure"
                        is_timeout = False
                    else:
                        reason, is_timeout = _termination_reason(raw_env, env_id, infos)
                        if bool(visual_failed[env_id].item()):
                            reason = "+".join([r for r in (reason, visual_reasons[env_id]) if r])
                            is_timeout = False
                    fell = not is_timeout
                    cell = stats[(row, col)]
                    cell.episodes += 1
                    cell.timeouts += int(is_timeout)
                    cell.falls += int(fell)
                    reason_counts[f"{terrain_name}/row{row}/{reason}"] += 1

                    for frame_meta in current_frames[env_id]:
                        frame_meta["episode_complete"] = True
                        frame_meta["episode_fell"] = fell
                        frame_meta["episode_timeout"] = is_timeout
                        frame_meta["termination_reason"] = reason
                        frame_meta["cell_episode_count_after"] = cell.episodes
                        frame_meta["cell_fall_rate_after"] = cell.fall_rate
                        frame_meta["cell_falls_after"] = cell.falls
                        frame_meta["cell_timeouts_after"] = cell.timeouts
                        frame_meta["cell_frames_after"] = cell.frames
                        _write_label_file(output_dir, frame_meta["png"], frame_meta)
                        metadata_file.write(json.dumps(frame_meta, ensure_ascii=False) + "\n")
                    current_frames[env_id].clear()

                    combo_idx, cursor = _next_combo(combos, stats, cursor)
                    if combo_idx is None:
                        inactive_envs.add(env_id)
                        current_assignment.pop(env_id, None)
                    else:
                        current_episode_id[env_id] += 1
                        assignment = combos[combo_idx]
                        current_assignment[env_id] = assignment
                        reassignments[env_id] = assignment
                    episode_steps[env_id] = 0
                    stuck_counts[env_id] = 0
                    base_contact_counts[env_id] = 0

                if reassignments:
                    _set_env_cells(raw_env, reassignments)
                    obs, _ = env.get_observations()

                if timestep % 500 == 0:
                    completed = sum(cell.episodes for cell in stats.values())
                    target = len(stats) * args_cli.episodes_per_cell
                    mean_fall = (
                        sum(cell.falls for cell in stats.values()) / completed if completed > 0 else 0.0
                    )
                    print(
                        f"[EVAL] step={timestep} completed={completed}/{target} "
                        f"active_envs={args_cli.num_envs - len(inactive_envs)} mean_fall={mean_fall:.3f}"
                    )
                    _write_stats(output_dir, stats, reason_counts)

                if all(cell.episodes >= args_cli.episodes_per_cell for cell in stats.values()):
                    print("[EVAL] completed all terrain-row targets.")
                    break

                timestep += 1

        # Some frames can be saved in episodes that are still running when all
        # terrain-row targets have completed. Keep them labeled, but mark them
        # explicitly so downstream dataset builders can filter them out.
        incomplete_frames = 0
        for env_id, frames in enumerate(current_frames):
            for frame_meta in frames:
                row = int(frame_meta["row"])
                col = int(frame_meta["col"])
                cell = stats[(row, col)]
                frame_meta["episode_complete"] = False
                frame_meta["episode_fell"] = None
                frame_meta["episode_timeout"] = None
                frame_meta["termination_reason"] = "incomplete_at_eval_stop"
                frame_meta["cell_episode_count_after"] = cell.episodes
                frame_meta["cell_fall_rate_after"] = cell.fall_rate
                frame_meta["cell_falls_after"] = cell.falls
                frame_meta["cell_timeouts_after"] = cell.timeouts
                frame_meta["cell_frames_after"] = cell.frames
                _write_label_file(output_dir, frame_meta["png"], frame_meta)
                metadata_file.write(json.dumps(frame_meta, ensure_ascii=False) + "\n")
                incomplete_frames += 1
        if incomplete_frames > 0:
            print(f"[EVAL] labeled {incomplete_frames} incomplete frames at shutdown.")

    _write_stats(output_dir, stats, reason_counts)
    env.close()
    simulation_app.close()
    print(f"[EVAL] done. Output: {output_dir}")


if __name__ == "__main__":
    main()
