"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import subprocess
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 地形名称列表（按 TRAINING_SUB_TERRAINS 中 dict 插入顺序，dedup 后）
# 来源：shared_terrain_cfg.py 的 TRAINING_SUB_TERRAINS（注意 raised_mound/pit_crater 重复定义被覆盖）
SUB_TERRAINS_KEYS = [
    "perlin_rough",
    "perlin_rough_stand",
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

# num_rows和num_cols用于计算terrain_idx
TERRAIN_NUM_ROWS = 10
TERRAIN_NUM_COLS = 20

def get_terrain_name(env_id, terrain_type_list):
    if terrain_type_list and len(terrain_type_list) > env_id and terrain_type_list[env_id] != "unknown":
        return terrain_type_list[env_id]
    # 正确计算 terrain_idx: (row * num_cols + col) % num_terrains
    row = env_id // TERRAIN_NUM_COLS
    col = env_id % TERRAIN_NUM_COLS
    terrain_idx = (row * TERRAIN_NUM_COLS + col) % len(SUB_TERRAINS_KEYS)
    return SUB_TERRAINS_KEYS[terrain_idx]

# add argparse arguments
parser = argparse.ArgumentParser(description="使用 Instinct-RL 播放RL智能体。")
parser.add_argument("--video", action="store_true", default=False, help="训练时录制视频。")
parser.add_argument("--video_length", type=int, default=3000, help="录制视频的长度(步数)。")
parser.add_argument("--video_start_step", type=int, default=0, help="开始录制的步数。")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="禁用fabric，使用USD I/O操作。"
)
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量。")
parser.add_argument("--task", type=str, default=None, help="任务名称。")
parser.add_argument("--exportonnx", action="store_true", default=False, help="将策略导出为ONNX模型。")
parser.add_argument("--useonnx", action="store_true", default=False, help="使用ONNX模型进行推理。")
parser.add_argument("--debug", action="store_true", default=False, help="启用调试模式。")
parser.add_argument("--no_resume", default=None, action="store_true", help="强制使用no_resume模式。")
parser.add_argument("--env_cfg", action="store_true", default=False, help="从文件加载环境配置。")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="从文件加载智能体配置。")
parser.add_argument("--sample", action="store_true", default=False, help="使用随机采样动作而非策略。")
parser.add_argument("--zero_act_until", type=int, default=0, help="到指定步数前动作为零。")
parser.add_argument(
    "--walk_load_run",
    type=str,
    default=None,
    help="启用双策略流程，并从该运行目录加载第二个行走策略。",
)
parser.add_argument(
    "--walk_checkpoint",
    type=str,
    default="model_3000.pt",
    help="双策略流程使用的行走策略checkpoint。",
)
parser.add_argument(
    "--stand_policy_seconds",
    type=float,
    default=3.0,
    help="低增益接管后，站立策略独立稳定站立的时间。",
)
parser.add_argument(
    "--walk_policy_blend_seconds",
    type=float,
    default=1.5,
    help="从站立策略平滑混合到行走策略的时间；设为0表示瞬时切换。",
)
parser.add_argument("--walk_cmd_vx", type=float, default=0.10, help="双策略流程最终前向速度命令。")
parser.add_argument("--walk_cmd_vy", type=float, default=0.0, help="双策略流程最终侧向速度命令。")
parser.add_argument("--walk_cmd_wz", type=float, default=0.0, help="双策略流程最终偏航角速度命令。")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="启用键盘控制(WASD走, QE转, X归零)。")
parser.add_argument("--auto_policy", action="store_true", default=False, help="自动策略模式：所有 env 跑 base_velocity 命令，不接收键盘输入。")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.1, help="键盘每次调整的线速度增量。")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="键盘控制的最大角速度。")
parser.add_argument("--keyboard_angvel_step", type=float, default=0.1, help="键盘每次调整的角速度增量。")
parser.add_argument("--free_view", action="store_true", default=False, help="自由视角（不跟随机器人）。")
parser.add_argument(
    "--follow_view",
    action="store_true",
    default=False,
    help="固定跟随env 0的robot根节点，适合观察完整起立和策略切换流程。",
)
parser.add_argument("--debug_ray", action="store_true", default=False, help="启用射线检测可视化。")
parser.add_argument(
    "--plot_leg_pd",
    action="store_true",
    default=False,
    help="实时显示12个腿电机的Kp和Kd曲线，并在增益变化时打印实际值。",
)
parser.add_argument(
    "--pd_plot_history",
    type=int,
    default=500,
    help="PD曲线保留的历史步数。",
)
parser.add_argument(
    "--pd_plot_interval",
    type=int,
    default=5,
    help="每隔多少个仿真步刷新一次PD曲线。",
)
parser.add_argument(
    "--disable_arm_disturbance",
    action="store_true",
    default=False,
    help="关闭 play 中机械臂末端载荷随机和机械臂 interval 姿态扰动，便于和无扰动策略做对照。",
)
parser.add_argument(
    "--no_terminate",
    action="store_true",
    default=False,
    help="放宽 play 终止条件，仅保留明显翻倒类终止，便于长时间手动测试。",
)
parser.add_argument(
    "--play_row",
    type=int,
    default=None,
    help="指定出生在第 N 行（0-based，最简单是 0，最难是 num_rows-1）。仅 num_envs=1 时生效。",
)
parser.add_argument(
    "--play_col",
    type=int,
    default=None,
    help="指定出生在第 N 列（0 ~ num_cols-1）。仅 num_envs=1 时生效。",
)
parser.add_argument(
    "--play_terrain",
    type=str,
    default=None,
    help="按地形名指定出生列（如 raised_mound / circle_track）。仅 num_envs=1 时生效。",
)
parser.add_argument(
    "--play_terrain_set",
    type=str,
    default=None,
    choices=[
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
    ],
    help="将 play 场景切换到 physical terrain 集合。注意：当前仅支持 static physical terrains，不包含 visualize_terrain_only.py 里额外动态生成的碎块/滚柱列。",
)
parser.add_argument(
    "--play_material",
    type=str,
    default=None,
    choices=["default", "low_friction", "springy", "high_grip", "slippery_bouncy", "damped_soft_like"],
    help="覆盖 play 地形的 physics material。常与 --play_terrain_set 搭配使用。",
)
parser.add_argument("--show_first_person_rgbd", action="store_true", default=False, help="显示机器人第一视角 RGB 和 Depth 两个实时小窗口。")
parser.add_argument(
    "--first_person_depth_source",
    type=str,
    default="raycaster",
    choices=["raycaster", "rgb_camera"],
    help="第一视角深度窗口显示来源：raycaster(策略原深度) 或 rgb_camera(额外RGBD相机深度)。",
)
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次俯视深度图，0表示禁用。")
parser.add_argument("--save_rgb_zhengshi_interval", type=int, default=0, help="每N步保存一次rgb_camera的RGB和深度图，0表示禁用。")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.save_rgb_zhengshi_interval > 0:
    args_cli.enable_cameras = True
if args_cli.show_first_person_rgbd:
    args_cli.enable_cameras = True

print(f"[DEBUG] args_cli.video = {args_cli.video}")
print(f"[DEBUG] args_cli.enable_cameras = {getattr(args_cli, 'enable_cameras', 'NOT_SET')}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from PIL import Image
import numpy as np
import cv2

import carb.input
import omni.appwindow
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab.utils.io import  load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg
from instinctlab.terrains.physical_terrain_cfg import (
    PHYSICAL_STUDY_TERRAIN_NAMES,
    PHYSICAL_DYNAMIC_ARENA_NAMES,
    PHYSICAL_MATERIAL_PRESETS,
    PHYSICAL_TERRAIN_COLLECTIONS,
)
from instinctlab.terrains.physical_dynamic_arenas import spawn_dynamic_arena_column
from instinctlab.terrains.physical_dynamic_arenas import dynamic_arena_center_y
from instinctlab.terrains.shared_terrain_cfg import SHARED_SUB_TERRAINS, TRAINING_SUB_TERRAINS
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator


RGB_WINDOW_NAME = "B2RM First-Person RGB"
DEPTH_WINDOW_NAME = "B2RM First-Person Depth"
KP_WINDOW_NAME = "B2RM Leg Kp"
KD_WINDOW_NAME = "B2RM Leg Kd"


class _LegPdPlotter:
    """Plot the effective gains of every leg motor in two live windows."""

    def __init__(self, env, history_steps: int, update_interval: int):
        from collections import deque
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        if "legs" not in robot.actuators:
            raise RuntimeError("Robot has no actuator group named 'legs'.")
        self._actuator = robot.actuators["legs"]
        self._step_dt = float(unwrapped.step_dt)
        self._history_steps = max(2, int(history_steps))
        self._update_interval = max(1, int(update_interval))
        self._times = deque(maxlen=self._history_steps)
        self._kp_history = deque(maxlen=self._history_steps)
        self._kd_history = deque(maxlen=self._history_steps)
        self._last_printed_kp = None
        self._last_printed_kd = None

        action_term = getattr(unwrapped, "_leg_action_term", None)
        if action_term is not None:
            self._joint_names = list(action_term._joint_names)
        else:
            indices = self._actuator.joint_indices
            if isinstance(indices, slice):
                indices = range(*indices.indices(len(robot.joint_names)))
            self._joint_names = [robot.joint_names[index] for index in indices]

        num_motors = int(self._actuator.stiffness.shape[-1])
        if len(self._joint_names) != num_motors:
            self._joint_names = [f"leg_motor_{index}" for index in range(num_motors)]

        self._kp_figure = Figure(figsize=(9.0, 4.8), dpi=100, tight_layout=True)
        self._kd_figure = Figure(figsize=(9.0, 4.8), dpi=100, tight_layout=True)
        self._kp_canvas = FigureCanvasAgg(self._kp_figure)
        self._kd_canvas = FigureCanvasAgg(self._kd_figure)
        self._kp_axis = self._kp_figure.add_subplot(111)
        self._kd_axis = self._kd_figure.add_subplot(111)
        colors = colormaps["tab20"].resampled(num_motors)
        self._kp_lines = []
        self._kd_lines = []
        for index, joint_name in enumerate(self._joint_names):
            color = colors(index)
            (kp_line,) = self._kp_axis.plot([], [], color=color, linewidth=1.6, label=joint_name)
            (kd_line,) = self._kd_axis.plot([], [], color=color, linewidth=1.6, label=joint_name)
            self._kp_lines.append(kp_line)
            self._kd_lines.append(kd_line)

        self._configure_axis(self._kp_axis, "Effective leg stiffness (Kp)", "Kp")
        self._configure_axis(self._kd_axis, "Effective leg damping (Kd)", "Kd")
        cv2.namedWindow(KP_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(KD_WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[PD PLOT] joint order: " + ", ".join(self._joint_names))

    @staticmethod
    def _configure_axis(axis, title: str, ylabel: str) -> None:
        axis.set_title(title)
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)

    @staticmethod
    def _env0_values(values: torch.Tensor) -> np.ndarray:
        if values.ndim == 1:
            return values.detach().cpu().numpy().copy()
        return values[0].detach().cpu().numpy().copy()

    def update(self, timestep: int) -> str | None:
        kp = self._env0_values(self._actuator.stiffness)
        kd = self._env0_values(self._actuator.damping)
        self._times.append(timestep * self._step_dt)
        self._kp_history.append(kp)
        self._kd_history.append(kd)

        gains_changed = (
            self._last_printed_kp is None
            or not np.allclose(kp, self._last_printed_kp, atol=1e-5)
            or not np.allclose(kd, self._last_printed_kd, atol=1e-5)
        )
        if gains_changed:
            print(
                f"[PD] step={timestep} t={timestep * self._step_dt:.3f}s "
                f"Kp={np.array2string(kp, precision=2, separator=',')} "
                f"Kd={np.array2string(kd, precision=2, separator=',')}"
            )
            self._last_printed_kp = kp.copy()
            self._last_printed_kd = kd.copy()

        if timestep % self._update_interval != 0:
            return self._read_key()
        times = np.asarray(self._times)
        kp_history = np.asarray(self._kp_history)
        kd_history = np.asarray(self._kd_history)
        self._draw_plot(self._kp_axis, self._kp_lines, self._kp_canvas, times, kp_history, KP_WINDOW_NAME)
        self._draw_plot(self._kd_axis, self._kd_lines, self._kd_canvas, times, kd_history, KD_WINDOW_NAME)
        return self._read_key()

    @staticmethod
    def _read_key() -> str | None:
        key_code = cv2.waitKey(1) & 0xFF
        if key_code in (ord("w"), ord("s"), ord("a"), ord("d"), ord("q"), ord("e"), ord("x")):
            return chr(key_code).upper()
        return None

    @staticmethod
    def _draw_plot(axis, lines, canvas, times, values, window_name: str) -> None:
        for index, line in enumerate(lines):
            line.set_data(times, values[:, index])
        axis.relim()
        axis.autoscale_view()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.imshow(window_name, bgr)

    @staticmethod
    def close() -> None:
        for window_name in (KP_WINDOW_NAME, KD_WINDOW_NAME):
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
        cv2.waitKey(1)


def _build_playable_physical_terrain_cfg(base_cfg, terrain_names: list[str]) -> FiledTerrainGeneratorCfg:
    """Rebuild a physical terrain cfg without stripping flat_patch_sampling.

    The visualization presets disable flat patch sampling to avoid patch-shape
    mismatches, but the play command generator needs a valid 'target' patch.
    """
    sub_terrains = {}
    for idx, name in enumerate(terrain_names):
        if name in TRAINING_SUB_TERRAINS:
            source_cfg = TRAINING_SUB_TERRAINS[name]
        elif name in SHARED_SUB_TERRAINS:
            source_cfg = SHARED_SUB_TERRAINS[name]
        else:
            raise RuntimeError(f"Unknown physical terrain name for play: {name}")
        copied_cfg = copy.deepcopy(source_cfg)
        # Relax target patch search for physical-test play scenes so the
        # terrain-based command generator always has reachable goals.
        copied_cfg.flat_patch_sampling = {
            "target": FlatPatchSamplingCfg(
                num_patches=4,
                patch_radius=[0.05, 0.10, 0.15],
                max_height_diff=0.35,
                x_range=(-0.6, 0.6),
                y_range=(-0.6, 0.6),
            )
        }
        sub_terrains[f"terrain_{idx}"] = copied_cfg

    cfg = copy.deepcopy(base_cfg)
    cfg.class_type = FiledTerrainGenerator
    cfg.terrain_layout = terrain_names
    cfg.sub_terrains = sub_terrains
    cfg.num_cols = len(terrain_names)
    return cfg


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert simulator RGB output into uint8 HWC image."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.nanmax(image)) if image.size > 0 else 0.0
        if max_val <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _colorize_depth(depth: np.ndarray, min_depth: float = 0.0, max_depth: float = 10.0) -> np.ndarray:
    """Map depth image to a colored uint8 BGR image for cv2 display."""
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    depth = np.nan_to_num(depth, nan=min_depth, posinf=max_depth, neginf=min_depth)
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / max(max_depth - min_depth, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def _show_first_person_rgbd_windows(env, timestep: int) -> None:
    """Render first-person RGB and depth into two small cv2 windows."""
    try:
        rgb_camera_output = env.unwrapped.scene["rgb_camera"].data.output
        rgb_tensor = rgb_camera_output.get("rgb")
        if rgb_tensor is None or len(rgb_tensor) == 0:
            return

        rgb_np = _ensure_uint8_rgb(rgb_tensor[0].cpu().numpy())
        depth_np = None
        if args_cli.first_person_depth_source == "raycaster":
            raycaster_output = env.unwrapped.scene["camera"].data.output
            depth_tensor = raycaster_output.get("distance_to_image_plane")
        else:
            depth_tensor = rgb_camera_output.get("distance_to_image_plane")

        if depth_tensor is not None and len(depth_tensor) > 0:
            depth_np = depth_tensor[0].cpu().numpy()

        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        if depth_np is None:
            return
        depth_bgr = _colorize_depth(depth_np)

        target_size = (384, 216)
        rgb_vis = cv2.resize(rgb_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        depth_vis = cv2.resize(depth_bgr, target_size, interpolation=cv2.INTER_NEAREST)

        depth_window_name = f"{DEPTH_WINDOW_NAME} ({args_cli.first_person_depth_source})"
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RGB_WINDOW_NAME, *target_size)
        cv2.resizeWindow(depth_window_name, *target_size)
        cv2.imshow(RGB_WINDOW_NAME, rgb_vis)
        cv2.imshow(depth_window_name, depth_vis)
        cv2.waitKey(1)
    except Exception as e:
        if timestep % 200 == 0:
            print(f"[DEBUG] Failed to display first-person RGBD windows: {e}")

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Play with Instinct-RL agent."""
    if args_cli.free_view and args_cli.follow_view:
        raise ValueError("--free_view and --follow_view cannot be enabled together.")
    if args_cli.walk_load_run is not None and args_cli.useonnx:
        raise ValueError("The dual-policy play pipeline does not support --useonnx; load both PyTorch checkpoints.")
    if args_cli.stand_policy_seconds < 0.0 or args_cli.walk_policy_blend_seconds < 0.0:
        raise ValueError("Policy phase durations must be non-negative.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    if args_cli.disable_arm_disturbance and hasattr(env_cfg, "events") and env_cfg.events is not None:
        if hasattr(env_cfg.events, "arm_tip_payload"):
            env_cfg.events.arm_tip_payload = None
        if hasattr(env_cfg.events, "arm_safe_carry_pose"):
            env_cfg.events.arm_safe_carry_pose = None
        if hasattr(env_cfg.events, "arm_workspace_target_reset"):
            env_cfg.events.arm_workspace_target_reset = None
        if hasattr(env_cfg.events, "arm_workspace_target_interval"):
            env_cfg.events.arm_workspace_target_interval = None
        if hasattr(env_cfg.events, "arm_pose_interval"):
            env_cfg.events.arm_pose_interval = None
        print("[INFO] --disable_arm_disturbance: 已关闭 play 中机械臂末端载荷随机与安全携带姿态扰动。")

    active_play_material = None
    if args_cli.play_terrain_set is not None:
        if args_cli.play_terrain_set not in PHYSICAL_TERRAIN_COLLECTIONS:
            raise RuntimeError(
                f"Unsupported --play_terrain_set={args_cli.play_terrain_set!r}. "
                f"Available: {sorted(PHYSICAL_TERRAIN_COLLECTIONS.keys())}"
            )
        terrain_collection = PHYSICAL_TERRAIN_COLLECTIONS[args_cli.play_terrain_set]
        terrain_generator_cfg = _build_playable_physical_terrain_cfg(
            base_cfg=terrain_collection["terrain_cfg"],
            terrain_names=PHYSICAL_STUDY_TERRAIN_NAMES,
        )
        env_cfg.scene.terrain.terrain_generator = terrain_generator_cfg
        if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
            env_cfg.scene.terrain.max_init_terrain_level = 0
        selected_material = args_cli.play_material or terrain_collection["default_material"]
        active_play_material = selected_material
        env_cfg.scene.terrain.physics_material = copy.deepcopy(PHYSICAL_MATERIAL_PRESETS[selected_material])
        print(
            f"[INFO] Overrode play terrain set to {args_cli.play_terrain_set} "
            f"with material={selected_material}, num_rows={terrain_generator_cfg.num_rows}, "
            f"num_cols={terrain_generator_cfg.num_cols}"
        )
        print("[INFO] Physical terrain-set override enabled.")

    dynamic_play_terrain_cols: dict[str, int] = {}
    if (
        args_cli.play_terrain_set is not None
        and "curriculum" in args_cli.play_terrain_set
        and args_cli.play_terrain in PHYSICAL_DYNAMIC_ARENA_NAMES
    ):
        terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
        num_rows = int(terrain_generator_cfg.num_rows)
        num_cols = int(terrain_generator_cfg.num_cols)
        total_cols = num_cols + len(PHYSICAL_DYNAMIC_ARENA_NAMES)
        target_dynamic_row = 0 if args_cli.play_row is None else int(args_cli.play_row)
        target_dynamic_row = max(0, min(num_rows - 1, target_dynamic_row))
        # Match the same centered grid convention used by terrain.env_origins / fallback relocation.
        row_centers_x = [
            -(row_idx - (num_rows - 1) / 2) * env_cfg.scene.env_spacing for row_idx in range(num_rows)
        ]
        material_name_for_dynamic = active_play_material or "default"
        for arena_offset, arena_name in enumerate(PHYSICAL_DYNAMIC_ARENA_NAMES):
            arena_col = num_cols + arena_offset
            y_center = dynamic_arena_center_y(arena_offset, num_cols, env_cfg.scene.env_spacing)
            dynamic_play_terrain_cols[arena_name] = arena_col
            if arena_name == args_cli.play_terrain:
                prim_path = f"/World/play_dynamic/{arena_name}_{material_name_for_dynamic}"
                spawn_dynamic_arena_column(
                    arena_name=arena_name,
                    root_path=prim_path,
                    material_name=material_name_for_dynamic,
                    y_center=y_center,
                    row_centers_x=row_centers_x,
                    row_indices=[target_dynamic_row],
                )
        print(
            f"[INFO] Pre-spawned dynamic play arena '{args_cli.play_terrain}' for row {target_dynamic_row}. "
            f"Dynamic column map: {dynamic_play_terrain_cols}"
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        # Normalize load_run to avoid trailing slash causing os.path.basename() to return ''
        # which would make get_checkpoint_path treat it as a regex that matches everything.
        agent_cfg.load_run = agent_cfg.load_run.rstrip("/\\")
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

    if args_cli.free_view and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (4.0, 4.0, 4.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    elif args_cli.follow_view and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (4.0, 0.75, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.75, 0.35)
        print("[INFO] --follow_view: camera follows env 0 robot root.")

    if args_cli.debug_ray:
        env_cfg.scene.left_height_scanner.debug_vis = True
        env_cfg.scene.right_height_scanner.debug_vis = True
        env_cfg.scene.leg_volume_points.debug_vis = True
        env_cfg.scene.camera.debug_vis = True

    if args_cli.no_terminate:
        env_cfg.terminations.terrain_out_bound = None
        env_cfg.terminations.root_height = None
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.leg_link_contact = None
        env_cfg.terminations.calf_link_contact = None
        print("[INFO] --no_terminate: 已关闭出界/高度/link接触终止，仅保留明显翻倒类终止。")

    import time
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_depth_dir = None
    save_rgb_zhengshi_dir = None

    if args_cli.save_depth_interval > 0 or args_cli.save_rgb_zhengshi_interval > 0:
        depth_run_dir = os.path.join(log_dir, f"depth_run_{run_id}")
        os.makedirs(depth_run_dir, exist_ok=True)
        print(f"[INFO] Depth run directory: {depth_run_dir}")

    if args_cli.save_depth_interval > 0:
        save_depth_dir = os.path.join(depth_run_dir, "raycaster")
        os.makedirs(save_depth_dir, exist_ok=True)
        print(f"[INFO] Saving raycaster depth to: {save_depth_dir}")
        print(f"[INFO] Will save every {args_cli.save_depth_interval} steps")

    if args_cli.save_rgb_zhengshi_interval > 0:
        save_rgb_zhengshi_dir = os.path.join(depth_run_dir, "rgbd_zhengshi")
        os.makedirs(save_rgb_zhengshi_dir, exist_ok=True)
        print(f"[INFO] Saving rgb_camera to: {save_rgb_zhengshi_dir}")
        print(f"[INFO] Will save every {args_cli.save_rgb_zhengshi_interval} steps")

    needs_rendered_camera = (
        args_cli.video
        or args_cli.save_rgb_zhengshi_interval > 0
        or args_cli.show_first_person_rgbd
        or args_cli.first_person_depth_source == "rgb_camera"
    )
    if not needs_rendered_camera:
        if hasattr(env_cfg.scene, "rgb_camera"):
            env_cfg.scene.rgb_camera = None
            print("[INFO] Disabled scene.rgb_camera for play/export; raycaster depth camera remains enabled.")
        if hasattr(env_cfg.scene, "camera_rgb_record"):
            env_cfg.scene.camera_rgb_record = None
            print("[INFO] Disabled scene.camera_rgb_record for play/export.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.play_terrain_set is not None and "curriculum" in args_cli.play_terrain_set:
        terrain = env.unwrapped.scene.terrain
        num_cols = int(terrain.cfg.terrain_generator.num_cols)
        for arena_offset, arena_name in enumerate(PHYSICAL_DYNAMIC_ARENA_NAMES):
            dynamic_play_terrain_cols[arena_name] = num_cols + arena_offset

    # 处理 --play_row / --play_col / --play_terrain：把单 env 重定位到指定网格
    if env_cfg.scene.num_envs == 1 and (
        args_cli.play_row is not None
        or args_cli.play_col is not None
        or args_cli.play_terrain is not None
    ):
        terrain = env.unwrapped.scene.terrain
        if not hasattr(terrain, "env_origins"):
            raise RuntimeError("terrain has no env_origins")

        num_rows = int(terrain.cfg.terrain_generator.num_rows)
        num_cols = int(terrain.cfg.terrain_generator.num_cols)
        total_cols = num_cols + len(dynamic_play_terrain_cols)
        dynamic_target = False

        # 默认 row = 0, col = 0
        target_row = 0 if args_cli.play_row is None else int(args_cli.play_row)
        target_col = 0 if args_cli.play_col is None else int(args_cli.play_col)

        # --play_terrain 按名查 col（取第一个匹配列）
        if args_cli.play_terrain is not None:
            if args_cli.play_terrain in dynamic_play_terrain_cols:
                target_col = dynamic_play_terrain_cols[args_cli.play_terrain]
                dynamic_target = True
            elif hasattr(terrain, "terrain_names"):
                names_2d = terrain.terrain_names
                matched_cols = [
                    c for c in range(num_cols) if any(names_2d[r, c] == args_cli.play_terrain for r in range(num_rows))
                ]
                if not matched_cols:
                    raise RuntimeError(
                        f"--play_terrain={args_cli.play_terrain!r} not found in terrain_names; "
                        f"available: {sorted(set(names_2d.flatten().tolist())) + list(dynamic_play_terrain_cols.keys())}"
                    )
                target_col = matched_cols[0]
            else:
                raise RuntimeError("terrain.terrain_names not available; cannot resolve --play_terrain")

        # 边界检查
        if not (0 <= target_row < num_rows):
            raise RuntimeError(
                f"--play_row={target_row} 越界, 合法范围 [0, {num_rows - 1}]。"
                f" 注意这里是 0-based 行号：最简单是 0，最难是 {num_rows - 1}。"
            )
        if not (0 <= target_col < total_cols):
            raise RuntimeError(f"--play_col={target_col} 越界, 合法范围 [0, {total_cols - 1}]")

        # Isaac Lab env_origins 形状是 (num_envs, 3)，单 env 覆盖 env 0
        # terrain.terrain_origins 是 (num_rows, num_cols, 3)，curriculum=True 时
        # env_origins[i] = terrain_origins[terrain_levels[i], terrain_types[i]]
        if not dynamic_target and hasattr(terrain, "terrain_origins") and terrain.terrain_origins is not None:
            new_origin = terrain.terrain_origins[target_row, target_col].clone()
        else:
            if dynamic_target:
                dynamic_idx = target_col - num_cols
                y_coord = dynamic_arena_center_y(dynamic_idx, num_cols, env_cfg.scene.env_spacing)
            else:
                y_coord = (target_col - (total_cols - 1) / 2) * env_cfg.scene.env_spacing
            new_origin = torch.tensor(
                [
                    -(target_row - (num_rows - 1) / 2) * env_cfg.scene.env_spacing,
                    y_coord,
                    0.0,
                ],
                device=terrain.env_origins.device,
                dtype=terrain.env_origins.dtype,
            )

        # 同步 terrain / scene 的 origin 与课程索引
        terrain.env_origins[0] = new_origin
        if hasattr(env.unwrapped.scene, "env_origins") and env.unwrapped.scene.env_origins is not None:
            env.unwrapped.scene.env_origins[0] = new_origin
        if hasattr(terrain, "terrain_levels") and terrain.terrain_levels is not None:
            terrain.terrain_levels[0] = target_row
        if not dynamic_target and hasattr(terrain, "terrain_types") and terrain.terrain_types is not None:
            terrain.terrain_types[0] = target_col

        # 仅重置 env 0，让机器人真正出生到目标行列，而不是只改元数据。
        env_id_tensor = torch.tensor([0], device=env.unwrapped.device, dtype=torch.long)
        env.unwrapped._reset_idx(env_id_tensor)

        # 打印定位信息
        if dynamic_target:
            tname = args_cli.play_terrain
        elif hasattr(terrain, "terrain_names"):
            tname = terrain.terrain_names[target_row, target_col]
        else:
            tname = "?"
        actual_row = int(terrain.terrain_levels[0].item()) if hasattr(terrain, "terrain_levels") else -1
        actual_col = target_col if dynamic_target else (int(terrain.terrain_types[0].item()) if hasattr(terrain, "terrain_types") else -1)
        print(
            f"[PLAY LOC] 重定位单 env: target_row={target_row}, target_col={target_col}, "
            f"actual_row={actual_row}, actual_col={actual_col}, "
            f"terrain={tname}, origin={new_origin.tolist()}"
        )

    # 打印 play 地图布局：第几行/列是什么地形（方便键盘巡检时知道走到哪）
    try:
        terrain = getattr(env.unwrapped.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "terrain_names"):
            names_2d = terrain.terrain_names  # (num_rows, num_cols) 数组
            num_rows, num_cols = names_2d.shape
            print(f"[PLAY MAP] {num_rows} 行 × {num_cols} 列:")
            if num_rows > 1:
                print(f"[PLAY MAP] 难度按行递增: row 0 最简单, row {num_rows - 1} 最难")
            # 按列优先打印：每一列的所有行一起
            for col in range(num_cols):
                col_entries = []
                for row in range(num_rows):
                    name = names_2d[row, col]
                    col_entries.append(f"r{row}={name}" if name else f"r{row}=?")
                print(f"[PLAY MAP]   col {col:>2d}: " + ", ".join(col_entries))
        elif terrain is not None and hasattr(terrain, "terrain_type_names"):
            # fallback：只有列名（无 row 信息）
            col_names = terrain.terrain_type_names
            print(f"[PLAY MAP] 列 → 地形 (总 {len(col_names)} 列):")
            for col, name in enumerate(col_names):
                if name:
                    print(f"[PLAY MAP]   col {col:>2d} = {name}")
        else:
            terrain_types = terrain.terrain_types.cpu().numpy() if terrain is not None else None
            if terrain_types is not None:
                num_cols = terrain.cfg.terrain_generator.num_cols
                print(f"[PLAY MAP] 总 {num_cols} 列 (按 curriculum 比例):")
                for col in range(num_cols):
                    col_ids = terrain_types[:, col]
                    uniq = list(dict.fromkeys([int(x) for x in col_ids]))
                    name = SUB_TERRAINS_KEYS[uniq[0]] if len(uniq) == 1 and uniq[0] < len(SUB_TERRAINS_KEYS) else f"mixed{uniq}"
                    print(f"[PLAY MAP]   col {col:>2d} = {name}")
    except Exception as e:
        print(f"[PLAY MAP] 无法打印地图布局: {e}")
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        print(f"[DEBUG] RecordVideo wrapper applied. env type: {type(env)}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    # OnPolicyRunner consumes parts of its configuration with pop(). Keep the
    # original dictionary intact so a second runner can load the walking policy.
    ppo_runner = OnPolicyRunner(
        env,
        copy.deepcopy(agent_cfg_dict),
        log_dir=None,
        device=agent_cfg.device,
    )
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    walk_policy = None
    walk_resume_path = None
    if args_cli.walk_load_run is not None:
        walk_load_run = args_cli.walk_load_run.rstrip("/\\")
        if os.path.isabs(walk_load_run):
            walk_resume_path = get_checkpoint_path(
                os.path.dirname(walk_load_run),
                os.path.basename(walk_load_run),
                args_cli.walk_checkpoint,
            )
        else:
            walk_resume_path = get_checkpoint_path(
                log_root_path,
                walk_load_run,
                args_cli.walk_checkpoint,
            )
        walk_runner = OnPolicyRunner(
            env,
            copy.deepcopy(agent_cfg_dict),
            log_dir=None,
            device=agent_cfg.device,
        )
        print(f"[PIPELINE] Loading walking policy checkpoint from: {walk_resume_path}")
        walk_runner.load(walk_resume_path)
        walk_policy = walk_runner.get_inference_policy(device=env.unwrapped.device)
        print(
            "[PIPELINE] scripted stand-up -> stand policy "
            f"({args_cli.stand_policy_seconds:.2f}s) -> walking blend "
            f"({args_cli.walk_policy_blend_seconds:.2f}s) -> walking policy; "
            f"final cmd=({args_cli.walk_cmd_vx:+.2f}, {args_cli.walk_cmd_vy:+.2f}, "
            f"{args_cli.walk_cmd_wz:+.2f})"
        )

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    # use the exported model for inference
    if args_cli.useonnx:
        from onnxer import load_parkour_onnx_model

        # NOTE: This is only applicable with parkour task
        obs_segments = env.get_obs_segments()
        proprio_components = [
            component
            for component in [
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "velocity_commands",
                "joint_pos",
                "joint_vel",
                "actions",
            ]
            if component in obs_segments
        ]
        onnx_policy = load_parkour_onnx_model(
            model_dir=os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                obs_segments,
                temporal=True,
            ),
            depth_shape=obs_segments["depth_image"],
            proprio_slice=slice(
                0,
                get_subobs_size(
                    obs_segments,
                    proprio_components,
                ),
            ),
        )

    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")
    keyboard_command_enabled = args_cli.keyboard_control or walk_policy is not None

    def set_velocity_command(observation: torch.Tensor, command: torch.Tensor) -> None:
        repeats = command_obs_slice[1][0] // 3
        observation[:, command_obs_slice[0]] = command.repeat(1, repeats)

    cmd_display_names = {"W": "前进", "S": "减速", "A": "左转", "D": "右转", "Q": "停转", "E": "停转", "X": "急停"}
    _print_vel_help = True

    print("=" * 60)
    print("键盘控制已启用:")
    print("  W          : 前进加速  (+" + str(args_cli.keyboard_linvel_step) + " m/s)")
    print("  S          : 前进减速  (-" + str(args_cli.keyboard_linvel_step) + " m/s, 最低 0)")
    print(
        "  A / D      : 左转 / 右转  (每次 "
        + str(args_cli.keyboard_angvel_step)
        + " rad/s, 上限 "
        + str(args_cli.keyboard_angvel)
        + " rad/s)"
    )
    print("  Q / E      : 停止转向")
    print("  X          : 急停归零")
    print("  长按 W / S / A / D 可累计调整速度")
    print("=" * 60)

    def apply_keyboard_command(name: str) -> None:
        if name == "X":
            override_command[:] = 0.0
        elif name == "W":
            override_command[:, 0] += args_cli.keyboard_linvel_step
        elif name == "S":
            override_command[:, 0] = torch.clamp(
                override_command[:, 0] - args_cli.keyboard_linvel_step,
                min=0.0,
            )
        elif name == "A":
            override_command[:, 2] = torch.clamp(
                override_command[:, 2] + args_cli.keyboard_angvel_step,
                min=-args_cli.keyboard_angvel,
                max=args_cli.keyboard_angvel,
            )
        elif name == "D":
            override_command[:, 2] = torch.clamp(
                override_command[:, 2] - args_cli.keyboard_angvel_step,
                min=-args_cli.keyboard_angvel,
                max=args_cli.keyboard_angvel,
            )
        elif name in ("Q", "E"):
            override_command[:, 2] = 0.0
        vx = override_command[0, 0].item()
        vy = override_command[0, 1].item()
        wz = override_command[0, 2].item()
        print(
            f"[键盘] {cmd_display_names.get(name, name):>4s} | "
            f"cmd=(v_x={vx:+.2f}, v_y={vy:+.2f}, ω_z={wz:+.2f})"
        )

    def on_keyboard_input(e):
        global _print_vel_help
        if not keyboard_command_enabled:
            return
        key_map = {
            carb.input.KeyboardInput.W: "W",
            carb.input.KeyboardInput.S: "S",
            carb.input.KeyboardInput.A: "A",
            carb.input.KeyboardInput.D: "D",
            carb.input.KeyboardInput.Q: "Q",
            carb.input.KeyboardInput.E: "E",
            carb.input.KeyboardInput.X: "X",
        }
        if e.input in key_map:
            name = key_map[e.input]
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                apply_keyboard_command(name)
                _print_vel_help = True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # 获取obs切片信息，用于打印实际速度
    obs_segments = env.get_obs_segments()
    vel_slice = get_obs_slice(obs_segments, "base_lin_vel") if "base_lin_vel" in obs_segments else None

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    episode_counts = {}  # track episodes per env
    num_envs = env.unwrapped.scene.num_envs
    last_debug_cmd = torch.tensor([999.0, 999.0, 999.0], device=env.device)
    pipeline_phase = None
    fixed_walk_command = torch.tensor(
        [args_cli.walk_cmd_vx, args_cli.walk_cmd_vy, args_cli.walk_cmd_wz],
        device=env.device,
    ).repeat(env.num_envs, 1)
    if walk_policy is not None:
        # In the dual-policy pipeline, fixed CLI commands are the keyboard
        # command's initial value. Keyboard input is enabled automatically.
        override_command.copy_(fixed_walk_command)
        print(
            "[PIPELINE] keyboard control enabled automatically; "
            f"initial cmd=({override_command[0, 0].item():+.2f}, "
            f"{override_command[0, 1].item():+.2f}, {override_command[0, 2].item():+.2f})"
        )
    pd_plotter = None
    if args_cli.plot_leg_pd:
        pd_plotter = _LegPdPlotter(
            env,
            history_steps=args_cli.pd_plot_history,
            update_interval=args_cli.pd_plot_interval,
        )

    def summarize_termination(env_id, infos):
        episode_info = infos.get("episode", {})
        active_reasons = []
        for key, value in episode_info.items():
            if "Episode_Termination" not in key:
                continue
            try:
                reason_value = float(value[env_id].item())
            except Exception:
                continue
            if reason_value > 0.5:
                active_reasons.append(f"{key}={reason_value:.3f}")
        if active_reasons:
            return ", ".join(active_reasons)

        time_outs = infos.get("time_outs", None)
        if time_outs is not None:
            try:
                if bool(time_outs[env_id].item()):
                    return "time_outs=1.000"
            except Exception:
                pass
        return "unknown"

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if walk_policy is not None:
                unwrapped = env.unwrapped
                policy_active = getattr(
                    unwrapped,
                    "handoff_policy_active",
                    torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                )
                policy_steps = getattr(
                    unwrapped,
                    "policy_step_buf",
                    torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
                )
                policy_age = policy_steps.float() * float(unwrapped.step_dt)
                blend_age = policy_age - args_cli.stand_policy_seconds
                if args_cli.walk_policy_blend_seconds > 0.0:
                    walk_alpha = (blend_age / args_cli.walk_policy_blend_seconds).clamp(0.0, 1.0)
                    walk_alpha = walk_alpha * walk_alpha * (3.0 - 2.0 * walk_alpha)
                else:
                    walk_alpha = (blend_age >= 0.0).float()
                walk_alpha = torch.where(policy_active, walk_alpha, torch.zeros_like(walk_alpha))

                target_walk_command = override_command
                stand_obs = obs.clone()
                walk_obs = obs.clone()
                set_velocity_command(stand_obs, torch.zeros_like(target_walk_command))
                set_velocity_command(walk_obs, target_walk_command)
                stand_actions = policy(stand_obs)
                walk_actions = walk_policy(walk_obs)
                actions = torch.lerp(stand_actions, walk_actions, walk_alpha[:, None])

                if not bool(policy_active[0].item()):
                    current_pipeline_phase = "scripted_stand_up"
                elif float(policy_age[0].item()) < args_cli.stand_policy_seconds:
                    current_pipeline_phase = "stand_policy"
                elif float(walk_alpha[0].item()) < 1.0:
                    current_pipeline_phase = "walking_blend"
                else:
                    current_pipeline_phase = "walking_policy"
                if current_pipeline_phase != pipeline_phase:
                    pipeline_phase = current_pipeline_phase
                    print(
                        f"[PIPELINE] env0 phase={pipeline_phase} "
                        f"policy_age={policy_age[0].item():.2f}s "
                        f"walk_alpha={walk_alpha[0].item():.3f}"
                    )
            else:
                if args_cli.keyboard_control:
                    set_velocity_command(obs, override_command)
                actions = policy(obs)
            if args_cli.useonnx:
                torch_actions = actions
                actions = onnx_policy(obs)
                if (actions - torch_actions).abs().max() > 1e-5:
                    print(
                        "[INFO]: ONNX model and PyTorch model have a difference of"
                        f" {(actions - torch_actions).abs().max()} in actions at joint"
                        f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                    )
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            if pd_plotter is not None:
                plot_key = pd_plotter.update(timestep)
                if keyboard_command_enabled and plot_key is not None:
                    apply_keyboard_command(plot_key)

            # 打印实际速度 vs 命令速度（命令变化时或每200步）
            cmd_changed = not torch.allclose(override_command[0], last_debug_cmd, atol=1e-4)
            if keyboard_command_enabled and vel_slice is not None and (cmd_changed or timestep % 200 == 0):
                vel_start = vel_slice[0].start if isinstance(vel_slice[0], slice) else vel_slice[0]
                actual_vel = obs[0, vel_start:vel_start+3].cpu()
                cmd = override_command[0].cpu()
                print(f"[实时] cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})  "
                      f"实际=({actual_vel[0]:+.2f}, {actual_vel[1]:+.2f}, {actual_vel[2]:+.2f})")
                last_debug_cmd = override_command[0].clone()

            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_counts[env_id] = episode_counts.get(env_id, 0) + 1
                    reasons_str = summarize_termination(env_id, infos)
                    print(
                        f"[PLAY] env {env_id} reset! episode={episode_counts[env_id]}, "
                        f"步数={timestep}, 终止原因: {reasons_str}"
                    )

            if args_cli.video:
                env.unwrapped.render()

            if args_cli.show_first_person_rgbd:
                _show_first_person_rgbd_windows(env, timestep)

            terrain_type_list = getattr(env.unwrapped, "terrain_type_list", None)
            terrain = getattr(env.unwrapped.scene, "terrain", None)
            if terrain is not None:
                terrain_types = terrain.terrain_types.cpu().numpy()
                terrain_levels = getattr(terrain, "terrain_levels", None)
                if terrain_levels is not None:
                    terrain_levels = terrain_levels.cpu().numpy()
            else:
                terrain_types = None
                terrain_levels = None

            if save_depth_dir is not None and timestep % args_cli.save_depth_interval == 0:
                try:
                    depth_data = env.unwrapped.scene["camera"].data.output["distance_to_image_plane"]
                    if depth_data is None:
                        print(f"[WARN] timestep {timestep}: depth_data is None")
                        continue
                    if len(depth_data) == 0:
                        print(f"[WARN] timestep {timestep}: depth_data is empty")
                        continue

                    num_envs = depth_data.shape[0]

                    for env_id in range(num_envs):
                        depth_np = depth_data[env_id].cpu().numpy()
                        if depth_np.ndim == 3:
                            depth_np = depth_np.squeeze(-1)
                        if depth_np.size == 0 or len(depth_np.shape) < 2 or depth_np.shape[0] == 0 or depth_np.shape[1] == 0:
                            continue
                        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=10.0, neginf=0.0)
                        d_min, d_max = depth_np.min(), depth_np.max()
                        if d_max - d_min > 1e-8:
                            depth_normalized = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                        else:
                            depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

                        if terrain_levels is not None and terrain_types is not None:
                            level = int(terrain_levels[env_id])
                            col = int(terrain_types[env_id])
                            terrain_type = f"level{level}_col{col}"
                        else:
                            terrain_type = f"env{env_id}"
                        env_save_dir = os.path.join(save_depth_dir, terrain_type)
                        os.makedirs(env_save_dir, exist_ok=True)

                        if timestep == 0:
                            print(f"[DEBUG] raycaster env_id={env_id}, level={level if terrain_levels is not None else -1}, col={col if terrain_types is not None else -1}, terrain={terrain_type}")

                        img_depth = Image.fromarray(depth_normalized)
                        img_depth.save(os.path.join(env_save_dir, f"step_{timestep:06d}_env{env_id:02d}_depth.png"))

                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                        img_colored = Image.fromarray(depth_colored_rgb)
                        img_colored.save(os.path.join(env_save_dir, f"step_{timestep:06d}_env{env_id:02d}_color.png"))

                        with open(os.path.join(env_save_dir, f"step_{timestep:06d}_env{env_id:02d}_info.txt"), "w") as f:
                            f.write(f"run_id: {run_id}\n")
                            f.write(f"step: {timestep}\n")
                            f.write(f"episode: {episode_counts.get(env_id, 0)}\n")
                            f.write(f"terrain: {terrain_type}\n")
                            f.write(f"env_id: {env_id}\n")
                            f.write(f"image_shape: {depth_np.shape}\n")
                            f.write(f"depth_range: [{d_min:.4f}, {d_max:.4f}]\n")

                    print(f"[INFO] timestep {timestep}: saved {num_envs} envs, episodes completed: {sum(episode_counts.values())}")
                except Exception as e:
                    print(f"[ERROR] timestep {timestep}: failed to save depth image: {e}")
                    import traceback
                    traceback.print_exc()

            if save_rgb_zhengshi_dir is not None and timestep % args_cli.save_rgb_zhengshi_interval == 0:
                try:
                    rgb_zhengshi_data = env.unwrapped.scene["rgb_camera"].data.output
                    if "rgb" in rgb_zhengshi_data and rgb_zhengshi_data["rgb"] is not None:
                        rgb_zhengshi = rgb_zhengshi_data["rgb"]
                        depth_zhengshi_data = rgb_zhengshi_data.get("distance_to_image_plane")

                        if depth_zhengshi_data is not None and len(depth_zhengshi_data) > 0:
                            num_cameras = rgb_zhengshi.shape[0]

                            for cam_id in range(num_cameras):
                                cam_rgb = rgb_zhengshi[cam_id].cpu().numpy()
                                cam_rgb = (cam_rgb * 255).astype(np.uint8) if cam_rgb.max() <= 1.0 else cam_rgb.astype(np.uint8)
                                cam_rgb_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

                                cam_depth = depth_zhengshi_data[cam_id].cpu().numpy()
                                if cam_depth.ndim == 3:
                                    cam_depth = cam_depth.squeeze(-1)
                                cam_depth = np.nan_to_num(cam_depth, nan=0.0, posinf=100.0, neginf=0.0)

                                if terrain_levels is not None and terrain_types is not None:
                                    level = int(terrain_levels[cam_id])
                                    col = int(terrain_types[cam_id])
                                    terrain_type = f"level{level}_col{col}"
                                else:
                                    terrain_type = f"env{cam_id}"
                                env_save_dir = os.path.join(save_rgb_zhengshi_dir, terrain_type)
                                os.makedirs(env_save_dir, exist_ok=True)

                                if timestep == 0:
                                    print(f"[DEBUG] rgb_camera cam_id={cam_id}, level={level if terrain_levels is not None else -1}, col={col if terrain_types is not None else -1}, terrain={terrain_type}")

                                fixed_min, fixed_max = 0.0, 10.0
                                depth_clipped = np.clip(cam_depth, fixed_min, fixed_max)
                                depth_normalized = ((depth_clipped - fixed_min) / (fixed_max - fixed_min) * 255).astype(np.uint8)

                                depth_filename = os.path.join(env_save_dir, f"zhengshi_t{timestep}_cam{cam_id}_depth.png")
                                cv2.imwrite(depth_filename, depth_normalized)

                                rgb_filename = os.path.join(env_save_dir, f"zhengshi_t{timestep}_cam{cam_id}_rgb.png")
                                cv2.imwrite(rgb_filename, cam_rgb_bgr)
                except Exception as e:
                    if timestep % 200 == 0:
                        print(f"[DEBUG] Failed to save rgb_camera: {e}")

            timestep += 1

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    if pd_plotter is not None:
        pd_plotter.close()
    if args_cli.show_first_person_rgbd:
        try:
            depth_window_name = f"{DEPTH_WINDOW_NAME} ({args_cli.first_person_depth_source})"
            cv2.destroyWindow(RGB_WINDOW_NAME)
            cv2.destroyWindow(depth_window_name)
            cv2.waitKey(1)
        except Exception:
            pass

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
