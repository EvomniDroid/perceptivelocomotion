"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 地形名称列表（按sub_terrains顺序）
SUB_TERRAINS_KEYS = [
    "perlin_rough", "perlin_rough_stand", "square_gaps", "pyramid_stairs", "pyramid_stairs_high",
    "pyramid_stairs_inv", "pyramid_stairs_inv_high", "boxes", "mesh_boxes", "hf_pyramid_slope_inv"
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
parser.add_argument("--keyboard_control", action="store_true", default=False, help="启用键盘控制(WASD走, QE转, X归零)。")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="键盘每次调整的线速度增量。")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="键盘控制的角速度。")
parser.add_argument("--free_view", action="store_true", default=False, help="自由视角（不跟随机器人）。")
parser.add_argument("--debug_ray", action="store_true", default=False, help="启用射线检测可视化。")
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次俯视深度图，0表示禁用。")
parser.add_argument("--save_record_rgb_interval", type=int, default=0, help="每N步保存一次camera_rgb_record的RGB和深度图，0表示禁用。")
parser.add_argument("--save_rgb_zhengshi_interval", type=int, default=0, help="每N步保存一次rgb_camera的RGB和深度图，0表示禁用。")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.save_record_rgb_interval > 0:
    args_cli.enable_cameras = True
if args_cli.save_rgb_zhengshi_interval > 0:
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
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab.utils.io import  load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

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
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

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
        if args_cli.free_view:
            if hasattr(env_cfg, "viewer"):
                env_cfg.viewer.origin_type = "world"
                env_cfg.viewer.eye = (4.0, 4.0, 4.0)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if args_cli.debug_ray:
        env_cfg.scene.left_height_scanner.debug_vis = True
        env_cfg.scene.right_height_scanner.debug_vis = True
        env_cfg.scene.leg_volume_points.debug_vis = True
        env_cfg.scene.camera.debug_vis = True

    import time
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_depth_dir = None
    save_record_rgb_dir = None
    save_rgb_zhengshi_dir = None

    if args_cli.save_depth_interval > 0 or args_cli.save_record_rgb_interval > 0 or args_cli.save_rgb_zhengshi_interval > 0:
        depth_run_dir = os.path.join(log_dir, f"depth_run_{run_id}")
        os.makedirs(depth_run_dir, exist_ok=True)
        print(f"[INFO] Depth run directory: {depth_run_dir}")

    if args_cli.save_depth_interval > 0:
        save_depth_dir = os.path.join(depth_run_dir, "raycaster")
        os.makedirs(save_depth_dir, exist_ok=True)
        print(f"[INFO] Saving raycaster depth to: {save_depth_dir}")
        print(f"[INFO] Will save every {args_cli.save_depth_interval} steps")

    if args_cli.save_record_rgb_interval > 0:
        save_record_rgb_dir = os.path.join(depth_run_dir, "rgbd_record")
        os.makedirs(save_record_rgb_dir, exist_ok=True)
        print(f"[INFO] Saving rgbd_record to: {save_record_rgb_dir}")
        print(f"[INFO] Will save every {args_cli.save_record_rgb_interval} steps")

    if args_cli.save_rgb_zhengshi_interval > 0:
        save_rgb_zhengshi_dir = os.path.join(depth_run_dir, "rgbd_zhengshi")
        os.makedirs(save_rgb_zhengshi_dir, exist_ok=True)
        print(f"[INFO] Saving rgb_camera to: {save_rgb_zhengshi_dir}")
        print(f"[INFO] Will save every {args_cli.save_rgb_zhengshi_interval} steps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

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
        onnx_policy = load_parkour_onnx_model(
            model_dir=os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                env.get_obs_segments(),
                temporal=True,
            ),
            depth_shape=env.get_obs_segments()["depth_image"],
            proprio_slice=slice(
                0,
                get_subobs_size(
                    env.get_obs_segments(),
                    [
                        "base_lin_vel",
                        "base_ang_vel",
                        "projected_gravity",
                        "velocity_commands",
                        "joint_pos",
                        "joint_vel",
                        "actions",
                    ],
                ),
            ),
        )

    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")

    cmd_display_names = {"W": "前", "S": "后", "A": "左移", "D": "右移", "Q": "左转", "E": "右转", "X": "急停"}
    _print_vel_help = True

    print("=" * 60)
    print("键盘控制已启用:")
    print("  W / S      : 前进 / 后退  (+/-" + str(args_cli.keyboard_linvel_step) + " m/s)")
    print("  A / D      : 左移 / 右移  (+/-" + str(args_cli.keyboard_linvel_step) + " m/s)")
    print("  Q / E      : 左转 / 右转  (" + str(args_cli.keyboard_angvel) + " rad/s)")
    print("  X          : 急停归零")
    print("  长按可累计叠加速度")
    print("=" * 60)

    def on_keyboard_input(e):
        global _print_vel_help
        key_map = {
            carb.input.KeyboardInput.W: (0, 1.0, "W"),
            carb.input.KeyboardInput.S: (0, -1.0, "S"),
            carb.input.KeyboardInput.A: (1, 1.0, "A"),
            carb.input.KeyboardInput.D: (1, -1.0, "D"),
            carb.input.KeyboardInput.Q: (2, 1.0, "Q"),
            carb.input.KeyboardInput.E: (2, -1.0, "E"),
            carb.input.KeyboardInput.X: (-1, 0.0, "X"),
        }
        if e.input in key_map:
            idx, sign, name = key_map[e.input]
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                if name == "X":
                    override_command[:] = 0.0
                elif name in ("Q", "E"):
                    override_command[:, 2] = sign * args_cli.keyboard_angvel
                else:
                    override_command[:, idx] += sign * args_cli.keyboard_linvel_step
                vx = override_command[0, 0].item()
                vy = override_command[0, 1].item()
                wz = override_command[0, 2].item()
                print(f"[键盘] {cmd_display_names.get(name, name):>4s} | cmd=(v_x={vx:+.2f}, v_y={vy:+.2f}, ω_z={wz:+.2f})")
                _print_vel_help = True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # 获取obs切片信息，用于打印实际速度
    obs_segments = env.get_obs_segments()
    vel_slice = get_obs_slice(obs_segments, "base_lin_vel")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    episode_counts = {}  # track episodes per env
    num_envs = env.unwrapped.scene.num_envs
    last_debug_cmd = torch.tensor([999.0, 999.0, 999.0], device=env.device)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.keyboard_control:
                obs[:, command_obs_slice[0]] = override_command.repeat(1, command_obs_slice[1][0] // 3)
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

            # 打印实际速度 vs 命令速度（命令变化时或每200步）
            cmd_changed = not torch.allclose(override_command[0], last_debug_cmd, atol=1e-4)
            if args_cli.keyboard_control and (cmd_changed or timestep % 200 == 0):
                vel_start = vel_slice[0].start if isinstance(vel_slice[0], slice) else vel_slice[0]
                actual_vel = obs[0, vel_start:vel_start+3].cpu()
                cmd = override_command[0].cpu()
                print(f"[实时] cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})  "
                      f"实际=({actual_vel[0]:+.2f}, {actual_vel[1]:+.2f}, {actual_vel[2]:+.2f})")
                last_debug_cmd = override_command[0].clone()

            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_counts[env_id] = episode_counts.get(env_id, 0) + 1
                    if timestep <= 10 and episode_counts.get(env_id, 1) == 1:
                        termination_reasons = [k for k in infos.get("episode", {}).keys() if "Episode_Termination" in k]
                        if termination_reasons:
                            reasons_str = ", ".join([f"{k}={infos['episode'][k][env_id].item():.3f}" for k in termination_reasons])
                            print(f"[PLAY] env {env_id} 死亡! 步数={timestep}, 终止原因: {reasons_str}")
                        else:
                            print(f"[PLAY] env {env_id} 死亡! 步数={timestep}")

            if args_cli.video:
                env.unwrapped.render()

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

            if save_record_rgb_dir is not None and timestep % args_cli.save_record_rgb_interval == 0:
                try:
                    record_rgb_data = env.unwrapped.scene["camera_rgb_record"].data.output
                    if "rgb" in record_rgb_data and record_rgb_data["rgb"] is not None:
                        record_rgb = record_rgb_data["rgb"]
                        record_depth_data = record_rgb_data.get("distance_to_image_plane")

                        if record_depth_data is not None and len(record_depth_data) > 0:
                            num_cameras = record_rgb.shape[0]
                            print(f"[DEBUG] camera_rgb_record rgb shape: {record_rgb.shape}, depth shape: {record_depth_data.shape}")

                            for cam_id in range(num_cameras):
                                cam_rgb = record_rgb[cam_id].cpu().numpy()
                                cam_rgb = (cam_rgb * 255).astype(np.uint8) if cam_rgb.max() <= 1.0 else cam_rgb.astype(np.uint8)
                                cam_rgb_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)

                                cam_depth = record_depth_data[cam_id].cpu().numpy()
                                if cam_depth.ndim == 3:
                                    cam_depth = cam_depth.squeeze(-1)
                                cam_depth = np.nan_to_num(cam_depth, nan=0.0, posinf=100.0, neginf=0.0)

                                # 使用与 raycaster 相同的地形逻辑
                                if terrain_levels is not None and terrain_types is not None:
                                    level = int(terrain_levels[cam_id])
                                    col = int(terrain_types[cam_id])
                                    terrain_type = f"level{level}_col{col}"
                                else:
                                    terrain_type = f"env{cam_id}"
                                env_save_dir = os.path.join(save_record_rgb_dir, terrain_type)
                                os.makedirs(env_save_dir, exist_ok=True)

                                if timestep == 0:
                                    print(f"[DEBUG] rgbd_record cam_id={cam_id}, level={level if terrain_levels is not None else -1}, col={col if terrain_types is not None else -1}, terrain={terrain_type}")

                                fixed_min, fixed_max = 0.0, 10.0
                                depth_clipped = np.clip(cam_depth, fixed_min, fixed_max)
                                depth_normalized = ((depth_clipped - fixed_min) / (fixed_max - fixed_min) * 255).astype(np.uint8)

                                depth_filename = os.path.join(env_save_dir, f"record_t{timestep}_cam{cam_id}_depth.png")
                                cv2.imwrite(depth_filename, depth_normalized)

                                rgb_filename = os.path.join(env_save_dir, f"record_t{timestep}_cam{cam_id}_rgb.png")
                                cv2.imwrite(rgb_filename, cam_rgb_bgr)
                except Exception as e:
                    if timestep % 200 == 0:
                        print(f"[DEBUG] Failed to save camera_rgb_record: {e}")

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
