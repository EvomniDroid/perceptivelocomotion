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
parser.add_argument("--keyboard_control", action="store_true", default=False, help="启用键盘控制。")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="键盘每次调整的线速度增量。")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="键盘控制的角速度。")
parser.add_argument("--debug_ray", action="store_true", default=False, help="启用射线检测可视化。")
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次俯视深度图，0表示禁用。")
parser.add_argument("--save_record_rgb_interval", type=int, default=0, help="每N步保存一次camera_rgb_record的RGB和深度图，0表示禁用。")

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

    if args_cli.debug_ray:
        env_cfg.scene.left_height_scanner.debug_vis = True
        env_cfg.scene.right_height_scanner.debug_vis = True
        env_cfg.scene.leg_volume_points.debug_vis = True
        env_cfg.scene.camera.debug_vis = True

    import time
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_depth_dir = None
    save_record_rgb_dir = None

    if args_cli.save_depth_interval > 0 or args_cli.save_record_rgb_interval > 0:
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

    def on_keyboard_input(e):
        if e.input == carb.input.KeyboardInput.W:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 0] += args_cli.keyboard_linvel_step
        if e.input == carb.input.KeyboardInput.S:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = 0.0
        if e.input == carb.input.KeyboardInput.F:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.G:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = -args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.X:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:] = 0.0

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    episode_counts = {}  # track episodes per env
    num_envs = env.unwrapped.scene.num_envs
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

            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_counts[env_id] = episode_counts.get(env_id, 0) + 1

            if args_cli.video:
                env.unwrapped.render()

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
                    terrain_type_list = getattr(env.unwrapped, "terrain_type_list", [])

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

                        terrain_type = terrain_type_list[env_id] if env_id < len(terrain_type_list) else "unknown"
                        env_save_dir = os.path.join(save_depth_dir, terrain_type)
                        os.makedirs(env_save_dir, exist_ok=True)

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
                        record_rgb = record_rgb_data["rgb"][0].cpu().numpy()
                        record_rgb = (record_rgb * 255).astype(np.uint8) if record_rgb.max() <= 1.0 else record_rgb.astype(np.uint8)
                        record_rgb_bgr = cv2.cvtColor(record_rgb, cv2.COLOR_RGB2BGR)

                        record_depth_data = record_rgb_data.get("distance_to_image_plane")
                        if record_depth_data is not None and len(record_depth_data) > 0:
                            record_depth_np = record_depth_data[0].cpu().numpy()
                            if record_depth_np.ndim == 3:
                                record_depth_np = record_depth_np.squeeze(-1)
                            record_depth_np = np.nan_to_num(record_depth_np, nan=0.0, posinf=100.0, neginf=0.0)

                            terrain_type_list = getattr(env.unwrapped, "terrain_type_list", [])
                            terrain_type = terrain_type_list[0] if len(terrain_type_list) > 0 else "unknown"
                            env_save_dir = os.path.join(save_record_rgb_dir, terrain_type)
                            os.makedirs(env_save_dir, exist_ok=True)

                            fixed_min, fixed_max = 0.0, 10.0
                            depth_clipped = np.clip(record_depth_np, fixed_min, fixed_max)
                            depth_normalized = ((depth_clipped - fixed_min) / (fixed_max - fixed_min) * 255).astype(np.uint8)

                            depth_filename = os.path.join(env_save_dir, f"record_t{timestep}_depth.png")
                            cv2.imwrite(depth_filename, depth_normalized)

                            rgb_filename = os.path.join(env_save_dir, f"record_t{timestep}_rgb.png")
                            cv2.imwrite(rgb_filename, record_rgb_bgr)
                except Exception as e:
                    if timestep % 200 == 0:
                        print(f"[DEBUG] Failed to save camera_rgb_record: {e}")

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
