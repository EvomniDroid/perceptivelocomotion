"""部署脚本：RL策略 + 摔倒率分类器 + 运动规划器"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="部署RL智能体，包含摔倒率分类器和运动规划器")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量")
parser.add_argument("--classifier_model", type=str, default=None, help="摔倒率分类器模型路径(.pth)，不提供则只加载RL策略")
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次深度图，0表示禁用")
parser.add_argument("--fall_rate_threshold", type=float, default=0.5, help="切换到安全模式的摔倒率阈值")
parser.add_argument("--use_vis_terrain", action="store_true", default=False, help="使用vis.py的地形配置进行泛化测试")
parser.add_argument("--vel_debug", action="store_true", default=False, help="启用速度调试模式，使用直接速度指令替代RL策略")
parser.add_argument("--vel", type=str, default="0.5,0.0,0.0", help="调试模式速度向量: vel_x,vel_y,ang_z (逗号分隔，默认0.5,0.0,0.0)")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="启用键盘控制速度 (WASD)")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="键盘每次调整的速度增量")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="键盘控制的角速度大小")
parser.add_argument("--termination_mode", type=str, default="full", help="终止模式: full=摔倒/出界等, time_only=仅超时, none=不禁用")
parser.add_argument("--debug_ray", action="store_true", default=False, help="启用射线检测可视化")
parser.add_argument("--target_pos", type=str, default=None, help="目标位置(x,y)，例如2.0,2.0，单位米")
parser.add_argument("--spawn_pos", type=str, default=None, help="出生位置(x,y)，例如0.0,0.0，单位米，默认随机")

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
import cli_args
cli_args.add_instinct_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, 'video', False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input
import omni.appwindow
from carb.input import KeyboardEventType

from instinct_rl.utils.utils import get_obs_slice
import gymnasium as gym
import torch
import numpy as np
from PIL import Image

from instinct_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg
from instinctlab.terrains.shared_terrain_cfg import MY_TERRAIN_CFG

sys.path.append("/home/zh/isaac/liveratemodel")
from model import create_model
from dataset import LABEL_TO_FALL_RATE, LABEL_TO_TERRAIN, build_model_input_from_depth_array, get_input_channels
import torch.nn.functional as F


class FallRateClassifier:
    """摔倒率分类器：基于深度图预测摔倒风险"""

    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})

        self.input_mode = config.get("input_mode", "depth_gradients")
        self.use_resnet = config.get("use_resnet", False)
        self.use_pointnet = config.get("use_pointnet", False) or self.input_mode == "point_cloud"
        self.patch_size = config.get("patch_size", 32)
        self.point_count = config.get("point_count", 1024)
        self.focal_length = config.get("focal_length", 24.0)
        self.horizontal_aperture = config.get("horizontal_aperture", 20.955)
        self.vertical_aperture = config.get("vertical_aperture", None)

        in_channels = get_input_channels(self.input_mode)
        self.model = create_model(
            model_type="classifier",
            num_classes=10,
            device=self.device,
            use_resnet=self.use_resnet,
            use_pointnet=self.use_pointnet,
            pretrained=False,
            in_channels=in_channels,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, depth_array):
        depth_max = float(depth_array.max())
        depth_scale = depth_max if depth_max > 1.0 else 1.0

        if self.input_mode != "point_cloud":
            img = Image.fromarray(depth_array).convert("L")
            img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            depth_array = np.array(img)

        model_input = build_model_input_from_depth_array(
            depth_array,
            input_mode=self.input_mode,
            depth_scale=depth_scale,
            point_count=self.point_count,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
            vertical_aperture=self.vertical_aperture,
        )
        model_input = model_input.to(self.device)
        model_input = model_input.unsqueeze(0)

        outputs = self.model(model_input)
        probabilities = F.softmax(outputs, dim=1)
        probs_np = probabilities.cpu().numpy()[0]
        pred_label = np.argmax(probs_np)
        fall_rate = LABEL_TO_FALL_RATE[pred_label]
        terrain_name = LABEL_TO_TERRAIN.get(pred_label, "unknown")

        return fall_rate, pred_label, terrain_name


class MotionPlanner:
    """运动规划器接口 - 根据摔倒率和地形类型调整动作"""

    def __init__(self, cfg=None, vel_debug=False, debug_vels=None, target_pos=None):
        self.cfg = cfg or {}
        self.safe_mode_lin_vel_scale = self.cfg.get("safe_mode_lin_vel_scale", 0.5)
        self.safe_mode_step_width = self.cfg.get("safe_mode_step_width", 0.5)
        self.vel_debug = vel_debug
        self.debug_vels = debug_vels or {}
        self.target_pos = target_pos
        self.init_pos = None
        self.Kp_lin = 0.5
        self.Kp_ang = 1.5
        self._keyboard_active = False

    def set_target(self, x, y):
        self.target_pos = (x, y)
        self.init_pos = None
        print(f"[规划] 目标位置设置为: ({x:.2f}, {y:.2f}) 相对坐标")

    def get_action(self, obs, fall_rate, terrain_type, command_obs_slice, vel_debug=False, keyboard_command=None, robot_pos=None, robot_yaw=None, timestep=0):
        if not vel_debug:
            return obs

        vel_x, vel_y, ang_z = self._get_blended_velocity(keyboard_command)
        obs = self._inject_velocity(obs, command_obs_slice, vel_x, vel_y, ang_z)
        return obs

    def _get_blended_velocity(self, keyboard_command):
        """混合速度选择：键盘优先，无键盘输入则用默认规划（一直往前）"""
        if keyboard_command is not None:
            kx = keyboard_command[0, 0].item()
            ky = keyboard_command[0, 1].item()
            kz = keyboard_command[0, 2].item()
            if self._keyboard_active or abs(kx) > 0.01 or abs(ky) > 0.01 or abs(kz) > 0.01:
                self._keyboard_active = True
                print(f"[规划] 键盘接管: vel_x={kx:.2f}, vel_y={ky:.2f}, ang_z={kz:.2f}")
                return kx, ky, kz

        vel_x = self.debug_vels.get("vel_x", 0.5)
        vel_y = self.debug_vels.get("vel_y", 0.0)
        ang_z = self.debug_vels.get("ang_z", 0.0)
        return vel_x, vel_y, ang_z

    def _position_to_velocity(self, robot_pos, robot_yaw=None, log_interval=100, timestep=None):
        """Pure Pursuit 视线导航：将相对位置转为速度命令（考虑机器人当前朝向）"""
        rx, ry = robot_pos[0], robot_pos[1]

        if self.init_pos is None:
            self.init_pos = (rx, ry)
            init_yaw = robot_yaw if robot_yaw is not None else 0.0
            print(f"[规划] 记录起始位置: ({rx:.2f}, {ry:.2f}), 朝向: {math.degrees(init_yaw):.1f}°")

        tx = self.init_pos[0] + self.target_pos[0]
        ty = self.init_pos[1] + self.target_pos[1]

        dx = tx - rx
        dy = ty - ry
        dist = math.sqrt(dx*dx + dy*dy)
        world_angle = math.atan2(dy, dx)

        rel_angle = world_angle - (robot_yaw if robot_yaw is not None else 0.0)
        while rel_angle > math.pi:
            rel_angle -= 2 * math.pi
        while rel_angle < -math.pi:
            rel_angle += 2 * math.pi

        if dist < 0.1:
            if timestep is None or timestep % log_interval == 0:
                print(f"[规划] 到达目标! dist={dist:.3f}")
            return 0.0, 0.0, 0.0

        vel_x = self.Kp_lin * dist
        vel_x = max(0.0, min(vel_x, 0.8))
        ang_z = self.Kp_ang * rel_angle
        ang_z = max(-1.5, min(ang_z, 1.5))

        if abs(ang_z) > 0.1:
            vel_x *= 0.5

        if timestep is None or timestep % log_interval == 0:
            print(f"[规划] 当前({rx:.2f},{ry:.2f}) 目标({tx:.2f},{ty:.2f}) 距离:{dist:.2f} 世界角:{math.degrees(world_angle):.1f}° 相对角:{math.degrees(rel_angle):.1f}° -> vel({vel_x:.2f},{ang_z:.2f})")
        return vel_x, 0.0, ang_z

    def _inject_velocity(self, obs, command_obs_slice, vel_x, vel_y, ang_z):
        obs = obs.clone()
        debug_cmd = torch.tensor([[vel_x, vel_y, ang_z]], device=obs.device)
        debug_cmd_repeated = debug_cmd.repeat(1, command_obs_slice[1][0] // 3)
        obs[:, command_obs_slice[0]] = debug_cmd_repeated
        return obs


def main():
    print("\n" + "="*60)
    print("部署 RL策略 + 摔倒率分类器 + 运动规划器")
    print("="*60)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, 'device', 'cuda:0'),
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 1,
        use_fabric=not getattr(args_cli, 'disable_fabric', False)
    )

    if getattr(args_cli, 'use_vis_terrain', False):
        print("[INFO] 使用vis地形配置进行泛化测试 (MY_TERRAIN_CFG)")
        env_cfg.scene.terrain.terrain_generator = MY_TERRAIN_CFG
        env_cfg.scene.terrain.curriculum = False

    spawn_pos_str = getattr(args_cli, 'spawn_pos', None)
    if spawn_pos_str:
        try:
            parts = [float(x) for x in spawn_pos_str.split(',')]
            if len(parts) >= 2:
                spawn_x, spawn_y = parts[0], parts[1]
                spawn_yaw = parts[2] if len(parts) > 2 else 0.0
                print(f"[INFO] spawn_pos: ({spawn_x}, {spawn_y}, yaw={spawn_yaw})")
                if hasattr(env_cfg.events, 'reset_base') and env_cfg.events.reset_base is not None:
                    env_cfg.events.reset_base.params["pose_range"] = {
                        "x": (spawn_x, spawn_x),
                        "y": (spawn_y, spawn_y),
                        "yaw": (spawn_yaw, spawn_yaw),
                    }
                    env_cfg.events.reset_base.params["velocity_range"] = {
                        "x": (0, 0), "y": (0, 0), "z": (0, 0),
                        "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0),
                    }
                    print("[INFO] 已设置固定出生位置")
                else:
                    print("[WARN] reset_base 不存在，无法设置出生位置")
        except Exception as e:
            print(f"[WARN] spawn_pos 解析失败: {e}")

    if getattr(args_cli, 'debug_ray', False):
        if hasattr(env_cfg.scene, 'left_height_scanner'):
            env_cfg.scene.left_height_scanner.debug_vis = True
        if hasattr(env_cfg.scene, 'right_height_scanner'):
            env_cfg.scene.right_height_scanner.debug_vis = True
        if hasattr(env_cfg.scene, 'leg_volume_points'):
            env_cfg.scene.leg_volume_points.debug_vis = True
        if hasattr(env_cfg.scene, 'camera'):
            env_cfg.scene.camera.debug_vis = True
        print("[INFO] 启用射线检测可视化")

    term_mode = getattr(args_cli, 'termination_mode', 'full')
    if term_mode == "none":
        env_cfg.terminations.time_out = None
        env_cfg.terminations.terrain_out_bound = None
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.bad_orientation = None
        env_cfg.terminations.root_height = None
        env_cfg.terminations.dataset_exhausted = None
        print("[INFO] termination_mode=none: 禁用所有终止检测")
    elif term_mode == "time_only":
        env_cfg.terminations.terrain_out_bound = None
        env_cfg.terminations.base_contact = None
        env_cfg.terminations.bad_orientation = None
        env_cfg.terminations.root_height = None
        env_cfg.terminations.dataset_exhausted = None
        print("[INFO] termination_mode=time_only: 仅超时终止")
    else:
        env_cfg.terminations.time_out = None
        env_cfg.terminations.dataset_exhausted = None
        print("[INFO] termination_mode=full: 摔倒/出界等终止，超时不禁用")

    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if agent_cfg.load_run is not None:
        agent_cfg.load_run = agent_cfg.load_run.rstrip("/\\")
        resume_path = get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run),
            os.path.basename(agent_cfg.load_run),
            args_cli.checkpoint
        )
        log_dir = os.path.dirname(resume_path)
    else:
        log_dir = os.path.join(log_root_path, "deploy_output")
        resume_path = args_cli.checkpoint

    agent_cfg_dict = agent_cfg.to_dict()

    print(f"[INFO] 日志目录: {log_dir}")
    print(f"[INFO] 恢复路径: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args_cli, 'video', False) else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = InstinctRlVecEnvWrapper(env)

    ppo_runner = None
    if getattr(args_cli, 'vel_debug', False):
        print("[INFO] vel_debug模式：加载RL策略用于关节控制，只覆盖velocity_commands")
        ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        if agent_cfg.load_run is not None:
            print(f"[INFO] 加载RL策略: {resume_path}")
            ppo_runner.load(resume_path)
    else:
        ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        if agent_cfg.load_run is not None:
            print(f"[INFO] 加载RL策略: {resume_path}")
            ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    print(f"\n[INFO] 加载摔倒率分类器: {args_cli.classifier_model}")
    if args_cli.classifier_model is not None:
        classifier = FallRateClassifier(args_cli.classifier_model, device=getattr(args_cli, 'device', 'cuda:0'))
    else:
        classifier = None
        print("[INFO] 未提供分类器模型，跳过分类器初始化")

    vel_str = getattr(args_cli, 'vel', '0.5,0.0,0.0')
    vel_parts = [float(x) for x in vel_str.split(',')]
    debug_vels = {
        "vel_x": vel_parts[0] if len(vel_parts) > 0 else 0.5,
        "vel_y": vel_parts[1] if len(vel_parts) > 1 else 0.0,
        "ang_z": vel_parts[2] if len(vel_parts) > 2 else 0.0,
    }

    target_pos = None
    target_pos_str = getattr(args_cli, 'target_pos', None)
    if target_pos_str:
        try:
            parts = [float(x) for x in target_pos_str.split(',')]
            if len(parts) >= 2:
                target_pos = (parts[0], parts[1])
                print(f"[INFO] 目标位置: ({target_pos[0]}, {target_pos[1]})")
        except:
            print(f"[WARN] 无效的目标位置格式: {target_pos_str}，使用默认速度模式")

    planner = MotionPlanner(
        cfg={"safe_mode_lin_vel_scale": 0.5},
        vel_debug=getattr(args_cli, 'vel_debug', False),
        debug_vels=debug_vels,
        target_pos=target_pos
    )

    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")

    run_id = np.random.randint(10000)
    save_depth_dir = None
    if getattr(args_cli, 'save_depth_interval', 0) > 0:
        save_depth_dir = os.path.join(log_dir, f"deploy_depth_{run_id}")
        os.makedirs(save_depth_dir, exist_ok=True)
        print(f"[INFO] 深度图保存目录: {save_depth_dir}")

    obs, _ = env.get_observations()
    timestep = 0

    raw_env = env.unwrapped
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    print(f"[DEBUG] raw_env type: {type(raw_env)}")
    if hasattr(raw_env, 'scene'):
        print(f"[DEBUG] env.scene keys: {list(raw_env.scene.keys())}")

    # 保存raw_env供后面循环使用
    env_scene = raw_env.scene

    keyboard_command = torch.zeros(env.num_envs, 3, device=env.device)
    keyboard_linvel_step = getattr(args_cli, 'keyboard_linvel_step', 0.5)
    keyboard_angvel = getattr(args_cli, 'keyboard_angvel', 1.0)

    emergency_stop = False
    last_emergency_state = False

    if getattr(args_cli, 'keyboard_control', False):
        def on_keyboard_input(e):
            if e.input == carb.input.KeyboardInput.W:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 0] += keyboard_linvel_step
                    print(f"[键盘] W: vel_x += {keyboard_linvel_step} -> {keyboard_command[0, 0].item():.2f}")
            if e.input == carb.input.KeyboardInput.S:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 0] = max(0.0, keyboard_command[:, 0].item() - keyboard_linvel_step)
                    print(f"[键盘] S: vel_x -= {keyboard_linvel_step} -> {keyboard_command[0, 0].item():.2f}")
            if e.input == carb.input.KeyboardInput.A:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 2] = keyboard_angvel
                    print(f"[键盘] A: 左转 ang_z = {keyboard_angvel}")
            if e.input == carb.input.KeyboardInput.Q:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 2] = 0.0
                    print(f"[键盘] Q: 停止转向")
            if e.input == carb.input.KeyboardInput.D:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 2] = -keyboard_angvel
                    print(f"[键盘] D: 右转 ang_z = {-keyboard_angvel}")
            if e.input == carb.input.KeyboardInput.E:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:, 2] = 0.0
                    print(f"[键盘] E: 停止转向")
            if e.input == carb.input.KeyboardInput.X:
                if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                    keyboard_command[:] = 0.0
                    emergency_stop = True
                    print(f"[键盘] X: 急停!")

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input = carb.input.acquire_input_interface()
        input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    print("\n" + "="*60)
    print("开始部署循环...")
    print("="*60 + "\n")

    try:
        while True:
            depth_np = None

            try:
                depth_data = raw_env.scene["camera"].data.output["distance_to_image_plane"]
                if depth_data is not None and len(depth_data) > 0:
                    depth_np = depth_data[0].cpu().numpy()
                    if depth_np.ndim == 3:
                        depth_np = depth_np.squeeze(-1)
                    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=10.0, neginf=0.0)
            except Exception as e:
                if timestep % 200 == 0:
                    print(f"[DEBUG] 获取深度图失败: {e}")

            if classifier is not None and depth_np is not None:
                fall_rate, label, terrain_name = classifier.predict(depth_np)
            else:
                fall_rate = 0.0
                terrain_name = "unknown"

            robot_pos = None
            robot_yaw = None
            try:
                robot_entity = env_scene["robot"]
                root_pos = robot_entity.data.root_link_pos_w
                root_quat = robot_entity.data.root_quat_w
                if torch.is_tensor(root_pos) and torch.is_tensor(root_quat):
                    rx = root_pos[0, 0].item()
                    ry = root_pos[0, 1].item()
                    robot_pos = (rx, ry)
                    q = root_quat[0]
                    w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
                    robot_yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
            except Exception as e:
                if timestep % 200 == 0:
                    print(f"[DEBUG] 获取位置失败: {e}")

            obs = planner.get_action(obs, fall_rate, terrain_name, command_obs_slice, vel_debug=getattr(args_cli, 'vel_debug', False), keyboard_command=keyboard_command if getattr(args_cli, 'keyboard_control', False) else None, robot_pos=robot_pos, robot_yaw=robot_yaw, timestep=timestep)

            if timestep % 100 == 0:
                pos_str = f"({robot_pos[0]:.2f}, {robot_pos[1]:.2f})" if robot_pos else "N/A"
                print(f"[t={timestep}] pos={pos_str} | 地形: {terrain_name} | 摔倒率: {fall_rate:.3f}")

            if emergency_stop:
                if timestep == 0:
                    actions = policy(obs)
                actions[:] = 0.0
                if timestep % 20 == 0:
                    print(f"[急停] actions[:3]={actions[0, :3].tolist()}")
            else:
                actions = policy(obs)

            obs, _, _, _ = env.step(actions)
            timestep += 1

    except KeyboardInterrupt:
        print("\n[INFO] 部署中断")

    env.close()
    print("[INFO] 部署完成")


if __name__ == "__main__":
    main()
