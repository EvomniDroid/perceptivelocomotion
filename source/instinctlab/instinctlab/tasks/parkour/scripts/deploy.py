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
parser.add_argument("--fall_rate_override", type=float, default=None, help="手动设置所有摔倒率值(0-1)，用于测试")
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次深度图，0表示禁用")
parser.add_argument("--fall_rate_threshold", type=float, default=0.5, help="切换到安全模式的摔倒率阈值")
parser.add_argument("--use_vis_terrain", action="store_true", default=False, help="使用vis.py的地形配置进行泛化测试")
parser.add_argument("--use_frontier_test_terrain", action="store_true", default=False, help="使用FRONTIER_TEST_TERRAIN地形（预设摔倒率的简单地形）")
parser.add_argument("--preset_fall_rate_map", action="store_true", default=False, help="使用预设摔倒率地图（棋盘格），可独立于terrain使用")
parser.add_argument("--vel_debug", action="store_true", default=False, help="启用速度调试模式，使用直接速度指令替代RL策略")
parser.add_argument("--vel", type=str, default="0.5,0.0,0.0", help="调试模式速度向量: vel_x,vel_y,ang_z (逗号分隔，默认0.5,0.0,0.0)")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="启用键盘控制速度 (WASD)")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="键盘每次调整的速度增量")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="键盘控制的角速度大小")
parser.add_argument("--termination_mode", type=str, default="full", help="终止模式: full=摔倒/出界等, time_only=仅超时, none=不禁用")
parser.add_argument("--debug_ray", action="store_true", default=False, help="启用射线检测可视化")
parser.add_argument("--target_pos", type=str, default=None, help="目标位置(x,y)，例如2.0,2.0，单位米")
parser.add_argument("--spawn_pos", type=str, default=None, help="出生位置(x,y)，例如0.0,0.0，单位米，默认随机")
parser.add_argument("--frontier_debug", action="store_true", default=False, help="启用前沿点检测调试")
parser.add_argument("--frontier_interval", type=int, default=100, help="前沿点打印间隔")
parser.add_argument("--frontier_save_interval", type=int, default=500, help="前沿点数据保存间隔")
parser.add_argument("--auto_frontier_nav", action="store_true", default=False, help="自动选择最优前沿点作为导航目标")

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
import cli_args
cli_args.add_instinct_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from instinctlab.terrains.shared_terrain_cfg import MY_TERRAIN_CFG, FRONTIER_TEST_TERRAIN_CFG

sys.path.append("/home/zh/isaac/liveratemodel")
from model import create_model
from dataset import LABEL_TO_FALL_RATE, LABEL_TO_TERRAIN, build_model_input_from_depth_array, get_input_channels
from local_planner import LocalFallRateMap, FrontierDetector
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

        vel_x, vel_y, ang_z = self._get_blended_velocity(keyboard_command, robot_pos, robot_yaw, timestep)
        obs = self._inject_velocity(obs, command_obs_slice, vel_x, vel_y, ang_z)
        return obs

    def _get_blended_velocity(self, keyboard_command, robot_pos=None, robot_yaw=None, timestep=None):
        """混合速度选择：键盘优先，无键盘输入则用 Pure Pursuit 导航到 target_pos"""
        log_interval = 100
        if keyboard_command is not None:
            kx = keyboard_command[0, 0].item()
            ky = keyboard_command[0, 1].item()
            kz = keyboard_command[0, 2].item()
            if self._keyboard_active or abs(kx) > 0.01 or abs(ky) > 0.01 or abs(kz) > 0.01:
                self._keyboard_active = True
                if timestep is None or timestep % log_interval == 0:
                    print(f"[规划] 键盘接管: vel_x={kx:.2f}, vel_y={ky:.2f}, ang_z={kz:.2f}")
                return kx, ky, kz

        if robot_pos is not None and self.target_pos is not None:
            return self._position_to_velocity(robot_pos, robot_yaw, timestep=timestep)

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

        tx = self.target_pos[0]
        ty = self.target_pos[1]

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
    print(f"[DEBUG] use_frontier_test_terrain CLI = {args_cli.use_frontier_test_terrain}")
    print(f"[DEBUG] use_vis_terrain CLI = {args_cli.use_vis_terrain}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, 'device', 'cuda:0'),
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 1,
        use_fabric=not getattr(args_cli, 'disable_fabric', False)
    )
    print(f"[DEBUG] parse_env_cfg 后 terrain_generator type: {type(env_cfg.scene.terrain.terrain_generator).__name__}")
    print(f"[DEBUG] use_frontier_test_terrain = {getattr(args_cli, 'use_frontier_test_terrain', False)}")
    print(f"[DEBUG] use_vis_terrain = {getattr(args_cli, 'use_vis_terrain', False)}")

    if getattr(args_cli, 'use_vis_terrain', False):
        print("[INFO] 使用vis地形配置进行泛化测试 (MY_TERRAIN_CFG)")
        env_cfg.scene.terrain.terrain_generator = MY_TERRAIN_CFG
        env_cfg.scene.terrain.curriculum = False
        env_cfg.scene.env_spacing = 0.0
        print("[INFO] env_spacing=0, env_origins 将全部为 (0,0,0)，坐标系统一")

    if getattr(args_cli, 'use_frontier_test_terrain', False):
        print("[INFO] 使用FRONTIER_TEST_TERRAIN地形（预设摔倒率的简单地形）")
        env_cfg.scene.terrain.terrain_generator = FRONTIER_TEST_TERRAIN_CFG
        env_cfg.scene.terrain.curriculum = False
        env_cfg.scene.env_spacing = 0.0
        print(f"[INFO] terrain_generator 已设置为 FRONTIER_TEST_TERRAIN_CFG, size={env_cfg.scene.terrain.terrain_generator.size}")

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

    print("[DEBUG] 1. gym.make 之前")
    print(f"[DEBUG] 1b. env_cfg.scene.terrain.terrain_generator type: {type(env_cfg.scene.terrain.terrain_generator).__name__}")
    print(f"[DEBUG] 1c. env_cfg.scene.terrain.terrain_generator id: {id(env_cfg.scene.terrain.terrain_generator)}")
    print(f"[DEBUG] 1d. env_cfg.scene.terrain.terrain_generator.size: {env_cfg.scene.terrain.terrain_generator.size}")
    print(f"[DEBUG] 1e. env_cfg.scene.terrain.terrain_generator.num_rows: {env_cfg.scene.terrain.terrain_generator.num_rows}")
    print(f"[DEBUG] 1f. env_cfg.scene.terrain.terrain_generator.num_cols: {env_cfg.scene.terrain.terrain_generator.num_cols}")
    print(f"[DEBUG] 1g. env_cfg.scene.terrain.terrain_generator.seed: {env_cfg.scene.terrain.terrain_generator.seed}")
    print(f"[DEBUG] 1h. env_cfg.scene.env_spacing: {env_cfg.scene.env_spacing}")
    
    import gc
    gc.collect()
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args_cli, 'video', False) else None)
    print("[DEBUG] 2. gym.make 完成")
    
    if getattr(args_cli, 'use_frontier_test_terrain', False) or getattr(args_cli, 'use_vis_terrain', False):
        print("[DEBUG] 2a. 强制设置固定的 env_origins 和 terrain_levels")
        raw_env = env.unwrapped
        while hasattr(raw_env, 'unwrapped') and not hasattr(raw_env, 'scene'):
            raw_env = raw_env.unwrapped
        if hasattr(raw_env, 'scene') and hasattr(raw_env.scene, 'env_origins'):
            fixed_origins = np.array([[-5.0, -3.0, 0.0]])
            with torch.no_grad():
                raw_env.scene.env_origins[:] = torch.tensor(fixed_origins, device=raw_env.scene.env_origins.device)
                if hasattr(raw_env.scene.terrain, 'terrain_levels'):
                    raw_env.scene.terrain.terrain_levels[:] = 0
                if hasattr(raw_env.scene.terrain, 'terrain_types'):
                    raw_env.scene.terrain.terrain_types[:] = 0
            print(f"[DEBUG] 2b. env_origins 已强制设置为: {fixed_origins}")
            if hasattr(raw_env.scene.terrain, 'terrain_levels'):
                print(f"[DEBUG] 2c. terrain_levels 已强制设置为: {raw_env.scene.terrain.terrain_levels.cpu().numpy()}")
            if hasattr(raw_env.scene.terrain, 'terrain_types'):
                print(f"[DEBUG] 2d. terrain_types 已强制设置为: {raw_env.scene.terrain.terrain_types.cpu().numpy()}")
    print(f"[DEBUG] 3. 准备访问 env.unwrapped...")
    tmp = env.unwrapped
    print(f"[DEBUG] 3b. env.unwrapped 类型: {type(tmp).__name__}")

    spawn_pos_str = getattr(args_cli, 'spawn_pos', None)
    print(f"[DEBUG] 4. spawn_pos_str = {spawn_pos_str}")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    print("[DEBUG] 5. multi_agent_to_single_agent 完成")

    inner_env = env.unwrapped
    print(f"[DEBUG] 6. 获取 inner_env = {type(inner_env).__name__}")
    loop_count = 0
    while hasattr(inner_env, 'unwrapped') and loop_count < 10:
        next_env = inner_env.unwrapped
        if next_env is inner_env:
            print(f"[DEBUG] 6b. unwrapped 返回自身，停止")
            break
        inner_env = next_env
        print(f"[DEBUG] 6b. inner_env = {type(inner_env).__name__}")
        loop_count += 1
    print(f"[DEBUG] 6c. 最终 inner_env = {type(inner_env).__name__}")

    print(f"[DEBUG] 6d. hasattr(inner_env, 'scene') = {hasattr(inner_env, 'scene')}")
    if hasattr(inner_env, 'scene'):
        scene_keys = list(inner_env.scene.keys()) if hasattr(inner_env.scene, 'keys') else dir(inner_env.scene)
        print(f"[DEBUG] 6e. scene keys = {scene_keys}")
        has_robot = hasattr(inner_env.scene, 'robot')
        print(f"[DEBUG] 6f. hasattr(inner_env.scene, 'robot') = {has_robot}")

    if spawn_pos_str:
        try:
            if hasattr(inner_env, 'scene'):
                try:
                    robot = inner_env.scene['robot']
                    parts = [float(x) for x in spawn_pos_str.split(',')]
                    spawn_x, spawn_y = parts[0], parts[1]
                    root_state = robot.data.default_root_state.clone()
                    print(f"[DEBUG] 7a. 修改前 default_root_state[0]: {root_state[0, :3]}")
                    root_state[:, 0] = spawn_x
                    root_state[:, 1] = spawn_y
                    robot.data.default_root_state[:] = root_state
                    print(f"[DEBUG] 7b. 修改后 default_root_state[0]: {root_state[0, :3]}")
                    print(f"[DEBUG] 机器人位置已设置为 ({spawn_x}, {spawn_y})")
                except KeyError as e:
                    print(f"[WARN] inner_env.scene['robot'] 不存在: {e}")
            else:
                print(f"[WARN] inner_env 没有 scene 属性，跳过 spawn_pos 设置")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] spawn_pos 设置失败: {e}")

    env = InstinctRlVecEnvWrapper(env)
    print("[DEBUG] 7. InstinctRlVecEnvWrapper 完成")

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

    use_preset_fall_rate = getattr(args_cli, 'preset_fall_rate_map', False)
    fall_rate_map = LocalFallRateMap(
        map_size=240,
        cell_size=0.05,
        max_depth=2.5,
        fov=87.0,
        focal_length=400.0,
        horizontal_aperture=640.0,
        use_preset_fall_rate=use_preset_fall_rate,
    )

    frontier_detector = FrontierDetector(
        map_size=240,
        cell_size=0.05,
        fov=87.0,
        max_depth=2.5,
    )

    frontier_vis_data = []
    frontier_nav_state = {
        'reached_cooldown': 0,
        'last_target': None,
    }
    save_depth_dir = None
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")

    import datetime
    run_id = np.random.randint(10000)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frontier_save_dir = os.path.join(log_dir, "frontier", f"run_{timestamp}_{run_id}")
    os.makedirs(frontier_save_dir, exist_ok=True)
    print(f"[INFO] 前沿数据保存目录: {frontier_save_dir}")

    if use_preset_fall_rate:
        fall_rate_map_path = os.path.join(frontier_save_dir, "preset_fall_rate_map.json")
        fall_rate_map.save_preset_map(fall_rate_map_path)

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
                if timestep % 200 == 0:
                    print(f"[DEBUG] depth_data type={type(depth_data)}, len={len(depth_data) if hasattr(depth_data, '__len__') else 'N/A'}")
                if depth_data is not None and len(depth_data) > 0:
                    depth_np = depth_data[0].cpu().numpy()
                    if depth_np.ndim == 3:
                        depth_np = depth_np.squeeze(-1)
                    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=10.0, neginf=0.0)
                else:
                    if timestep % 200 == 0:
                        print(f"[DEBUG] 深度数据为空或无效")
            except Exception as e:
                if timestep % 200 == 0:
                    print(f"[DEBUG] 获取深度图失败: {e}")

            if classifier is not None and depth_np is not None:
                fall_rate, label, terrain_name = classifier.predict(depth_np)
            else:
                fall_rate = 0.0
                terrain_name = "unknown"

            fall_rate_override = getattr(args_cli, 'fall_rate_override', None)
            if fall_rate_override is not None:
                fall_rate = fall_rate_override
                terrain_name = f"override({fall_rate_override:.2f})"

            if timestep == 0:
                print(f"[DEBUG] ===== 部署开始 =====")
                print(f"[DEBUG] target_pos: ({target_pos[0]}, {target_pos[1]})")
                print(f"[DEBUG] spawn_pos: {spawn_pos_str}")
                print(f"[DEBUG] env_spacing: {env_cfg.scene.env_spacing}")
                try:
                    env_origins_debug = raw_env.scene.env_origins.cpu().numpy() if hasattr(raw_env.scene, 'env_origins') else np.array([[0.0, 0.0, 0.0]])
                    print(f"[DEBUG] env_origins shape: {env_origins_debug.shape}")
                    print(f"[DEBUG] env_origins[0]: ({env_origins_debug[0, 0]:.2f}, {env_origins_debug[0, 1]:.2f})")
                    print(f"[DEBUG] env_origins all:\n{env_origins_debug}")
                    root_state_debug = inner_env.scene['robot'].data.default_root_state[0].cpu().numpy()
                    print(f"[DEBUG] default_root_state[0]: ({root_state_debug[0]:.2f}, {root_state_debug[1]:.2f}, {root_state_debug[2]:.2f})")
                    if hasattr(raw_env.scene.terrain, 'terrain_levels'):
                        tl = raw_env.scene.terrain.terrain_levels.cpu().numpy()
                        print(f"[DEBUG] terrain_levels: {tl}")
                    if hasattr(raw_env.scene.terrain, 'terrain_types'):
                        tt = raw_env.scene.terrain.terrain_types.cpu().numpy()
                        print(f"[DEBUG] terrain_types: {tt}")
                except Exception as e:
                    print(f"[DEBUG] 获取 env_origins 失败: {e}")

            robot_pos = None
            robot_yaw = None
            env_origins = np.array([0.0, 0.0, 0.0])
            try:
                robot_entity = env_scene["robot"]
                root_pos = robot_entity.data.root_link_pos_w
                root_quat = robot_entity.data.root_quat_w
                env_origins = raw_env.scene.env_origins[0].cpu().numpy() if hasattr(raw_env.scene, 'env_origins') else np.array([0.0, 0.0, 0.0])
                if torch.is_tensor(root_pos) and torch.is_tensor(root_quat):
                    rx = root_pos[0, 0].item() - env_origins[0]
                    ry = root_pos[0, 1].item() - env_origins[1]
                    robot_pos = (rx, ry)
                    q = root_quat[0]
                    w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
                    robot_yaw = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
                    if timestep % 200 == 0:
                        print(f"[DEBUG] env_origins={env_origins[:2]}, robot_rel=({rx:.3f}, {ry:.3f})")
            except Exception as e:
                if timestep % 200 == 0:
                    print(f"[DEBUG] 获取位置失败: {e}")

            if getattr(args_cli, 'frontier_debug', False) and robot_pos is not None:
                frontier_debug_print = timestep % 200 == 0
                if depth_np is not None:
                    if frontier_debug_print:
                        print(f"[DEBUG] depth_np shape={depth_np.shape}, dtype={depth_np.dtype}, min={depth_np.min():.3f}, max={depth_np.max():.3f}")
                    fall_rate_map.update_map(
                        depth_np, fall_rate, robot_pos[0], robot_pos[1], robot_yaw if robot_yaw else 0.0,
                        skip_fall_rate_update=use_preset_fall_rate
                    )
                    if frontier_debug_print:
                        print(f"[DEBUG] after update_map explored_sum={fall_rate_map.get_explored_mask().sum()}")
                else:
                    if frontier_debug_print:
                        print(f"[前沿 DEBUG] depth_np=None, explored_sum={fall_rate_map.get_explored_mask().sum()}")
                frontier_interval = getattr(args_cli, 'frontier_interval', 100)
                if timestep % frontier_interval == 0:
                    explored_mask = fall_rate_map.get_explored_mask()
                    height_map = fall_rate_map.get_height_map()
                    height_count_map = fall_rate_map.get_height_count_map()
                    frontiers = frontier_detector.detect_frontiers_by_layer(
                        explored_mask, robot_pos[0], robot_pos[1], robot_yaw if robot_yaw else 0.0,
                        height_map=height_map, height_count_map=height_count_map
                    )

                    final_target = planner.target_pos if getattr(args_cli, 'auto_frontier_nav', False) else None
                    if final_target is not None:
                        fx, fy = final_target
                        dist_to_goal = math.sqrt((fx - robot_pos[0])**2 + (fy - robot_pos[1])**2)
                    else:
                        dist_to_goal = 0.0

                    for f in frontiers:
                        f['fall_rate'] = fall_rate_map.get_fall_rate_at(f['x'], f['y'], robot_pos[0], robot_pos[1])

                        goal_dx = f['x'] - robot_pos[0]
                        goal_dy = f['y'] - robot_pos[1]
                        dist_to_frontier = math.sqrt(goal_dx**2 + goal_dy**2)

                        if final_target is not None:
                            to_goal_x = fx - robot_pos[0]
                            to_goal_y = fy - robot_pos[1]
                            to_goal_dist = math.sqrt(to_goal_x**2 + to_goal_y**2)
                            if to_goal_dist > 0.1 and dist_to_frontier > 0.1:
                                to_goal_x /= to_goal_dist
                                to_goal_y /= to_goal_dist
                                goal_dir_x = goal_dx / dist_to_frontier
                                goal_dir_y = goal_dy / dist_to_frontier
                                dot = goal_dir_x * to_goal_x + goal_dir_y * to_goal_y
                                direction_score = max(0.1, min(1.0, dot))
                            else:
                                direction_score = 0.5
                        else:
                            direction_score = 1.0

                        conf_val = float(f['conf']) if f['conf'] is not None else 0.5
                        conf_val = max(0.1, min(2.0, conf_val))
                        fr_val = max(0.0, min(1.0, f['fall_rate']))
                        dist_val = max(0.3, float(f['dist']))
                        f['score'] = (conf_val * direction_score * (1.0 - fr_val)) / dist_val

                    frontiers.sort(key=lambda x: x['score'], reverse=True)

                    if getattr(args_cli, 'auto_frontier_nav', False) and frontiers:
                        if frontier_nav_state['reached_cooldown'] > 0:
                            frontier_nav_state['reached_cooldown'] -= 1
                            if timestep % 100 == 0:
                                print(f"[规划] 冷却中... ({frontier_nav_state['reached_cooldown']})")
                        elif final_target is not None and dist_to_goal < 0.3:
                            frontier_nav_state['reached_cooldown'] = 50
                            frontier_nav_state['last_target'] = (fx, fy)
                            if timestep % 100 == 0:
                                print(f"[规划] 到达前沿目标 ({fx:.2f}, {fy:.2f})，等待新前沿...")
                        else:
                            best_frontier = frontiers[0]
                            if best_frontier['score'] > 0.1:
                                planner.set_target(best_frontier['x'], best_frontier['y'])

                    frontier_vis_data.append({
                        'timestep': timestep,
                        'robot_pos': (robot_pos[0], robot_pos[1]),
                        'env_origins': (env_origins[0], env_origins[1]),
                        'frontiers': frontiers,
                        'explored_mask': explored_mask.copy(),
                    })
                    print(f"[前沿] t={timestep} pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f}) explored={explored_mask.sum()} 前沿数={len(frontiers)}")
                    for i, f in enumerate(frontiers[:5]):
                        print(f"  [L{f['layer']}] world=({f['x']:.2f},{f['y']:.2f}) dist={f['dist']:.2f}m h={f['height']:.2f}m fr={f['fall_rate']:.2f} score={f['score']:.3f}")
                    frontier_save_interval = getattr(args_cli, 'frontier_save_interval', 500)
                    if frontier_save_interval > 0 and timestep % frontier_save_interval == 0 and frontier_vis_data:
                        import json
                        vis_file = os.path.join(frontier_save_dir, f"frontier_vis_t{timestep}.json")
                        serializable_data = []
                        for d in frontier_vis_data[-10:]:
                            serializable_frontiers = []
                            for f in d['frontiers']:
                                serializable_frontiers.append({
                                    'layer': int(f['layer']),
                                    'dist': float(f['dist']),
                                    'x': float(f['x']),
                                    'y': float(f['y']),
                                    'conf': float(f['conf']),
                                    'height': float(f['height']),
                                    'fall_rate': float(f['fall_rate']),
                                    'score': float(f['score']),
                                    'near_depth': float(f['near_depth']),
                                    'far_depth': float(f['far_depth']),
                                })
                            serializable_data.append({
                                'timestep': int(d['timestep']),
                                'robot_pos': (float(d['robot_pos'][0]), float(d['robot_pos'][1])),
                                'env_origins': (float(d['env_origins'][0]), float(d['env_origins'][1])),
                                'frontiers': serializable_frontiers,
                            })
                        with open(vis_file, 'w') as f:
                            json.dump(serializable_data, f)
                        print(f"[INFO] 前沿可视化数据已保存: {vis_file}")

            obs = planner.get_action(obs, fall_rate, terrain_name, command_obs_slice, vel_debug=getattr(args_cli, 'vel_debug', False), keyboard_command=keyboard_command if getattr(args_cli, 'keyboard_control', False) else None, robot_pos=robot_pos, robot_yaw=robot_yaw, timestep=timestep)

            if timestep % 100 == 0:
                pos_str = f"({robot_pos[0]:.2f}, {robot_pos[1]:.2f})" if robot_pos else "N/A"
                print(f"[t={timestep}] pos={pos_str} | 地形: {terrain_name} | 摔倒率: {fall_rate:.3f}")
                if timestep == 0 and robot_pos:
                    print(f"[DEBUG] 机器人实际位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")

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

    if frontier_vis_data:
        import json
        vis_file = os.path.join(log_dir, f"frontier_vis_{run_id}.json")
        with open(vis_file, 'w') as f:
            json.dump([
                {k: v for k, v in d.items() if k != 'explored_mask'}
                for d in frontier_vis_data
            ], f)
        print(f"[INFO] 前沿可视化数据已保存: {vis_file}")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            for d in frontier_vis_data:
                rx, ry = d['robot_pos']
                for fx, fy, fw in d['frontiers']:
                    ax.plot([rx, fx], [ry, fy], 'b-', alpha=0.3, linewidth=0.5)
                ax.scatter(rx, ry, c='green', s=100, marker='^')
                ax.scatter([fx for fx, fy, fw in d['frontiers']],
                          [fy for fx, fy, fw in d['frontiers']], c='red', s=20, alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Frontier Detection (green=robot, red=frontiers)')
            ax.grid(True)
            fig.savefig(vis_file.replace('.json', '.png'), dpi=150)
            print(f"[INFO] 前沿可视化图已保存: {vis_file.replace('.json', '.png')}")
        except Exception as e:
            print(f"[WARN] 可视化失败: {e}")


if __name__ == "__main__":
    main()
