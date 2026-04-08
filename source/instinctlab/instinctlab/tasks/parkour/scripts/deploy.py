"""部署脚本：RL策略 + 摔倒率分类器 + 运动规划器"""

import argparse
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

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))
import cli_args
cli_args.add_instinct_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, 'video', False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

        outputs = self.model(model_input)
        probabilities = F.softmax(outputs, dim=1)
        probs_np = probabilities.cpu().numpy()[0]
        pred_label = np.argmax(probs_np)
        fall_rate = LABEL_TO_FALL_RATE[pred_label]
        terrain_name = LABEL_TO_TERRAIN.get(pred_label, "unknown")

        return fall_rate, pred_label, terrain_name


class MotionPlanner:
    """运动规划器接口 - 根据摔倒率和地形类型调整动作"""

    def __init__(self, cfg=None, vel_debug=False, debug_vels=None):
        self.cfg = cfg or {}
        self.safe_mode_lin_vel_scale = self.cfg.get("safe_mode_lin_vel_scale", 0.5)
        self.safe_mode_step_width = self.cfg.get("safe_mode_step_width", 0.5)
        self.vel_debug = vel_debug
        self.debug_vels = debug_vels or {}

    def get_action(self, obs, fall_rate, terrain_type, command_obs_slice, vel_debug=False):
        if vel_debug:
            obs = self._inject_debug_velocity(obs, command_obs_slice)
            return obs

        return obs

    def _inject_debug_velocity(self, obs, command_obs_slice):
        obs = obs.clone()
        vel_x = self.debug_vels.get("vel_x", 0.5)
        vel_y = self.debug_vels.get("vel_y", 0.0)
        ang_z = self.debug_vels.get("ang_z", 0.0)
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
    else:
        print("[INFO] 使用训练地形配置 (ROUGH_TERRAINS_CFG)")

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
    planner = MotionPlanner(
        cfg={"safe_mode_lin_vel_scale": 0.5},
        vel_debug=getattr(args_cli, 'vel_debug', False),
        debug_vels=debug_vels
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

    print("\n" + "="*60)
    print("开始部署循环...")
    print("="*60 + "\n")

    try:
        while True:
            depth_np = None

            if classifier is not None and depth_np is not None:
                fall_rate, label, terrain_name = classifier.predict(depth_np)
            else:
                fall_rate = 0.0
                terrain_name = "unknown"

            obs = planner.get_action(obs, fall_rate, terrain_name, command_obs_slice, vel_debug=getattr(args_cli, 'vel_debug', False))

            if timestep % 100 == 0:
                print(f"[t={timestep}] 运行中 | 模式: vel_debug | 地形: {terrain_name} | 摔倒率: {fall_rate:.3f}")

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            timestep += 1

    except KeyboardInterrupt:
        print("\n[INFO] 部署中断")

    env.close()
    print("[INFO] 部署完成")


if __name__ == "__main__":
    main()
