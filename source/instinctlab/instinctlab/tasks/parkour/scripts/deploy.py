# Deploy script: Run RL policy with fall rate classifier and motion planner
# 部署脚本：使用摔倒率分类器和运动规划器运行RL策略

import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

import cli_args

# 命令行参数定义
parser = argparse.ArgumentParser(description="部署RL智能体，包含摔倒率分类器和运动规划器")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--num_envs", type=int, default=1, help="仿真环境数量")
parser.add_argument("--load_run", type=str, default=None, help="RL检查点目录路径")
parser.add_argument("--checkpoint", type=str, default="model_40000.pt", help="检查点文件名")
parser.add_argument("--classifier_model", type=str, required=True, help="摔倒率分类器模型路径")
parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
parser.add_argument("--save_depth_interval", type=int, default=0, help="每N步保存一次深度图，0表示禁用")
parser.add_argument("--fall_rate_threshold", type=float, default=0.5, help="切换到安全模式的摔倒率阈值")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="禁用fabric，使用USD I/O操作")
parser.add_argument("--video", action="store_true", default=False, help="录制部署视频")
parser.add_argument("--video_length", type=int, default=3000, help="视频长度（步数）")
parser.add_argument("--video_start_step", type=int, default=0, help="仿真开始步数")
parser.add_argument("--useonnx", action="store_true", default=False, help="使用ONNX模型进行推理")
parser.add_argument("--exportonnx", action="store_true", default=False, help="导出策略为ONNX模型")
parser.add_argument("--debug", action="store_true", default=False, help="启用调试模式")

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from PIL import Image
import cv2

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

import sys
sys.path.append("/home/zh/isaac/liveratemodel")
from model import create_model
from dataset import LABEL_TO_FALL_RATE, LABEL_TO_TERRAIN, build_model_input_from_depth_array, get_input_channels
import torch.nn.functional as F


class FallRateClassifier:
    """摔倒率分类器：基于深度图预测摔倒风险"""

    def __init__(self, model_path, device="cuda:0"):
        # 初始化摔倒率分类器
        # Args:
        #     model_path: 分类器模型文件路径(.pt)
        #     device: 推理设备(cuda:0 或 cpu)
        self.device = device

        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})

        # 模型配置参数
        self.input_mode = config.get("input_mode", "depth_gradients")
        self.patch_size = config.get("patch_size", 32)
        self.point_count = config.get("point_count", 1024)
        self.focal_length = config.get("focal_length", 24.0)
        self.horizontal_aperture = config.get("horizontal_aperture", 20.955)
        self.vertical_aperture = config.get("vertical_aperture", None)

        # 创建模型
        in_channels = get_input_channels(self.input_mode)
        self.model = create_model(
            model_type="classifier",
            num_classes=10,
            device=device,
            use_resnet=config.get("use_resnet", False),
            use_pointnet=self.input_mode == "point_cloud",
            pretrained=False,
            in_channels=in_channels,
        )

        # 加载权重
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        print(f"[INFO] 摔倒率分类器加载自: {model_path}")
        print(f"[INFO] 输入模式: {self.input_mode}, patch_size: {self.patch_size}")

    @torch.no_grad()
    def predict(self, depth_array):
        # 从深度图预测摔倒率
        # Args:
        #     depth_array: 深度图 numpy数组，形状为 (H, W)
        # Returns:
        #     fall_rate: 摔倒率 (0-1 之间)
        #     terrain_label: 地形类别标签 (int)
        #     terrain_name: 地形类别名称 (str)
        # 计算深度缩放因子
        depth_max = float(depth_array.max())
        depth_scale = depth_max if depth_max > 1.0 else 1.0

        # 根据输入模式预处理深度图
        if self.input_mode != "point_cloud":
            img = Image.fromarray(depth_array).convert("L")
            img = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            depth_array = np.array(img)

        # 构建模型输入
        tensor = build_model_input_from_depth_array(
            depth_array,
            input_mode=self.input_mode,
            depth_scale=depth_scale,
            point_count=self.point_count,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
            vertical_aperture=self.vertical_aperture,
        )
        tensor = tensor.unsqueeze(0).to(self.device)

        # 模型推理
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        label = int(np.argmax(probs))
        fall_rate = LABEL_TO_FALL_RATE.get(label, 0.5)

        return fall_rate, label, LABEL_TO_TERRAIN.get(label, "unknown")


class MotionPlanner:
    # 运动规划器接口 - 这是你自定义规划器的基类/框架，你需要根据实际需求实现具体的规划逻辑

    def __init__(self, cfg=None):
        # 初始化运动规划器
        # Args:
        #     cfg: 配置字典，包含安全模式参数等
        self.cfg = cfg or {}
        # 安全模式下的线速度缩放因子（小于1表示降低速度）
        self.safe_mode_lin_vel_scale = self.cfg.get("safe_mode_lin_vel_scale", 0.5)
        # 安全模式下的步幅宽度
        self.safe_mode_step_width = self.cfg.get("safe_mode_step_width", 0.5)

    def get_action(self, obs, fall_rate, terrain_type, policy_fn=None):
        # 根据当前观测和摔倒率获取动作
        # 这是核心规划函数，你需要在这里实现：
        # - 基于摔倒率选择不同模式（安全/正常/激进）
        # - 基于地形类型调整步态参数
        # - 调用RL策略或使用预设动作
        # Args:
        #     obs: 当前观测（字典或tensor）
        #     fall_rate: 预测的摔倒率 (0-1)
        #     terrain_type: 地形类型字符串
        #     policy_fn: 可选的RL策略函数
        # Returns:
        #     action: 发送给机器人的动作tensor
        #     mode: 当前模式 ("normal", "safe", 或自定义)
        # 根据摔倒率判断模式
        if fall_rate > 0.5:
            mode = "safe"
            print(f"[规划器] 安全模式激活! 摔倒率={fall_rate:.3f}, 地形={terrain_type}")
        else:
            mode = "normal"

        # 调用策略获取动作
        if policy_fn is not None:
            action = policy_fn(obs)
            # 安全模式下缩小动作幅度
            if mode == "safe":
                action = action * self.safe_mode_lin_vel_scale
            return action, mode
        else:
            # 如果没有策略，返回零动作
            return torch.zeros(12), mode


def main():
    # 主部署函数
    print("\n" + "="*60)
    print("部署 RL策略 + 摔倒率分类器 + 运动规划器")
    print("="*60)

    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # 解析智能体配置
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # 设置日志目录
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run

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

    # 创建仿真环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 转换为单智能体环境（如需要）
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装环境
    env = InstinctRlVecEnvWrapper(env)

    # 加载RL策略
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO] 加载RL策略: {resume_path}")
        ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 加载摔倒率分类器
    print(f"\n[INFO] 加载摔倒率分类器: {args_cli.classifier_model}")
    classifier = FallRateClassifier(args_cli.classifier_model, device=args_cli.device)

    # 初始化运动规划器
    planner = MotionPlanner(cfg={"safe_mode_lin_vel_scale": 0.5})

    # 设置深度图保存
    run_id = np.random.randint(10000)
    save_depth_dir = None
    if args_cli.save_depth_interval > 0:
        save_depth_dir = os.path.join(log_dir, f"deploy_depth_{run_id}")
        os.makedirs(save_depth_dir, exist_ok=True)
        print(f"[INFO] 深度图保存目录: {save_depth_dir}")

    # 初始化环境
    obs, _ = env.get_observations()
    timestep = 0
    episode_counts = {}
    num_envs = env.unwrapped.scene.num_envs

    print("\n" + "="*60)
    print("开始部署循环...")
    print("="*60 + "\n")

    # 主循环
    while simulation_app.is_running():
        with torch.inference_mode():
            try:
                # 获取当前帧深度图
                depth_data = env.unwrapped.scene["camera"].data.output["distance_to_image_plane"]
                depth_np = depth_data[0].cpu().numpy()

                # 处理深度图维度
                if depth_np.ndim == 3:
                    depth_np = depth_np.squeeze(-1)
                depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=10.0, neginf=0.0)

                # 调用分类器预测摔倒率
                fall_rate, label, terrain_name = classifier.predict(depth_np)

                if timestep % 100 == 0:
                    print(f"[t={timestep}] 地形: {terrain_name}, 摔倒率: {fall_rate:.3f}")

            except Exception as e:
                print(f"[警告] 获取深度图或分类失败: {e}")
                fall_rate = 0.0
                terrain_name = "unknown"
                label = 0

            # 获取动作（通过规划器）
            action, mode = planner.get_action(obs, fall_rate, terrain_name, policy_fn=policy)

            # 执行动作
            obs, rewards, dones, infos = env.step(action)

            # 统计episode完成情况
            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_counts[env_id] = episode_counts.get(env_id, 0) + 1

            # 渲染（如需要）
            if args_cli.video:
                env.unwrapped.render()

            # 保存深度图（如需要）
            if save_depth_dir is not None and timestep % args_cli.save_depth_interval == 0:
                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max - d_min > 1e-8:
                    depth_normalized = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)

                img_depth = Image.fromarray(depth_normalized)
                img_path = os.path.join(save_depth_dir, f"step_{timestep:06d}_fall{fall_rate:.2f}_{terrain_name}.png")
                img_depth.save(img_path)

            timestep += 1

        # 视频录制模式，录制到指定长度后退出
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] 达到视频长度 {args_cli.video_length}，退出...")
            break

    print("\n[INFO] 部署完成")


if __name__ == "__main__":
    main()
