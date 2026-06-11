# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 Instinct-RL 训练强化学习智能体的脚本。"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse
import copy
import multiprocessing as mp
import os
import sys
from collections import OrderedDict

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip


# 添加 argparse 各项参数
parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--logroot", type=str, default=None, help="Override default log root path, typically `log/instinct_rl/`."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Enable distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument(
    "--local-rank",
    type=int,
    help="Local rank for distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# train.py 专属参数
parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")
parser.add_argument("--stage2_at", type=int, default=10000, help="在 N 轮后从平地切换到全地形（0=禁用两阶段）")
parser.add_argument(
    "--stage2_plan",
    type=str,
    default="full",
    choices=("full", "mound_pit"),
    help="阶段2训练计划：full=全地形；mound_pit=仅训练凸台+坑专项。",
)
# 附加 Instinct-RL 的命令行参数
cli_args.add_instinct_rl_args(parser)
# 附加 AppLauncher 的命令行参数
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True

# 如果要录制视频，则始终启用相机
if args_cli.video:
    args_cli.enable_cameras = True

# 为 Hydra 清空 sys.argv
sys.argv = [sys.argv[0]] + hydra_args

# 启动 omniverse 应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余所有部分在之后执行。"""

import gymnasium as gym
import torch
import torch.distributed as dist
from datetime import datetime

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg
from instinctlab.terrains.shared_terrain_cfg import FLAT_TRAINING_SUB_TERRAINS, TRAINING_SUB_TERRAINS

# 如果在调试模式下，则等待附加 (attach) 调试器
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

# 导入扩展包以设置环境任务
import instinctlab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# 在多进程模式下设置自动核心亲和性 (affinity)
def auto_affinity():
    rank = int(os.environ["RANK"])  # 获取由 torch 分配的 rank
    num_cores = mp.cpu_count() // torch.cuda.device_count()
    core_range = range(rank * num_cores, (rank + 1) * num_cores)
    core_mask = ",".join(map(str, core_range))
    os.system(f"taskset -cp {core_mask} {os.getpid()}")
    print("Affinity auto updated to:", core_mask, "for rank:", rank)


def _get_checkpoint_iteration(path: str) -> int:
    """从 checkpoint 中读取已完成的训练轮数。"""
    loaded_dict = torch.load(path, map_location="cpu", weights_only=False)
    return int(loaded_dict.get("iter", 0))


def _get_remaining_iterations(current_iteration: int, target_iteration: int) -> int:
    """将目标总轮数转换成还需要继续训练的增量轮数。"""
    return max(target_iteration - current_iteration, 0)


def _clone_subterrains_by_keys(keys: list[str]) -> OrderedDict:
    """从训练地形中按 key 挑出一个独立的子集，避免就地改坏全局配置。"""
    return OrderedDict((key, copy.deepcopy(TRAINING_SUB_TERRAINS[key])) for key in keys)


def _build_stage2_plan_subterrains(plan_name: str) -> OrderedDict:
    if plan_name == "full":
        return copy.deepcopy(TRAINING_SUB_TERRAINS)
    if plan_name == "mound_pit":
        sub_terrains = _clone_subterrains_by_keys(["perlin_rough", "raised_mound", "pit_crater"])
        sub_terrains["perlin_rough"].proportion = 0.2
        sub_terrains["raised_mound"].proportion = 0.4
        sub_terrains["pit_crater"].proportion = 0.4
        return sub_terrains
    raise ValueError(f"Unknown stage2 plan: {plan_name}")


def _apply_stage2_plan_overrides(env_cfg, plan_name: str):
    """按专项计划覆写阶段2训练配置。

    设计原则：
    - full: 保持当前 stage2 逻辑不变
    - mound_pit: 聚焦凸台/坑，降低初始难度和速度上限，让策略先学会过障再学更快
    """
    env_cfg.scene.terrain.terrain_generator.sub_terrains = _build_stage2_plan_subterrains(plan_name)

    if plan_name == "full":
        return "全地形 stage2"

    if plan_name == "mound_pit":
        env_cfg.scene.terrain.max_init_terrain_level = min(getattr(env_cfg.scene.terrain, "max_init_terrain_level", 3), 2)

        base_velocity = getattr(env_cfg.commands, "base_velocity", None)
        if base_velocity is not None and hasattr(base_velocity, "velocity_ranges"):
            if "raised_mound" in base_velocity.velocity_ranges:
                base_velocity.velocity_ranges["raised_mound"]["lin_vel_x"] = (0.0, 0.6)
                base_velocity.velocity_ranges["raised_mound"]["ang_vel_z"] = (-0.08, 0.08)
            if "pit_crater" in base_velocity.velocity_ranges:
                base_velocity.velocity_ranges["pit_crater"]["lin_vel_x"] = (0.0, 0.5)
                base_velocity.velocity_ranges["pit_crater"]["ang_vel_z"] = (-0.08, 0.08)

        return "凸台+坑专项 stage2"

    raise ValueError(f"Unknown stage2 plan: {plan_name}")


@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg):
    """使用 Instinct-RL 智能体进行训练主函数。"""
    
    # === 在这里打断点 ===
    # import pdb; pdb.set_trace()
    print("1")
    
    # 使用来自命令行的非 hydra 参数覆盖相关配置
    agent_cfg = cli_args.update_instinct_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置环境的随机种子
    # 注意：在环境初始化的过程中会发生某些随机化操作，因此我们需要在这里提前设定好种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 准备用于分布式训练的各项配置
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            rank=app_launcher.local_rank,
            world_size=int(os.getenv("WORLD_SIZE", 1)),
        )
        auto_affinity()
        local_rank, world_size = dist.get_rank(), dist.get_world_size()
        env_cfg.seed += local_rank
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        print(
            f"[INFO] 根据以下参数启动分布式训练 -- 局部 rank: {local_rank}, 数据并行 world size: {world_size}, 全局 rank: {os.environ['RANK']}"
        )

    # 为记录实验日志指定目录
    if args_cli.logroot is None:
        log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
    else:
        log_root_path = args_cli.logroot

    print(f"[INFO] 将实验记录到以下目录: {log_root_path}")
    print(f"[INFO] 阶段2计划: {args_cli.stage2_plan}")
    # 指定每次运行所在的日志目录名字格式: {时间戳}_{运行名称}
    log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(env_cfg, "run_name", None):
        log_dir += f"_{env_cfg.run_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
        for h_args in hydra_args:
            log_dir += "_"
            log_dir += h_args.split("=")[0].split(".")[-1]
            log_dir += "-"
            log_dir += h_args.split("=")[1]
    if args_cli.stage2_plan != "full":
        log_dir += f"_stage2-{args_cli.stage2_plan}"
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint)  # type: ignore
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from directory: {resume_path}")
        resume_run_name = os.path.basename(os.path.dirname(resume_path))
        log_dir += f"_from{resume_run_name.split('_')[0]}_{resume_run_name.split('_')[1]}"
    # import pdb; pdb.set_trace()
    print("2")
    # 判断两阶段训练
    stage2_at = args_cli.stage2_at
    is_resuming = agent_cfg.resume
    resume_iter = 0
    saved_sub_terrains = copy.deepcopy(TRAINING_SUB_TERRAINS)
    stage2_plan_desc = "全地形 stage2"

    if is_resuming:
        resume_iter = _get_checkpoint_iteration(resume_path)
        print(f"[恢复] checkpoint 当前轮数: {resume_iter}")

    if stage2_at > 0 and resume_iter < stage2_at:
        stage1_target_iter = min(stage2_at, agent_cfg.max_iterations)
        print(f"[两阶段] 阶段1目标轮数: {stage1_target_iter}")
        env_cfg.scene.terrain.terrain_generator.sub_terrains = FLAT_TRAINING_SUB_TERRAINS
        stage2_mode = True
    else:
        stage2_mode = False
        stage2_plan_desc = _apply_stage2_plan_overrides(env_cfg, args_cli.stage2_plan)
        if stage2_at <= 0:
            print(f"[训练] 单阶段模式: 直接使用 {stage2_plan_desc}")
        else:
            print(f"[训练] 当前 checkpoint 已达到/超过 stage1 目标轮数 {stage2_at}: 直接进入{stage2_plan_desc}")

    # 创建 isaac 仿真环境
    print("进入环境构造")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("2.99")
    print("环境构造完成")
    # 为视频录制套上 Wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 已启用训练期间录制视频的功能。")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果强化学习算法有必须要求，则将多智能体实例转化为单智能体实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 封装环境适配 Instinct-RL 的读取要求
    env = InstinctRlVecEnvWrapper(env)
    print("3")
    # 从 instinct-rl 创建跑者 (runner)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    # 加载已有的模型检查点 (checkpoint)
    if is_resuming:
        print(f"[INFO]: 从此处加载模型参数与检查点: {resume_path}")
        runner.load(resume_path)

    # 将所有配置参数原样转储到日志目录
    if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.cprofile:
        import cProfile
        cprofile = cProfile.Profile()
        print("cProfile 性能分析已启用。")
        cprofile.enable()

    print("4")
    # ---- 阶段1: 平地训练 ----
    if stage2_mode:
        stage1_remaining = _get_remaining_iterations(runner.current_learning_iteration, stage1_target_iter)
        stage1_ckpt = os.path.join(log_dir, f"model_flat_stage_{stage1_target_iter}.pt")

        if stage1_remaining > 0:
            print(f"[两阶段] 阶段1: 从第 {runner.current_learning_iteration} 轮继续平地训练 {stage1_remaining} 轮，到 {stage1_target_iter} 轮")
            runner.learn(
                num_learning_iterations=stage1_remaining,
                init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
            )
            runner.save(stage1_ckpt)
            print(f"[两阶段] 阶段1 完成! 保存 checkpoint: {stage1_ckpt}")
        else:
            stage1_ckpt = resume_path if is_resuming else stage1_ckpt
            print(f"[两阶段] 阶段1已完成，直接使用 checkpoint: {stage1_ckpt}")

        env.close()

        # ---- 阶段2: 全地形 ----
        stage2_remaining = _get_remaining_iterations(stage1_target_iter, agent_cfg.max_iterations)
        stage2_plan_desc = _apply_stage2_plan_overrides(env_cfg, args_cli.stage2_plan)
        print(f"[两阶段] 阶段2: 切换到{stage2_plan_desc}，目标总轮数 {agent_cfg.max_iterations}，剩余 {stage2_remaining} 轮")
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = InstinctRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.load(stage1_ckpt)
        if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
            dump_yaml(os.path.join(log_dir, "params", "env_stage2.yaml"), env_cfg)

        stage2_remaining = _get_remaining_iterations(runner.current_learning_iteration, agent_cfg.max_iterations)
        if stage2_remaining > 0:
            runner.learn(
                num_learning_iterations=stage2_remaining,
                init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
            )
        else:
            print(f"[两阶段] 阶段2无需继续训练: 当前轮数 {runner.current_learning_iteration} 已达到目标 {agent_cfg.max_iterations}")
    else:
        # ---- 单阶段 (全地形或恢复) ----
        remaining_iterations = _get_remaining_iterations(runner.current_learning_iteration, agent_cfg.max_iterations)
        if remaining_iterations > 0:
            runner.learn(
                num_learning_iterations=remaining_iterations,
                init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
            )
        else:
            print(f"[训练] 当前 checkpoint 已达到目标总轮数 {agent_cfg.max_iterations}，跳过训练。")

    print("5")
    print("迭代完成")
    print("迭代")
    if args_cli.cprofile:
        cprofile.disable()
        cprofile.dump_stats(os.path.join(log_dir, "cprofile_stats.profile"))

    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()
    # 彻底关闭模拟器
    env.close()


if __name__ == "__main__":
    # 启动运行主函数
    main()
    # 完全关闭仿真 app 以结束残留进程
    simulation_app.close()
