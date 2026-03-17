# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 Instinct-RL 训练强化学习智能体的脚本。"""

"""首先启动 Isaac Sim 模拟器。"""

import argparse
import multiprocessing as mp
import os
import sys

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


@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg):
    """使用 Instinct-RL 智能体进行训练主函数。"""
    
    # === 在这里打断点 ===
    import pdb; pdb.set_trace()
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
    log_dir = os.path.join(log_root_path, log_dir)

    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint)  # type: ignore
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from directory: {resume_path}")
        resume_run_name = os.path.basename(os.path.dirname(resume_path))
        log_dir += f"_from{resume_run_name.split('_')[0]}_{resume_run_name.split('_')[1]}"
    import pdb; pdb.set_trace()
    print("2")
    # 创建 isaac 仿真环境
    print("进入环境构造")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
    print("3")
    # 从 instinct-rl 创建跑者 (runner)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # 将当前的 git 代码仓库状态写入日志中
    runner.add_git_repo_to_log(__file__)
    # 加载已有的模型检查点 (checkpoint)
    if agent_cfg.resume:
        print(f"[INFO]: 从此处加载模型参数与检查点: {resume_path}")
        # 加载之前训练过的模型
        runner.load(resume_path)

    # 将所有配置参数原样转储 (dump) 到日志目录记录
    if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
        # 通过判断 rank>0 ，以防止非 rank-0 零的主进程重复转储配置导致冲突
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.cprofile:
        import cProfile

        cprofile = cProfile.Profile()
        print(
            "cProfile 性能分析已启用。程序完成运行后，会自动在日志目录下保存为以 .profile 结尾的日志文件。"
        )
        cprofile.enable()
    import pdb; pdb.set_trace()
    print("4")
    print("按s")
    # 开始执行主训练环节
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )
    import pdb; pdb.set_trace()
    print("5")
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
