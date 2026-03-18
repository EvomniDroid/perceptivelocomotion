"""Reward manager for computing multiple reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardManager, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .manager_term_cfg import MultiRewardCfg


class MultiRewardManager(RewardManager):
    """管理并计算给定世界中多个奖励信号组的管理器。
    
    该奖励管理器类似于 RewardManager 类，但它计算的是多个组的奖励信号。
    
    奖励项应集中在 RewardGroups 中。然后返回的 reward_buf 
    形状将是 (num_envs, num_groups)，其中每一列对应于总奖励。
    """

    def __init__(self, cfg: MultiRewardCfg, env: ManagerBasedRLEnv):
        """初始化奖励管理器。

        Args:
            cfg: 配置对象或字典 (``dict[str, RewardTermCfg]``)。
            env: 环境实例。
        """
        super().__init__(cfg, env)
        # 准备额外信息，用于存储各个独立奖励项的回合总和（用于 Tensorboard 日志记录）
        self._episode_sums = dict()
        for group_name in self.__group_term_names.keys():
            for term_name in self.__group_term_names[group_name]:
                # 初始化每个环境每个奖励项的得分为 0
                self._episode_sums["_".join([group_name, term_name])] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )
        
        # 初始化最终计算出来并交给 RL 算法的核心奖励缓冲区 dict -> Tensor(num_envs,)
        self._reward_buf = dict()
        for group_name in self.__group_term_cfgs.keys():
            self._reward_buf[group_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 一个用于记录当前步骤中每组各个奖励项的日志缓冲区（单步瞬时记录）
        self._termwise_reward_buf: dict[str, dict[str, torch.Tensor]] = dict()
        for group_name in self.__group_term_cfgs.keys():
            self._termwise_reward_buf[group_name] = dict()
            for term_name in self.__group_term_names[group_name]:
                self._termwise_reward_buf[group_name][term_name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )

    def __str__(self) -> str:
        """Returns: 奖励管理器的字符串表示形式。"""
        msg = f"<MultiRewardManager> contains {len(self.__group_term_names)} active groups.\n"
        msg += f"and {sum(len(terms) for terms in self.__group_term_names.values())} active reward terms.\n"

        # 创建用于展示奖励项信息的表格
        table = PrettyTable()
        table.title = "Active Reward Group Terms"
        table.field_names = ["Index", "Group", "Name", "Weight"]
        # 设置表格列的对齐方式
        table.align["Group"] = "l"
        table.align["Weight"] = "r"
        # 逐一添加每个奖励项的信息(编号, 组别, 名字, 权重)
        index = 0
        for group_name in self.__group_term_names.keys():
            for term_name, term_cfg in zip(self.__group_term_names[group_name], self.__group_term_cfgs[group_name]):
                table.add_row([index, group_name, term_name, term_cfg.weight])
                index += 1
        # 将表格转换为字符串并添加至信息末尾
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    属性 (Properties).
    """

    @property
    def active_terms(self):
        """获取激活的奖励项名称。"""
        return self.__group_term_names

    @property
    def num_rewards(self) -> int:
        """获取奖励组的数量（传给 Critic 的维度）。"""
        return len(self.__group_term_names)

    """
    核心操作 (Operations).
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """环境重置时调用，用于结算历史奖励并清零"""
        # 解析环境ID (如果没有指定，则视为所有环境都需要重置)
        if env_ids is None:
            env_ids = range(self.num_envs)
        
        # 存储日志信息 (extras字典最后会被送去跑 Tensorboard 画图)
        extras = {}
        for key in self._episode_sums.keys():
            # 计算该奖励项在需要重置的环境中的平均总得分
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # 根据时间步或最大步骤数归一化得分写入 extras
            extras["Episode_Reward/" + key + "/max_episode_len_s"] = episodic_sum_avg / self._env.max_episode_length_s
            extras["Episode_Reward/" + key + "/sum"] = episodic_sum_avg
            extras["Episode_Reward/" + key + "/timestep"] = torch.mean(
                self._episode_sums[key][env_ids] / self._env.episode_length_buf[env_ids]
            )
            # 重置环境中的该项奖励的累计分数为 0
            self._episode_sums[key][env_ids] = 0.0
            
        # 重置底层奖励函数内部的状态 (有些带有记忆的复杂的 reward 类在初始化中需要归零)
        for group_class_term_cfg in self.__group_class_term_cfgs.values():
            for term_cfg in group_class_term_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # 返回被整理并结算好的 log 信息
        return extras

    def compute(self, dt: float) -> dict[str, torch.Tensor]:
        """
        计算奖励值。这一步是从物理引擎出来后被调用的核心。
        Returns:
            为了每一个奖励组返回带形状为 (num_envs,) 的信号字典。
        """
        # 在每步开始时，先将缓冲区的分数值置 0 或者初始化步长 dt
        for group_name in self.__group_term_cfgs.keys():
            term_combine_method = self.__group_term_combine_methods.get(group_name, "sum")
            if term_combine_method == "sum":
                self._reward_buf[group_name][:] = 0.0
            elif term_combine_method == "prod":
                self._reward_buf[group_name][:] = dt
            else:
                raise ValueError(f"Invalid term combine method: {term_combine_method}")
                
        # 加一个标志位，只在计算第一个奖励项时触发 pdb 断点
        _has_pdb_triggered = False
        
        # 遍历配置表上所有的奖励项和它们的组别
        for group_name, terms_cfgs in self.__group_term_cfgs.items():
            term_combine_method = self.__group_term_combine_methods.get(group_name, "sum")
            # 内层循环：遍历这个组里的所有奖励项（这就是你说的十几个项）
            for term_name, term_cfg in zip(self.__group_term_names[group_name], terms_cfgs):
                # ！！！如果权重被设置为 0.0，跳过不计算这可以作为关闭某个奖励特征的微优化
                if term_cfg.weight == 0.0:
                    continue
                # 🔥最核心的一行！这行代码去运行配置表里注册的奖励函数(如动作模仿、越界惩罚等)，将返回值乘以配置好的权重
                
                # 仅在第一次循环时触发断点
                if not _has_pdb_triggered:
                    # import pdb; pdb.set_trace()
                    print("这里是奖励")
                    print("4.2.0.1")
                    print(f"！！！！当前计算的奖励项是: {term_name} ！！！！")
                    print("！！！！从step里面进的这里reward，可以按下s，注意下这里有几个value！！！！")
                    _has_pdb_triggered = True
                
                value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
                
                # 根据合并方法把小项的分数更新追加到整体组奖励池里
                if term_combine_method == "sum":
                    self._reward_buf[group_name] += value * dt
                elif term_combine_method == "prod":
                    self._reward_buf[group_name] *= value
                else:
                    raise ValueError(f"Invalid term combine method: {term_combine_method}")
                    
                # 更新本步记录的局部字典缓冲区用于日志记录
                self._termwise_reward_buf[group_name][term_name] = value  # (num_envs,)
                # 更新当前整局实验大记录的全局字典
                self._episode_sums["_".join([group_name, term_name])] += value * dt
                
        # 将算好的所有组的加权总分返回给管家
        return self._reward_buf

    def get_term_cfg(self, term_name: str, group_name: str | None = None) -> RewardTermCfg:
        """获取给定术语名称和组名称的配置。

        Args:
            term_name: 术语（奖励项）的名称。
            group_name: 组的名称。如果为None，将使用第一个组。

        Returns:
            奖励项配置。
        """
        if group_name is None:
            group_name = list(self.__group_term_names.keys())[0]
        if group_name not in self.__group_term_names:
            raise ValueError(f"Group '{group_name}' not found.")
        if term_name not in self.__group_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        index = self.__group_term_names[group_name].index(term_name)
        return self.__group_term_cfgs[group_name][index]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for group_name in self.__group_term_cfgs.keys():
            for term_name in self.__group_term_names[group_name]:
                # NOTE: there are some shitty conventions in feeding back to manager_live_visualizer.
                # You need to return a list[tuple[str, Iterable[float]]] where the first element is the name of the term
                terms.append(
                    (
                        f"{group_name}-{term_name}",
                        [self._termwise_reward_buf[group_name][term_name][env_idx].cpu().item()],
                    )
                )
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepare the reward group and each term in the groups for computation."""
        self.__group_term_names: dict[str, list[str]] = dict()
        self.__group_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self.__group_class_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self.__group_term_combine_methods: dict[str, str] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            groups_cfg_items = self.cfg.items()
        else:
            groups_cfg_items = self.cfg.__dict__.items()
        for group_name, group_cfg in groups_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms
            for term_name, term_cfg in group_cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # check configs for the group specifically
                if term_name == "combine_method":
                    assert isinstance(
                        term_cfg, str
                    ), f"Configuration for the term '{term_name}' in group '{group_name}' is not of type str."
                    self.__group_term_combine_methods[group_name] = term_cfg
                    continue
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' in group '{group_name}' is not of type"
                        f" RewardTermCfg. Received: '{type(term_cfg)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add the term to the group
                if group_name not in self.__group_term_names:
                    self.__group_term_names[group_name] = list()
                    self.__group_term_cfgs[group_name] = list()
                    self.__group_class_term_cfgs[group_name] = list()
                self.__group_term_names[group_name].append(term_name)
                self.__group_term_cfgs[group_name].append(term_cfg)
                if isinstance(term_cfg.func, ManagerTermBase):
                    self.__group_class_term_cfgs[group_name].append(term_cfg)
