import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.ui.widgets import ManagerLiveVisualizer

from instinctlab.managers import DummyRewardCfg, MultiRewardCfg, MultiRewardManager
from instinctlab.monitors import MonitorManager


class InstinctRlEnv(ManagerBasedRLEnv):
    # 输出所有地形 patch 的 name，便于检查
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        terrain = getattr(self, 'terrain', None)
        if terrain is not None and hasattr(terrain, "terrain_generator"):
            patch_cfgs = getattr(terrain.terrain_generator, "subterrain_specific_cfgs", None)
            if patch_cfgs:
                print("\n[CHECK] 所有地形patch的name:")
                for i, cfg in enumerate(patch_cfgs):
                    print(f"[CHECK] patch {i}: name={getattr(cfg, 'name', None)}")
                # 修正地形类型列表，按 env 顺序分配
                num_envs = getattr(self, 'num_envs', len(patch_cfgs))
                self.terrain_type_list = [patch_cfgs[i % len(patch_cfgs)].name for i in range(num_envs)]
                print(f"[CHECK] 修正后的 terrain_type_list: {self.terrain_type_list}")
    """This class adds additional logging mechanism on sensors to get more
    comprehensive running statistics.
    """

    def load_managers(self):

        # check and routing the reward manager to the multi reward manager
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()
        super().load_managers()
        # replace the parent class's reward manager
        if "reward_group_cfg" in locals():
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi-Reward Manager: ", self.reward_manager)

        self.monitor_manager = MonitorManager(self.cfg.monitors, self)
        print("[INFO] Monitor Manager: ", self.monitor_manager)

    def setup_manager_visualizers(self):
        super().setup_manager_visualizers()
        self.manager_visualizers["monitor_manager"] = ManagerLiveVisualizer(manager=self.monitor_manager)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        return_ = super().step(action)
        monitor_infos = self.monitor_manager.update(dt=self.step_dt)
        self.extras["step"] = self.extras.get("step", {})
        self.extras["step"].update(monitor_infos)
        return return_

    def _reset_idx(self, env_ids: Sequence[int]):
        monitor_infos = self.monitor_manager.reset(env_ids, is_episode=True)
        return_ = super()._reset_idx(env_ids)
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"].update(monitor_infos)

        # 更健壮的地形类型分配逻辑
        terrain = getattr(self.scene, "terrain", None)
        subterrain_cfgs = None
        num_cols = None
        # 优先用subterrain_specific_cfgs属性（兼容无terrain_generator场景）
        if terrain is not None:
            # 先直接取subterrain_specific_cfgs属性
            subterrain_cfgs = getattr(terrain, "subterrain_specific_cfgs", None)
            print(f"[DEBUG] terrain.subterrain_specific_cfgs: {type(subterrain_cfgs)} len={len(subterrain_cfgs) if subterrain_cfgs else None}")
            # 再尝试取terrain_generator的num_cols
            terrain_gen = getattr(terrain, "terrain_generator", None)
            print(f"[DEBUG] terrain.terrain_generator: {type(terrain_gen)}")
            if terrain_gen is not None:
                num_cols = getattr(terrain_gen.cfg, "num_cols", None)
                print(f"[DEBUG] terrain_gen.cfg.num_cols: {num_cols}")
            # 如果terrain_gen没有，尝试从terrain.cfg获取num_cols
            if num_cols is None:
                cfg = getattr(terrain, "cfg", None)
                if cfg is not None:
                    num_cols = getattr(cfg, "num_cols", None)
                    print(f"[DEBUG] terrain.cfg.num_cols: {num_cols}")

        if not hasattr(self, "terrain_type_list"):
            self.terrain_type_list = ["unknown"] * self.num_envs

        for env_id in env_ids:
            # 修正：直接用env_id % len(subterrain_cfgs)循环分配
            terrain_name = "unknown"
            if subterrain_cfgs and len(subterrain_cfgs) > 0:
                idx = env_id % len(subterrain_cfgs)
                cfg = subterrain_cfgs[idx]
                terrain_name = getattr(cfg, "name", "unknown")
                print(f"[DEBUG][RESET] env_id={env_id}, idx={idx}, terrain_name={terrain_name}, cfg type={type(cfg)}")
            else:
                print(f"[DEBUG][RESET] env_id={env_id}, idx={env_id}, terrain_name=unknown (no cfg)")
            self.terrain_type_list[env_id] = terrain_name
        print(f"[DEBUG][RESET] terrain_type_list: {self.terrain_type_list}")

        # 若subterrain_cfgs依然为None，输出一次性警告
        if subterrain_cfgs is None:
            print("[WARN] 地形类型分配失败：未找到subterrain_specific_cfgs，所有地形将标记为unknown")

        return return_

    """
    Properties.
    """

    @property
    def num_rewards(self) -> int:
        return getattr(self.reward_manager, "num_rewards", 1)
