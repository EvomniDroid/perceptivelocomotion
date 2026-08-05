from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import B2RMVelocityCommandCfg


class B2RMVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with straight-line oversampling and command-local gait time."""

    cfg: B2RMVelocityCommandCfg

    def __init__(self, cfg: B2RMVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gait_time = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.gait_time[env_ids] = 0.0

        moving_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if moving_ids.numel() == 0:
            return
        forward = torch.rand(moving_ids.numel(), device=self.device) < self.cfg.rel_forward_envs
        forward_ids = moving_ids[forward & ~self.is_standing_env[moving_ids]]
        self.vel_command_b[forward_ids, 1:] = 0.0

    def _update_command(self):
        super()._update_command()
        moving = torch.logical_or(
            torch.norm(self.vel_command_b[:, :2], dim=-1) >= 0.05,
            torch.abs(self.vel_command_b[:, 2]) >= 0.05,
        )
        self.gait_time[moving] += self._env.step_dt
        self.gait_time[~moving] = 0.0
