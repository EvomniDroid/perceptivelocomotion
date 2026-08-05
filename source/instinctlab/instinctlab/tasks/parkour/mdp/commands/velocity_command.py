from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import B2RMVelocityCommandCfg


class B2RMVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with optional speed and straight-line oversampling."""

    cfg: B2RMVelocityCommandCfg

    def __init__(self, cfg: B2RMVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_curriculum_update_step = 0

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        moving_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if moving_ids.numel() == 0:
            return
        active_ids = moving_ids[~self.is_standing_env[moving_ids]]
        if active_ids.numel() > 0 and self.cfg.low_speed_fraction + self.cfg.mid_speed_fraction > 0.0:
            if self.cfg.low_speed_fraction + self.cfg.mid_speed_fraction > 1.0:
                raise ValueError("low_speed_fraction + mid_speed_fraction must not exceed 1.0.")
            bucket = torch.rand(active_ids.numel(), device=self.device)
            low = bucket < self.cfg.low_speed_fraction
            mid = torch.logical_and(
                bucket >= self.cfg.low_speed_fraction,
                bucket < self.cfg.low_speed_fraction + self.cfg.mid_speed_fraction,
            )
            high = ~(low | mid)
            for mask, speed_range in (
                (low, self.cfg.low_speed_range),
                (mid, self.cfg.mid_speed_range),
                (high, self.cfg.high_speed_range),
            ):
                ids = active_ids[mask]
                if ids.numel() > 0:
                    self.vel_command_b[ids, 0] = torch.empty(ids.numel(), device=self.device).uniform_(*speed_range)

        forward = torch.rand(moving_ids.numel(), device=self.device) < self.cfg.rel_forward_envs
        forward_ids = moving_ids[forward & ~self.is_standing_env[moving_ids]]
        self.vel_command_b[forward_ids, 1:] = 0.0
