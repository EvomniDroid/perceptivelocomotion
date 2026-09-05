from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp import JointPositionAction
from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ActionTerm, ActionTermCfg

    from . import action_cfg


class ActionOverridenMixin:
    """Override some action dimensions with the provided values constantly."""

    def __init__(self: ActionTerm, cfg: ActionTermCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)  # type: ignore
        self._override_action_ids = self._env.scene[cfg.asset_cfg.name].find_joints(cfg.asset_cfg.joint_names)[0]
        self._override_value = cfg.override_value

    def process_actions(self: ActionTerm, action: torch.Tensor):
        _raw_actions = action
        action = _raw_actions.clone()
        action[:, self._override_action_ids] = self._override_value
        super().process_actions(action)
        self._raw_actions[:] = _raw_actions


class ActionOverridenJointPositionAction(ActionOverridenMixin, JointPositionAction):
    """Delayed joint position action term that overrides some action dimensions with the provided values constantly."""

    cfg: action_cfg.ActionOverridenJointPositionActionCfg


class DynamicTargetJointPositionAction(JointPositionAction):
    """Joint position action that follows a per-env target stored on the articulation.

    This keeps the policy action dimensions unchanged while allowing reset/interval
    events to own a subset of joints, such as a carried arm pose.
    """

    cfg: action_cfg.DynamicTargetJointPositionActionCfg

    def __init__(self, cfg: action_cfg.DynamicTargetJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._ee_body_idx = None
        self._jacobi_body_idx = None
        if cfg.ee_body_name is not None:
            body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name, preserve_order=True)
            if len(body_ids) != 1:
                raise ValueError(
                    f"Expected exactly one body for ee_body_name={cfg.ee_body_name!r}, got {body_names}."
                )
            self._ee_body_idx = body_ids[0]
            self._jacobi_body_idx = self._ee_body_idx - 1 if self._asset.is_fixed_base else self._ee_body_idx

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        ee_target_pos_w = None
        if self.cfg.ee_target_pos_attr_name is not None:
            ee_target_pos_w = getattr(self._asset, self.cfg.ee_target_pos_attr_name, None)

        if ee_target_pos_w is not None and self._ee_body_idx is not None and self._jacobi_body_idx is not None:
            ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
            delta_pos = ee_target_pos_w - ee_pos_w
            delta_norm = torch.norm(delta_pos, dim=-1, keepdim=True).clamp_min(1e-6)
            max_delta = float(self.cfg.max_ik_delta)
            delta_pos = delta_pos * torch.clamp(max_delta / delta_norm, max=1.0)

            jacobian = self._asset.root_physx_view.get_jacobians()[
                :, self._jacobi_body_idx, :3, self._joint_ids
            ]
            jacobian_t = torch.transpose(jacobian, 1, 2)
            damping = float(self.cfg.ik_damping)
            lhs = torch.bmm(jacobian, jacobian_t) + (damping**2) * torch.eye(
                3, device=self.device, dtype=jacobian.dtype
            ).unsqueeze(0)
            delta_joint_pos = torch.bmm(jacobian_t, torch.linalg.solve(lhs, delta_pos.unsqueeze(-1))).squeeze(-1)
            joint_pos_des = (
                self._asset.data.joint_pos[:, self._joint_ids] + float(self.cfg.ik_gain) * delta_joint_pos
            )
            joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._processed_actions[:] = joint_pos_des.clamp(joint_limits[..., 0], joint_limits[..., 1])
            return

        dynamic_target = getattr(self._asset, self.cfg.target_attr_name, None)
        if dynamic_target is None:
            return
        self._processed_actions[:] = dynamic_target[:, self._joint_ids]


class FilteredJointPositionAction(JointPositionAction):
    """Apply a first-order filter while preserving the policy's raw action."""

    cfg: action_cfg.FilteredJointPositionActionCfg

    def __init__(self, cfg: action_cfg.FilteredJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0.0 < cfg.filter_alpha <= 1.0:
            raise ValueError(f"filter_alpha must be in (0, 1], got {cfg.filter_alpha}.")
        self._filtered_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        alpha = float(self.cfg.filter_alpha)
        self._filtered_actions.mul_(1.0 - alpha).add_(actions, alpha=alpha)
        super().process_actions(self._filtered_actions)
        # Action history and saturation rewards must retain the network output,
        # while processed_actions contains the command that reaches the PD loop.
        self._raw_actions[:] = actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._filtered_actions[env_ids] = 0.0


class FixedJointPositionAction(ActionTerm):
    """Hold selected joints at fixed positions without consuming policy actions."""

    cfg: action_cfg.FixedJointPositionActionCfg

    def __init__(self, cfg: action_cfg.FixedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        if len(self._joint_ids) == 0:
            raise ValueError("FixedJointPositionAction did not resolve any joints.")

        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        for joint_name, target in cfg.joint_pos.items():
            if joint_name not in self._joint_names:
                raise ValueError(f"Fixed joint target {joint_name!r} was not resolved.")
            joint_index = self._joint_names.index(joint_name)
            self._processed_actions[:, joint_index] = float(target)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[1] != 0:
            raise ValueError(f"FixedJointPositionAction expected zero actions, got {actions.shape[1]}.")

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass
