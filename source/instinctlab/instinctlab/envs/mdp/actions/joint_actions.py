from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp import JointPositionAction

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
