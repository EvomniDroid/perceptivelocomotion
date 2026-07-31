from __future__ import annotations

from collections.abc import Sequence

import torch

from .manager_based_rl_env import InstinctRlEnv


class B2RMHandoffRlEnv(InstinctRlEnv):
    """B2RM environment with a scripted stand-up and policy handoff after reset."""

    def __init__(self, *args, **kwargs):
        self._handoff_ready = False
        super().__init__(*args, **kwargs)

        self.handoff_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.policy_step_buf = torch.zeros_like(self.handoff_step_buf)
        self.handoff_policy_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.handoff_action_alpha = torch.zeros(self.num_envs, device=self.device)
        self.handoff_gain_alpha = torch.zeros(self.num_envs, device=self.device)

        self._robot = self.scene["robot"]
        self._leg_actuator = self._robot.actuators["legs"]
        self._leg_action_term = self.action_manager.get_term("leg_joint_pos")
        self._leg_action_start = 0
        for term_name in self.action_manager._term_names:
            if term_name == "leg_joint_pos":
                break
            self._leg_action_start += self.action_manager.get_term(term_name).action_dim
        self._leg_action_end = self._leg_action_start + self._leg_action_term.action_dim

        self._target1 = self._joint_targets(
            {
                "FR_hip_joint": 0.0,
                "FL_hip_joint": 0.0,
                "RR_hip_joint": -0.2,
                "RL_hip_joint": 0.2,
                ".*_thigh_joint": 1.36,
                ".*_calf_joint": -2.65,
            }
        )
        self._target2 = self._joint_targets(
            {
                "FR_hip_joint": -0.1,
                "FL_hip_joint": 0.1,
                "RR_hip_joint": -0.1,
                "RL_hip_joint": 0.1,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            }
        )
        self._prone = self._joint_targets(
            {
                "FR_hip_joint": -0.5,
                "FL_hip_joint": 0.5,
                "RR_hip_joint": -0.5,
                "RL_hip_joint": 0.5,
                ".*_thigh_joint": 1.36,
                ".*_calf_joint": -2.65,
            }
        )

        self._target1_steps = self._seconds_to_steps(self.cfg.handoff_target1_seconds)
        self._target2_steps = self._seconds_to_steps(self.cfg.handoff_target2_seconds)
        self._hold_steps = self._seconds_to_steps(self.cfg.handoff_hold_seconds)
        self._gain_blend_steps = self._seconds_to_steps(self.cfg.handoff_gain_blend_seconds)
        self._action_blend_steps = self._seconds_to_steps(self.cfg.handoff_action_blend_seconds)
        self._target1_end = self._target1_steps
        self._target2_end = self._target1_end + self._target2_steps
        self._hold_end = self._target2_end + self._hold_steps
        self._policy_blend_end = self._hold_end + self._action_blend_steps
        self._gain_blend_end = self._policy_blend_end + self._gain_blend_steps
        self._termination_enable_step = self._gain_blend_end + self._seconds_to_steps(
            self.cfg.handoff_termination_grace_seconds
        )
        self._debug_phase = None

        self._handoff_ready = True
        self._reset_handoff_state(torch.arange(self.num_envs, device=self.device))
        self.obs_buf = self.observation_manager.compute(update_history=False)

        print(
            "[B2RM HANDOFF] prone -> target1 "
            f"({self.cfg.handoff_target1_seconds:.1f}s) -> target2 "
            f"({self.cfg.handoff_target2_seconds:.1f}s) -> hold "
            f"({self.cfg.handoff_hold_seconds:.1f}s) -> policy blend at stand gains "
            f"({self.cfg.handoff_action_blend_seconds:.1f}s) -> gain blend "
            f"({self.cfg.handoff_gain_blend_seconds:.1f}s); termination grace "
            f"{self.cfg.handoff_termination_grace_seconds:.1f}s"
        )

    def _seconds_to_steps(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.step_dt)))

    def _joint_targets(self, values: dict[str, float]) -> torch.Tensor:
        targets = torch.empty(self._leg_action_term.action_dim, device=self.device)
        for index, name in enumerate(self._leg_action_term._joint_names):
            if name in values:
                targets[index] = values[name]
            elif name.endswith("_thigh_joint"):
                targets[index] = values[".*_thigh_joint"]
            elif name.endswith("_calf_joint"):
                targets[index] = values[".*_calf_joint"]
            else:
                raise KeyError(f"No handoff target configured for leg joint {name!r}.")
        return targets

    @staticmethod
    def _smoothstep(alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.clamp(0.0, 1.0)
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _normalized_leg_action(self, target: torch.Tensor) -> torch.Tensor:
        offset = self._leg_action_term._offset
        scale = self._leg_action_term._scale
        if isinstance(offset, torch.Tensor):
            offset = offset[0]
        if isinstance(scale, torch.Tensor):
            scale = scale[0]
        return (target - offset) / scale

    def _set_leg_gains(self, gain_alpha: torch.Tensor) -> None:
        alpha = gain_alpha[:, None]
        self._leg_actuator.stiffness[:] = (
            self.cfg.handoff_stand_kp * (1.0 - alpha) + self.cfg.handoff_policy_kp * alpha
        )
        self._leg_actuator.damping[:] = (
            self.cfg.handoff_stand_kd * (1.0 - alpha) + self.cfg.handoff_policy_kd * alpha
        )

    def _build_handoff_action(self, policy_action: torch.Tensor) -> torch.Tensor:
        step = self.handoff_step_buf
        action = policy_action.clone()
        leg_action = action[:, self._leg_action_start : self._leg_action_end]

        target1_mask = step < self._target1_end
        if target1_mask.any():
            alpha = self._smoothstep(step[target1_mask].float() / self._target1_steps)
            target = torch.lerp(self._prone, self._target1, alpha[:, None])
            leg_action[target1_mask] = self._normalized_leg_action(target)

        target2_mask = torch.logical_and(step >= self._target1_end, step < self._target2_end)
        if target2_mask.any():
            alpha = self._smoothstep(
                (step[target2_mask] - self._target1_end).float() / self._target2_steps
            )
            target = torch.lerp(self._target1, self._target2, alpha[:, None])
            leg_action[target2_mask] = self._normalized_leg_action(target)

        stand_mask = torch.logical_and(step >= self._target2_end, step < self._hold_end)
        if stand_mask.any():
            leg_action[stand_mask] = 0.0

        policy_mask = step >= self._hold_end
        action_alpha = torch.zeros(self.num_envs, device=self.device)
        if policy_mask.any():
            policy_age = step[policy_mask] - self._hold_end
            alpha = self._smoothstep(policy_age.float() / self._action_blend_steps)
            action_alpha[policy_mask] = alpha
            leg_action[policy_mask] *= alpha[:, None]

        gain_alpha = torch.zeros(self.num_envs, device=self.device)
        gain_mask = torch.logical_and(step >= self._policy_blend_end, step < self._gain_blend_end)
        gain_alpha[gain_mask] = self._smoothstep(
            (step[gain_mask] - self._policy_blend_end).float() / self._gain_blend_steps
        )
        gain_alpha[step >= self._gain_blend_end] = 1.0
        self.handoff_action_alpha.copy_(action_alpha)
        self.handoff_gain_alpha.copy_(gain_alpha)
        self._set_leg_gains(gain_alpha)
        return action

    def _phase_name(self, step: int) -> str:
        if step < self._target1_end:
            return "target1"
        if step < self._target2_end:
            return "target2"
        if step < self._hold_end:
            return "hold"
        if step < self._policy_blend_end:
            return "policy_blend"
        if step < self._gain_blend_end:
            return "gain_blend"
        return "policy"

    def step(self, action: torch.Tensor):
        transformed_action = self._build_handoff_action(action.to(self.device))
        reward_active_this_step = self.handoff_step_buf >= self._hold_end

        self.handoff_step_buf += 1
        self.policy_step_buf[:] = torch.clamp(
            self.handoff_step_buf - self._hold_end,
            min=0,
        )
        self.handoff_policy_active[:] = self.handoff_step_buf >= self._termination_enable_step

        if self.cfg.handoff_debug:
            phase = self._phase_name(int(self.handoff_step_buf[0].item()))
            if phase != self._debug_phase:
                self._debug_phase = phase
                root_z = float(self._robot.data.root_pos_w[0, 2].item())
                print(
                    f"[B2RM HANDOFF] env0 phase={phase} "
                    f"t={self.handoff_step_buf[0].item() * self.step_dt:.2f}s root_z={root_z:.3f}"
                )

        obs, reward, terminated, truncated, extras = super().step(transformed_action)
        if self.cfg.handoff_debug and bool((terminated[0] | truncated[0]).item()):
            last_dones = self.termination_manager._last_episode_dones[0]
            reasons = [
                name
                for name, done in zip(self.termination_manager._term_names, last_dones, strict=True)
                if bool(done.item())
            ]
            print(f"[B2RM HANDOFF] env0 reset reasons={reasons}")

        startup_mask = ~reward_active_this_step
        if isinstance(reward, torch.Tensor):
            reward[startup_mask] = 0.0
        elif isinstance(reward, dict):
            for reward_component in reward.values():
                reward_component[startup_mask] = 0.0
        else:
            for reward_component in reward:
                reward_component[startup_mask] = 0.0
        return obs, reward, terminated, truncated, extras

    def _reset_handoff_state(self, env_ids: torch.Tensor) -> None:
        self.handoff_step_buf[env_ids] = 0
        self.policy_step_buf[env_ids] = 0
        self.handoff_policy_active[env_ids] = False
        self.handoff_action_alpha[env_ids] = 0.0
        self.handoff_gain_alpha[env_ids] = 0.0

        root_pose = self._robot.data.root_pose_w[env_ids].clone()
        root_pose[:, 2] = (
            self.scene.env_origins[env_ids, 2] + self.cfg.handoff_initial_root_height
        )
        self._robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device),
            env_ids=env_ids,
        )

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self._leg_action_term._joint_ids] = self._prone
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._set_leg_gains_for_envs(env_ids, self.cfg.handoff_stand_kp, self.cfg.handoff_stand_kd)

    def _set_leg_gains_for_envs(
        self,
        env_ids: torch.Tensor,
        stiffness: float,
        damping: float,
    ) -> None:
        self._leg_actuator.stiffness[env_ids] = stiffness
        self._leg_actuator.damping[env_ids] = damping

    def _reset_idx(self, env_ids: Sequence[int]):
        result = super()._reset_idx(env_ids)
        if self._handoff_ready:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._reset_handoff_state(env_ids_tensor)
        return result
