from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp

from .b2rm_velocity_cfg import (
    B2RMVelocityCriticObsCfg,
    B2RMVelocityEnvCfg,
    B2RMVelocityEnvCfg_PLAY,
    B2RMVelocityPolicyObsCfg,
)


def _gate_termination(
    term_cfg,
    term_name: str,
    minimum_duration_s: float = 0.0,
) -> None:
    wrapped_func = term_cfg.func
    wrapped_params = dict(term_cfg.params)
    term_cfg.func = mdp.handoff_gated_termination
    term_cfg.params = {
        "wrapped_func": wrapped_func,
        "wrapped_params": wrapped_params,
        "counter_name": term_name,
        "minimum_duration_s": minimum_duration_s,
    }


class B2RMVelocityHandoffCfgMixin:
    handoff_initial_root_height: float = 0.22
    handoff_target1_seconds: float = 0.8
    handoff_target2_seconds: float = 0.8
    handoff_hold_seconds: float = 0.4
    handoff_gain_blend_seconds: float = 0.0
    handoff_action_blend_seconds: float = 0.5
    handoff_termination_grace_seconds: float = 1.0
    handoff_stand_kp: float = 250.0
    handoff_stand_kd: float = 5.0
    handoff_policy_kp: float = 250.0
    handoff_policy_kd: float = 5.0
    handoff_policy_action_clip: float = 1.5
    handoff_debug: bool = False

    def _configure_handoff(self) -> None:
        # Preserve 20 seconds of policy-controlled time after the scripted startup.
        self.episode_length_s += (
            self.handoff_target1_seconds
            + self.handoff_target2_seconds
            + self.handoff_hold_seconds
            + self.handoff_gain_blend_seconds
            + self.handoff_action_blend_seconds
        )
        self.events.base_external_force_torque = None
        # Handoff transients can briefly touch merged shell, thigh, or calf collision
        # bodies. Keep these contacts as reward penalties instead of ending nearly
        # every rollout before the policy can recover. Height and orientation still
        # reset genuinely fallen robots.
        self.terminations.base_contact = None
        self.terminations.leg_link_contact = None
        self.terminations.calf_link_contact = None
        for name in (
            "time_out",
            "terrain_out_bound",
            "root_height",
            "bad_orientation",
        ):
            term_cfg = getattr(self.terminations, name, None)
            if term_cfg is not None:
                _gate_termination(term_cfg, name)


@configclass
class B2RMVelocityHandoffPolicyObsCfg(B2RMVelocityPolicyObsCfg):
    handoff_blend = ObsTerm(func=mdp.handoff_blend_state)


@configclass
class B2RMVelocityHandoffCriticObsCfg(B2RMVelocityCriticObsCfg):
    handoff_blend = ObsTerm(func=mdp.handoff_blend_state)


@configclass
class B2RMVelocityHandoffObservationsCfg:
    policy: ObsGroup = B2RMVelocityHandoffPolicyObsCfg()
    critic: ObsGroup = B2RMVelocityHandoffCriticObsCfg()


@configclass
class B2RMVelocityHandoffEnvCfg(B2RMVelocityHandoffCfgMixin, B2RMVelocityEnvCfg):
    observations: B2RMVelocityHandoffObservationsCfg = B2RMVelocityHandoffObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_handoff()


@configclass
class B2RMVelocityHandoffEnvCfg_PLAY(
    B2RMVelocityHandoffCfgMixin,
    B2RMVelocityEnvCfg_PLAY,
):
    observations: B2RMVelocityHandoffObservationsCfg = B2RMVelocityHandoffObservationsCfg()
    handoff_debug: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._configure_handoff()


@configclass
class B2RMVelocityHandoffStandEnvCfg(B2RMVelocityHandoffEnvCfg):
    """Stage-1 handoff task: zero command on flat ground, no gait demand."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.only_positive_lin_vel_x = False
        command.ranges = mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        command.velocity_ranges = {}
        command.random_velocity_terrain = []


@configclass
class B2RMVelocityHandoffStandEnvCfg_PLAY(B2RMVelocityHandoffStandEnvCfg):
    handoff_debug: bool = True
