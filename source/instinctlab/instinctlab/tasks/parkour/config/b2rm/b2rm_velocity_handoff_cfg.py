from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.terrains.shared_terrain_cfg import FLAT_TRAINING_SUB_TERRAINS

from .b2rm_velocity_cfg import (
    B2RMLegOnlyActionsCfg,
    B2RMLegOnlyVelocityObservationsCfg,
    B2RMLegOnlyVelocityRewardsCfgFinal,
    B2RMVelocityCriticObsCfg,
    B2RMVelocityEnvCfg,
    B2RMVelocityEnvCfg_PLAY,
    B2RMVelocityPolicyObsCfg,
)
from .b2rm_parkour_cfg import B2RMEventsCfg


B2RM_TARGET2_LEG_POS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.67,
    "FR_thigh_joint": 0.67,
    "RL_thigh_joint": 0.67,
    "RR_thigh_joint": 0.67,
    "FL_calf_joint": -1.30,
    "FR_calf_joint": -1.30,
    "RL_calf_joint": -1.30,
    "RR_calf_joint": -1.30,
}

B2RM_HISTORY_GAIT_PERIOD = 0.7


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
    handoff_stand_root_height: float = 0.58
    handoff_start_from_stand: bool = False
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
        scripted_seconds = 0.0
        if not self.handoff_start_from_stand:
            scripted_seconds = (
                self.handoff_target1_seconds
                + self.handoff_target2_seconds
                + self.handoff_hold_seconds
            )
        self.episode_length_s += scripted_seconds + max(
            self.handoff_gain_blend_seconds,
            self.handoff_action_blend_seconds,
        )
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # The B2RM model uses merged collision shells that can report transient
        # base/thigh contacts during an otherwise valid takeover. Rear-calf
        # contact, however, directly captures the persistent rear-knee kneeling
        # failure without rejecting a brief scrape while recovering.
        self.terminations.base_contact = None
        self.terminations.leg_link_contact = None
        rear_calf_contact = self.terminations.calf_link_contact
        rear_calf_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["RL_calf", "RR_calf"],
            preserve_order=True,
        )
        _gate_termination(
            rear_calf_contact,
            "rear_calf_contact",
            minimum_duration_s=0.30,
        )
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
    def __post_init__(self):
        super().__post_init__()
        self._configure_handoff()


@configclass
class B2RMVelocityHandoffEnvCfg_PLAY(
    B2RMVelocityHandoffCfgMixin,
    B2RMVelocityEnvCfg_PLAY,
):
    handoff_debug: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._configure_handoff()


@configclass
class B2RMVelocityHandoffStandEnvCfg(B2RMVelocityHandoffEnvCfg):
    """Stage 1: learn the instantaneous 1000/10 to 250/5 stand takeover."""

    handoff_stand_kp: float = 1000.0
    handoff_stand_kd: float = 10.0
    handoff_policy_kp: float = 250.0
    handoff_policy_kd: float = 5.0
    handoff_gain_blend_seconds: float = 0.0
    handoff_action_blend_seconds: float = 0.0
    handoff_termination_grace_seconds: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # Keep the pretrained velocity policy's action-to-joint mapping exactly
        # unchanged while fine-tuning the instantaneous gain takeover.
        self.actions.leg_joint_pos.scale = 0.4
        self.scene.terrain.terrain_generator.sub_terrains = copy.deepcopy(
            FLAT_TRAINING_SUB_TERRAINS
        )
        contact_balance = self.rewards.rewards.foot_contact_balance
        contact_balance.func = mdp.feet_contact_deficit
        contact_balance.weight = -5.0
        contact_balance.params = {
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "force_threshold": 5.0,
        }
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


@configclass
class B2RMVelocityStandEnvCfg(B2RMVelocityHandoffStandEnvCfg):
    """Pretrain zero-command policy standing directly from the target2 pose."""

    handoff_start_from_stand: bool = True
    handoff_stand_kp: float = 250.0
    handoff_stand_kd: float = 5.0
    handoff_policy_kp: float = 250.0
    handoff_policy_kd: float = 5.0
    handoff_action_blend_seconds: float = 0.0
    # The policy controls and receives reward immediately; only hard fall
    # termination is delayed briefly to allow initial contact settling.
    handoff_termination_grace_seconds: float = 0.5


@configclass
class B2RMVelocityStandEnvCfg_PLAY(B2RMVelocityStandEnvCfg):
    handoff_debug: bool = True


@configclass
class B2RMVelocityHandoffWalkEnvCfg(B2RMVelocityEnvCfg):
    """Historical velocity baseline with Kp=250 and 0-0.5 m/s commands."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = copy.deepcopy(
            FLAT_TRAINING_SUB_TERRAINS
        )
        # Match the action authority used by the previously converged
        # proprioceptive velocity baseline (commit b8329be).
        self.actions.leg_joint_pos.scale = 0.4
        command = self.commands.base_velocity
        command.only_positive_lin_vel_x = True
        command.ranges = mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        command.velocity_ranges = {}
        command.random_velocity_terrain = []


@configclass
class B2RMVelocityHandoffWalkEnvCfg_PLAY(B2RMVelocityHandoffWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.debug_vis = True


@configclass
class B2RMLegOnlyVelocityHandoffWalkEnvCfg(B2RMVelocityHandoffWalkEnvCfg):
    """Deprecated 50-D velocity baseline retained for old checkpoints."""

    observations: B2RMLegOnlyVelocityObservationsCfg = B2RMLegOnlyVelocityObservationsCfg()
    actions: B2RMLegOnlyActionsCfg = B2RMLegOnlyActionsCfg()
    rewards: B2RMLegOnlyVelocityRewardsCfgFinal = B2RMLegOnlyVelocityRewardsCfgFinal()


@configclass
class B2RMLegOnlyVelocityHandoffWalkEnvCfg_PLAY(B2RMLegOnlyVelocityHandoffWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.debug_vis = True


@configclass
class B2RMLegOnlyVelocityHistoryPolicyObsCfg(ObsGroup):
    """Eight-frame leg-only history for generic velocity control."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-10, 10),
        history_length=8,
        flatten_history_dim=True,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-50, 50),
        history_length=8,
        flatten_history_dim=True,
    )
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-10, 10),
        history_length=8,
        flatten_history_dim=True,
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-20, 20),
        history_length=8,
        flatten_history_dim=True,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=8,
        flatten_history_dim=True,
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        history_length=8,
        flatten_history_dim=True,
    )
    actions = ObsTerm(func=instinct_mdp.last_action, history_length=8, flatten_history_dim=True)
    gait_phase = ObsTerm(
        func=mdp.gait_phase,
        params={"period": B2RM_HISTORY_GAIT_PERIOD},
        history_length=8,
        flatten_history_dim=True,
    )
    foot_contacts = ObsTerm(
        func=mdp.foot_contacts,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "force_threshold": 20.0,
        },
        history_length=8,
        flatten_history_dim=True,
    )
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class B2RMLegOnlyVelocityHistoryCriticObsCfg(B2RMLegOnlyVelocityHistoryPolicyObsCfg):
    """Noise-free counterpart of the actor history."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        history_length=8,
        flatten_history_dim=True,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        history_length=8,
        flatten_history_dim=True,
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=8, flatten_history_dim=True)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=8, flatten_history_dim=True)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=8, flatten_history_dim=True)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class B2RMLegOnlyVelocityHistoryObservationsCfg:
    policy: B2RMLegOnlyVelocityHistoryPolicyObsCfg = B2RMLegOnlyVelocityHistoryPolicyObsCfg()
    critic: B2RMLegOnlyVelocityHistoryCriticObsCfg = B2RMLegOnlyVelocityHistoryCriticObsCfg()


@configclass
class B2RMLegOnlyVelocityHistoryEventsCfg(B2RMEventsCfg):
    """Moderate sim-to-real randomization for the generic velocity policy."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    scale_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.015, 0.015),
            },
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class B2RMLegOnlyVelocityHistoryEnvCfg(B2RMVelocityEnvCfg):
    """Standing-start generic velocity control with eight-frame history."""

    observations: B2RMLegOnlyVelocityHistoryObservationsCfg = B2RMLegOnlyVelocityHistoryObservationsCfg()
    actions: B2RMLegOnlyActionsCfg = B2RMLegOnlyActionsCfg()
    rewards: B2RMLegOnlyVelocityRewardsCfgFinal = B2RMLegOnlyVelocityRewardsCfgFinal()
    events: B2RMLegOnlyVelocityHistoryEventsCfg = B2RMLegOnlyVelocityHistoryEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the stable SDK2/deployment handoff pose used before policy takeover.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.58)
        self.scene.robot.init_state.joint_pos.update(B2RM_TARGET2_LEG_POS)
        self.scene.terrain.terrain_generator.sub_terrains = copy.deepcopy(FLAT_TRAINING_SUB_TERRAINS)
        # UniformVelocityCommand does not expose the Parkour target-command
        # tracking metrics consumed by the inherited terrain curriculum.
        self.curriculum.terrain_levels = None
        self.actions.leg_joint_pos.scale = 0.4
        self.actions.leg_joint_pos.clip = {".*": (-1.5, 1.5)}
        self.rewards.rewards.trot_phase_contact.params["period"] = B2RM_HISTORY_GAIT_PERIOD
        self.rewards.rewards.trot_phase_foot_velocity.params["period"] = B2RM_HISTORY_GAIT_PERIOD
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.125,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.5, 0.5),
            ),
        )


@configclass
class B2RMLegOnlyVelocityHistoryEnvCfg_PLAY(B2RMLegOnlyVelocityHistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.debug_vis = True


@configclass
class B2RMLegOnlyVelocityHandoffStandEnvCfg(B2RMVelocityHandoffStandEnvCfg):
    """Instantaneous stand handoff with a 50-D observation and 12-D action."""

    observations: B2RMLegOnlyVelocityObservationsCfg = B2RMLegOnlyVelocityObservationsCfg()
    actions: B2RMLegOnlyActionsCfg = B2RMLegOnlyActionsCfg()
    rewards: B2RMLegOnlyVelocityRewardsCfgFinal = B2RMLegOnlyVelocityRewardsCfgFinal()


@configclass
class B2RMLegOnlyVelocityHandoffStandEnvCfg_PLAY(B2RMLegOnlyVelocityHandoffStandEnvCfg):
    handoff_debug: bool = True
