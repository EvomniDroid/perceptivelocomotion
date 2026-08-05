from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.managers import MultiRewardCfg
from instinctlab.terrains.shared_terrain_cfg import FLAT_TRAINING_SUB_TERRAINS
from instinctlab.tasks.parkour.config.parkour_env_cfg import CurriculumCfg

from .b2rm_velocity_cfg import (
    B2RMLegOnlyActionsCfg,
    B2RMLegOnlyVelocityRewardsCfg,
    B2RMVelocityEnvCfg,
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

B2RM_GAIT_ACTIVATION_START = 0.02
B2RM_GAIT_ACTIVATION_FULL = 0.15


@configclass
class B2RMFilteredLegOnlyActionsCfg(B2RMLegOnlyActionsCfg):
    """Leg-only action path with the same low-pass filter used at deployment."""

    leg_joint_pos = instinct_mdp.FilteredJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.3,
        use_default_offset=True,
        filter_alpha=0.8,
    )


@configclass
class B2RMLegOnlyVelocityHistoryPolicyObsCfg(ObsGroup):
    """Eight-frame deployable history using only real-robot observables."""

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
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class B2RMLegOnlyVelocityHistoryCriticObsCfg(B2RMLegOnlyVelocityHistoryPolicyObsCfg):
    """Noise-free critic with simulation-only privileged state."""

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
    # Contact remains available to the critic and rewards during training, but
    # is deliberately excluded from the deployable actor observation. The B2RM
    # low-state force fields did not provide a separable loaded/unloaded signal.
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
class B2RMLegOnlyVelocityHistoryRewardTermsCfg(B2RMLegOnlyVelocityRewardsCfg):
    """Velocity rewards augmented for quiet zero-command deployment."""

    # Let the policy discover its own gait from state history. These inherited
    # terms encode a fixed diagonal-trot clock and would reintroduce phase
    # supervision even though phase is no longer part of the observation.
    trot_phase_contact = None
    trot_phase_foot_velocity = None

    # Prevent the policy from improving its apparent return by deliberately
    # touching a calf/base and ending a difficult episode early. Timeouts are
    # truncations, so successful full-length episodes are not penalized.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # The inherited 0.5 kernels are too broad for this task's +/-0.2 command
    # range: a visibly wrong velocity still receives almost full reward.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.15},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.20},
    )
    dont_wait = RewTerm(
        func=mdp.velocity_command_deficit,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "target_ratio": 0.75,
        },
    )
    zero_command_action = RewTerm(
        func=mdp.zero_command_action_l2,
        weight=-0.08,
        params={"command_name": "base_velocity", "command_threshold": B2RM_GAIT_ACTIVATION_START},
    )
    zero_command_motion = RewTerm(
        func=mdp.zero_command_base_motion_l2,
        weight=-2.0,
        params={"command_name": "base_velocity", "command_threshold": B2RM_GAIT_ACTIVATION_START},
    )
    zero_command_contacts = RewTerm(
        func=mdp.zero_command_feet_contact_deficit,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "force_threshold": 5.0,
            "command_threshold": B2RM_GAIT_ACTIVATION_START,
        },
    )
    action_saturation = RewTerm(
        func=mdp.action_saturation_l2,
        weight=-0.50,
        params={"soft_limit": 0.7},
    )


@configclass
class B2RMLegOnlyVelocityHistoryRewardsCfg(MultiRewardCfg):
    rewards: B2RMLegOnlyVelocityHistoryRewardTermsCfg = B2RMLegOnlyVelocityHistoryRewardTermsCfg()


@configclass
class B2RMVelocityHistoryCurriculumCfg(CurriculumCfg):
    """Flat-ground command curriculum for the deployable velocity policy."""

    terrain_levels = None
    velocity_command_levels = CurrTerm(
        func=mdp.velocity_command_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "reward_group_name": "rewards",
            "success_ratio": 0.8,
            "lin_x_step": 0.1,
            "lin_y_step": 0.1,
            "yaw_step": 0.1,
        },
    )


@configclass
class B2RMLegOnlyVelocityHistoryEnvCfg(B2RMVelocityEnvCfg):
    """Deployable target2-start velocity control with eight-frame history."""

    observations: B2RMLegOnlyVelocityHistoryObservationsCfg = B2RMLegOnlyVelocityHistoryObservationsCfg()
    actions: B2RMFilteredLegOnlyActionsCfg = B2RMFilteredLegOnlyActionsCfg()
    rewards: B2RMLegOnlyVelocityHistoryRewardsCfg = B2RMLegOnlyVelocityHistoryRewardsCfg()
    events: B2RMLegOnlyVelocityHistoryEventsCfg = B2RMLegOnlyVelocityHistoryEventsCfg()
    curriculum: B2RMVelocityHistoryCurriculumCfg = B2RMVelocityHistoryCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the stable SDK2/deployment handoff pose used before policy takeover.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.58)
        self.scene.robot.init_state.joint_pos.update(B2RM_TARGET2_LEG_POS)
        self.scene.terrain.terrain_generator.sub_terrains = copy.deepcopy(FLAT_TRAINING_SUB_TERRAINS)
        self.actions.leg_joint_pos.scale = 0.3
        # IsaacLab clips processed joint targets after applying scale/offset.
        # These limits are exactly target2 + 0.3 * [-1.0, 1.0], matching the
        # real adapter's raw-action clip instead of clipping every target to
        # the unrelated absolute range [-1.5, 1.5].
        self.actions.leg_joint_pos.clip = {
            ".*_hip_joint": (-0.30, 0.30),
            ".*_thigh_joint": (0.37, 0.97),
            ".*_calf_joint": (-1.60, -1.00),
        }
        self.rewards.rewards.feet_air_time.params.update(
            vel_threshold=B2RM_GAIT_ACTIVATION_START,
            activation_full=B2RM_GAIT_ACTIVATION_FULL,
        )
        self.rewards.rewards.feet_height.params.update(
            target_height=0.08,
            minimum_target_height=0.03,
            vel_threshold=B2RM_GAIT_ACTIVATION_START,
            activation_full=B2RM_GAIT_ACTIVATION_FULL,
        )
        self.rewards.rewards.action_rate_l2.weight = -0.03
        self.rewards.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.rewards.flat_orientation_l2.weight = -3.5
        self.rewards.rewards.roll_l2.weight = -2.5
        self.commands.base_velocity = mdp.B2RMVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.10,
            rel_forward_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.10, 0.10),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.20, 0.20),
            ),
            limit_ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.00, 1.00),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-0.80, 0.80),
            ),
        )


@configclass
class B2RMLegOnlyVelocityHistoryEnvCfg_PLAY(B2RMLegOnlyVelocityHistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.debug_vis = True
        # Keep evaluation deterministic while preserving reset events. Removing
        # the entire event config also removes terrain-relative base/joint
        # resets, causing every post-contact reset to respawn inside the ground.
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.scale_body_mass = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_leg_joints.params["position_range"] = (0.0, 0.0)
