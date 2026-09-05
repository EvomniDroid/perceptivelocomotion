from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.managers import MultiRewardCfg

from .b2rm_parkour_cfg import (
    ARM_FOLDED_OFFSET,
    B2RMActionsCfg,
    B2RMParkourEnvCfg,
    B2RMParkourEnvCfg_PLAY,
    B2RMRewardsCfg,
)


LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]


@configclass
class B2RMVelocityPolicyObsCfg(ObsGroup):
    """Parkour policy observations without the depth image."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10, 10))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-50, 50))
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10, 10))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-20, 20))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    actions = ObsTerm(func=instinct_mdp.last_action)
    gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class B2RMVelocityCriticObsCfg(ObsGroup):
    """Parkour critic observations without the depth image."""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    actions = ObsTerm(func=instinct_mdp.last_action)
    gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class B2RMVelocityObservationsCfg:
    policy: B2RMVelocityPolicyObsCfg = B2RMVelocityPolicyObsCfg()
    critic: B2RMVelocityCriticObsCfg = B2RMVelocityCriticObsCfg()


@configclass
class B2RMLegOnlyVelocityPolicyObsCfg(B2RMVelocityPolicyObsCfg):
    """Velocity observations containing leg joints only."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-10, 10),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-50, 50),
    )


@configclass
class B2RMLegOnlyVelocityCriticObsCfg(B2RMVelocityCriticObsCfg):
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )


@configclass
class B2RMLegOnlyVelocityObservationsCfg:
    policy: B2RMLegOnlyVelocityPolicyObsCfg = B2RMLegOnlyVelocityPolicyObsCfg()
    critic: B2RMLegOnlyVelocityCriticObsCfg = B2RMLegOnlyVelocityCriticObsCfg()


@configclass
class B2RMLegOnlyActionsCfg(B2RMActionsCfg):
    """Twelve leg actions plus a zero-dimensional fixed arm holder."""

    arm_joint_pos = instinct_mdp.FixedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_FOLDED_OFFSET),
        joint_pos=ARM_FOLDED_OFFSET,
        preserve_order=True,
    )


@configclass
class B2RMVelocityRewardsCfg(B2RMRewardsCfg):
    """Parkour rewards with an explicit, observable alternating trot clock."""

    # These contact-conditioned terms accept a permanently fixed diagonal.
    tracking_contacts_shaped_force = None
    tracking_contacts_shaped_vel = None
    # A valid trot intentionally alternates which diagonal has accumulated air time.
    feet_air_time_balance = None

    foot_contact_balance = RewTerm(
        func=mdp.foot_contact_balance,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "max_air_time": 0.55,
        },
    )
    trot_phase_contact = RewTerm(
        func=mdp.trot_phase_contact_reward,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "period": 0.8,
            "force_scale": 120.0,
            "sigma": 0.25,
            "command_threshold": 0.05,
        },
    )
    trot_phase_foot_velocity = RewTerm(
        func=mdp.trot_phase_foot_velocity_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "period": 0.8,
            "min_swing_speed": 0.35,
            "command_threshold": 0.05,
        },
    )


@configclass
class B2RMVelocityRewardsCfgFinal(MultiRewardCfg):
    rewards: B2RMVelocityRewardsCfg = B2RMVelocityRewardsCfg()


@configclass
class B2RMLegOnlyVelocityRewardsCfg(B2RMVelocityRewardsCfg):
    """Locomotion rewards with no arm-dependent learning objective."""

    joint_deviation_arm = None


@configclass
class B2RMLegOnlyVelocityRewardsCfgFinal(MultiRewardCfg):
    rewards: B2RMLegOnlyVelocityRewardsCfg = B2RMLegOnlyVelocityRewardsCfg()


@configclass
class B2RMVelocityEnvCfg(B2RMParkourEnvCfg):
    """B2RM Parkour dynamics and rewards with proprioceptive observations only."""

    observations: B2RMVelocityObservationsCfg = B2RMVelocityObservationsCfg()
    rewards: B2RMVelocityRewardsCfgFinal = B2RMVelocityRewardsCfgFinal()

    def __post_init__(self):
        super().__post_init__()
        self.scene.camera = None
        self.scene.rgb_camera = None


@configclass
class B2RMVelocityEnvCfg_PLAY(B2RMParkourEnvCfg_PLAY):
    observations: B2RMVelocityObservationsCfg = B2RMVelocityObservationsCfg()
    rewards: B2RMVelocityRewardsCfgFinal = B2RMVelocityRewardsCfgFinal()

    def __post_init__(self):
        super().__post_init__()
        self.scene.camera = None
        self.scene.rgb_camera = None
