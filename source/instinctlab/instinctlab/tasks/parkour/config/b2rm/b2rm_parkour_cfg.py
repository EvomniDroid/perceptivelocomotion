import copy
import os
import math

from isaaclab.envs import ViewerCfg, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from instinctlab.managers import MultiRewardCfg
from instinctlab.monitors import FootStatMonitorTerm, MonitorTermCfg
from instinctlab.tasks.parkour.config.parkour_env_cfg import CurriculumCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinctlab.tasks.parkour.mdp as mdp
import instinctlab.envs.mdp as instinct_mdp

from instinctlab.assets.unitree_b2rm import (
    B2RM_CFG,
    b2rm_18dof_delayed_actuators,
)
from instinctlab.sensors import Grid3dPointsGeneratorCfg, VolumePointsCfg, NoisyGroupedRayCasterCameraCfg
from instinctlab.terrains import (
    TerrainImporterCfg,
    TerrainImporter,
)
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_generator import FiledTerrainGenerator
from instinctlab.terrains.shared_terrain_cfg import TRAINING_SUB_TERRAINS
from instinctlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RandomGaussianNoiseCfg,
    RangeBasedGaussianNoiseCfg,
)

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

ARM_FOLDED_OFFSET = {
    "arm_joint_1": 0.0,
    "arm_joint_2": 1.5707963267948966,
    "arm_joint_3": -3.036872898470133,
    "arm_joint_4": 0.0,
    "arm_joint_5": -0.15707963267948966,
    "arm_joint_6": 0.017453292519943295,
}

ROUGH_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    class_type=FiledTerrainGenerator,
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains=TRAINING_SUB_TERRAINS,
)


@configclass
class B2RMSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 2.5

    terrain = TerrainImporterCfg(
        class_type=TerrainImporter,
        prim_path="/World/ground",
        terrain_type="hacked_generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = B2RM_CFG

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )

    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 0.1)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )

    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=[
            "/World/ground",
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.41251, 0.024997, 0.04765),
            rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
            convention="world",
        ),
        min_distance=0.1,
        noise_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )

    rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rgb_camera",
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1e6),
        ),
        width=640,
        height=360,
        offset=CameraCfg.OffsetCfg(
            pos=(0.41251, 0.024997, 0.05765),
            rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
            convention="world",
        ),
        data_types=["rgb", "distance_to_image_plane"],
        update_period=0.02,
    )

    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        points_generator=Grid3dPointsGeneratorCfg(
            x_min=-0.025,
            x_max=0.12,
            x_num=10,
            y_min=-0.03,
            y_max=0.03,
            y_num=5,
            z_min=-0.04,
            z_max=0.0,
            z_num=2,
        ),
        debug_vis=False,
    )


@configclass
class B2RMRewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    heading_error = RewTerm(
        func=mdp.heading_error,
        weight=-1.5,
        params={"command_name": "base_velocity"},
    )
    dont_wait = RewTerm(
        func=mdp.dont_wait,
        weight=-2.0,
        params={"command_name": "base_velocity"},
    )
    must_turn = RewTerm(
        func=mdp.must_turn,
        weight=-2.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.05, "min_turn_rate": 0.05, "target_ratio": 0.6},
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "offset": 0.0, "threshold": 0.05},
    )
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-2.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="leg_volume_points",
                body_names=[".*_foot"],
            ),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "vel_threshold": 0.15,
        },
    )
    # 新增：强制 4 足都参与 - 任何一只脚腾空超 1.0s 直接扣分（防"三条腿走路"局部最优）
    # 1.0s 阈值能覆盖：爬台子 (~0.6s)、跨越 gap (~0.8s)、复杂地形 stance 调整 (~1.0s)
    # 配合 feet_air_time 的 asymmetry 软约束（max > 1.5 * mean 扣分）做两层防御
    foot_contact_balance = RewTerm(
        func=mdp.foot_contact_balance,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_air_time": 1.0,
        },
    )
    feet_air_time_balance = RewTerm(
        func=mdp.feet_air_time_balance,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "vel_threshold": 0.15,
        },
    )
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    feet_close_xy_gauss = RewTerm(
        func=mdp.feet_close_xy_gauss,
        weight=0.0,
        params={
            "threshold": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "std": 0.05,
        },
    )
    joint_deviation_arm = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["arm_joint_1", "arm_joint_2", "arm_joint_3",
                             "arm_joint_4", "arm_joint_5", "arm_joint_6"],
            )
        },
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    roll_l2 = RewTerm(func=mdp.roll_l2, weight=-2.0)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            )
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-7.5e-07,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
            )
        },
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
            )
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_pitch_l2 = RewTerm(func=mdp.base_pitch_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-4.0,
        params={"target_height": 0.55},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            )
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            )
        },
    )
    # ===== HYT 新增项 =====
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_height": 0.3,
        },
    )
    feet_height_balance = RewTerm(
        func=mdp.feet_height_balance,
        weight=-4.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_height": 0.36,
        },
    )
    work_l2 = RewTerm(
        func=mdp.work_l2,
        weight=-0.003,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            )
        },
    )
    delta_torques = RewTerm(
        func=mdp.delta_torques,
        weight=-1.0e-07,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            )
        },
    )
    feet_jerk = RewTerm(
        func=mdp.feet_jerk,
        weight=-0.0002,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    contact_forces_penalty = RewTerm(
        func=mdp.contact_forces_penalty,
        weight=-0.001,
        params={
            "threshold": 120.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    tracking_contacts_shaped_force = RewTerm(
        func=mdp.tracking_contacts_shaped_force,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "sigma": 0.5,
            "kappa": 0.07,
        },
    )
    tracking_contacts_shaped_vel = RewTerm(
        func=mdp.tracking_contacts_shaped_vel,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "sigma": 0.5,
        },
    )
    walking_dof = RewTerm(
        func=mdp.walking_dof,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            ),
        },
    )


@configclass
class B2RMPolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10, 10))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-50, 50))
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-10, 10))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-20, 20))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    actions = ObsTerm(func=instinct_mdp.last_action)
    depth_image = ObsTerm(
        func=mdp.delayed_visualizable_image,
        params={
            "data_type": "distance_to_image_plane_noised_history",
            "sensor_cfg": SceneEntityCfg("camera"),
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "debug_vis": False,
        },
        noise=None,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class B2RMCriticObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    actions = ObsTerm(func=instinct_mdp.last_action)
    depth_image = ObsTerm(
        func=mdp.delayed_visualizable_image,
        params={
            "data_type": "distance_to_image_plane_noised_history",
            "sensor_cfg": SceneEntityCfg("camera"),
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "debug_vis": False,
        },
        noise=None,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class B2RMObservationsCfg:
    policy: B2RMPolicyObsCfg = B2RMPolicyObsCfg()
    critic: B2RMCriticObsCfg = B2RMCriticObsCfg()


@configclass
class B2RMCommandsCfg:
    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        only_positive_lin_vel_x=True,
        rel_standing_envs=0.0,
        ranges=mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0)
        ),
        random_velocity_terrain=["perlin_rough_stand", "boxes", "mesh_boxes"],
        random_ang_vel_threshold=0.0,
        velocity_ranges={
            "perlin_rough": {"lin_vel_x": (0.0, 0.6), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.8, 0.8)},
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "square_gaps": {"lin_vel_x": (0.15, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
            "pyramid_stairs": {"lin_vel_x": (0.12, 0.70), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.08, 0.08)},
            "pyramid_stairs_high": {"lin_vel_x": (0.0, 0.60), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.06, 0.06)},
            "pyramid_stairs_inv": {"lin_vel_x": (0.12, 0.70), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.08, 0.08)},
            "pyramid_stairs_inv_high": {"lin_vel_x": (0.0, 0.60), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.06, 0.06)},
            "boxes": {"lin_vel_x": (0.15, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "mesh_boxes": {"lin_vel_x": (0.15, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "hf_pyramid_slope_inv": {"lin_vel_x": (0.0, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
            "raised_mound": {"lin_vel_x": (0.0, 0.9), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.15, 0.15)},
            "pit_crater": {"lin_vel_x": (0.0, 0.9), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.15, 0.15)},
            "wave": {"lin_vel_x": (0.0, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.6, 0.6)},
        },
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
    )


@configclass
class B2RMTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_bound = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.25})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 1.0,
        },
    )
    leg_link_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh"]),
            "threshold": 15.0,
        },
    )
    # calf_link_contact 实质就是"膝关节触地"检测：
    # 物理引擎只在 rigid body (link) 层面追踪接触力，calf_joint 本身没有碰撞体。
    # calf 顶端（靠近 thigh 一侧）就是"膝盖"位置，所以 calf_link 触地 = 膝盖触地。
    # 阈值 50N：正常走路时 calf 力 < 10N，跌倒时 calf 砸地 > 50N。
    calf_link_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_calf"]),
            "threshold": 50.0,
        },
    )
    # base_link_illegal_contact 同样关闭：阈值 1N 极严，平地训练时可能因 pose 微小偏差误杀。
    # base_link_illegal_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
    #         "threshold": 1.0,
    #     },
    # )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.3})
    # bad_pitch 暂时注释：因为 calf/foot/base 触地 termination 已经能在机器人跌倒时立即 reset，
    # bad_pitch 会让策略在学习"低头"时过早终止，反而拖慢收敛。
    # bad_pitch = DoneTerm(func=mdp.bad_pitch, params={"max_pitch": 0.35})


@configclass
class B2RMRewardsCfgFinal(MultiRewardCfg):
    rewards: B2RMRewardsCfg = B2RMRewardsCfg()


@configclass
class B2RMActionsCfg:
    # Keep locomotion authority on legs.
    leg_joint_pos = instinct_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.4,
        use_default_offset=True,
    )
    # Keep arm in action space for tracking, but tightly around folded pose.
    arm_joint_pos = instinct_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6"],
        scale=0.05,
        offset=ARM_FOLDED_OFFSET,
        use_default_offset=False,
    )


@configclass
class B2RMMonitorsCfg:
    foot_stats = MonitorTermCfg(
        func=FootStatMonitorTerm,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "foot_names": ["FL", "FR", "RL", "RR"],
        },
    )


@configclass
class B2RMEventsCfg:
    # Keep mild base randomization at reset.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # Randomize only leg joints; keep arm out of random reset.
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Force arm to default folded pose on every reset.
    reset_arm_joints_folded = EventTerm(
        func=mdp.reset_joints_to_targets,
        mode="reset",
        params={
            "joint_pos_targets": ARM_FOLDED_OFFSET,
            "joint_vel_target": 0.0,
        },
    )


@configclass
class B2RMParkourEnvCfg(ManagerBasedRLEnvCfg):
    scene: B2RMSceneCfg = B2RMSceneCfg()
    observations: B2RMObservationsCfg = B2RMObservationsCfg()
    actions: B2RMActionsCfg = B2RMActionsCfg()
    commands: B2RMCommandsCfg = B2RMCommandsCfg()
    rewards: B2RMRewardsCfgFinal = B2RMRewardsCfgFinal()
    terminations: B2RMTerminationsCfg = B2RMTerminationsCfg()
    monitors: B2RMMonitorsCfg = B2RMMonitorsCfg()
    events: B2RMEventsCfg = B2RMEventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.sim.physx.max_depenetration_velocity = 1.0
        self.sim.physx.default_buffered_penetration_count = 0

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.scene.robot = B2RM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = b2rm_18dof_delayed_actuators


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]
    sub_terrain_cfg.proportion = 1.0  # 强制每种地形至少一列


@configclass
class B2RMParkourEnvCfg_PLAY(B2RMParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # play 只用于人工巡检，不需要训练期的 terrain curriculum 在 reset 时再改行号。
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # 等比例行×列 = 16 列（每种地形至少一列） × 1 行
        n_terrains = len(ROUGH_TERRAINS_CFG_PLAY.sub_terrains)
        self.scene.num_envs = n_terrains
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.5],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.base_height = None
        self.terminations.base_contact = None
        if self.scene.terrain.terrain_generator is not None:
            # 10 行 × N 列：每列固定一种地形；play 时按行固定难度，便于人工巡检。
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = n_terrains
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.deterministic_curriculum_rows = True
            # 单 env 出生在 row 0（最简单那一行），方便键盘巡检
            self.scene.terrain.max_init_terrain_level = 0

        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges = mdp.PoseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5)
        )
