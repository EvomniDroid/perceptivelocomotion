"""
Customized Unitree B2 asset for Isaac Sim
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

B2_12DOF_LINKS = [
    "base_link",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
]

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

b2_12dof_actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
        ],
        effort_limit_sim={
            ".*_hip_joint": 200.0,
            ".*_thigh_joint": 200.0,
            ".*_calf_joint": 320.0,
        },
        velocity_limit_sim={
            ".*_hip_joint": 23.0,
            ".*_thigh_joint": 23.0,
            ".*_calf_joint": 14.0,
        },
        stiffness={
            ".*_hip_joint": STIFFNESS_7520_22,
            ".*_thigh_joint": STIFFNESS_7520_14,
            ".*_calf_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_joint": DAMPING_7520_22,
            ".*_thigh_joint": DAMPING_7520_14,
            ".*_calf_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_joint": ARMATURE_7520_22,
            ".*_thigh_joint": ARMATURE_7520_14,
            ".*_calf_joint": ARMATURE_7520_22,
        },
    ),
}

b2_12dof_delayed_actuators = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
        ],
        effort_limit={
            ".*_hip_joint": 200.0,
            ".*_thigh_joint": 200.0,
            ".*_calf_joint": 320.0,
        },
        velocity_limit={
            ".*_hip_joint": 23.0,
            ".*_thigh_joint": 23.0,
            ".*_calf_joint": 14.0,
        },
        stiffness={
            ".*_hip_joint": STIFFNESS_7520_22,
            ".*_thigh_joint": STIFFNESS_7520_14,
            ".*_calf_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_joint": DAMPING_7520_22,
            ".*_thigh_joint": DAMPING_7520_14,
            ".*_calf_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_joint": ARMATURE_7520_22,
            ".*_thigh_joint": ARMATURE_7520_14,
            ".*_calf_joint": ARMATURE_7520_22,
        },
        min_delay=0,
        max_delay=2,
    ),
}

B2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{__file_dir__}/resources/unitree_b2/urdf/b2rm.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=b2_12dof_actuators,
)
