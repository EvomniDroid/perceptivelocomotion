from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    EstimatorCfgMixin,
    InstinctRlMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
)

from .instinct_rl_ppo_cfg import B2RMPPOAlgorithmCfg


@configclass
class B2RMVelocityPolicyCfg(EstimatorCfgMixin, InstinctRlMoEActorCriticCfg):
    """MoE policy with a BasicLoco-style concurrent velocity estimator."""

    class_name = (
        "instinctlab.tasks.parkour.config.b2rm.agents.velocity_estimator_moe:"
        "B2RMEstimatorMoEActorCritic"
    )
    init_noise_std = 0.4
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    estimator_obs_components = [
        "joint_pos",
        "joint_vel",
        "base_lin_acc",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "actions",
    ]
    estimator_target_components = ["base_lin_vel"]
    estimator_configs = {
        "hidden_sizes": [128, 128, 128],
        "nonlinearity": "ReLU",
    }
    replace_state_prob = 1.0


@configclass
class B2RMVelocityEstimatorPPOAlgorithmCfg(B2RMPPOAlgorithmCfg):
    class_name = "EstimatorPPO"


@configclass
class B2RMVelocityPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 100
    experiment_name = "b2rm_velocity"
    resume = False
    load_run = ""
    load_checkpoint = ""
    empirical_normalization = False
    policy = B2RMVelocityPolicyCfg()
    algorithm = B2RMVelocityEstimatorPPOAlgorithmCfg()
