from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
)

from .instinct_rl_ppo_cfg import B2RMPPOAlgorithmCfg


@configclass
class B2RMVelocityPolicyCfg(InstinctRlMoEActorCriticCfg):
    """The Parkour MoE policy without visual encoders."""

    init_noise_std = 0.4
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"


@configclass
class B2RMVelocityPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "b2rm_velocity"
    resume = False
    load_run = ""
    load_checkpoint = ""
    empirical_normalization = False
    policy = B2RMVelocityPolicyCfg()
    algorithm = B2RMPPOAlgorithmCfg()
