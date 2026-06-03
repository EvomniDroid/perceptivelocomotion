from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [4]
    kernel_sizes = [3]
    strides = [1]
    hidden_sizes = [256, 256]
    paddings = [1]
    nonlinearity = "ReLU"
    use_maxpool = True
    component_names = [
        "depth_image",
    ]


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConv2dCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 0.4
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class B2RMPPOAlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 0.5
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.003
    num_learning_epochs = 3
    num_mini_batches = 8
    learning_rate = 7.5e-5
    schedule = "adaptive"
    gamma = 0.995
    lam = 0.97
    desired_kl = 0.008
    max_grad_norm = 0.5


@configclass
class B2RMParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "b2rm_parkour"
    resume = False
    load_run = ""
    load_checkpoint = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = B2RMPPOAlgorithmCfg()
