import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.b2rm"


gym.register(
    id="Instinct-Parkour-Target-Amp-B2RM-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2rm_parkour_cfg:B2RMParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:B2RMParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-B2RM-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2rm_parkour_cfg:B2RMParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:B2RMParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2rm_velocity_cfg:B2RMVelocityEnvCfg",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2rm_velocity_cfg:B2RMVelocityEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Stand-v0",
    entry_point="instinctlab.envs:B2RMHandoffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityStandEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Stand-Play-v0",
    entry_point="instinctlab.envs:B2RMHandoffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityStandEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Handoff-Walk-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityHandoffWalkEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Handoff-Walk-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityHandoffWalkEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Handoff-Stand-v0",
    entry_point="instinctlab.envs:B2RMHandoffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityHandoffStandEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-Velocity-Handoff-Stand-Play-v0",
    entry_point="instinctlab.envs:B2RMHandoffRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMVelocityHandoffStandEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)
