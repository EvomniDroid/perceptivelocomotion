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
    id="Instinct-Parkour-Target-Amp-B2RM-ActionScale04-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2rm_parkour_cfg:B2RMParkourActionScale04EnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:B2RMParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-B2RM-LegOnly-Velocity-History-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMLegOnlyVelocityHistoryEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-B2RM-LegOnly-Velocity-History-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.b2rm_velocity_handoff_cfg:B2RMLegOnlyVelocityHistoryEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_velocity_ppo_cfg:B2RMVelocityPPORunnerCfg"
        ),
    },
)
