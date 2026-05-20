# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.b2"


gym.register(
    id="Instinct-Parkour-Target-Amp-B2-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2_parkour_cfg:B2ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:B2ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-B2-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.b2_parkour_cfg:B2ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:B2ParkourPPORunnerCfg",
    },
)
