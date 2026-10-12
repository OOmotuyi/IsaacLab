# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck training tasks using Newton MJWarp."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-MicroDuck-Velocity-Flat",
    entry_point=f"{__name__}.microduck_env:MicroDuckEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.microduck_env_cfg:MicroDuckFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroDuckPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


for task, config in (("Stand", "MicroDuckStandEnvCfg"), ("Ground-Pick", "MicroDuckPickupEnvCfg")):
    gym.register(
        id=f"IsaacContrib-MicroDuck-{task}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.skill_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroDuckSkillPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )


for task, config in (("Stand", "MicroDuckStandScratchEnvCfg"), ("Ground-Pick", "MicroDuckPickupScratchEnvCfg")):
    gym.register(
        id=f"IsaacContrib-MicroDuck-{task}-Scratch",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.skill_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroDuckScratchPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )


gym.register(
    id="IsaacContrib-MicroDuck-Marker-Grasp-Scratch",
    entry_point=f"{__name__}.microduck_env:MicroDuckGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:MicroDuckMarkerGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:MicroDuckGraspPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
