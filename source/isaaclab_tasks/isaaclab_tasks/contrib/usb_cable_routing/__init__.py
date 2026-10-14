# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e routing of a pre-grasped USB cable through open guides."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Route-USB-UR10e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UsbRoutingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:UsbRoutingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
