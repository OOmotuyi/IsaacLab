# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A separate UR10e task routing a USB cable through staggered open guides."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Route-USB-Zigzag-UR10e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UsbZigzagEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:UsbZigzagPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
