# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S-curve cable routing followed by USB-C socket insertion."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Route-USB-SCurve-Insert-UR10e",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UsbInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:UsbInsertPPORunnerCfg",
        "rsl_rl_residual_cfg_entry_point": f"{__name__}.agents:UsbInsertResidualPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
