# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USB cable routing with a physical Robotiq grasp and randomized guide poses."""

import gymnasium as gym

for _suffix, _configuration, _runner in (
    ("SCurve-Contact", "UsbContactSCurveEnvCfg", "UsbContactSCurvePPORunnerCfg"),
    ("SCurve-Insert-Contact", "UsbContactInsertEnvCfg", "UsbContactInsertPPORunnerCfg"),
):
    gym.register(
        id=f"IsaacContrib-Route-USB-{_suffix}-UR10e",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_configuration}",
            "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:{_runner}",
            "default_agent": "rsl_rl",
        },
    )

del _suffix, _configuration, _runner
