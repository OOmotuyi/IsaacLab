# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zigzag layout and independent PPO configuration using the existing USB assets."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.usb_cable_routing.agents import UsbRoutingPPORunnerCfg
from isaaclab_tasks.contrib.usb_cable_routing.env_cfg import RoutingCommandCfg, UsbRoutingEnvCfg

GUIDES = ((0.30, -0.11, 0.0), (0.44, -0.19, 0.0), (0.56, -0.11, 0.0))
"""Guide center positions [m]; neighboring guides alternate laterally by 80 mm."""

_MDP = "isaaclab_tasks.contrib.usb_cable_routing_zigzag.mdp:"


@configclass
class ZigzagCommandCfg(RoutingCommandCfg):
    """Guide-specific clearance and seating targets for progressive cable bending."""

    class_type = _MDP + "ZigzagRoutingCommand"
    lift_targets: tuple[tuple[float, float, float], ...] = (
        (0.665, -0.02, 0.35),
        (0.70, -0.32, 0.21),
        (0.68, -0.04, 0.20),
    )
    """Free-end clearance targets for each guide [m]."""
    seat_targets: tuple[tuple[float, float, float], ...] = (
        (0.73, 0.035, 0.035),
        (0.70, -0.30, 0.035),
        (0.705, -0.035, 0.035),
    )
    """Free-end seating targets for each guide [m]."""


@configclass
class UsbZigzagEnvCfg(UsbRoutingEnvCfg):
    """Independent staggered-guide task; the straight task retains its original settings."""

    def __post_init__(self):
        self.commands.routing = ZigzagCommandCfg()
        self.actions.arm.class_type = _MDP + "ZigzagRoutingAction"
        self.observations.policy.state.func = _MDP + "zigzag_observation"
        self.rewards.bend = RewardTermCfg(func=_MDP + "bend_penalty", weight=-2.0)
        self.rewards.dropped.weight = -6000.0
        self.rewards.skipped.weight = -6000.0
        self.rewards.stretched_failure = RewardTermCfg(
            func="isaaclab_tasks.contrib.usb_cable_routing.mdp:cable_overstretched", weight=-6000.0
        )
        self.rewards.lost_guide = RewardTermCfg(func=_MDP + "lost_guide_penalty", weight=-8.0)
        self.episode_length_s = 24.0
        for i, position in enumerate(GUIDES, 1):
            getattr(self.scene, f"guide{i}").init_state.pos = position


@configclass
class UsbZigzagPPORunnerCfg(UsbRoutingPPORunnerCfg):
    """PPO run namespace and horizon for the longer zigzag manipulation sequence."""

    experiment_name = "usb_cable_routing_zigzag"
    max_iterations = 1500
