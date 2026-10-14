# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent S-curve layout using the existing UR10e/Robotiq Newton environment."""

import math
from pathlib import Path

from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import (
    UsbZigzagEnvCfg,
    UsbZigzagPPORunnerCfg,
    ZigzagCommandCfg,
)

from .assets import BLUE, GUIDE_COLOR_NAMES, GUIDE_YAWS, GUIDES, spawn_reference_cable
from .parameters import BEND_MODULUS, CABLE_DENSITY, DYNAMIC_FRICTION, STRETCH_MODULUS, TWIST_MODULUS

_MDP = "isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp:"


@configclass
class SCurveCommandCfg(ZigzagCommandCfg):
    """Ordered S-curve shaping targets validated with physical routing demonstrations."""

    class_type = _MDP + "SCurveRoutingCommand"
    friction_range = (0.9 * DYNAMIC_FRICTION, 1.1 * DYNAMIC_FRICTION)
    lift_targets = ((0.68, -0.15, 0.32), (0.64, 0.07, 0.19), (0.64, -0.23, 0.13))
    seat_targets = ((0.60, -0.15, 0.075), (0.60, 0.10, 0.060), (0.64, -0.23, 0.065))


@configclass
class UsbSCurveEnvCfg(UsbZigzagEnvCfg):
    """Narrow upright channels following a gentle S-curve above the tabletop."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.routing = SCurveCommandCfg()
        self.observations.policy.state.func = _MDP + "scurve_observation"
        self.rewards.approach.func = _MDP + "guide_approach_reward"
        self.rewards.collision.func = _MDP + "collision_penalty"
        self.rewards.collision.weight = -80.0
        self.rewards.bend.func = _MDP + "bend_penalty"
        self.rewards.bend_failure = RewardTermCfg(func=_MDP + "cable_overbent", weight=-6000.0)
        self.terminations.overbent = TerminationTermCfg(func=_MDP + "cable_overbent")
        material = self.scene.cable.spawn.physics_material
        material.density = CABLE_DENSITY
        material.stretch_stiffness = STRETCH_MODULUS
        material.bend_stiffness = BEND_MODULUS
        material.twist_stiffness = TWIST_MODULUS
        self.sim.physics.default_shape_cfg.mu = DYNAMIC_FRICTION
        self.scene.cable.spawn.func = spawn_reference_cable
        self.scene.cable.spawn.visual_material.diffuse_color = BLUE
        self.scene.table.spawn.visual_material.diffuse_color = (0.065, 0.075, 0.085)
        for i, (position, yaw) in enumerate(zip(GUIDES, GUIDE_YAWS, strict=True), 1):
            guide = getattr(self.scene, f"guide{i}")
            guide.spawn.usd_path = str(Path(__file__).parent / "assets" / "guide.usda")
            guide.spawn.variants = {"Color": GUIDE_COLOR_NAMES[i - 1]}
            guide.init_state.pos = position
            guide.init_state.rot = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


@configclass
class UsbSCurvePPORunnerCfg(UsbZigzagPPORunnerCfg):
    """Separate PPO checkpoint namespace for ordered S-curve routing."""

    experiment_name = "usb_cable_routing_scurve"
