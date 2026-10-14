# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Route the S-curve cable, then align and seat its USB-C plug in a physical socket."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.usb_cable_routing.env_cfg import RoutingActionCfg
from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import (
    SCurveCommandCfg,
    UsbSCurveEnvCfg,
    UsbSCurvePPORunnerCfg,
)

from .assets import SOCKET_FACE, spawn_insert_cable

_MDP = "isaaclab_tasks.contrib.usb_cable_routing_insert.mdp:"


@configclass
class InsertActionCfg(RoutingActionCfg):
    """Bounded translation [m] and axis-angle rotation [rad] increments."""

    class_type = _MDP + "InsertionAction"
    lock_wrist_while_routing: bool = True
    """Retain the side grasp until routing is complete and the cable has been lowered for alignment."""
    rotation_scale: float = 0.025
    """Maximum rotation increment per control step [rad]."""
    insertion_scale: float = 0.001
    """Translation increment scale after routing [m]."""


@configclass
class InsertCommandCfg(SCurveCommandCfg):
    """All three guides must remain seated throughout the final insertion dwell."""

    class_type = _MDP + "InsertionCommand"
    guide_count = 3
    curriculum = False


@configclass
class UsbInsertEnvCfg(UsbSCurveEnvCfg):
    """Independent six-action routing-and-insertion task using Newton VBD and MJWarp."""

    def __post_init__(self):
        super().__post_init__()
        for name, value in self.scene.robot.init_state.joint_pos.items():
            if "finger" in name or "knuckle" in name:
                self.scene.robot.init_state.joint_pos[name] = 0.730 if value > 0 else -0.730
        self.commands.routing = InsertCommandCfg()
        self.actions.arm = InsertActionCfg()
        self.scene.cable.class_type = "isaaclab_tasks.contrib.usb_cable_routing_insert.cable:InsertionCable"
        self.scene.cable.spawn.func = spawn_insert_cable
        self.scene.socket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Socket",
            spawn=sim_utils.UsdFileCfg(usd_path=str(Path(__file__).parent / "assets" / "usb_c_socket.usda")),
            init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_FACE),
        )
        self.sim.physics.solver_cfg.entries[1].solver_cfg.rigid_body_contact_buffer_size = 4096
        self.sim.physics.solver_cfg.entries[1].bodies.append(r"/World/envs/env_.*/Socket")
        self.observations.policy.state.func = _MDP + "insertion_observation"
        self.rewards.angular_progress = RewardTermCfg(func=_MDP + "angular_progress_reward", weight=30.0)
        self.rewards.alignment = RewardTermCfg(func=_MDP + "alignment_reward", weight=4.0)
        self.rewards.insertion_progress = RewardTermCfg(func=_MDP + "insertion_progress", weight=80.0)
        self.rewards.premature_entry = RewardTermCfg(func=_MDP + "premature_entry", weight=-6000.0)
        self.rewards.insertion_force = RewardTermCfg(func=_MDP + "insertion_force_penalty", weight=-20.0)
        self.rewards.routing_stage = RewardTermCfg(func=_MDP + "curriculum_routing_complete", weight=5000.0)
        self.terminations.routing_stage = TerminationTermCfg(func=_MDP + "curriculum_routing_complete")
        self.terminations.premature_entry = TerminationTermCfg(func=_MDP + "premature_entry")
        self.episode_length_s = 60.0


@configclass
class UsbInsertPPORunnerCfg(UsbSCurvePPORunnerCfg):
    """Separate PPO checkpoint namespace for the six-action insertion policy."""

    experiment_name = "usb_cable_routing_insert"
    clip_actions = 1.0
