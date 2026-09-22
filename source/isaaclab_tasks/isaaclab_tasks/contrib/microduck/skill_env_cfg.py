# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standing and reaching skill fine-tuning with the cleanup robot in Newton."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .constants import JOINT_NAMES
from .microduck_env_cfg import _ASSETS, MicroDuckActionCfg, MicroDuckFlatEnvCfg

_SKILLS = "isaaclab_tasks.contrib.microduck.mdp.skills:"
_ROOT = Path(__file__).parents[5]


@configclass
class MicroDuckSkillActionCfg(MicroDuckActionCfg):
    class_type = _SKILLS + "MicroDuckSkillAction"
    teacher_path: str = str(_ASSETS / "alpha_stand.onnx")
    delay_steps = (0, 0)
    battery_range = (7.3, 7.5)
    sag_range = (0.1, 0.1)
    friction_range = (0.98, 1.02)


@configclass
class MicroDuckSkillCommandCfg(CommandTermCfg):
    class_type = _SKILLS + "MicroDuckSkillCommand"
    resampling_time_range = (100.0, 100.0)
    skill: str = "stand"
    reference_path: str = os.environ.get("MICRODUCK_SKILL_REFERENCE", str(_ASSETS / "pickup_reference.json"))


@configclass
class MicroDuckSkillRewardsCfg:
    head_target = RewTerm(func=_SKILLS + "skill_head_tracking", weight=4.0)
    balance = RewTerm(func=_SKILLS + "skill_balance", weight=2.0)
    height = RewTerm(func=_SKILLS + "skill_height", weight=1.0)
    source_prior = RewTerm(func=_SKILLS + "skill_source_prior", weight=4.0)
    stationary = RewTerm(func=_SKILLS + "skill_stationary", weight=0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class MicroDuckStandEnvCfg(MicroDuckFlatEnvCfg):
    """Fine-tune standing body control; the jaw stays scripted and open."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = MicroDuckSkillActionCfg()
        self.commands.twist = MicroDuckSkillCommandCfg()
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.rewards = MicroDuckSkillRewardsCfg()
        self.curriculum.walking = None
        self.episode_length_s = 7.0
        self.scene.robot.spawn.usd_path = str(
            _ROOT / "logs/microduck/marker_pick/assets_lip/robot_marker/robot_marker.usda"
        )
        self.scene.robot.init_state.joint_pos["mouth"] = 0.5235987756
        self.scene.robot.actuators["mouth"] = ImplicitActuatorCfg(
            joint_names_expr=["mouth"], stiffness=3.0, damping=0.06, joint_effort_limit=0.15
        )
        self.sim.physics.solver_cfg.cone = "elliptic"
        self.sim.physics.solver_cfg.impratio = 10.0
        self.sim.physics.solver_cfg.iterations = 100
        for name in ("calibration", "mass", "base_com", "reset_com", "armature", "push"):
            setattr(self.events, name, None)
        self.events.foot_friction.params["static_friction_range"] = (0.9, 1.1)
        self.events.foot_friction.params["dynamic_friction_range"] = (0.9, 1.1)
        self.events.reset_root.params["pose_range"] = {
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.1, 0.1),
        }
        self.events.reset_joints.params["position_range"] = (-0.005, 0.005)
        self.observations.policy.ang_vel.func = mdp.base_ang_vel
        self.observations.policy.ang_vel.params = {}
        self.observations.policy.gravity.func = mdp.projected_gravity
        self.observations.policy.gravity.params = {}
        self.observations.policy.joint_vel.func = mdp.joint_vel
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)
        }

    def play_mode(self):
        super().play_mode()
        self.observations.policy.ang_vel.params = {}
        self.observations.policy.gravity.params = {}
        self.events.reset_root.params["pose_range"] = {}
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)


@configclass
class MicroDuckPickupEnvCfg(MicroDuckStandEnvCfg):
    """Fine-tune phase-conditioned reaching and lifting body motion."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.teacher_path = str(_ASSETS / "alpha_ground_pick.onnx")
        self.commands.twist.skill = "pickup"
