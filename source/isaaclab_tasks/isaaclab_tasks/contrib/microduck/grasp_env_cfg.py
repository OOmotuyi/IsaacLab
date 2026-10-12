# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-conditioned MicroDuck grasping with learned body and jaw control."""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonArticulationCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ActionTermCfg, CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .constants import MARKER_SUPPORT_SIZE
from .microduck_env_cfg import MicroDuckObservationsCfg
from .skill_env_cfg import MicroDuckStandScratchEnvCfg

_MDP = "isaaclab_tasks.contrib.microduck.mdp.grasp:"
_ROOT = Path(__file__).parents[5]
_ASSETS = _ROOT / "logs/microduck/marker_pick/assets_lip"


@configclass
class MicroDuckGraspActionCfg(ActionTermCfg):
    """Absolute body-joint offsets and jaw targets [rad]."""

    class_type = _MDP + "MicroDuckGraspAction"
    asset_name = "robot"
    motor_parameters: str = str(_ROOT / "logs/microduck/assets/bam_m6.json")


@configclass
class MicroDuckGraspCommandCfg(CommandTermCfg):
    """Grasp goals derived from the simulated object, with heights in [m]."""

    class_type = _MDP + "MicroDuckGraspCommand"
    resampling_time_range = (1000.0, 1000.0)
    initial_object_height: float = 0.0395
    initial_reach_height: float = 0.215
    minimum_object_height: float = 0.0395
    lift_distance: float = 0.05
    curriculum: bool = True
    observation_contract: str = "grasp_proprio_object_v3"


@configclass
class MicroDuckGraspCommandsCfg:
    grasp = MicroDuckGraspCommandCfg()


@configclass
class MicroDuckGraspObservationsCfg:
    @configclass
    class PolicyCfg(MicroDuckObservationsCfg.PolicyCfg):
        twist = None
        head_pose = None
        body_pose = None
        grasp = ObsTerm(func=mdp.generated_commands, params={"command_name": "grasp"})

    @configclass
    class CriticCfg(PolicyCfg):
        linear_velocity = ObsTerm(func="isaaclab_tasks.contrib.microduck.mdp.observations:linear_velocity")

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class MicroDuckGraspRewardsCfg:
    approach = RewTerm(func=_MDP + "grasp_reward", weight=12.0, params={"kind": "approach"})
    orientation = RewTerm(func=_MDP + "grasp_reward", weight=2.0, params={"kind": "orientation"})
    alignment = RewTerm(func=_MDP + "grasp_reward", weight=2.0, params={"kind": "alignment"})
    jaw = RewTerm(func=_MDP + "grasp_reward", weight=1.0, params={"kind": "jaw"})
    contacts = RewTerm(func=_MDP + "grasp_reward", weight=3.0, params={"kind": "contacts"})
    lift = RewTerm(func=_MDP + "grasp_reward", weight=8.0, params={"kind": "lift"})
    success = RewTerm(func=_MDP + "grasp_reward", weight=15.0, params={"kind": "success"})
    balance = RewTerm(func=_MDP + "grasp_reward", weight=0.5, params={"kind": "balance"})
    posture = RewTerm(func=_MDP + "grasp_reward", weight=2.0, params={"kind": "posture"})
    neck = RewTerm(func=_MDP + "grasp_reward", weight=1.0, params={"kind": "neck"})
    height = RewTerm(func=_MDP + "grasp_reward", weight=1.0, params={"kind": "height"})
    anchor = RewTerm(func=_MDP + "grasp_reward", weight=1.0, params={"kind": "anchor"})
    movement = RewTerm(func=_MDP + "grasp_reward", weight=-0.02, params={"kind": "movement"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    termination = RewTerm(func=mdp.is_terminated, weight=-8.0)


@configclass
class MicroDuckMarkerGraspEnvCfg(MicroDuckStandScratchEnvCfg):
    """Learn a real contact grasp and sustained lift with Isaac-only policy lineage."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos = MicroDuckGraspActionCfg()
        self.commands = MicroDuckGraspCommandsCfg()
        self.observations = MicroDuckGraspObservationsCfg()
        self.rewards = MicroDuckGraspRewardsCfg()
        self.observations.policy.joint_pos.func = "isaaclab_tasks.contrib.microduck.mdp.observations:joint_position"
        self.observations.policy.joint_pos.params = {}
        self.observations.critic.joint_pos.func = self.observations.policy.joint_pos.func
        self.observations.critic.joint_pos.params = {}
        self.scene.num_envs = 256
        self.scene.robot.spawn.articulation_props = [NewtonArticulationCfg(self_collision_enabled=False)]
        self.scene.robot.actuators["mouth"] = ImplicitActuatorCfg(
            joint_names_expr=["mouth"],
            stiffness=1.0,
            damping=0.008,
            armature=0.0001,
            friction=0.002,
            joint_effort_limit=0.05,
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_ASSETS / "marker.usda"),
                collision_props=[
                    MujocoCollisionCfg(
                        priority=1,
                        condim=6,
                        solref=(0.01, 1.0),
                        solimp=(0.99, 0.9999, 0.001, 0.5, 2.0),
                    )
                ],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.068, 0.0, 0.16)),
        )
        self.scene.support = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Support",
            spawn=sim_utils.CuboidCfg(
                size=MARKER_SUPPORT_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.36, 0.4)),
            ),
        )
        self.scene.support_other = self.scene.support.replace(prim_path="{ENV_REGEX_NS}/SupportOther")
        self.events.reset_root = None
        self.events.reset_joints = None
        self.events.grasp_reset = EventTerm(func=_MDP + "reset_grasp", mode="reset")
        self.events.foot_friction = None
        self.terminations.orientation.params["limit_angle"] = math.radians(85)
        self.terminations.grasp_failure = DoneTerm(func=_MDP + "grasp_failure")
        self.terminations.invalid = DoneTerm(func=_MDP + "grasp_invalid_state")
        self.sim.physics.num_substeps = 8
        self.episode_length_s = 8.0

    def play_mode(self) -> None:
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.commands.grasp.curriculum = False
        self.commands.grasp.initial_reach_height = self.commands.grasp.minimum_object_height
        self.commands.grasp.initial_object_height = self.commands.grasp.minimum_object_height
