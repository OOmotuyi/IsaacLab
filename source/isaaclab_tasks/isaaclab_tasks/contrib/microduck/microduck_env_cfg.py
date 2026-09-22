# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-ground training configuration for Newton MJWarp."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, CommandTermCfg, ObservationGroupCfg, SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.visualizers import VisualizerCfg

from .constants import DEFAULT_POSE, JOINT_NAMES

_ASSETS = Path(os.environ.get("MICRODUCK_ASSET_DIR", Path(__file__).parents[5] / "logs/microduck/assets"))
_MDP = "isaaclab_tasks.contrib.microduck.mdp."


@configclass
class MicroDuckActionCfg(ActionTermCfg):
    """BAM motor supply [V], delay [physics steps], and friction scaling."""

    class_type = _MDP + "actions:MicroDuckAction"
    asset_name = "robot"
    motor_parameters: str = str(_ASSETS / "bam_m6.json")
    scale: float = 1.0
    delay_steps: tuple[int, int] = (3, 6)
    battery_range: tuple[float, float] = (6.5, 8.2)
    sag_range: tuple[float, float] = (0.0, 0.2)
    friction_range: tuple[float, float] = (0.9, 1.1)


@configclass
class MicroDuckVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type = _MDP + "commands:MicroDuckVelocityCommand"
    turn_in_place_fraction: float = 0.15


@configclass
class MicroDuckPoseCommandCfg(CommandTermCfg):
    class_type = _MDP + "commands:MicroDuckPoseCommand"
    resampling_time_range = (2.0, 5.0)
    ranges: tuple[tuple[float, float], ...] = ()


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    num_envs = 1024
    env_spacing = 2.0
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0)
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2500.0))
    robot: ArticulationCfg = None
    feet = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_(left|right)",
        track_air_time=True,
        track_friction_forces=True,
        force_threshold=0.01,
    )


@configclass
class MicroDuckActionsCfg:
    joint_pos = MicroDuckActionCfg()


@configclass
class MicroDuckCommandsCfg:
    twist = MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )
    head_pose = MicroDuckPoseCommandCfg(ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)))
    body_pose = MicroDuckPoseCommandCfg(ranges=((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3)


@configclass
class MicroDuckObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03))
        gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            noise=UniformNoiseCfg(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        twist = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        head_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        linear_velocity = ObsTerm(func=_MDP + "observations:linear_velocity")
        feet = ObsTerm(func=_MDP + "feet:foot_observation")

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class MicroDuckRewardsCfg:
    track_linear = RewTerm(func=_MDP + "rewards:track_velocity", weight=2.0)
    track_angular = RewTerm(func=_MDP + "rewards:track_velocity", weight=2.0, params={"angular": True, "variance": 0.5})
    upright = RewTerm(func=_MDP + "rewards:upright", weight=2.0)
    posture = RewTerm(func=_MDP + "rewards:posture", weight=1.0)
    head_pose = RewTerm(func=_MDP + "rewards:head_pose", weight=2.0)
    head_bias = RewTerm(func=_MDP + "rewards:MicroDuckHeadBias", weight=0.0)
    body_ang_vel = RewTerm(func=_MDP + "rewards:body_angular_velocity", weight=-0.05)
    angular_momentum = RewTerm(func=_MDP + "rewards:angular_momentum", weight=-0.02)
    self_collision = RewTerm(func=_MDP + "rewards:self_collision", weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time = RewTerm(func=_MDP + "feet:air_time", weight=3.0)
    foot_clearance = RewTerm(func=_MDP + "feet:clearance", weight=-2.0)
    foot_slip = RewTerm(func=_MDP + "feet:slip", weight=-0.1)
    foot_swing = RewTerm(func=_MDP + "feet:MicroDuckSwingHeight", weight=-0.25)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)},
    )


@configclass
class MicroDuckEventsCfg:
    calibration = EventTerm(func=_MDP + "events:sensor_calibration", mode="startup")
    mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["trunk_base"]),
            "mass_distribution_params": (0.95, 1.05),
            "distribution": "log_uniform",
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    base_com = EventTerm(
        func=_MDP + "events:MicroDuckCenterOfMass",
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.025, 0.025, 0.03)},
    )
    reset_com = EventTerm(
        func=_MDP + "events:MicroDuckCenterOfMass",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["trunk_base", "neck", "neck_pitch_1", "yaw_roll_motion", "jaw_soft", "bearing_roll"],
            ),
            "ranges": (0.003, 0.003, 0.003),
        },
    )
    armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_left", "ankle_right"]),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.01), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )
    push = EventTerm(
        func=_MDP + "events:velocity_push",
        mode="interval",
        interval_range_s=(3.0, 6.0),
    )


@configclass
class MicroDuckTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70)})
    invalid = DoneTerm(func=_MDP + "rewards:invalid_state")


@configclass
class MicroDuckCurriculumCfg:
    walking = CurriculumTerm(
        func=_MDP + "curricula:walking_schedule", params={"step_offset": 0, "action_rate_scale": 1.0}
    )


@configclass
class MicroDuckFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Native manager environment with the 61-value MicroDuck policy contract."""

    scene = MicroDuckSceneCfg()
    actions = MicroDuckActionsCfg()
    commands = MicroDuckCommandsCfg()
    observations = MicroDuckObservationsCfg()
    rewards = MicroDuckRewardsCfg()
    events = MicroDuckEventsCfg()
    terminations = MicroDuckTerminationsCfg()
    curriculum = MicroDuckCurriculumCfg()
    decimation = 4
    episode_length_s = 20.0
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        use_newton_actuators=False,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                integrator="implicitfast", njmax=600, nconmax=128, cone="pyramidal", iterations=100, ls_iterations=50
            ),
            use_cuda_graph=True,
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(0.6, -0.7, 0.4), lookat=(0.0, 0.0, 0.12)),
    )

    def __post_init__(self):
        params_path = Path(self.actions.joint_pos.motor_parameters)
        if not params_path.is_file():
            raise FileNotFoundError("Run scripts/demos/microduck/prepare.py before creating the training environment.")
        params = json.loads(params_path.read_text())
        actor = self.observations.policy
        actor.ang_vel.func = _MDP + "observations:MicroDuckDelayedObservation"
        actor.ang_vel.params = {"kind": "angular_velocity"}
        actor.gravity.func = _MDP + "observations:MicroDuckDelayedObservation"
        actor.gravity.params = {"kind": "gravity"}
        actor.joint_vel.func = _MDP + "observations:MicroDuckDelayedObservation"
        actor.joint_vel.params = {"kind": "joint_velocity"}
        actor.joint_pos.func = _MDP + "observations:joint_position"
        actor.joint_pos.params = {}

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_ASSETS / "robot_walk/robot_walk.usda"),
                variants={"Physics": "mujoco"},
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.12), joint_pos=dict(zip(JOINT_NAMES, DEFAULT_POSE)), joint_vel={".*": 0.0}
            ),
            actuators={
                "servos": ImplicitActuatorCfg(
                    joint_names_expr=list(JOINT_NAMES),
                    stiffness=0.0,
                    damping=0.0,
                    armature=params["armature"],
                    friction=params["friction_base"],
                    joint_effort_limit=max(self.actions.joint_pos.battery_range) * params["kt"] / params["R"],
                )
            },
        )

    def play_mode(self):
        self.scene.num_envs = 16
        self.curriculum.walking = None
        self.observations.policy.enable_corruption = False
        self.events.push = None
        self.events.calibration = None
        self.events.mass = None
        self.events.base_com = None
        self.events.reset_com = None
        self.events.armature = None
        self.events.foot_friction = None
        self.observations.policy.ang_vel.params["random_delay"] = False
        self.observations.policy.gravity.params["random_delay"] = False
        self.actions.joint_pos.battery_range = (7.4, 7.4)
        self.actions.joint_pos.sag_range = (0.1, 0.1)
        self.actions.joint_pos.friction_range = (1.0, 1.0)
