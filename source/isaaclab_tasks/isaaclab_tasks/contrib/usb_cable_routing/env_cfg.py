# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e / Newton VBD cable routing with ordered guide curriculum."""

from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG

from .assets import (
    ANCHOR,
    ARM_JOINTS,
    CABLE_DIAMETER,
    GRIPPER_POSITION,
    GUIDES,
    SEGMENT_LENGTH,
    SEGMENTS,
    TCP_OFFSET,
    spawn_robot,
    spawn_usb_cable,
)
from .parameters import BEND_MODULUS, CABLE_DENSITY, DYNAMIC_FRICTION, STRETCH_MODULUS, TWIST_MODULUS

_MDP = "isaaclab_tasks.contrib.usb_cable_routing.mdp:"
_ASSETS = Path(__file__).parent / "assets"


@configclass
class RoutingCommandCfg(CommandTermCfg):
    """Routing geometry, episode randomization, and success-based stage advancement."""

    class_type = _MDP + "RoutingCommand"
    resampling_time_range = (1e9, 1e9)
    guide_count: int = 1
    curriculum: bool = True
    curriculum_episodes: int = 128
    curriculum_success_rate: float = 0.8
    hold_steps: int = 12
    randomize: bool = True
    stiffness_range: tuple[float, float] = (0.85, 1.15)
    friction_range: tuple[float, float] = (0.36, 0.44)
    position_jitter: float = 0.005
    """Initial cable centerline perturbation [m], vanishing at the clamped ends."""


@configclass
class RoutingActionCfg(base_mdp.DifferentialInverseKinematicsActionCfg):
    class_type = _MDP + "RoutingAction"
    asset_name = "robot"
    joint_names = ARM_JOINTS
    body_name = "wrist_3_link"
    body_offset = base_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET)
    controller = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    scale = 0.010
    maximum_joint_speed: float = 2.0
    """Maximum commanded joint-target change divided by the physics step [rad/s]."""
    minimum_tcp_height: float = 0.045
    """Minimum commanded grasp-point height above the tabletop [m]."""


def _guide(index):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Guide{index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSETS / "guide.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=GUIDES[index - 1]),
    )


@configclass
class RoutingSceneCfg(InteractiveSceneCfg):
    num_envs = 64
    env_spacing = 2.5
    replicate_physics = True
    robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.func = spawn_robot
    for actuator in (robot.actuators["shoulder"], robot.actuators["elbow"], robot.actuators["wrist"]):
        actuator.stiffness = 10000.0
        actuator.damping = 200.0
        actuator.armature = 0.1
    del actuator
    robot.init_state.pos = (0.10, -0.85, 0.0)
    robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    robot.init_state.joint_pos = dict(
        zip(ARM_JOINTS, (-1.1461, -1.7117, 2.6062, -2.4654, -1.5708, -2.7169), strict=True)
    )
    robot.init_state.joint_pos.update(
        {
            "finger_joint": GRIPPER_POSITION,
            "right_outer_knuckle_joint": GRIPPER_POSITION,
            "left_inner_finger_joint": -GRIPPER_POSITION,
            "right_inner_finger_joint": GRIPPER_POSITION,
            ".*_inner_finger_knuckle_joint": -GRIPPER_POSITION,
        }
    )
    robot.actuators["gripper_drive"].stiffness = 40.0
    robot.actuators["gripper_drive"].damping = 1.0
    robot.actuators["gripper_finger"].stiffness = 10.0
    robot.actuators["gripper_finger"].damping = 0.05
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.1, 1.6, 0.055),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.65, 0.68)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.44, -0.13, -0.0275)),
    )
    anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.UsdFileCfg(usd_path=str(_ASSETS / "anchor.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(ANCHOR[0], ANCHOR[1], 0.0)),
    )
    guide1 = _guide(1)
    guide2 = _guide(2)
    guide3 = _guide(3)
    cable = CableObjectCfg(
        class_type="isaaclab_tasks.contrib.usb_cable_routing.cable:UsbCable",
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            func=spawn_usb_cable,
            positions=[(i * SEGMENT_LENGTH, 0, 0) for i in range(SEGMENTS + 1)],
            physics_material=sim_utils.CableMaterialCfg(
                thickness=CABLE_DIAMETER,
                density=CABLE_DENSITY,
                stretch_stiffness=STRETCH_MODULUS,
                bend_stiffness=BEND_MODULUS,
                twist_stiffness=TWIST_MODULUS,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.025, 0.025, 0.028)),
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=ANCHOR),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))


@configclass
class ActionsCfg:
    arm = RoutingActionCfg()


@configclass
class CommandsCfg:
    routing = RoutingCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=_MDP + "routing_observation")
        last_action = ObsTerm(func=base_mdp.last_action)
        concatenate_terms = True
        enable_corruption = False

    policy = PolicyCfg()


@configclass
class EventsCfg:
    reset = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    approach = RewTerm(func=_MDP + "approach_reward", weight=100.0)
    guide = RewTerm(func=_MDP + "guide_reward", weight=1000.0)
    success = RewTerm(func=_MDP + "routing_success", weight=5000.0)
    collision = RewTerm(func=_MDP + "collision_penalty", weight=-8.0)
    stretch = RewTerm(func=_MDP + "stretch_penalty", weight=-20.0)
    dropped = RewTerm(func=_MDP + "cable_dropped", weight=-100.0)
    skipped = RewTerm(func=_MDP + "skipped_guide", weight=-60.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    arm_posture = RewTerm(func=_MDP + "arm_posture_penalty", weight=-4.0)
    arm_clearance = RewTerm(func=_MDP + "arm_clearance_penalty", weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=_MDP + "routing_success")
    dropped = DoneTerm(func=_MDP + "cable_dropped")
    unsafe_arm = DoneTerm(func=_MDP + "unsafe_arm_posture")
    stretched = DoneTerm(func=_MDP + "cable_overstretched")
    skipped = DoneTerm(func=_MDP + "skipped_guide")


@configclass
class UsbRoutingEnvCfg(ManagerBasedRLEnvCfg):
    scene = RoutingSceneCfg()
    commands = CommandsCfg()
    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    decimation = 4
    episode_length_s = 12.0
    sim = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics=NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name="rigid",
                        solver_cfg=MJWarpSolverCfg(integrator="implicitfast"),
                        bodies=[r"/World/envs/env_.*/Robot"],
                    ),
                    CouplerEntryCfg(
                        name="cable",
                        solver_cfg=VBDSolverCfg(iterations=12),
                        bodies=[r"/World/envs/env_.*/Cable", r"/World/envs/env_.*/Guide.*"],
                        include_static_shapes=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rigid", destination="cable", bodies=[r"/World/envs/env_.*/Robot/.*"], collide_interval=1
                    )
                ],
                iterations=1,
            ),
            default_shape_cfg=NewtonShapeCfg(ke=2e4, kd=20.0, mu=DYNAMIC_FRICTION),
            num_substeps=8,
            use_cuda_graph=True,
        ),
        default_visualizer_cfg=VisualizerCfg(eye=(1.3, -1.35, 1.05), lookat=(0.4, -0.12, 0.12)),
    )
