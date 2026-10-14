# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separate contact-grasp tasks; the assisted-grasp benchmark remains unchanged."""

from dataclasses import fields

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab.assets import CableObjectCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlRNNModelCfg

from isaaclab_tasks.contrib.usb_cable_routing_insert.env_cfg import (
    InsertCommandCfg,
    UsbInsertEnvCfg,
)
from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import (
    SCurveCommandCfg,
    UsbSCurveEnvCfg,
    UsbSCurvePPORunnerCfg,
)

from .assets import spawn_contact_cable, spawn_contact_insert_cable
from .cable_cfg import ContactCableCfg

_MDP = "isaaclab_tasks.contrib.usb_cable_routing_contact.mdp:"


@configclass
class ContactGraspCommandCfg:
    """Physical gripper closure and reset-time guide pose variation."""

    gripper_initial_position: float = 0.735
    """Uniform initial angle [rad] when the calibrated joint map is disabled."""
    gripper_initial_joint_positions: dict[str, float] | None = {
        "finger_joint": 0.7452058792114258,
        "left_inner_finger_joint": -0.7545701265335083,
        "left_inner_finger_knuckle_joint": -0.7449366450309753,
        "right_outer_knuckle_joint": 0.7409436702728271,
        "right_inner_finger_joint": 0.7549059987068176,
        "right_inner_finger_knuckle_joint": -0.7447932958602905,
    }
    """Measured preloaded finger positions [rad], applied only at episode reset."""
    gripper_closed_position: float = 0.80
    """Closing drive target [rad]; contact and actuator limits determine the actual opening."""
    randomize_guides: bool = True
    guide_xy_jitter: float = 0.010
    """Uniform guide translation half-range along each horizontal axis [m]."""
    guide_yaw_jitter: float = 0.08726646259971647
    """Uniform guide yaw half-range [rad], equivalent to five degrees."""
    minimum_layout_slack: float = 0.015
    """Minimum cable length remaining beyond the guide-center polyline [m]."""


@configclass
class ContactSCurveCommandCfg(ContactGraspCommandCfg, SCurveCommandCfg):
    """Routing command with measured grasp retention and transformed guide targets."""

    class_type = _MDP + "ContactSCurveCommand"


@configclass
class ContactInsertCommandCfg(ContactGraspCommandCfg, InsertCommandCfg):
    """Insertion command measuring socket load independently of pinch load."""

    class_type = _MDP + "ContactInsertCommand"
    guide_count = 1
    curriculum = True
    gripper_initial_joint_positions = {
        "finger_joint": 0.745245635509491,
        "left_inner_finger_joint": -0.7553189992904663,
        "left_inner_finger_knuckle_joint": -0.7449785470962524,
        "right_outer_knuckle_joint": 0.7406142354011536,
        "right_inner_finger_joint": 0.7552012205123901,
        "right_inner_finger_knuckle_joint": -0.7449257969856262,
    }


def _configure_contact_scene(cfg: UsbSCurveEnvCfg, *, insertion: bool) -> None:
    previous = cfg.scene.cable
    cable_values = {field.name: getattr(previous, field.name) for field in fields(CableObjectCfg)}
    cable_values.pop("class_type", None)
    cfg.scene.cable = ContactCableCfg(**cable_values)
    cfg.scene.cable.spawn.func = spawn_contact_insert_cable if insertion else spawn_contact_cable
    cfg.scene.robot.actuators["gripper_finger"].joint_effort_limit = 10.0
    # Convert inherited legacy properties before combining backend schema fragments.
    previous_rigid = cfg.scene.robot.spawn.rigid_props
    rigid_values = {field.name: getattr(previous_rigid, field.name) for field in fields(previous_rigid)}
    cfg.scene.robot.spawn.rigid_props = [
        fragment(**{field.name: rigid_values[field.name] for field in fields(fragment) if field.name in rigid_values})
        for fragment in (UsdPhysicsRigidBodyCfg, PhysxRigidBodyCfg)
    ]
    cfg.scene.robot.spawn.rigid_props.append(MujocoRigidBodyCfg(gravcomp=1.0))
    cable_solver = cfg.sim.physics.solver_cfg.entries[1].solver_cfg
    cable_solver.rigid_compliant_alm = True
    cable_solver.rigid_body_contact_buffer_size = max(getattr(cable_solver, "rigid_body_contact_buffer_size", 64), 512)
    cfg.events.guide_poses = EventTermCfg(func=_MDP + "randomize_guide_poses", mode="reset")
    cfg.observations.policy.state.func = (
        "isaaclab_tasks.contrib.usb_cable_routing_contact.observations:contact_policy_observation"
    )
    cfg.observations.policy.last_action = None
    if insertion:
        cfg.actions.arm.class_type = "isaaclab_tasks.contrib.usb_cable_routing_contact.actions:ContactInsertionAction"
        cfg.actions.arm.lock_wrist_while_routing = False


@configclass
class UsbContactSCurveEnvCfg(UsbSCurveEnvCfg):
    """Route a cable held only by actuated finger contact and friction."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.routing = ContactSCurveCommandCfg()
        _configure_contact_scene(self, insertion=False)


@configclass
class UsbContactInsertEnvCfg(UsbInsertEnvCfg):
    """Route and insert a physically pinched USB-C connector."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.routing = ContactInsertCommandCfg()
        self.rewards.insertion_force.func = _MDP + "socket_force_penalty"
        _configure_contact_scene(self, insertion=True)


@configclass
class UsbContactSCurvePPORunnerCfg(UsbSCurvePPORunnerCfg):
    """PPO output namespace for physical contact-grasp routing."""

    experiment_name = "usb_cable_routing_contact"
    clip_actions = 1.0
    actor = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )


@configclass
class UsbContactInsertPPORunnerCfg(UsbContactSCurvePPORunnerCfg):
    """PPO output namespace for physical contact-grasp insertion."""

    experiment_name = "usb_cable_routing_insert_contact"
