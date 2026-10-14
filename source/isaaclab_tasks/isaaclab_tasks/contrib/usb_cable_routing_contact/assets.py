# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Existing USB geometry with physical connector–Robotiq collisions enabled."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from newton.solvers import SolverMuJoCo

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import _fit_insert_connector
from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import _fit_reference_connector

if TYPE_CHECKING:
    from newton import ModelBuilder


@clone
def spawn_contact_cable(
    prim_path: str,
    cfg: sim_utils.CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the routing cable with a rigid, physically grasped connector."""
    from isaaclab_newton.physics import NewtonManager

    if _fit_contact_connector not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_fit_contact_connector)
    return sim_utils.spawn_cable(prim_path, cfg, translation, orientation)


@clone
def spawn_contact_insert_cable(
    prim_path: str,
    cfg: sim_utils.CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the hollow insertion plug with physical Robotiq contact."""
    from isaaclab_newton.physics import NewtonManager

    if _fit_contact_insert_connector not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_fit_contact_insert_connector)
    return sim_utils.spawn_cable(prim_path, cfg, translation, orientation)


def _enable_connector_contacts(builder: ModelBuilder, world_index: int) -> None:
    """Remove only this world's artificial-grasp collision exclusions."""
    cable_bodies = {
        index
        for index, label in enumerate(builder.body_label)
        if "/Cable/" in label and builder.body_world[index] == world_index
    }
    connector_shapes = {
        index
        for index, body in enumerate(builder.shape_body)
        if body in cable_bodies and "mesh_edge_capsule_" not in builder.shape_label[index]
    }
    gripper_shapes = {
        index
        for index, body in enumerate(builder.shape_body)
        if body >= 0 and builder.body_world[body] == world_index and "/Robotiq_2F_85/" in builder.shape_label[index]
    }
    builder.shape_collision_filter_pairs[:] = [
        (a, b)
        for a, b in builder.shape_collision_filter_pairs
        if not ((a in connector_shapes and b in gripper_shapes) or (b in connector_shapes and a in gripper_shapes))
    ]


def _fit_contact_connector(builder: ModelBuilder, world_index: int, position, orientation) -> None:
    _fit_reference_connector(builder, world_index, position, orientation)
    _enable_connector_contacts(builder, world_index)
    _configure_contact_linkage(builder, world_index)


def _fit_contact_insert_connector(builder: ModelBuilder, world_index: int, position, orientation) -> None:
    _fit_insert_connector(builder, world_index, position, orientation)
    _enable_connector_contacts(builder, world_index)
    _configure_contact_linkage(builder, world_index)


def _configure_contact_linkage(builder: ModelBuilder, world_index: int) -> None:
    """Set Robotiq mimic equality time constants to 0.002 [s] for physical pinching.

    The default 0.02 [s] linkage compliance allowed asymmetric finger motion
    under contact load. Metadata targets the existing mimic constraints, so
    MuJoCo does not create duplicate constraints or attach anything to the plug.
    """
    # Newton's equality schema uses target kind 2 for a native mimic and
    # object kind 3 for a joint; add_custom_values is its public authoring API.
    mimic_target_kind = 2
    joint_object_kind = 3
    attributes = builder.custom_attributes
    targets = attributes["mujoco:equality_constraint_target"].values
    kinds = attributes["mujoco:equality_constraint_target_kind"].values
    existing = {
        int(value): index
        for index, value in enumerate(targets)
        if value is not None and index < len(kinds) and kinds[index] == mimic_target_kind
    }
    for mimic, (follower, leader) in enumerate(
        zip(builder.constraint_mimic_joint0, builder.constraint_mimic_joint1, strict=True)
    ):
        labels = (builder.joint_label[follower], builder.joint_label[leader])
        child = builder.joint_child[follower]
        if builder.body_world[child] != world_index or not all("/Robotiq_2F_85/" in label for label in labels):
            continue
        if mimic in existing:
            values = attributes["mujoco:eq_solref"].values
            index = existing[mimic]
            values.extend([None] * max(0, index + 1 - len(values)))
            values[index] = wp.vec2(0.002, 1.0)
            continue
        builder.add_custom_values(
            **{
                "mujoco:equality_constraint_type": int(SolverMuJoCo.EqType.JOINT),
                "mujoco:equality_constraint_objtype": joint_object_kind,
                "mujoco:equality_constraint_joint1": follower,
                "mujoco:equality_constraint_joint2": leader,
                "mujoco:equality_constraint_polycoef": [
                    builder.constraint_mimic_coef0[mimic],
                    builder.constraint_mimic_coef1[mimic],
                    0.0,
                    0.0,
                    0.0,
                ],
                "mujoco:equality_constraint_enabled": builder.constraint_mimic_enabled[mimic],
                "mujoco:equality_constraint_world": world_index,
                "mujoco:equality_constraint_label": labels[0] + "/contact_grasp_linkage",
                "mujoco:equality_constraint_target_kind": mimic_target_kind,
                "mujoco:equality_constraint_target": mimic,
                "mujoco:eq_solref": wp.vec2(0.002, 1.0),
            }
        )
