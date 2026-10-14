# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USB-C receptacle fixture and an insertion-compatible version of the blue plug."""

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

from isaaclab_tasks.contrib.usb_cable_routing.assets import SEGMENT_LENGTH, SEGMENTS, _box
from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import _fit_reference_connector
from isaaclab_tasks.contrib.usb_cable_routing_scurve.parameters import CABLE_BODY_MASS, CONNECTOR_MASS

CONNECTOR_OFFSET = SEGMENT_LENGTH / 2 + 0.015
"""Rigid plug offset along the final rod local z, placing its rear at the cable endpoint [m]."""
SOCKET_FACE = (0.680, -0.150, 0.065)
"""Receptacle entrance center in the environment [m]; insertion follows world +x."""
INSERT_START = (0.580, 0.080, 0.350 - CONNECTOR_OFFSET)
"""Raised cable endpoint [m]; the wrist approaches the upright plug from the side."""
INSERT_DIRECTION = (0.0, 0.0, 1.0)
"""Initial cable/plug forward unit axis, perpendicular to the wrist approach axis."""
INSERT_TOOL_QUAT = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
"""Initial wrist orientation (xyzw); tool approach follows +y across the upright plug."""
SOCKET_QUAT = (0.5, 0.5, 0.5, 0.5)
"""Socket frame orientation (xyzw); local z is insertion and local x is the wide axis."""
GRASP_LOCAL_Z = CONNECTOR_OFFSET - 0.005
"""Grasp location along the final segment local z axis [m], behind the mating shell."""
PLUG_FRONT_OFFSET = CONNECTOR_OFFSET + 0.016
"""Plug shell nose relative to the final rod body's center along local z [m]."""
SOCKET_WIDTH = 0.0088
"""Receptacle clear width along its wide cross-section axis [m]."""
SOCKET_HEIGHT = 0.0030
"""Receptacle clear height along its narrow cross-section axis [m]."""
INSERTION_DEPTH = 0.006
"""Nominal shell nose depth from the socket entrance [m]."""
PREINSERT_GAP = 0.008
"""Shell nose clearance in front of the entrance for alignment [m]."""


def generate_assets(directory: Path | None = None) -> Path:
    """Create the raised USB-C receptacle USD, with dimensions in [m]."""
    directory = directory or Path(__file__).parent / "assets"
    directory.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(directory / "usb_c_socket.usda"))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1)
    grey, silver = (0.10, 0.13, 0.17), (0.69, 0.73, 0.78)
    # The mounting block is behind the protruding socket so the fingers can reach the plug.
    _box(stage, "/Asset/base", (0.065, 0.080, 0.006), (0.040, 0, 0.003 - SOCKET_FACE[2]), grey)
    post_height = SOCKET_FACE[2] + 0.027
    _box(stage, "/Asset/post", (0.032, 0.046, post_height), (0.038, 0, 0.006 + post_height / 2 - SOCKET_FACE[2]), grey)
    _box(stage, "/Asset/rear_stop", (0.008, 0.012, 0.016), (0.014, 0, 0), grey)
    for sign in (-1, 1):
        _box(
            stage,
            f"/Asset/bolt_{sign + 1}",
            (0.008, 0.008, 0.003),
            (0.040, sign * 0.031, 0.0075 - SOCKET_FACE[2]),
            silver,
        )
    # In world coordinates, the wide USB-C axis is y and the narrow axis is vertical.
    half_straight = (SOCKET_WIDTH - SOCKET_HEIGHT) / 2
    wall = 0.0005
    radius = SOCKET_HEIGHT / 2 + wall / 2
    for sign in (-1, 1):
        _box(
            stage, f"/Asset/shell_flat_{sign + 1}", (0.010, 2 * half_straight, wall), (0.005, 0, sign * radius), silver
        )
        for i in range(12):
            angle = -math.pi / 2 + (i + 0.5) * math.pi / 12 + (math.pi if sign < 0 else 0)
            # Tangential boxes provide a rounded opening without a convex hull sealing it.
            center = (0.005, -(sign * half_straight + radius * math.cos(angle)), radius * math.sin(angle))
            prim = _box(
                stage,
                f"/Asset/shell_round_{sign + 1}_{i}",
                (0.010, 2 * radius * math.tan(math.pi / 24), wall),
                center,
                silver,
            )
            geom = UsdGeom.Xformable(prim)
            ops = geom.GetOrderedXformOps()
            rotation = geom.AddRotateXOp()
            rotation.Set(90 - math.degrees(angle))
            geom.SetXformOpOrder([ops[0], rotation, ops[1]])
    _box(stage, "/Asset/tongue", (0.0075, 0.0060, 0.0006), (0.00625, 0, 0), (0.04, 0.05, 0.07))
    for sign in (-1, 1):
        for i in range(12):
            _box(
                stage,
                f"/Asset/contact_{sign + 1}_{i}",
                (0.004, 0.00025, 0.00004),
                (0.005, -(i - 5.5) * 0.0005, sign * 0.00032),
                (0.83, 0.65, 0.18),
                False,
            )
    _box(stage, "/Asset/target_led", (0.001, 0.005, 0.005), (0.0215, -0.015, 0.021), (0.08, 0.85, 0.28), False)
    stage.GetRootLayer().Save()
    _export_plug(directory / "usb_c_plug.usda")
    return directory


def _fit_insert_connector(builder, world_index, position, orientation):
    """Reuse the 4 g plug while opening its central cavity for the receptacle tongue."""
    import warp as wp
    from newton import ShapeFlags

    _fit_reference_connector(builder, world_index, position, orientation, connector_offset=CONNECTOR_OFFSET)
    ends = [
        i for i, label in enumerate(builder.body_label) if "/Cable/" in label and builder.body_world[i] == world_index
    ]
    if not ends:
        return
    end = ends[-1]
    # The compliant pre-grasp substitutes for housing contact only, never cable contact.
    cable_shapes = {
        i
        for i, label in enumerate(builder.shape_label)
        if builder.shape_body[i] in ends and "mesh_edge_capsule_" in label
    }
    gripper_shapes = {i for i, label in enumerate(builder.shape_label) if "/Robotiq_2F_85/" in label}
    builder.shape_collision_filter_pairs[:] = [
        (a, b)
        for a, b in builder.shape_collision_filter_pairs
        if not ((a in cable_shapes and b in gripper_shapes) or (b in cable_shapes and a in gripper_shapes))
    ]
    for i, label in enumerate(builder.shape_label):
        if "/Socket/" in label:
            builder.shape_flags[i] |= ShapeFlags.VISIBLE
        if builder.shape_body[i] == end and label.endswith("/usb_c_tongue"):
            # Retain its lumped electronics mass, but leave the physical mating cavity empty.
            builder.shape_flags[i] &= ~(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.VISIBLE)
    for sign in (-1, 1):
        for i in range(12):
            material = builder.default_shape_cfg.copy()
            material.density = 0.0
            shape = builder.add_shape_box(
                end,
                xform=wp.transform(((i - 5.5) * 0.0005, sign * 0.00092, CONNECTOR_OFFSET + 0.0125), wp.quat_identity()),
                hx=0.000125,
                hy=0.000025,
                hz=0.0018,
                cfg=material,
                color=(0.83, 0.65, 0.18),
                label=builder.body_label[end] + f"/insert_contact_{sign + 1}_{i}",
            )
            builder.shape_flags[shape] &= ~ShapeFlags.COLLIDE_SHAPES


@clone
def spawn_insert_cable(
    prim_path: str,
    cfg: sim_utils.CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the existing VBD cable with a hollow USB-C plug for physical insertion."""
    from isaaclab_newton.physics import NewtonManager

    if _fit_insert_connector not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_fit_insert_connector)
    return sim_utils.spawn_cable(prim_path, cfg, translation, orientation)


def _export_plug(path: Path) -> None:
    """Export the runtime plug primitives as a reusable 4 g USD asset."""
    import warp as wp
    from newton import ModelBuilder, ShapeFlags

    builder = ModelBuilder()
    builder.begin_world()
    for i in range(SEGMENTS):
        end = builder.add_body(
            mass=CABLE_BODY_MASS / SEGMENTS,
            inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8),
            label=f"/Export/Cable/segment_{i}",
        )
    _fit_insert_connector(builder, 0, None, None)
    builder.end_world()
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(CONNECTOR_MASS)
    for i, body in enumerate(builder.shape_body):
        flags = builder.shape_flags[i]
        if body != end or not flags & ShapeFlags.VISIBLE:
            continue
        name = builder.shape_label[i].rsplit("/", 1)[-1].replace("-", "neg_")
        transform = builder.shape_transform[i]
        position, q = wp.transform_get_translation(transform), wp.transform_get_rotation(transform)
        prim = _box(
            stage,
            "/Asset/" + name,
            tuple(2 * v for v in builder.shape_scale[i]),
            tuple(position),
            builder.shape_color[i],
            bool(flags & ShapeFlags.COLLIDE_SHAPES),
        )
        geom = UsdGeom.Xformable(prim)
        ops = geom.GetOrderedXformOps()
        rotate = geom.AddOrientOp()
        rotate.Set(Gf.Quatf(q[3], Gf.Vec3f(q[0], q[1], q[2])))
        geom.SetXformOpOrder([ops[0], rotate, ops[1]])
    stage.GetRootLayer().Save()
