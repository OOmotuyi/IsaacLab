# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric routing fixtures and the maintained UR10e/Robotiq USD asset."""

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

CABLE_LENGTH = 0.6
CABLE_DIAMETER = 0.004
SEGMENTS = 40
SEGMENT_LENGTH = CABLE_LENGTH / SEGMENTS
ANCHOR = (0.18, -0.15, 0.022)
GUIDES = ((0.30, -0.15, 0.0), (0.44, -0.14, 0.0), (0.58, -0.15, 0.0))
CONNECTOR_OFFSET = SEGMENT_LENGTH / 2 + 0.015
"""Plug center offset from the final rod center [m]; strain relief starts at the rod end."""
GRASP_LOCAL_Z = CONNECTOR_OFFSET - 0.005
"""Grasp point on the rigid housing along the last rod local z-axis [m]."""
START = (0.580, 0.080, 0.350 - CONNECTOR_OFFSET)
START_DIRECTION = (0.0, 0.0, 1.0)
TOOL_QUAT = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
"""Initial wrist orientation (xyzw), approaching the upright plug from the side."""
HOME_ARM_JOINT_POS = (0.736648, -1.065324, 1.536466, -0.471142, 0.736648, -math.pi)
"""Validated reset arm joints [rad], with the wrist above the hand."""
BLUE = (0.035, 0.25, 0.65)
TCP_OFFSET = (0.0, 0.0, 0.143)
GRIPPER_POSITION = 0.730
ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _box(stage, path, size, position, color, collision=True):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    geom.AddTranslateOp().Set(Gf.Vec3d(*position))
    geom.AddScaleOp().Set(Gf.Vec3f(*size))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mat = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return geom.GetPrim()


def generate_assets(directory: Path | None = None) -> Path:
    """Write the open guide and anchor USD assets, with all dimensions in [m]."""
    directory = directory or Path(__file__).parent / "assets"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("guide", "anchor"):
        stage = Usd.Stage.CreateNew(str(directory / f"{name}.usda"))
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        if name == "guide":
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.03)
            orange = (0.95, 0.28, 0.025)
            _box(stage, "/Asset/base", (0.026, 0.074, 0.010), (0, 0, 0.005), orange)
            for sign, label in ((-1, "left"), (1, "right")):
                _box(stage, "/Asset/" + label, (0.018, 0.010, 0.048), (0, sign * 0.026, 0.030), orange)
                _box(
                    stage,
                    "/Asset/bolt_" + label,
                    (0.005, 0.005, 0.002),
                    (0.009, sign * 0.030, 0.011),
                    (0.4, 0.4, 0.42),
                    False,
                )
        else:
            _box(stage, "/Asset/base", (0.046, 0.048, 0.006), (-0.022, 0, 0.003), (0.5, 0.52, 0.54))
            _box(stage, "/Asset/clamp", (0.010, 0.028, 0.026), (-0.030, 0, 0.018), (0.45, 0.47, 0.49))
            _box(stage, "/Asset/usb_a_shell", (0.016, 0.012, 0.0045), (-0.044, 0, 0.022), (0.75, 0.77, 0.8))
            _box(stage, "/Asset/usb_a_housing", (0.025, 0.015, 0.008), (-0.024, 0, 0.022), (0.025, 0.025, 0.03))
            _box(stage, "/Asset/strain_relief", (0.012, 0.006, 0.006), (-0.006, 0, 0.022), (0.035, 0.035, 0.04))
        stage.GetRootLayer().Save()
    return directory


@clone
def spawn_robot(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the combined USD and fill missing Robotiq inertials for Newton.

    Missing link masses [kg] sum with the authored links to approximately 0.90 kg.
    Their bounding-box inertias are estimates, not manufacturer calibration data.
    """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    estimates = {"base_link": 0.61, "left_outer_knuckle": 0.04, "right_outer_knuckle": 0.04}
    for link in Usd.PrimRange(prim):
        if "/Robotiq_2F_85/" not in str(link.GetPath()) or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass = UsdPhysics.MassAPI(link)
        if mass and mass.GetMassAttr().Get() and mass.GetMassAttr().Get() > 0:
            continue
        value = estimates[link.GetName()]
        box = bounds.ComputeLocalBound(link).ComputeAlignedRange()
        size = box.GetSize()
        mass = UsdPhysics.MassAPI.Apply(link)
        mass.CreateMassAttr(value)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*box.GetMidpoint()))
        mass.CreateDiagonalInertiaAttr(
            Gf.Vec3f(*[value * (size[(i + 1) % 3] ** 2 + size[(i + 2) % 3] ** 2) / 12 for i in range(3)])
        )
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    return prim


def _fit_usb_connector(builder, world_index, position, orientation, *, connector_offset: float = CONNECTOR_OFFSET):
    """Normalize cable mass and add a rigid 4 g plug with an optional axial offset [m]."""
    import warp as wp
    from newton import ShapeFlags

    from .parameters import CABLE_BODY_MASS, CONNECTOR_MASS, DYNAMIC_FRICTION

    for i, label in enumerate(builder.shape_label):
        if "/Guide" in label:
            builder.shape_flags[i] |= ShapeFlags.VISIBLE
    bodies = [
        i for i, label in enumerate(builder.body_label) if "/Cable/" in label and builder.body_world[i] == world_index
    ]
    if not bodies:
        return
    for i, body in enumerate(builder.shape_body):
        if body in bodies:
            builder.shape_color[i] = BLUE
    if len(bodies) != SEGMENTS:
        raise ValueError(f"Expected {SEGMENTS} cable segments, got {len(bodies)}")
    # Scale inertia with mass so importer volume conventions cannot change the mass budget.
    for body in bodies:
        scale = (CABLE_BODY_MASS / SEGMENTS) / builder.body_mass[body]
        builder.body_mass[body] *= scale
        builder.body_inv_mass[body] /= scale
        builder.body_inertia[body] *= scale
        builder.body_inv_inertia[body] /= scale
    end = bodies[-1]
    # The effective component densities include unmodeled internal electronics and overmolding.
    parts = [
        ("blue_usb_c_housing", (0, 0, 0), (0.005, 0.003, 0.009), 0.0, 1100.0, BLUE),
        ("blue_strain_relief", (0, 0, -0.012), (0.003, 0.003, 0.003), 0.0, 1100.0, BLUE),
        ("usb_c_tongue", (0, 0, 0.012), (0.003, 0.00035, 0.0025), 0.0, 1100.0, (0.04, 0.05, 0.06)),
    ]
    silver = (0.72, 0.75, 0.80)
    for sign in (-1, 1):
        parts.append(
            (f"usb_c_flat_{sign}", (0, sign * 0.001125, 0.0125), (0.0029, 0.000175, 0.0035), 0.0, 7800.0, silver)
        )
        # Eight facets on each semicircle approximate the USB-C shell's rounded ends.
        for i in range(8):
            theta = -math.pi / 2 + (i + 0.5) * math.pi / 8 + (math.pi if sign < 0 else 0)
            center = (sign * 0.0029 + 0.001125 * math.cos(theta), 0.001125 * math.sin(theta), 0.0125)
            parts.append(
                (
                    f"usb_c_round_{sign}_{i}",
                    center,
                    (0.001125 * math.sin(math.pi / 16), 0.000175, 0.0035),
                    theta + math.pi / 2,
                    7800.0,
                    silver,
                )
            )
    raw_mass = sum(8 * math.prod(half_size) * density for _, _, half_size, _, density, _ in parts)
    for name, center, half_size, yaw, density, color in parts:
        material = builder.default_shape_cfg.copy()
        material.density = density * CONNECTOR_MASS / raw_mass
        material.mu = DYNAMIC_FRICTION
        builder.add_shape_box(
            end,
            xform=wp.transform(
                (center[0], center[1], center[2] + connector_offset), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw)
            ),
            hx=half_size[0],
            hy=half_size[1],
            hz=half_size[2],
            cfg=material,
            color=color,
            label=builder.body_label[end] + "/" + name,
        )
    wrist = [
        i
        for i, label in enumerate(builder.body_label)
        if "/Robotiq_2F_85/" in label and builder.body_world[i] == world_index
    ]
    tool_shapes = [i for i, body in enumerate(builder.shape_body) if body in wrist]
    # The compliant grasp replaces contact on the rigid plug only.
    held_shapes = [
        i
        for i, body in enumerate(builder.shape_body)
        if body == end and "mesh_edge_capsule_" not in builder.shape_label[i]
    ]
    for a in tool_shapes:
        for b in held_shapes:
            builder.shape_collision_filter_pairs.append((min(a, b), max(a, b)))


@clone
def spawn_usb_cable(
    prim_path: str,
    cfg: sim_utils.CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a native elastic cable and register its passive USB-C fitting."""
    from isaaclab_newton.physics import NewtonManager

    if _fit_usb_connector not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_fit_usb_connector)
    return sim_utils.spawn_cable(prim_path, cfg, translation, orientation)
