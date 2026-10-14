# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured fixture layout and approximate reference-image connector geometry."""

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone

from isaaclab_tasks.contrib.usb_cable_routing.assets import ANCHOR, CONNECTOR_OFFSET, _box, _fit_usb_connector

GUIDE_OFFSETS = ((0.120, 0.0), (0.260, 0.100), (0.400, 0.0))
"""Guide center offsets from the fixed cable endpoint in the tabletop plane [m]."""
GUIDES = tuple((ANCHOR[0] + x, ANCHOR[1] + y, 0.0) for x, y in GUIDE_OFFSETS)
GUIDE_YAWS = tuple(math.radians(degrees) for degrees in (10, 0, -10))
"""Guide passage yaw angles about vertical [rad]; zero follows table +x."""
CRADLE_HEIGHT = 0.010
"""Height of every channel floor above the tabletop [m]."""
CHANNEL_WIDTH = 0.018
"""Clear distance between the upright guide walls [m]."""
PASSAGE_LENGTH = 0.020
"""Length of the two guide walls along the passage axis [m]."""
WALL_TOP = 0.040
"""Height of the upright wall tops above the tabletop [m]."""
LIP_OVERHANG = 0.004
"""Horizontal inward projection from each guide wall at its tip [m]."""
LIP_THICKNESS = 0.004
"""Vertical thickness of each fixed horizontal lip [m]."""
LIP_UNDERSIDE = WALL_TOP - LIP_THICKNESS
"""Height beneath the right-angle retaining lips [m]."""
ENTRY_WIDTH = CHANNEL_WIDTH - 2 * LIP_OVERHANG
"""Clear opening between the inward lips [m]."""
FLARE_START = LIP_UNDERSIDE
"""Deprecated alias for LIP_UNDERSIDE [m]; the tips are now perpendicular inward lips."""
FLARE_OFFSET = LIP_OVERHANG
"""Deprecated alias for LIP_OVERHANG [m]; projection is now inward."""
GUIDE_COLOR_NAMES = ("red", "yellow", "green")
GUIDE_COLORS = ((0.9, 0.10, 0.055), (0.98, 0.67, 0.025), (0.055, 0.63, 0.22))
"""Guide colors in routing order, as linear RGB values."""
BLUE = (0.035, 0.25, 0.65)


def generate_assets(directory: Path | None = None) -> Path:
    """Write the upright U-guide with fixed 90-degree inward lips, in [m]."""
    directory = directory or Path(__file__).parent / "assets"
    directory.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(directory / "guide.usda"))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.03)
    white = (0.88, 0.89, 0.86)
    _box(stage, "/Asset/base", (0.028, 0.068, 0.006), (0, 0, 0.003), white)
    _box(stage, "/Asset/cradle", (0.022, 0.030, 0.004), (0, 0, 0.008), white)
    for sign, label in ((-1, "left"), (1, "right")):
        _box(
            stage,
            "/Asset/" + label,
            (PASSAGE_LENGTH, 0.006, WALL_TOP - CRADLE_HEIGHT),
            (0, sign * 0.012, (WALL_TOP + CRADLE_HEIGHT) / 2),
            white,
        )
        _box(
            stage,
            "/Asset/lip_" + label,
            (PASSAGE_LENGTH, LIP_OVERHANG, LIP_THICKNESS),
            (0, sign * (CHANNEL_WIDTH / 2 - LIP_OVERHANG / 2), WALL_TOP - LIP_THICKNESS / 2),
            white,
        )
        _box(stage, "/Asset/bolt_" + label, (0.005, 0.005, 0.002), (0, sign * 0.026, 0.007), (0.5, 0.52, 0.54), False)
    # Direct authored colors override variant opinions, so move colors into variants.
    for part in ("base", "cradle", "left", "right", "lip_left", "lip_right"):
        prim_path = "/Asset/" + part
        UsdGeom.Gprim(stage.GetPrimAtPath(prim_path)).GetDisplayColorAttr().Clear()
        UsdShade.Shader.Get(stage, prim_path + "/Material/Shader").GetInput("diffuseColor").GetAttr().Clear()
    variants = root.GetPrim().GetVariantSets().AddVariantSet("Color")
    for name, color in zip((*GUIDE_COLOR_NAMES, "white"), (*GUIDE_COLORS, white), strict=True):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            for part in ("base", "cradle", "left", "right", "lip_left", "lip_right"):
                prim_path = "/Asset/" + part
                UsdGeom.Gprim(stage.GetPrimAtPath(prim_path)).CreateDisplayColorAttr([Gf.Vec3f(*color)])
                UsdShade.Shader.Get(stage, prim_path + "/Material/Shader").GetInput("diffuseColor").Set(
                    Gf.Vec3f(*color)
                )
    variants.SetVariantSelection("white")
    stage.GetRootLayer().Save()
    return directory


def _fit_reference_connector(
    builder, world_index, position, orientation, *, connector_offset: float = CONNECTOR_OFFSET
):
    """Fit the shared 4 g connector with an axial offset [m]."""
    _fit_usb_connector(builder, world_index, position, orientation, connector_offset=connector_offset)


@clone
def spawn_reference_cable(
    prim_path: str,
    cfg: sim_utils.CableCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn the native Newton cable with the reference-style blue connector."""
    from isaaclab_newton.physics import NewtonManager

    if _fit_reference_connector not in NewtonManager._per_world_builder_hooks:
        NewtonManager._per_world_builder_hooks.append(_fit_reference_connector)
    return sim_utils.spawn_cable(prim_path, cfg, translation, orientation)
