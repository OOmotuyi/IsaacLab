# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build marker/bin USD assets and an experimental articulated MicroDuck mouth."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from prepare import repair_material_bindings

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

MARKER_LENGTH = 0.100
MARKER_RADIUS = 0.009
MARKER_MASS = 0.012
BIN_INNER = (0.180, 0.160)
BIN_HEIGHT = 0.045
BIN_WALL = 0.004
BIN_FLOOR = 0.003
MOUTH_CLOSED = math.radians(-5)
MOUTH_OPEN = math.radians(30)


def _stage(path: Path, name: str) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def _color(geometry: UsdGeom.Gprim, rgb: tuple[float, float, float]) -> None:
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _material(stage: Usd.Stage, root: Usd.Prim, friction: float) -> None:
    material = UsdShade.Material.Define(stage, root.GetPath().AppendChild("Material"))
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose="physics")


def build_objects(destination: Path) -> None:
    """Create a capped marker [m, kg] and a static open bin with separate wall colliders."""
    stage, root = _stage(destination / "marker.usda", "Marker")
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(MARKER_MASS)
    radial = MARKER_MASS * (3 * MARKER_RADIUS**2 + MARKER_LENGTH**2) / 12
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(radial, MARKER_MASS * MARKER_RADIUS**2 / 2, radial))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    for name, length, y, radius, color in (
        ("Barrel", 0.074, -0.013, 0.0085, (0.94, 0.94, 0.92)),
        ("Cap", 0.026, 0.037, 0.009, (0.02, 0.22, 0.95)),
        ("Band", 0.006, 0.020, 0.0091, (0.03, 0.04, 0.07)),
    ):
        cylinder = UsdGeom.Cylinder.Define(stage, f"/Marker/{name}")
        cylinder.CreateAxisAttr("Y")
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(length)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(0, y, 0))
        _color(cylinder, color)
    collision = UsdGeom.Cylinder.Define(stage, "/Marker/Collision")
    collision.CreateAxisAttr("Y")
    collision.CreateRadiusAttr(MARKER_RADIUS)
    collision.CreateHeightAttr(MARKER_LENGTH)
    collision.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    _material(stage, root, 0.8)
    stage.GetRootLayer().Save()

    stage, root = _stage(destination / "bin.usda", "Bin")
    x, y = BIN_INNER
    w, h, f = BIN_WALL, BIN_HEIGHT, BIN_FLOOR
    parts = [
        ("Floor", (x + 2 * w, y + 2 * w, f), (0, 0, f / 2)),
        ("Front", (w, y + 2 * w, h), (-(x + w) / 2, 0, f + h / 2)),
        ("Back", (w, y + 2 * w, h), ((x + w) / 2, 0, f + h / 2)),
        ("Left", (x, w, h), (0, (y + w) / 2, f + h / 2)),
        ("Right", (x, w, h), (0, -(y + w) / 2, f + h / 2)),
    ]
    for name, size, position in parts:
        cube = UsdGeom.Cube.Define(stage, f"/Bin/{name}")
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3d(*size))
        _color(cube, (0.03, 0.42, 0.40))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _material(stage, root, 0.7)
    stage.GetRootLayer().Save()


def _parallel(position: np.ndarray) -> np.ndarray:
    return np.dot(position, position) * np.eye(3) - np.outer(position, position)


def build_mouth(source: Path, destination: Path) -> Path:
    """Split the lower beak into a hinge; preserve closed-pose total mass and inertia.

    The pivot [m], jaw mass [kg], contact pads, and drive are prototype estimates.
    Joint travel [-5, 30] [deg] follows the official hardware mouth controller.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((source.parent / "assets").resolve()))
    head = root.find(".//body[@name='jaw_soft']")
    pivot = np.array([0.003, 0.0, -0.018])
    center = np.array([-0.010, 0.0, -0.055])
    lower_mass = 0.015
    lower_dims = np.array([0.008, 0.080, 0.050])
    lower_inertia = lower_mass / 12 * (np.sum(lower_dims**2) - lower_dims**2)
    inertial = head.find("inertial")
    original_mass = float(inertial.get("mass"))
    original_center = np.fromstring(inertial.get("pos"), sep=" ")
    xx, yy, zz, xy, xz, yz = np.fromstring(inertial.get("fullinertia"), sep=" ")
    original_inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    head_mass = original_mass - lower_mass
    head_center = (original_mass * original_center - lower_mass * center) / head_mass
    head_inertia = (
        original_inertia
        + original_mass * _parallel(original_center)
        - np.diag(lower_inertia)
        - lower_mass * _parallel(center)
        - head_mass * _parallel(head_center)
    )
    assert np.linalg.eigvalsh(head_inertia).min() > 0
    inertial.set("mass", str(head_mass))
    inertial.set("pos", " ".join(map(str, head_center)))
    inertial.set(
        "fullinertia",
        " ".join(
            map(
                str,
                [
                    *np.diag(head_inertia),
                    head_inertia[0, 1],
                    head_inertia[0, 2],
                    head_inertia[1, 2],
                ],
            )
        ),
    )
    lower = ET.SubElement(head, "body", name="lower_beak", pos=" ".join(map(str, pivot)))
    ET.SubElement(
        lower,
        "joint",
        name="mouth",
        type="hinge",
        axis="0 1 0",
        range=f"{MOUTH_CLOSED} {MOUTH_OPEN}",
        damping="0.02",
        armature="0.0005",
    )
    ET.SubElement(
        lower,
        "inertial",
        mass=str(lower_mass),
        pos=" ".join(map(str, center - pivot)),
        diaginertia=" ".join(map(str, lower_inertia)),
    )
    for geom in list(head.findall("geom")):
        mesh = geom.get("mesh")
        if mesh in ("jaw", "jaw_soft"):
            head.remove(geom)
            if geom.get("class") == "visual":
                geom.set("pos", " ".join(map(str, np.fromstring(geom.get("pos"), sep=" ") - pivot)))
                lower.append(geom)
        elif mesh == "bottom_head_shell" and geom.get("class") == "collision":
            # Its full convex hull bridges the mouth aperture. Use a rear-shell proxy.
            head.remove(geom)
    for body, name, position, size in [
        (head, "upper_mouth_pad", (-0.0055, 0, -0.063), (0.0015, 0.041, 0.014)),
        (lower, "lower_mouth_pad", tuple(np.array([-0.013, 0, -0.063]) - pivot), (0.002, 0.041, 0.014)),
        (lower, "lower_mouth_lip", tuple(np.array([-0.008, 0, -0.079]) - pivot), (0.004, 0.041, 0.002)),
        (head, "rear_lower_shell", (-0.005, 0, 0.003), (0.009, 0.040, 0.032)),
    ]:
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="box",
            **{"class": "collision"},
            pos=" ".join(map(str, position)),
            size=" ".join(map(str, size)),
            friction="0.8 0.005 0.0001",
            condim="4",
        )
    # A visible rounded-mouth surrogate: the lip prevents a cylinder rolling out of flat pads.
    for geom in lower.findall("geom"):
        if geom.get("name") == "lower_mouth_lip":
            visual = ET.SubElement(lower, "geom", **geom.attrib)
            visual.set("name", "lower_mouth_lip_visual")
            visual.set("class", "visual")
            visual.set("rgba", "1 0.45 0.015 1")
            break
    ET.SubElement(
        root.find("actuator"), "position", name="mouth", joint="mouth", kp="3", kv="0.06", forcerange="-0.15 0.15"
    )
    # Exported sensor names still reference the unchanged 14 policy joints.
    output = destination / "robot_marker.xml"
    ET.indent(tree)
    tree.write(output, encoding="unicode")
    return output


def main() -> None:
    """Generate assets separately from the released robot and retain their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_dir", type=Path, default=Path("logs/microduck/assets"))
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/marker_pick/assets_lip"))
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    build_objects(destination)
    xml = build_mouth(args.asset_dir.resolve() / "source/robot/robot_groundcontact.xml", destination)
    converter = Path(__file__).resolve().parents[2] / "tools/convert_mjcf.py"
    subprocess.run([sys.executable, str(converter), str(xml), str(destination / "robot_marker.usd")], check=True)
    robot_usd = destination / "robot_marker/robot_marker.usda"
    repair_material_bindings(robot_usd)
    provenance = {
        "source_robot": json.loads((args.asset_dir / "sources.json").read_text()),
        "official_mouth_controller": "https://github.com/pollen-robotics/microduck/blob/"
        "344925c9f8fa031f85428a305b1e8ec2eaae29c1/duck-control/src/model.rs",
        "mouth_range_rad": [MOUTH_CLOSED, MOUTH_OPEN],
        "prototype_estimates": [
            "jaw pivot",
            "jaw mass distribution",
            "rigid contact pads and retaining lip",
            "mouth actuator gains",
        ],
        "marker": {"length_m": MARKER_LENGTH, "radius_m": MARKER_RADIUS, "mass_kg": MARKER_MASS},
        "bin": {"inner_size_m": list(BIN_INNER), "height_m": BIN_HEIGHT},
        "grasp_attachment": False,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Created {destination / 'marker.usda'}, {destination / 'bin.usda'}, {robot_usd}", flush=True)


if __name__ == "__main__":
    main()
