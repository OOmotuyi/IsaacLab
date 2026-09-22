# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author a cutaway child's bedroom and a sewn, hollow cloth sock in meters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from bin_geometry import BASKET_GEOMETRY
from marker_assets import _color, _material, _stage

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _box(stage: Usd.Stage, path: str, size: tuple, position: tuple, color: tuple, collision: bool = True) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    _color(cube, color)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def build_room(destination: Path, photorealistic: bool = False) -> None:
    """Create a 2.8 × 2.4 [m] cutaway bedroom with a bed and chair."""
    stage, root = _stage(destination / "children_room.usda", "Room")
    _material(stage, root, 0.7)
    wood = (0.72, 0.48, 0.27)
    cream = (0.91, 0.84, 0.69)
    blue = (0.18, 0.48, 0.66)
    _box(stage, "/Room/Floor", (2.8, 2.4, 0.025), (0.1, 0, -0.0123), cream, False)
    for i in range(15):
        _box(stage, f"/Room/FloorSeam{i}", (2.8, 0.002, 0.001), (0.1, -1.12 + i * 0.16, 0.0004), wood, False)
    _box(stage, "/Room/BackWall", (2.8, 0.06, 1.25), (0.1, 1.2, 0.625), (0.73, 0.86, 0.84))
    _box(stage, "/Room/SideWall", (0.06, 2.4, 1.25), (-1.3, 0, 0.625), (0.91, 0.79, 0.72))
    _box(stage, "/Room/Baseboard", (2.8, 0.025, 0.06), (0.1, 1.16, 0.03), (0.97, 0.95, 0.9))
    # Child-sized bed: the mattress and bedding are static scene geometry.
    bx, by = -0.42, 0.72
    _box(stage, "/Room/Bed/Frame", (1.5, 0.72, 0.12), (bx, by, 0.21), wood)
    _box(stage, "/Room/Bed/Mattress", (1.43, 0.68, 0.11), (bx, by, 0.325), (0.97, 0.93, 0.83))
    _box(stage, "/Room/Bed/Duvet", (1.00, 0.69, 0.04), (bx + 0.20, by, 0.4), blue)
    for i in range(7):
        _box(
            stage,
            f"/Room/Bed/Stripe{i}",
            (0.055, 0.695, 0.003),
            (bx - 0.25 + i * 0.13, by, 0.422),
            (0.76, 0.89, 0.89),
            False,
        )
    _box(stage, "/Room/Bed/Pillow", (0.28, 0.5, 0.075), (bx - 0.51, by, 0.405), (1.0, 0.83, 0.49))
    _box(stage, "/Room/Bed/Headboard", (0.055, 0.79, 0.61), (bx - 0.75, by, 0.355), wood)
    for i, (dx, dy) in enumerate([(-0.67, -0.28), (-0.67, 0.28), (0.67, -0.28), (0.67, 0.28)]):
        _box(stage, f"/Room/Bed/Leg{i}", (0.055, 0.055, 0.18), (bx + dx, by + dy, 0.09), wood)
    cx, cy = 0.97, 0.62
    _box(stage, "/Room/Chair/Seat", (0.32, 0.32, 0.045), (cx, cy, 0.285), (0.89, 0.57, 0.25))
    _box(stage, "/Room/Chair/Back", (0.32, 0.045, 0.27), (cx, cy + 0.14, 0.43), (0.89, 0.57, 0.25))
    for i, (dx, dy) in enumerate([(-0.12, -0.12), (-0.12, 0.12), (0.12, -0.12), (0.12, 0.12)]):
        _box(stage, f"/Room/Chair/Leg{i}", (0.035, 0.035, 0.27), (cx + dx, cy + dy, 0.135), wood)
    # Wall decorations and a window make the room legible without external assets.
    _box(stage, "/Room/Window/Frame", (0.65, 0.035, 0.55), (0.59, 1.155, 0.88), cream, False)
    _box(stage, "/Room/Window/Glass", (0.58, 0.015, 0.48), (0.59, 1.13, 0.88), (0.49, 0.76, 0.91), False)
    _box(stage, "/Room/Window/Vertical", (0.025, 0.02, 0.49), (0.59, 1.115, 0.88), cream, False)
    _box(stage, "/Room/Window/Horizontal", (0.59, 0.02, 0.025), (0.59, 1.115, 0.88), cream, False)
    for i, rgb in enumerate([(0.91, 0.52, 0.40), (0.97, 0.76, 0.29), (0.27, 0.63, 0.63)]):
        _box(stage, f"/Room/Picture{i}/Frame", (0.18, 0.025, 0.22), (-0.9 + i * 0.26, 1.15, 0.92), cream, False)
        _box(stage, f"/Room/Picture{i}/Print", (0.14, 0.008, 0.18), (-0.9 + i * 0.26, 1.13, 0.92), rgb, False)
    if photorealistic:
        from bedroom_materials import enrich_room

        texture = destination / "textures" / "oak_albedo.png"
        if not texture.is_file():
            raise FileNotFoundError(texture)
        enrich_room(stage, texture)
    stage.GetRootLayer().Save()


ROOM_BIN_INNER = (0.18, 0.52)


def build_bin(destination: Path) -> None:
    """Create a lidless tapered wire basket with a 0.13 [m] rim."""
    from bedroom_materials import bind, material

    stage, root = _stage(destination / "cleanup_bin.usda", "Bin")
    geometry = BASKET_GEOMETRY
    metal = material(stage, "BlackPowderCoatedSteel", (0.012, 0.014, 0.017), 0.32, 0.65)

    def wire(name, start, end, radius):
        delta = Gf.Vec3d(*end) - Gf.Vec3d(*start)
        capsule = UsdGeom.Capsule.Define(stage, "/Bin/" + name)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(delta.GetLength())
        capsule.CreateAxisAttr("Z")
        capsule.AddTranslateOp().Set((Gf.Vec3d(*start) + Gf.Vec3d(*end)) / 2)
        capsule.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), delta).GetQuat())
        _color(capsule, (0.018, 0.020, 0.024))
        bind(capsule.GetPrim(), metal)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

    floor = UsdGeom.Cylinder.Define(stage, "/Bin/Bottom")
    floor.CreateRadiusAttr(geometry.bottom_radius)
    floor.CreateHeightAttr(geometry.floor_height)
    floor.AddTranslateOp().Set((0, 0, geometry.floor_height / 2))
    bind(floor.GetPrim(), metal)
    _color(floor, (0.018, 0.020, 0.024))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    rings = [
        ("BaseRing", geometry.floor_height, geometry.bottom_radius, geometry.rim_radius),
        ("Rim", geometry.height - geometry.rim_radius, geometry.top_radius, geometry.rim_radius),
    ]
    # Horizontal wires match the reference and keep the marker from passing through the weave.
    for level in range(1, 7):
        fraction = level / 7
        rings.append(
            (
                f"Horizontal_{level}",
                geometry.floor_height + fraction * (geometry.height - geometry.rim_radius - geometry.floor_height),
                geometry.bottom_radius + fraction * (geometry.top_radius - geometry.bottom_radius),
                geometry.wire_radius,
            )
        )
    for ring, z, radius, wire_radius in rings:
        for j in range(48):
            angles = [2 * math.pi * k / 48 for k in (j, j + 1)]
            endpoints = [(radius * math.cos(a), radius * math.sin(a), z) for a in angles]
            wire(f"{ring}_{j}", *endpoints, wire_radius)
    for direction in (-1, 1):
        for strand in range(16):
            points = []
            for j in range(7):
                t = j / 6
                angle = 2 * math.pi * strand / 16 + direction * t * math.pi / 3
                radius = geometry.bottom_radius + t * (geometry.top_radius - geometry.bottom_radius)
                points.append(
                    (
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        geometry.floor_height + t * (geometry.height - geometry.rim_radius - geometry.floor_height),
                    )
                )
            for j in range(6):
                wire(
                    f"Weave_{'L' if direction < 0 else 'R'}_{strand}_{j}",
                    points[j],
                    points[j + 1],
                    geometry.wire_radius,
                )
    _material(stage, root, 0.7)
    stage.GetRootLayer().Save()
    geometry.save(destination)


def build_sock(destination: Path) -> None:
    """Create a 12 cm hollow sock, open at the cuff and closed at the toe.

    Triangles describe textile surface, with two separated layers and no rigid
    interior. The launch script assigns the Newton surface material.
    """
    rings, sides = 25, 16
    points = []
    for i in range(rings):
        t = i / (rings - 1)
        # The heel bends sideways like a sock laid on its side.
        x = 0.120 * t
        y = 0.020 * max(0.0, (t - 0.57) / 0.43)
        width = 0.014 * (1.0 - 0.12 * t)
        thickness = 0.006
        if t > 0.88:
            taper = math.sqrt(max(0.04, 1 - ((t - 0.88) / 0.13) ** 2))
            width *= taper
            thickness *= taper
        for j in range(sides):
            theta = 2 * math.pi * j / sides
            points.append((x, y + width * math.cos(theta), thickness * math.sin(theta)))
    faces = []
    colors = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = (i + 1) * sides + j, (i + 1) * sides + (j + 1) % sides
            faces.extend([(a, b, c), (b, d, c)])
            color = (
                (0.98, 0.54, 0.37) if i < 4 or i > 21 else ((0.25, 0.63, 0.67) if (i // 3) % 2 else (0.96, 0.91, 0.76))
            )
            colors.extend([color, color])
    toe = len(points)
    points.append(tuple(np.mean(points[-sides:], axis=0)))
    for j in range(sides):
        faces.append(((rings - 1) * sides + j, (rings - 1) * sides + (j + 1) % sides, toe))
        colors.append((0.98, 0.54, 0.37))
    stage, root = _stage(destination / "child_sock.usda", "Sock")
    mesh = UsdGeom.Mesh.Define(stage, "/Sock/Textile")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*c) for c in colors])
    mesh.GetDisplayColorPrimvar().SetInterpolation("uniform")
    stage.GetRootLayer().Save()
    (destination / "sock_geometry.json").write_text(
        json.dumps(
            {
                "vertices": len(points),
                "triangles": len(faces),
                "open_cuff": True,
                "length_m": 0.120,
                "width_m": 0.028,
                "rigid_interior": False,
                "material_calibrated_to_real_sock": False,
            },
            indent=2,
        )
        + "\n"
    )


def build_folded_sock(destination: Path) -> None:
    """Author the hollow sock loosely rolled into a 3.4 cm bundle [m]."""
    stage = Usd.Stage.Open(str(destination / "child_sock.usda"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Sock/Textile"))
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()
    t = points[:, 0] / 0.120
    theta = 3.0 * math.pi * t
    radius = 0.010 + 0.007 * t + points[:, 2] * 0.15
    points[:, 0] = radius * np.cos(theta)
    points[:, 2] = radius * np.sin(theta)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in points])
    stage.GetRootLayer().Export(str(destination / "child_sock_folded.usda"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("logs/microduck/room_cleanup/assets"))
    parser.add_argument("--photorealistic", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_room(args.output_dir, photorealistic=args.photorealistic)
    build_sock(args.output_dir)
    build_bin(args.output_dir)
    build_folded_sock(args.output_dir)


if __name__ == "__main__":
    main()
