# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a photo-referenced NVIDIA Endeavor atrium with a flat walking floor.

Reference: Connie Zhou's 2018 Endeavor atrium photograph, published by Interior
Design. Dimensions are estimates from the image, not surveyed building data.
The scene contains no robot or policy and has no external texture dependencies.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

REFERENCE_URL = (
    "https://interiordesign.net/projects/nvidia-campus-by-gensler-2018-best-of-year-winner-for-large-tech-office/"
)
PHOTO_URL = (
    "https://sdg-migration-id.s3.amazonaws.com/"
    "ID-may-2018-IIDA-Awards-gensler-nvidia-santa-clara-california-interior-01-0518.jpg"
)


def build_endeavor(path: Path) -> None:
    """Write the environment-only USD with its floor at z = 0 [m].

    Args:
        path: Destination USD file. Existing files are not overwritten.
    """
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = _AtriumBuilder()
    builder.build()
    builder.stage.GetRootLayer().Export(str(path))


class _AtriumBuilder:
    """Author the atrium in meters using portable USD materials and lighting."""

    def __init__(self):
        self.stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(self.stage, "/Endeavor")
        self.stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.GetRootLayer().customLayerData = {
            "reference": REFERENCE_URL,
            "reference_photo": PHOTO_URL,
            "reference_photographer": "Connie Zhou",
            "description": "Photo-based approximation of Endeavor's atrium; dimensions are estimated.",
            "walking_area": "x=[-12,12], y=[-12,8], z=0 meters; only floor collision is authored.",
        }
        self.materials: dict[str, UsdShade.Material] = {}

    def build(self) -> None:
        self._material("floor", (0.58, 0.59, 0.57), 0.3)
        self._material("white", (0.83, 0.85, 0.84), 0.45)
        self._material("ceiling", (0.82, 0.84, 0.82), 0.72)
        self._material("dark", (0.018, 0.023, 0.025), 0.3, 0.5)
        self._material("wood", (0.31, 0.17, 0.075), 0.5)
        self._material("furniture", (0.06, 0.065, 0.065), 0.72)
        self._material("window", (0.73, 0.85, 0.91), 0.1, 0.25)
        for index in range(8):
            value = 0.014 + index * 0.004
            self._material(f"facet_{index}", (value, value * 1.06, value * 1.08), 0.25 + index * 0.025, 0.65)
        floor = self._box("Floor", (0, 7, -0.125), (70, 80, 0.25), "floor")
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        physics_material = UsdShade.Material.Define(self.stage, "/Endeavor/Looks/FloorPhysics")
        physics = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
        physics.CreateStaticFrictionAttr(1.0)
        physics.CreateDynamicFrictionAttr(1.0)
        physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI(floor).Bind(physics_material, materialPurpose="physics")
        self._core()
        self._roof()
        self._perimeter()
        self._lighting()
        camera = UsdGeom.Camera.Define(self.stage, "/Endeavor/ReferenceCamera")
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr((0.05, 300.0))
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, -21, 3.0), Gf.Vec3d(0, 18, 7.1), Gf.Vec3d(0, 0, 1))
        camera.AddTransformOp().Set(matrix.GetInverse())

    def _material(self, name: str, color: tuple[float, float, float], roughness: float, metallic: float = 0) -> None:
        material = UsdShade.Material.Define(self.stage, f"/Endeavor/Looks/{name}")
        shader = UsdShade.Shader.Define(self.stage, f"{material.GetPath()}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.materials[name] = material

    def _bind(self, geom: UsdGeom.Gprim, material: str) -> None:
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(self.materials[material])

    def _box(self, name: str, position: tuple, size: tuple, material: str) -> UsdGeom.Cube:
        geom = UsdGeom.Cube.Define(self.stage, f"/Endeavor/{name}")
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*position))
        geom.AddScaleOp().Set(Gf.Vec3f(*size))
        self._bind(geom, material)
        return geom

    def _beam(self, name: str, start: tuple, end: tuple, radius: float, material: str) -> None:
        first, last = Gf.Vec3d(*start), Gf.Vec3d(*end)
        delta = last - first
        geom = UsdGeom.Cylinder.Define(self.stage, f"/Endeavor/{name}")
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(delta.GetLength())
        geom.AddTranslateOp().Set((first + last) / 2)
        geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Rotation(Gf.Vec3d(0, 0, 1), delta).GetQuat())
        self._bind(geom, material)

    def _mesh(self, name: str, points: list, faces: list, material: str) -> None:
        geom = UsdGeom.Mesh.Define(self.stage, f"/Endeavor/{name}")
        geom.CreatePointsAttr(points)
        geom.CreateFaceVertexCountsAttr([len(face) for face in faces])
        geom.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        geom.CreateSubdivisionSchemeAttr("none")
        geom.CreateDoubleSidedAttr(True)
        self._bind(geom, material)

    def _core(self) -> None:
        # A faceted metal belt encloses the central meeting spaces, as in the photo.
        sections = 48
        rings = ((0.89, 7.6), (1.0, 9.25), (0.96, 11.0), (0.85, 12.0))
        points = []
        for scale, z in rings:
            for i in range(sections):
                theta = math.tau * i / sections
                points.append((17.5 * scale * math.cos(theta), 23 + 12.8 * scale * math.sin(theta), z))
        for row in range(len(rings) - 1):
            for i in range(sections):
                j = (i + 1) % sections
                vertices = [
                    points[row * sections + i],
                    points[row * sections + j],
                    points[(row + 1) * sections + j],
                    points[(row + 1) * sections + i],
                ]
                center = np.mean(vertices, axis=0)
                vertices = [tuple(center + (np.asarray(point) - center) * 0.996) for point in vertices]
                for half, face in enumerate(((0, 1, 2), (0, 2, 3))):
                    self._mesh(f"Cocoon/Panel_{row}_{i}_{half}", vertices, [face], f"facet_{(i * 3 + row + half) % 8}")
        for side in (-1, 1):
            tag = "Left" if side < 0 else "Right"
            self._box(f"Core/{tag}Wall", (side * 6.2, 20, 2.5), (8.0, 0.35, 5.0), "white")
            self._box(f"Core/{tag}Side", (side * 10.0, 26, 3.6), (0.3, 12, 7.2), "white")
            self._beam(f"Braces/{tag}Outer", (side * 10.5, 13, 0), (side * 14.3, 15.65, 9.25), 0.105, "dark")
            self._beam(f"Braces/{tag}Inner", (side * 10.5, 13, 0), (side * 3.3, 11.87, 7.6), 0.105, "dark")
            self._beam(f"Braces/{tag}Rear", (side * 10.5, 13, 0), (side * 15.1, 25.4, 7.6), 0.10, "dark")
        self._box("Core/DoorHeader", (0, 20, 4.3), (4.6, 0.35, 1.4), "white")
        self._box("Core/UpperSlab", (0, 25, 5.15), (20, 12, 0.3), "white")
        self._box("Core/BalconyParapet", (0, 17.3, 5.8), (18.5, 0.18, 1.0), "white")
        self._box("Core/UpperRearWall", (0, 30.6, 6.4), (20, 0.3, 2.4), "white")
        for x in (-7, -3.5, 0, 3.5, 7):
            self._beam(f"Core/UpperColumn_{int((x + 8) * 10)}", (x, 26, 5.3), (x, 26, 8.2), 0.07, "white")
        # Elevated circulation is scenic; the robot's walking zone remains on the ground floor.
        for side in (-1, 1):
            tag = "Left" if side < 0 else "Right"
            self._box(f"Gallery/{tag}Deck", (side * 25, 22, 5.0), (20, 5, 0.35), "white")
            self._box(f"Gallery/{tag}Rail", (side * 25, 19.6, 5.75), (20, 0.12, 1.1), "white")
            for x in (20, 29):
                self._beam(f"Gallery/{tag}Column_{x}", (side * x, 24, 0), (side * x, 24, 5), 0.15, "white")

    def _roof(self) -> None:
        # White triangulated roof with small triangular skylights and visible purlins.
        for row, y in enumerate(range(-28, 49, 7)):
            for col, x in enumerate(range(-35, 36, 7)):
                z = 16.0 - 0.0014 * x * x + 0.14 * math.cos(y / 7)
                self._beam(f"Roof/Across_{row}_{col}", (x, y, z), (x + 7, y, z - 0.12), 0.105, "white")
                self._beam(f"Roof/Along_{row}_{col}", (x, y, z), (x, y + 7, z), 0.08, "white")
                self._beam(f"Roof/Diagonal_{row}_{col}", (x, y, z), (x + 7, y + 7, z - 0.12), 0.11, "white")
                corners = [(x, y, z + 0.18), (x + 7, y, z + 0.06), (x + 7, y + 7, z + 0.06), (x, y + 7, z + 0.18)]
                self._mesh(f"Roof/Panel_{row}_{col}", corners, [(0, 2, 3)], "ceiling")
                outer = [np.asarray(corners[index]) for index in (0, 1, 2)]
                center = np.mean(outer, axis=0)
                inner = [center + (point - center) * 0.50 for point in outer]
                ring = [tuple(point) for point in outer + inner]
                self._mesh(f"Roof/Skylight_{row}_{col}", ring, [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)], "ceiling")
                for i in range(3):
                    self._beam(
                        f"Roof/SkylightFrame_{row}_{col}_{i}",
                        tuple(inner[i]),
                        tuple(inner[(i + 1) % 3]),
                        0.035,
                        "white",
                    )
        for side in (-1, 1):
            for index, y in enumerate((-8, 15, 37)):
                self._box(f"Columns/C_{side + 1}_{index}", (side * 21.5, y, 7.7), (0.62, 0.62, 15.4), "white")

    def _perimeter(self) -> None:
        self._box("Walls/Rear", (0, 47, 8), (70, 0.25, 16), "white")
        for side in (-1, 1):
            self._box(f"Walls/Glass_{side + 1}", (side * 35, 7, 7.2), (0.08, 80, 14.4), "window")
            for index, y in enumerate(range(-32, 48, 4)):
                self._box(f"Walls/Mullion_{side + 1}_{index}", (side * 34.8, y, 7.2), (0.10, 0.10, 14.4), "white")
            for index, y in enumerate((5, 14, 30)):
                x = side * 27
                self._box(f"Furniture/Table_{side + 1}_{index}", (x, y, 0.76), (3.5, 1.1, 0.08), "wood")
                for leg, dx in enumerate((-1.3, 1.3)):
                    self._box(f"Furniture/Leg_{side + 1}_{index}_{leg}", (x + dx, y, 0.35), (0.1, 0.7, 0.7), "dark")
                for chair in range(4):
                    cx = x - 1.3 + chair * 0.86
                    self._box(
                        f"Furniture/Chair_{side + 1}_{index}_{chair}",
                        (cx, y - 1.0, 0.44),
                        (0.56, 0.52, 0.09),
                        "furniture",
                    )
                    self._box(
                        f"Furniture/Back_{side + 1}_{index}_{chair}",
                        (cx, y - 1.22, 0.70),
                        (0.56, 0.08, 0.55),
                        "furniture",
                    )

    def _lighting(self) -> None:
        sky = UsdLux.DomeLight.Define(self.stage, "/Endeavor/Lights/Sky")
        sky.CreateIntensityAttr(1800.0)
        sky.CreateColorAttr((0.94, 0.97, 1.0))
        sun = UsdLux.DistantLight.Define(self.stage, "/Endeavor/Lights/Sun")
        sun.CreateIntensityAttr(900.0)
        sun.CreateAngleAttr(8.0)
        sun.CreateColorAttr((1.0, 0.97, 0.91))
        sun.AddRotateXYZOp().Set((-10, -22, -15))
        for index, y in enumerate((-12, 1, 14, 29)):
            light = UsdLux.RectLight.Define(self.stage, f"/Endeavor/Lights/Ceiling_{index}")
            light.CreateWidthAttr(24.0)
            light.CreateHeightAttr(10.0)
            light.CreateIntensityAttr(900.0)
            light.AddTranslateOp().Set((0, y, 14.5))
        fill = UsdLux.RectLight.Define(self.stage, "/Endeavor/Lights/DaylightFill")
        fill.CreateWidthAttr(45.0)
        fill.CreateHeightAttr(14.0)
        fill.CreateIntensityAttr(450.0)
        fill.AddTranslateOp().Set((0, -25, 7))
        fill.AddRotateXYZOp().Set((90, 0, 0))
        for index, x in enumerate((-7, 0, 7)):
            light = UsdLux.DiskLight.Define(self.stage, f"/Endeavor/Lights/Core_{index}")
            light.CreateRadiusAttr(0.14)
            light.CreateIntensityAttr(5500.0)
            light.AddTranslateOp().Set((x, 22, 7.5))
