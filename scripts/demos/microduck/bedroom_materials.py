# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical surface materials, soft furnishing meshes, and daylight for the bedroom."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade


def material(
    stage: Usd.Stage,
    name: str,
    color: tuple,
    roughness: float = 0.5,
    metallic: float = 0.0,
    texture: Path | None = None,
) -> UsdShade.Material:
    """Author a portable USD Preview Surface, optionally with an sRGB albedo."""
    mat = UsdShade.Material.Define(stage, f"{stage.GetDefaultPrim().GetPath()}/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if texture is not None:
        reader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Albedo")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture.resolve()))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        if texture.name == "cotton_albedo.png":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color, 1.0))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return mat


def bind(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    """Bind a visual material without changing physics-material bindings."""
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _mesh(stage, path, points, quads, mat, uv=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(q) for q in quads])
    mesh.CreateFaceVertexIndicesAttr([i for q in quads for i in q])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if uv is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(uv)
    bind(mesh.GetPrim(), mat)
    return mesh


def _soft_box(stage, path, center, size, mat, wrinkle=0.0):
    points, quads = [], []
    rows, cols = 33, 64

    def power(x, exponent):
        return math.copysign(abs(x) ** exponent, x)

    for i in range(rows):
        lat = -math.pi / 2 + math.pi * (i + 0.001) / (rows - 1 + 0.002)
        for j in range(cols):
            lon = 2 * math.pi * j / cols
            x = power(math.cos(lat), 0.4) * power(math.cos(lon), 0.3)
            y = power(math.cos(lat), 0.4) * power(math.sin(lon), 0.3)
            z = power(math.sin(lat), 0.4)
            w = wrinkle * math.sin(x * 19 + y * 9) * math.sin(y * 23) * (1 - abs(x) ** 8)
            points.append((center[0] + x * size[0] / 2, center[1] + y * size[1] / 2, center[2] + z * size[2] / 2 + w))
    for i in range(rows - 1):
        for j in range(cols):
            a = i * cols + j
            b = i * cols + (j + 1) % cols
            quads.append((a, b, b + cols, a + cols))
    quads.extend([tuple(reversed(range(cols))), tuple((rows - 1) * cols + j for j in range(cols))])
    mesh = _mesh(stage, path, points, quads, mat)
    # Planar face coordinates keep real-world grain and weave sizes through rounded edges.
    period = 0.055 if mat.GetPrim().GetName() in {"CottonTeal", "Linen"} else 0.65
    positions = np.asarray(points)
    uv = []
    for face in quads:
        p0, p1, p2 = positions[list(face[:3])]
        axis = int(np.argmax(abs(np.cross(p1 - p0, p2 - p0))))
        axes = [index for index in range(3) if index != axis]
        uv.extend([tuple(map(float, (positions[index] - center)[axes] / period)) for index in face])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(uv)
    mesh.CreateSubdivisionSchemeAttr("catmullClark")


def enrich_room(stage: Usd.Stage, texture: Path) -> None:
    """Improve bedroom appearance while retaining the existing furniture colliders."""
    oak = material(stage, "Oak", (0.55, 0.32, 0.14), 0.38, texture=texture)
    sage = material(stage, "SagePaint", (0.19, 0.34, 0.28), 0.88)
    ivory = material(stage, "IvoryPaint", (0.80, 0.76, 0.66), 0.84)
    white = material(stage, "WarmWhite", (0.88, 0.85, 0.76), 0.68)
    cotton = texture.with_name("cotton_albedo.png")
    cotton = cotton if cotton.is_file() else None
    teal = material(stage, "CottonTeal", (0.025, 0.19, 0.25), 0.96, texture=cotton)
    linen = material(stage, "Linen", (0.85, 0.77, 0.60), 0.97, texture=cotton)
    ochre = material(stage, "OchreLacquer", (0.55, 0.24, 0.055), 0.36)
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if "FloorSeam" in path or "/Stripe" in path:
            prim.SetActive(False)
        elif path.endswith("/BackWall"):
            bind(prim, sage)
        elif path.endswith("/SideWall"):
            bind(prim, ivory)
        elif "/Bed/" in path or "/Chair/Leg" in path:
            bind(prim, oak)
            if prim.IsA(UsdGeom.Cube) and not any(name in path for name in ("Mattress", "Duvet", "Pillow")):
                # The original cube remains the exact physical collision proxy.
                center = prim.GetAttribute("xformOp:translate").Get()
                size = prim.GetAttribute("xformOp:scale").Get()
                UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
                _soft_box(stage, path + "Visual", center, size, oak)
        else:
            color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            bind(
                prim,
                material(
                    stage,
                    str(prim.GetPath()).strip("/").replace("/", "_"),
                    tuple(color[0]) if color else (0.8, 0.8, 0.8),
                    0.65,
                ),
            )
    floor = stage.GetPrimAtPath("/Room/Floor")
    UsdGeom.Imageable(floor).CreateVisibilityAttr("invisible")
    _mesh(
        stage,
        "/Room/OakFloor",
        [(-1.3, -1.2, 0.0005), (1.5, -1.2, 0.0005), (1.5, 1.2, 0.0005), (-1.3, 1.2, 0.0005)],
        [(0, 1, 2, 3)],
        oak,
        [(0, 0), (2.0, 0), (2.0, 1.7), (0, 1.7)],
    )
    for path, center, size, mat, wrinkle in [
        ("Bed/Mattress", (-0.42, 0.72, 0.325), (1.43, 0.68, 0.11), white, 0.001),
        ("Bed/Duvet", (-0.22, 0.72, 0.395), (1.04, 0.73, 0.065), teal, 0.004),
        ("Bed/Pillow", (-0.93, 0.72, 0.412), (0.30, 0.50, 0.09), linen, 0.0015),
        ("Chair/Seat", (0.97, 0.62, 0.285), (0.33, 0.33, 0.048), ochre, 0.0),
        ("Chair/Back", (0.97, 0.76, 0.43), (0.33, 0.05, 0.27), ochre, 0.0),
    ]:
        UsdGeom.Imageable(stage.GetPrimAtPath("/Room/" + path)).CreateVisibilityAttr("invisible")
        _soft_box(stage, "/Room/" + path + "Visual", center, size, mat, wrinkle)
    _soft_box(stage, "/Room/Bed/BlanketFold", (-0.69, 0.72, 0.427), (0.13, 0.71, 0.03), linen, 0.002)
    # Draped curtains frame the existing window; their folds cast real shadows.
    for side, x0 in enumerate((0.23, 0.84)):
        points = []
        quads = []
        cols = 32
        rows = 12
        for i in range(rows):
            z = 0.47 + i * 0.72 / (rows - 1)
            for j in range(cols):
                u = j / (cols - 1)
                points.append(
                    (x0 + u * 0.13, 1.075 + 0.017 * math.cos(u * 8 * math.pi), z + 0.003 * math.sin(u * 8 * math.pi))
                )
        for i in range(rows - 1):
            for j in range(cols - 1):
                a = i * cols + j
                quads.append((a, a + 1, a + 1 + cols, a + cols))
        uv = [(j / (cols - 1), i / (rows - 1) * 5) for i in range(rows) for j in range(cols)]
        _mesh(stage, f"/Room/Window/Curtain{side}", points, quads, linen, uv)
    sky = UsdLux.DomeLight.Define(stage, "/Room/Lights/Sky")
    sky.CreateIntensityAttr(350)
    sky.CreateColorAttr((0.82, 0.89, 1.0))
    window = UsdLux.RectLight.Define(stage, "/Room/Lights/Window")
    window.CreateWidthAttr(0.6)
    window.CreateHeightAttr(0.48)
    window.CreateIntensityAttr(2500)
    window.CreateColorAttr((1.0, 0.92, 0.80))
    window.AddTranslateOp().Set((0.59, 1.08, 0.91))
    window.AddRotateXYZOp().Set((-72, 0, -18))
    fill = UsdLux.RectLight.Define(stage, "/Room/Lights/Softbox")
    fill.CreateWidthAttr(2.2)
    fill.CreateHeightAttr(1.7)
    fill.CreateIntensityAttr(500)
    fill.CreateColorAttr((1.0, 0.93, 0.83))
    fill.AddTranslateOp().Set((0.2, -0.3, 2.5))


def enrich_sock(stage: Usd.Stage, mesh: UsdGeom.Mesh, texture: Path) -> None:
    """Apply cotton weave to the sock's existing colored faces without changing its geometry."""
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    colors = np.asarray(mesh.GetDisplayColorAttr().Get())
    if len(colors) != len(indices):
        raise ValueError("Expected the authored sock's per-face stripe colors.")
    uv = []
    for face in indices:
        circumferential = (face % 16) / 16.0
        if np.ptp(circumferential) > 0.5:
            circumferential = np.where(circumferential < 0.5, circumferential + 1.0, circumferential)
        lengthwise = np.minimum(face // 16, 24) / 24.0
        uv.extend(zip((circumferential * 1.2).tolist(), (lengthwise * 2.4).tolist()))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(uv)
    palette, groups = np.unique(colors, axis=0, return_inverse=True)
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    for index, color in enumerate(palette):
        cotton = material(stage, f"SockCotton{index}", tuple(map(float, color)), 0.96, texture=texture)
        subset = binding.CreateMaterialBindSubset(f"Stripe{index}", np.flatnonzero(groups == index).tolist())
        bind(subset.GetPrim(), cotton)
