# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path-trace the bedroom and actual Newton cleanup poses, without rerunning physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _camera(stage: Usd.Stage, overview: bool, closeup: bool = False) -> None:
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(40.0 if closeup else 27.0 if overview else 32.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr((0.01, 100.0))
    eye = Gf.Vec3d(2.5, -3.25, 2.0) if overview else Gf.Vec3d(1.15, -1.40, 0.80)
    target = Gf.Vec3d(0.05, 0.24, 0.45) if overview else Gf.Vec3d(0.20, 0.13, 0.20)
    if closeup:
        eye, target = Gf.Vec3d(0.78, -0.50, 0.43), Gf.Vec3d(0.28, 0.13, 0.12)
    camera.ClearXformOpOrder()
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())


def _pose_ops(prim: Usd.Prim):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.SetResetXformStack(True)
    return (
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "measured"),
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "measured"),
    )


def _set_pose(ops, pose, time: float) -> None:
    ops[0].Set(Gf.Vec3d(*map(float, pose[:3])), time)
    orientation = Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))
    orientation.Normalize()
    ops[1].Set(orientation, time)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_animation(run_dir: Path) -> Path:
    """Author recorded world poses [m, xyzw] and cloth vertices [m] at measured times [s]."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    room_asset = Path(metrics["parameters"]["room_dir"]).resolve() / "children_room.usda"
    path = run_dir / "cleanup_render.usdc"
    provenance_path = run_dir / "render_provenance.json"
    inputs = [
        run_dir / "render_states.npz",
        run_dir / "scene.usda",
        run_dir / "metrics.json",
        room_asset,
        Path(__file__),
        Path(__file__).with_name("bedroom_materials.py"),
    ]
    input_hashes = {str(item.resolve()): _sha256(item) for item in inputs}
    if path.is_file() and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text())
        if previous.get("input_sha256") == input_hashes and previous.get("animation_sha256") == _sha256(path):
            return path
    with np.load(run_dir / "render_states.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    stage = Usd.Stage.Open(str(run_dir / "scene.usda"))
    layer = stage.Flatten()
    layer.Export(str(path))
    stage = Usd.Stage.Open(str(path))
    # Appearance can be refined offline; all recorded moving geometry stays measured.
    stage.RemovePrim("/World/Room")
    stage.DefinePrim("/World/Room", "Xform").GetReferences().AddReference(str(room_asset))
    for name in ("/World/Ground", "/World/Light"):
        if stage.GetPrimAtPath(name):
            stage.RemovePrim(name)
    stage.SetTimeCodesPerSecond(1.0)
    stage.SetFramesPerSecond(50.0)
    stage.SetStartTimeCode(float(data["time_s"][0]))
    stage.SetEndTimeCode(float(data["time_s"][-1]))
    body_paths = {
        p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")) if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    body_ops = [_pose_ops(body_paths[name]) for name in data["body_names"]]
    marker_ops = _pose_ops(stage.GetPrimAtPath("/World/Marker"))
    cloth_prim = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Sock")) if p.IsA(UsdGeom.Mesh))
    cloth = UsdGeom.Mesh(cloth_prim)
    cloth.ClearXformOpOrder()
    cloth.SetResetXformStack(True)
    for frame, time in enumerate(data["time_s"]):
        for ops, pose in zip(body_ops, data["body_pose_m_xyzw"][frame]):
            _set_pose(ops, pose, float(time))
        _set_pose(marker_ops, data["marker_pose_m_xyzw"][frame], float(time))
        points = data["cloth_points_m"][frame]
        cloth.GetPointsAttr().Set(points, float(time))
        cloth.CreateExtentAttr().Set(
            [Gf.Vec3f(*map(float, points.min(0))), Gf.Vec3f(*map(float, points.max(0)))], float(time)
        )
    cotton_texture = room_asset.parent / "textures" / "cotton_albedo.png"
    if cotton_texture.is_file():
        from bedroom_materials import enrich_sock

        enrich_sock(stage, cloth, cotton_texture)
    _camera(stage, False)
    stage.GetRootLayer().Save()
    provenance_path.write_text(
        json.dumps(
            {
                "renderer": "OVRTX PathTracing",
                "input_sha256": input_hashes,
                "animation_sha256": _sha256(path),
                "physics": "Isaac Lab Newton: MJWarp rigid bodies + VBD cloth",
                "animation": (
                    "Measured world body poses and all cloth vertices at 50 Hz; physics not modified during rendering"
                ),
                "frames_recorded": len(data["time_s"]),
                "room_appearance_asset": str(room_asset),
                "time_codes_per_second": 1,
                "assisted_grasp_enabled": metrics["assisted_grasp_enabled"],
                "maximum_attached_nodes": (metrics.get("assistance") or {}).get("maximum_pinned_nodes", 0),
                "native_body_skills": metrics["native_body_skills"],
            },
            indent=2,
        )
        + "\n"
    )
    return path


def render(args: argparse.Namespace) -> None:
    """Render a still or a full recorded cleanup with RTX path tracing."""
    os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
    import ovrtx
    import ovstage
    from endeavor import _render_pixels
    from PIL import Image, ImageDraw, ImageFont

    if args.scene_only:
        stage = Usd.Stage.CreateInMemory()
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        stage.DefinePrim("/World/Room", "Xform").GetReferences().AddReference(
            str((args.room_dir / "children_room.usda").resolve())
        )
        basket = stage.DefinePrim("/World/Bin", "Xform")
        basket.GetReferences().AddReference(str((args.room_dir / "cleanup_bin.usda").resolve()))
        UsdGeom.Xformable(basket).AddTranslateOp().Set((0.38, 0.13, 0))
    else:
        path = build_animation(args.run_dir)
        stage = Usd.Stage.Open(str(path))
    _camera(stage, args.overview, args.closeup)
    usd = (
        stage.ExportToString()
        + f"""
def Scope "Render" {{
    def RenderProduct "Camera" {{
        int2 resolution = ({args.width}, {args.height})
        rel camera = </World/Camera>
        rel orderedVars = [</Render/LdrColor>]
        token omni:rtx:rendermode = "PathTracing"
        uint omni:rtx:pt:samplesPerPixel = {args.samples}
        bool omni:rtx:pt:denoising:enabled = true
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
    }}
    def RenderVar "LdrColor" {{
        string sourceName = "LdrColor"
    }}
}}
"""
    )
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(
            log_level="warn",
            log_file_path=str(args.output.with_suffix(".log")),
            suppress_deprecation_warnings=True,
            texture_streaming_mode="synchronous",
        )
    )
    render_stage = ovstage.Stage("microduck.bedroom")
    renderer.attach_ovstage(render_stage)
    writer = None
    try:
        ovstage.population.open_usd_from_string(render_stage, usd, ordinal=1, time_code=args.time)
        render_stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        for _ in range(4):
            pixels = _render_pixels(renderer, 1, 1 / args.fps)[..., :3]
        if args.output.suffix.lower() != ".mp4":
            Image.fromarray(pixels).save(args.output)
            print(f"Saved {args.output}", flush=True)
            return
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        with np.load(args.run_dir / "render_states.npz") as archive:
            data = {name: archive[name] for name in archive.files}
        metrics = json.loads((args.run_dir / "metrics.json").read_text())
        policy_label = (
            "Isaac Lab + Newton trained body skills"
            if metrics["native_body_skills"]
            else "Existing body-policy checkpoints"
        )
        font_size = max(14, args.width // 64)
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        line_height, padding = font_size + 6, font_size // 3
        writer = FFMPEG_VideoWriter(
            str(args.output),
            (args.width, args.height),
            fps=args.fps,
            codec="libx264",
            preset="medium",
            ffmpeg_params=["-crf", "18"],
        )
        frames = round((float(data["time_s"][-1]) - args.time) * args.fps) + 1
        for frame in range(frames):
            t = args.time + frame / args.fps
            ordinal = frame + 2
            ovstage.population.update_from_usd_time_async(render_stage, ordinal=ordinal, time_code=t).wait()
            render_stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            pixels = _render_pixels(renderer, ordinal, 1 / args.fps)[..., :3]
            overlay = Image.fromarray(pixels)
            draw = ImageDraw.Draw(overlay)
            index = min(len(data["time_s"]) - 1, int(np.searchsorted(data["time_s"], t)))
            active = bool(data["assistance_active"][index])
            draw.rectangle((0, 0, args.width, 2 * (line_height + padding)), fill=(22, 28, 34))
            draw.text(
                (12, padding),
                (
                    f"ASSISTED SOCK GRASP: {'attached' if active else 'inactive'} | Marker: physical contact grasp"
                    if metrics["assisted_grasp_enabled"]
                    else "Contact-only grasping | No grasp attachment"
                ),
                font=font,
                fill=(255, 218, 125),
            )
            draw.text(
                (12, padding + line_height),
                policy_label + " | Scripted cleanup | RTX render of recorded simulation",
                font=font,
                fill="white",
            )
            writer.write_frame(np.asarray(overlay))
            if frame % args.fps == 0:
                print(f"Rendered {frame + 1}/{frames}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        renderer.detach_ovstage()
        render_stage.destroy()
        renderer.destroy()


def main() -> None:
    """Parse rendering options and export the requested image or video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--room_dir", type=Path)
    parser.add_argument("--scene_only", action="store_true")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--closeup", action="store_true", help="Frame the beak, objects, and basket closely.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--time", type=float, default=0.0)
    args = parser.parse_args()
    if args.overview and args.closeup:
        parser.error("Choose either --overview or --closeup.")
    if args.scene_only and (args.room_dir is None or args.output.suffix == ".mp4"):
        parser.error("A scene preview requires --room_dir and an image output.")
    if not args.scene_only and args.run_dir is None:
        parser.error("Recorded rendering requires --run_dir.")
    if min(args.width, args.height, args.fps, args.samples) < 1 or args.width % 2 or args.height % 2:
        parser.error("Use positive dimensions, frame rate and sample count, and even image dimensions.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args)


if __name__ == "__main__":
    main()
