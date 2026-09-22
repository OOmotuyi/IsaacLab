# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record measured cleanup states in Newton's native GUI without rerunning physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image
from pyglet.math import Vec3

from pxr import Usd, UsdGeom


def record(args: argparse.Namespace) -> None:
    """Capture recorded poses [m, xyzw] and cloth vertices [m] in the native viewer."""
    started = time.monotonic()
    with np.load(args.run_dir / "render_states.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    trace = [json.loads(line) for line in (args.run_dir / "trace.jsonl").read_text().splitlines()]
    trace_times = np.array([row["time_s"] for row in trace])
    original = Usd.Stage.Open(str(args.run_dir / "scene.usda"))
    stage = Usd.Stage.Open(original.Flatten())
    cloth = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Sock/Textile"))
    if set(cloth.GetFaceVertexCountsAttr().Get()) != {3}:
        raise ValueError("The recorded sock must have triangular topology.")
    triangles = np.array(cloth.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    face_colors = np.array(cloth.GetDisplayColorPrimvar().ComputeFlattened())
    colors, color_ids = np.unique(face_colors, axis=0, return_inverse=True)
    for name in ("/World/Sock", "/World/Ground", "/World/Light"):
        stage.RemovePrim(name)
    builder = newton.ModelBuilder()
    # USD visibility keeps robot guide proxies hidden and the physical basket wires visible.
    builder.add_usd(stage, collapse_fixed_joints=False, hide_collision_shapes=False)
    model = builder.finalize(device="cuda:0")
    state = model.state()
    body_ids = []
    for name in data["body_names"]:
        matches = [i for i, label in enumerate(model.body_label) if label == f"/World/Robot/{name}"]
        if len(matches) != 1:
            matches = [i for i, label in enumerate(model.body_label) if label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map recorded body {name}: {matches}")
        body_ids.append(matches[0])
    marker_id = model.body_label.index("/World/Marker")
    poses = state.body_q.numpy()
    points = wp.array(data["cloth_points_m"][0], dtype=wp.vec3, device=model.device)
    groups = [
        (tuple(map(float, color)), wp.array(triangles[color_ids == i].ravel(), dtype=wp.int32, device=model.device))
        for i, color in enumerate(colors)
    ]
    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        vsync=False,
        headless=False,
        enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE,
    )
    if viewer.gui is None or not viewer.gui.is_available:
        viewer.close()
        raise RuntimeError("A working native GUI is required; use an available DISPLAY.")
    viewer.set_model(model)
    viewer.camera.pos = Vec3(1.15, -1.40, 0.80)
    viewer.camera.look_at((0.20, 0.13, 0.20))
    viewer.camera.fov = math.degrees(2 * math.atan(20.25 / (2 * 32.0)))
    viewer.show_joints = False
    viewer.show_com = False
    viewer.show_particles = False
    current = {"time": 0.0, "frame": 0, "trace": trace[0]}

    def panel(imgui) -> None:
        imgui.text("MicroDuck bedroom cleanup")
        imgui.text("RECORDED STATE REPLAY")
        imgui.text("Isaac Lab + Newton")
        imgui.text("Source: MJWarp + VBD cloth")
        imgui.separator()
        imgui.text(f"Recorded time: {current['time']:05.2f} s")
        row = current["trace"]
        imgui.text(f"Phase: {row['phase']}")
        imgui.text(f"Actor: {row['policy']}")
        imgui.text(f"Measured speed: {np.linalg.norm(row['actual_velocity_m_s'][:2]):.3f} m/s")
        attached = bool(data["assistance_active"][current["frame"]])
        imgui.text(f"Sock attachment active: {'YES' if attached else 'no'}")
        imgui.text("Sock: assisted grasp")
        imgui.text("Marker: contact grasp")
        imgui.text("Trained body skills;")
        imgui.text("scripted jaw and sequence")
        imgui.separator()

    viewer.register_ui_callback(panel, position="panel")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = float(data["time_s"][-1] - data["time_s"][0] + np.median(np.diff(data["time_s"])))
    total = int(round(duration * args.fps))
    if args.frames is not None:
        total = min(total, args.frames)
    writer = FFMPEG_VideoWriter(
        str(args.output),
        (args.width, args.height),
        args.fps,
        codec="libx264",
        preset="medium",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )
    maximum_pose_error = 0.0
    maximum_cloth_error = 0.0
    recorded = 0
    try:
        for frame in range(total):
            if not viewer.is_running():
                raise RuntimeError("Viewer closed before the video finished.")
            stamp = float(data["time_s"][0]) + frame / args.fps
            index = int(np.argmin(abs(data["time_s"] - stamp)))
            trace_index = int(np.argmin(abs(trace_times - data["time_s"][index])))
            current.update(time=float(data["time_s"][index]), frame=index, trace=trace[trace_index])
            poses[body_ids] = data["body_pose_m_xyzw"][index]
            poses[marker_id] = data["marker_pose_m_xyzw"][index]
            state.body_q.assign(poses)
            points.assign(data["cloth_points_m"][index])
            viewer.begin_frame(stamp)
            viewer.log_state(state)
            for group, (color, indices) in enumerate(groups):
                viewer.log_mesh(
                    f"/recorded_sock/stripe_{group}",
                    points,
                    indices,
                    color=color,
                    roughness=0.95,
                    backface_culling=False,
                )
            viewer.end_frame()
            pixels = viewer.get_frame(render_ui=True).numpy()
            if pixels.shape != (args.height, args.width, 3):
                raise ValueError(f"Unexpected framebuffer size: {pixels.shape}")
            writer.write_frame(pixels)
            recorded += 1
            if frame in {0, min(total - 1, round(14.3 * args.fps)), total - 1}:
                Image.fromarray(pixels).save(args.output.with_name(f"{args.output.stem}_{frame:04d}.jpg"))
                actual = state.body_q.numpy()
                maximum_pose_error = max(
                    maximum_pose_error,
                    float(np.max(abs(actual[body_ids] - data["body_pose_m_xyzw"][index]))),
                    float(np.max(abs(actual[marker_id] - data["marker_pose_m_xyzw"][index]))),
                )
                maximum_cloth_error = max(
                    maximum_cloth_error, float(np.max(abs(points.numpy() - data["cloth_points_m"][index])))
                )
            if frame % args.fps == 0:
                print(f"Recorded {frame + 1}/{total} frames ({stamp:.2f} s)", flush=True)
    finally:
        writer.close()
        viewer.close()
    inputs = [args.run_dir / "render_states.npz", args.run_dir / "scene.usda", Path(__file__)]
    hashes = {}
    for path in inputs:
        with path.open("rb") as stream:
            hashes[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "renderer": "newton.viewer.ViewerGL",
        "native_gui_captured": True,
        "playback": "Recorded state replay; no new physics steps or policy inference",
        "source_physics": "Isaac Lab Newton MJWarp rigid bodies + VBD cloth",
        "newton_version": newton.__version__,
        "input_sha256": hashes,
        "frames": recorded,
        "fps": args.fps,
        "duration_s": recorded / args.fps,
        "resolution": [args.width, args.height],
        "elapsed_wall_s": time.monotonic() - started,
        "body_mapping": dict(zip(map(str, data["body_names"]), map(int, body_ids))),
        "sampled_body_pose_max_abs_error": maximum_pose_error,
        "sampled_cloth_position_max_abs_error_m": maximum_cloth_error,
        "source_objects_in_bin": metrics["objects_in_bin"],
        "sock_grasp": "assisted",
        "marker_grasp": "contact",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames", type=int, help="Limit output frames for a short capture check.")
    record(parser.parse_args())
