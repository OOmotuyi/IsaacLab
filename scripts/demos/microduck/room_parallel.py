# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show independent Isaac Lab cleanup rooms together using live measured states.

Each worker runs complete MJWarp/VBD physics and policy inference in its own
Isaac Lab process. The shared Newton GUI displays their live state streams;
it does not step physics or substitute a prerecorded trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image
from pyglet.math import Vec3

from pxr import Usd, UsdGeom


def _stop_workers(workers: list[subprocess.Popen]) -> None:
    for proc in workers:
        if proc.poll() is None:
            proc.terminate()
    for proc in workers:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _capture_frame(viewer: newton.viewer.ViewerGL) -> np.ndarray:
    if not np.isfinite(viewer.camera.get_view_matrix()).all():
        raise ValueError("The live viewer camera became non-finite.")
    return viewer.get_frame(render_ui=True).numpy()


def run(args: argparse.Namespace) -> None:
    """Run full rooms concurrently and capture their live native GUI view."""
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.reference_run / "metrics.json").read_text())
    parameters = reference["parameters"]
    with np.load(args.reference_run / "render_states.npz") as archive:
        initial = {name: archive[name][0] for name in ("body_pose_m_xyzw", "marker_pose_m_xyzw", "cloth_points_m")}
        body_names = archive["body_names"].tolist()
    original = Usd.Stage.Open(str(args.reference_run / "scene.usda"))
    stage = Usd.Stage.Open(original.Flatten())
    cloth = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Sock/Textile"))
    triangles = np.asarray(cloth.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    face_colors = np.asarray(cloth.GetDisplayColorPrimvar().ComputeFlattened())
    colors, color_ids = np.unique(face_colors, axis=0, return_inverse=True)
    for name in ("/World/Sock", "/World/Ground", "/World/Light"):
        stage.RemovePrim(name)
    blueprint = newton.ModelBuilder()
    blueprint.add_usd(stage, collapse_fixed_joints=False, hide_collision_shapes=False)
    local_bodies = []
    for name in body_names:
        matches = [i for i, label in enumerate(blueprint.body_label) if label.endswith(f"/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely map robot body {name}: {matches}")
        local_bodies.append(matches[0])
    marker_index = blueprint.body_label.index("/World/Marker")
    builder = newton.ModelBuilder()
    columns = math.ceil(math.sqrt(args.num_rooms))
    offsets = [np.array([(i % columns) * 3.1, (i // columns) * 2.8, 0.0]) for i in range(args.num_rooms)]
    body_ids = []
    marker_ids = []
    for i, offset in enumerate(offsets):
        first = builder.body_count
        builder.add_world(blueprint, xform=wp.transform(offset, wp.quat_identity()), label_prefix=f"room_{i}")
        body_ids.append(np.asarray(local_bodies) + first)
        marker_ids.append(marker_index + first)
    model = builder.finalize(device="cuda:0")
    state = model.state()
    poses = state.body_q.numpy()
    points = [wp.array(initial["cloth_points_m"] + offset, dtype=wp.vec3, device=model.device) for offset in offsets]
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
        raise RuntimeError("A working DISPLAY is required for the native multi-room GUI.")
    viewer.set_model(model)
    # Room transforms already include their grid offsets.
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    viewer.show_joints = viewer.show_com = viewer.show_particles = False
    center = np.mean(offsets, axis=0)

    def overview() -> None:
        viewer.camera.pos = Vec3(*map(float, center + np.array([4.9, -6.6, 6.0])))
        viewer.camera.look_at(tuple(center + np.array([0.1, 0.0, 0.25])))
        viewer.camera.fov = 44.0

    overview()
    current = [{"time_s": 0.0, "phase": "initializing", "policy": "", "updated": False} for _ in offsets]
    workers: list[subprocess.Popen] = []
    logs = []
    commands = []
    seen_times: list[list[float]] = [[] for _ in offsets]
    live_paths = [args.output_dir / f"room_{i}" / "live.npz" for i in range(args.num_rooms)]
    for i, offset in enumerate(offsets):
        poses[body_ids[i]] = initial["body_pose_m_xyzw"]
        poses[body_ids[i], :3] += offset
        poses[marker_ids[i]] = initial["marker_pose_m_xyzw"]
        poses[marker_ids[i], :3] += offset
    state.body_q.assign(poses)

    def panel(imgui) -> None:
        imgui.text(f"{args.num_rooms} complete Isaac Lab cleanup rooms")
        imgui.text("LIVE independent physics workers")
        imgui.text("Newton MJWarp + VBD cloth in each room")
        imgui.text("Scratch-trained body policies")
        imgui.text(
            "Learned grasp/jaw; scripted transport/release"
            if parameters.get("grasp_policy")
            else "Scripted sequence/jaw"
        )
        imgui.text("Sock assistance enabled")
        imgui.separator()
        if imgui.button("Show all rooms"):
            overview()
        for i, row in enumerate(current):
            active = i < len(workers) and workers[i].poll() is None
            imgui.text(f"Room {i + 1}: {float(row['time_s']):.2f} s | {'running' if active else 'finished'}")
            imgui.text(str(row["phase"]))
            if imgui.button(f"Focus room {i + 1}"):
                viewer.camera.pos = Vec3(*map(float, offsets[i] + np.array([1.15, -1.4, 0.8])))
                viewer.camera.look_at(tuple(offsets[i] + np.array([0.2, 0.13, 0.2])))
                viewer.camera.fov = 35.0
            imgui.separator()

    viewer.register_ui_callback(panel, position="panel")
    writer = None
    last_write = 0.0
    finished_at = None
    captured = 0
    try:
        for i, path in enumerate(live_paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).with_name("room_cleanup.py")),
                "--mode",
                "cleanup",
                "--first_object",
                "sock",
                "--assisted_grasp",
                "--record_states",
                "--visualizer",
                "none",
                "--duration",
                str(args.duration),
                "--live_state",
                str(path),
                "--output_dir",
                str(path.parent),
            ]
            for name in (
                "room_dir",
                "scene_dir",
                "asset_dir",
                "standing_policy",
                "pickup_policy",
                "grasp_policy",
                "walking_policy",
                "sock_close_time",
                "physics_substeps",
                "cloth_iterations",
                "coupling_relaxation",
            ):
                if parameters.get(name) is not None:
                    command += ["--" + name, str(parameters[name])]
            log = (path.parent / "worker.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "PXR_WORK_THREAD_LIMIT": "4",
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                },
            )
            workers.append(proc)
            commands.append({"room": i, "pid": proc.pid, "command": command, "launch_wall_time": time.time()})
        (args.output_dir / "workers.json").write_text(json.dumps(commands, indent=2) + "\n")
        while viewer.is_running():
            now = time.monotonic()
            for i, path in enumerate(live_paths):
                if not path.exists():
                    continue
                with np.load(path) as archive:
                    stamp = float(archive["time_s"])
                    if current[i]["updated"] and stamp <= current[i]["time_s"]:
                        continue
                    body = archive["body_pose_m_xyzw"]
                    marker = archive["marker_pose_m_xyzw"]
                    cloth_points = archive["cloth_points_m"]
                    current[i] = {
                        "time_s": stamp,
                        "phase": str(archive["phase"]),
                        "policy": str(archive["policy"]),
                        "updated": True,
                    }
                poses[body_ids[i]] = body
                poses[body_ids[i], :3] += offsets[i]
                poses[marker_ids[i]] = marker
                poses[marker_ids[i], :3] += offsets[i]
                points[i].assign(cloth_points + offsets[i])
                seen_times[i].append(stamp)
            state.body_q.assign(poses)
            viewer.begin_frame(now - started)
            viewer.log_state(state)
            for i, point_array in enumerate(points):
                for group, (color, indices) in enumerate(groups):
                    viewer.log_mesh(
                        f"/live_sock_{i}/stripe_{group}",
                        point_array,
                        indices,
                        color=color,
                        roughness=0.95,
                        backface_culling=False,
                    )
            viewer.end_frame()
            if not viewer.is_running():
                break
            if writer is None and any(row["updated"] for row in current):
                writer = FFMPEG_VideoWriter(
                    str(args.output_dir / "parallel_rooms.mp4"),
                    (args.width, args.height),
                    args.fps,
                    codec="libx264",
                    preset="medium",
                    ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
                )
            if writer is not None and now - last_write >= 1 / args.fps:
                pixels = _capture_frame(viewer)
                writer.write_frame(pixels)
                Image.fromarray(pixels).save(args.output_dir / "latest.jpg")
                captured += 1
                last_write = now
            if all(proc.poll() is not None for proc in workers):
                finished_at = now if finished_at is None else finished_at
                if now - finished_at >= args.hold_seconds:
                    break
            if now - started > args.wall_timeout:
                raise TimeoutError("Parallel rooms exceeded the wall-time limit.")
            time.sleep(0.005)
    finally:
        _stop_workers(workers)
        for log in logs:
            log.close()
        if writer is not None:
            writer.close()
        viewer.close()
        report = {
            "architecture": "Independent Isaac Lab physics/policy workers; shared live Newton GUI",
            "prerecorded_motion_used": False,
            "num_rooms": args.num_rooms,
            "workers": commands,
            "worker_exit_codes": [proc.returncode for proc in workers],
            "final_live_states": current,
            "observed_simulation_times_s": seen_times,
            "frames_captured": captured,
            "elapsed_wall_s": time.monotonic() - started,
            "reference_run": str(args.reference_run.resolve()),
            "policy_sha256": {
                name: hashlib.sha256(Path(parameters[name]).read_bytes()).hexdigest()
                for name in ("standing_policy", "pickup_policy", "grasp_policy", "walking_policy")
                if parameters.get(name) is not None
            },
        }
        (args.output_dir / "parallel_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(proc.returncode != 0 for proc in workers) or not all(row["updated"] for row in current):
        raise RuntimeError("A parallel worker failed or did not publish simulation states; inspect worker.log.")


def main() -> None:
    """Parse complete-room count, recording settings, and bounded run duration [s]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_run", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_rooms", type=int, default=4)
    parser.add_argument("--duration", type=float, default=70.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--hold_seconds", type=float, default=10.0)
    parser.add_argument("--wall_timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if args.num_rooms < 1 or min(args.duration, args.fps, args.width, args.height, args.wall_timeout) <= 0:
        parser.error("Room count, durations, frame rate, and dimensions must be positive.")
    run(args)


if __name__ == "__main__":
    main()
