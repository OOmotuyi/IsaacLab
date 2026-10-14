# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render measured direct-Astra episodes without loading any policy or advancing physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
from isaaclab_newton.physics import NewtonManager
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from PIL import Image

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main() -> None:
    """Export matching native-GUI and clean 1080p videos of a verified successful episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.episode_dir / "success.json").read_text())
    if not (report.get("terminal") or {}).get("success") or report["rl_policy_loaded"]:
        parser.error("Require a verified successful episode that did not load an RL policy.")
    insertion = report["variant"] == "insert"
    task = "IsaacContrib-Route-USB-SCurve-Insert-UR10e" if insertion else "IsaacContrib-Route-USB-SCurve-UR10e"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = report["seed"]
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.guide_count = 3
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            window_width=1920,
            window_height=1080,
            headless=False,
            enable_picking=False,
            enable_markers=False,
            enable_live_plots=False,
            streaming_view=False,
        )
    ]
    archive = np.load(args.episode_dir / "trajectory.npz")
    trace = json.loads((args.episode_dir / "trace.json").read_text())
    poses = archive["body_pose_m_xyzw"]
    if len(trace) != len(poses) or len(poses) != report["policy_steps"] + 1:
        raise ValueError("Trace and measured pose counts do not match the physical episode.")
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        writers = {}
        try:
            env.reset(seed=cfg.seed)
            visualizer = env.sim.visualizers[0]
            viewer = visualizer._viewer
            viewer.renderer.window.set_visible(False)
            model = NewtonManager.get_model()
            if list(archive["body_labels"]) != list(model.body_label):
                raise ValueError("Recorded body ordering differs from the rendering model.")
            state = model.state()
            current = {"index": 0}

            def panel(imgui):
                row = trace[current["index"]]
                imgui.text("DIRECT ASTRA CONTROL")
                imgui.text("Model-selected moves + robot IK")
                imgui.text("No RL policy loaded")
                imgui.text("Recorded physical states, 30 fps")
                imgui.separator()
                imgui.text(f"Time: {row['time_s']:.2f} s")
                imgui.text(f"Guides: {row['guides_completed']} / 3")
                if insertion and row["ready_to_approach"]:
                    imgui.text(f"Insertion: {row['depth_m'] * 1000:.2f} mm")
                if current["index"] == len(trace) - 1:
                    imgui.text("SUCCESS verified by environment")

            viewer.register_ui_callback(panel, position="panel")
            visualizer._log_pending_meshes()
            for kind in ("clean", "newton_gui"):
                writer = imageio_ffmpeg.write_frames(
                    str(args.episode_dir / f"astra_{report['variant']}_{kind}_1080p.mp4"),
                    (1920, 1080),
                    fps=30,
                    codec="libx264",
                    pix_fmt_out="yuv420p",
                    macro_block_size=1,
                    output_params=["-crf", "16", "-preset", "medium", "-movflags", "+faststart"],
                )
                writer.send(None)
                writers[kind] = writer
            maximum_pose_error = 0.0
            for index, row in enumerate(trace):
                current["index"] = index
                if insertion and row["ready_to_approach"]:
                    eye, lookat, focal = (0.663, -0.40, 0.095), (0.668, -0.15, 0.065), 38.0
                else:
                    eye, lookat, focal = (0.86, 0.16, 0.48), (0.53, -0.15, 0.10), 24.0
                visualizer.set_camera_view(eye, lookat)
                visualizer.cfg.focal_length = focal
                visualizer._apply_camera_focal_length()
                state.body_q.assign(poses[index])
                viewer.begin_frame(row["time_s"])
                viewer.log_state(state)
                viewer.end_frame()
                clean = viewer.get_frame(render_ui=False).numpy()
                gui = viewer.get_frame(render_ui=True).numpy()
                if clean.shape != (1080, 1920, 3) or gui.shape != clean.shape:
                    raise ValueError("Unexpected native framebuffer dimensions.")
                writers["clean"].send(clean)
                writers["newton_gui"].send(gui)
                maximum_pose_error = max(maximum_pose_error, float(np.abs(state.body_q.numpy() - poses[index]).max()))
                if index in (0, len(trace) - 1):
                    Image.fromarray(clean).save(args.episode_dir / f"frame_{index:04d}_clean.png")
                    Image.fromarray(gui).save(args.episode_dir / f"frame_{index:04d}_gui.png")
                if index % 150 == 0:
                    print(f"ENCODE {index + 1}/{len(trace)}", flush=True)
            for _ in range(60):
                writers["clean"].send(clean)
                writers["newton_gui"].send(gui)
            for writer in writers.values():
                writer.close()
            writers.clear()
            result = {
                "resolution": [1920, 1080],
                "fps": 30,
                "frames": len(trace) + 60,
                "duration_s": (len(trace) + 60) / 30,
                "final_pose_hold_s": 2.0,
                "maximum_body_pose_replay_error": maximum_pose_error,
                "rl_policy_loaded": False,
                "source": "Measured direct Astra-selected action rollout; recorded-state rendering only.",
                "trajectory_sha256": hashlib.sha256((args.episode_dir / "trajectory.npz").read_bytes()).hexdigest(),
            }
            (args.episode_dir / "render_report.json").write_text(json.dumps(result, indent=2) + "\n")
            print("VIDEOS_COMPLETE " + json.dumps(result), flush=True)
        finally:
            for writer in writers.values():
                writer.close()
            env.close()


if __name__ == "__main__":
    main()
