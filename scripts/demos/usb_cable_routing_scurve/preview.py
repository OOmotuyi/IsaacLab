# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show the untrained S-curve scene with the robot holding the cable at reset."""

import argparse
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab.app import launch_simulation
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_tasks.contrib.usb_cable_routing.assets import ANCHOR, CABLE_LENGTH, SEGMENTS
from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import (
    CHANNEL_WIDTH,
    CRADLE_HEIGHT,
    ENTRY_WIDTH,
    GUIDE_COLOR_NAMES,
    GUIDE_COLORS,
    GUIDE_OFFSETS,
    GUIDE_YAWS,
    LIP_OVERHANG,
    LIP_UNDERSIDE,
)
from isaaclab_tasks.contrib.usb_cable_routing_scurve.env_cfg import UsbSCurveEnvCfg
from isaaclab_tasks.contrib.usb_cable_routing_scurve.parameters import (
    CABLE_BODY_MASS,
    DYNAMIC_FRICTION,
    MIN_BEND_RADIUS,
    RESTITUTION_TARGET,
    STATIC_FRICTION_TARGET,
)

EYE = (1.45, 0.55, 1.20)
LOOKAT = (0.40, -0.35, 0.32)


def main() -> None:
    """Run zero-action physics and optionally save simulation views and measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="Control steps; zero keeps the viewer open.")
    parser.add_argument("--output_dir", type=Path, default=Path("logs/usb_cable_routing_scurve/preview"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error("Headless preview requires a positive --steps limit.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = UsbSCurveEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 43
    cfg.commands.routing.guide_count = 3
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.randomize = False
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            eye=EYE,
            lookat=LOOKAT,
            window_width=1280,
            window_height=720,
            headless=args.headless,
        )
    ]
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output_dir),
                output_filename_prefix="preview",
                video_length=180,
                step_offset=35,
            )
        ]
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-SCurve-UR10e", cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            command = env.command_manager.get_term("routing")
            initial = command.points()[0].clone()
            count = resets = 0
            peak_force = peak_gap = 0.0
            min_radius = float(command.minimum_bend_radius.min())
            model = NewtonManager.get_model()
            masses = wp.to_torch(model.body_mass)[command.cable.body_ids[0]]
            report = {
                "task": "IsaacContrib-Route-USB-SCurve-UR10e",
                "trained": False,
                "controller": "zero Cartesian increments; fixed pre-grasp pose",
                "guide_offsets_m": GUIDE_OFFSETS,
                "guide_color_names": GUIDE_COLOR_NAMES,
                "guide_colors_rgb": GUIDE_COLORS,
                "cable_plus_free_connector_mass_kg": float(masses.sum()),
                "flexible_body_mass_kg": float(masses[:-1].sum()) + CABLE_BODY_MASS / SEGMENTS,
                "free_connector_mass_kg": float(masses[-1]) - CABLE_BODY_MASS / SEGMENTS,
                "segment_body_masses_kg": masses.cpu().tolist(),
                "connector_type": "approximate USB-C; rounded 8.4 x 2.6 x 7 mm metal shell",
                "segment_count": SEGMENTS,
                "contact_parameters": {
                    "active_vbd_friction": DYNAMIC_FRICTION,
                    "measured_cable_shape_friction": wp.to_torch(model.shape_material_mu)[command.cable_shapes[0]]
                    .unique()
                    .cpu()
                    .tolist(),
                    "static_friction_calibration_target": STATIC_FRICTION_TARGET,
                    "restitution_calibration_target": RESTITUTION_TARGET,
                    "separate_static_friction_supported": False,
                    "restitution_applied_by_vbd": False,
                },
                "bend_limit": {
                    "minimum_radius_m": MIN_BEND_RADIUS,
                    "enforcement": "task termination at control steps; no hard VBD rod limit",
                    "measurement": "mean adjacent chord length / (2 sin(turn_angle / 2))",
                },
                "mechanical_parameters_are_estimates": True,
                "calibration_note": (
                    "Fit effective moduli and contact targets to measured video; no video calibration performed."
                ),
                "cable_material": {
                    name: getattr(cfg.scene.cable.spawn.physics_material, name)
                    for name in (
                        "thickness",
                        "density",
                        "stretch_stiffness",
                        "bend_stiffness",
                        "shear_stiffness",
                        "twist_stiffness",
                    )
                },
                "guide_yaws_deg": [math.degrees(a) for a in GUIDE_YAWS],
                "cradle_height_m": CRADLE_HEIGHT,
                "channel_width_m": CHANNEL_WIDTH,
                "mouth_width_m": ENTRY_WIDTH,
                "lip_underside_m": LIP_UNDERSIDE,
                "lip_inward_projection_m": LIP_OVERHANG,
                "guide_tip_type": "fixed horizontal inward lips at 90 degrees to upright walls",
                "cable_rest_length_m": CABLE_LENGTH,
                "initial_centerline_m": initial.cpu().tolist(),
                "initial_sampled_length_m": float((initial[1:] - initial[:-1]).norm(dim=-1).sum()),
                "endpoint_distance_m": float((initial[-1] - initial[0]).norm()),
                "observation_shape": list(obs["policy"].shape),
            }
            print("S-curve preview running: three upright guides; no training or policy loaded.", flush=True)
            while (args.steps == 0 or count < args.steps) and env.sim.is_headless_or_exist_active_visualizer():
                before = time.monotonic()
                obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 3, device=env.device))
                count += 1
                resets += int((terminated | truncated).sum())
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                sampled_radius = (
                    command.last_minimum_bend_radius
                    if bool((terminated | truncated).any())
                    else command.minimum_bend_radius
                )
                min_radius = min(min_radius, float(sampled_radius.min()))
                peak_force = max(peak_force, float(wp.to_torch(command.cable.tension).max()))
                peak_gap = max(peak_gap, float(command.stretch().max()))
                if count == 30:
                    positions, rotations = [], []
                    for index in range(1, 4):
                        pose = env.scene[f"guide{index}"].data.root_state_w.torch[0]
                        positions.append((pose[:3] - env.scene.env_origins[0]).cpu().tolist())
                        rotations.append(pose[3:7].cpu().tolist())
                    report.update(guide_origins_m=positions, guide_rotations_xyzw=rotations)
                    measured = torch.tensor(positions)[:, :2] - torch.tensor(ANCHOR[:2])
                    torch.testing.assert_close(measured, torch.tensor(GUIDE_OFFSETS), atol=1e-5, rtol=0)
                    viewer = env.sim.visualizers[0]
                    tip = command.tip()[0].cpu().tolist()
                    cameras = (
                        ("overview", EYE, LOOKAT),
                        ("guides", (0.53, -0.64, 0.57), (0.41, -0.12, 0.015)),
                        ("top", (0.43, -0.119, 0.90), (0.43, -0.12, 0.0)),
                        ("lips", (0.405, -0.132, 0.070), (0.30, -0.15, 0.025)),
                        ("tip", (tip[0] + 0.15, tip[1], tip[2] + 0.025), tuple(tip)),
                    )
                    for name, eye, target in cameras:
                        viewer.set_camera_view(eye, target)
                        for _ in range(3):
                            env.sim.render()
                        frame = viewer.render_rgb_array()
                        if frame is None:
                            raise RuntimeError("The simulation visualizer returned no preview frame.")
                        iio.imwrite(args.output_dir / f"{name}.png", frame)
                    viewer.set_camera_view(EYE, LOOKAT)
                    (args.output_dir / "scene.json").write_text(json.dumps(report, indent=2) + "\n")
                    print("Saved overview, guide, top, lip and tip views, and measured scene geometry.", flush=True)
                if not args.headless:
                    time.sleep(max(0, env.step_dt - (time.monotonic() - before)))
            report.update(
                steps=count,
                automatic_resets=resets,
                max_grasp_force_n=peak_force,
                max_joint_gap_ratio=peak_gap,
                minimum_sampled_bend_radius_m=min_radius,
            )
            (args.output_dir / "scene.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            env.close()


if __name__ == "__main__":
    main()
