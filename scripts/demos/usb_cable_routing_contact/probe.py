# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure an actuated contact grasp, then optionally open the fingers to test release.

This is a stationary mechanical probe, not a routing policy or a scored evaluation.
It captures the first native terminal state before reset and never retries an episode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from PIL import Image

from isaaclab.app import launch_simulation
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def write_json(path: Path, value: dict | list) -> None:
    """Write finite JSON measurements, replacing nonfinite float values with null."""

    def clean(item):
        if isinstance(item, dict):
            return {key: clean(element) for key, element in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(element) for element in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


class ContactProbe:
    """Record spring-free holding, actual pad forces [N], and relative connector slip [m, rad]."""

    def __init__(self, env: ManagerBasedRLEnv, args: argparse.Namespace):
        self.env = env
        self.args = args
        self.command = env.command_manager.get_term("routing")
        self.arm = env.action_manager.get_term("arm")
        self.cable = self.command.cable
        self.robot = self.command.robot
        self.step_count = 0
        self.opened = False
        self.open_step: int | None = None
        self.terminal: dict | None = None
        self.trace: list[dict] = []
        self.poses: list[np.ndarray] = []
        self.writer = None
        self.visualizer = env.sim.visualizers[0] if env.sim.visualizers else None
        if self.visualizer is not None:
            self.visualizer._viewer.renderer.window.set_visible(args.show_gui)
            self.visualizer._runtime_headless = not args.show_gui
            tcp, _ = self.arm._compute_frame_pose()
            tcp += self.robot.data.root_pos_w.torch - env.scene.env_origins
            self.visualizer.set_camera_view((tcp[0] + tcp.new_tensor((0.16, 0.18, 0.10))).tolist(), tcp[0].tolist())
        if args.video:
            self.writer = imageio_ffmpeg.write_frames(
                str(args.output_dir / "grasp_probe.mp4"),
                (args.width, args.height),
                fps=round(1 / env.step_dt),
                codec="libx264",
                pix_fmt_out="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "18", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"],
            )
            self.writer.send(None)
        self._original_reset = env._reset_idx

        def capture_terminal(env_ids):
            if bool(env.termination_manager.dones[0]):
                self.snapshot()
                self.terminal = {
                    **self.trace[-1],
                    "terminations": {
                        name: bool(env.termination_manager.get_term(name)[0])
                        for name in env.termination_manager.active_terms
                    },
                }
                self.capture_frame("terminal.png")
                # Stop after env.step returns, preserving the measured physical terminal state.
                return
            raise RuntimeError("The mechanical probe does not allow an additional reset.")

        env._reset_idx = capture_terminal
        self.snapshot()
        self.capture_frame("reset.png")
        self.save_metadata()

    def measure(self) -> dict:
        """Read cable, joints, contact loads [N, N m], and relative housing pose [m, xyzw]."""
        c = self.command
        pose = wp.to_torch(NewtonManager.get_state().body_q)
        wrist = pose[wp.to_torch(self.cable._wrist_ids).long()][0]
        plug = pose[self.cable.body_ids[0, -1]]
        relative_position = quat_apply_inverse(wrist[3:].unsqueeze(0), (plug[:3] - wrist[:3]).unsqueeze(0))[0]
        relative_quat = quat_mul(quat_inv(wrist[3:]), plug[3:])
        row = {
            "step": self.step_count,
            "time_s": self.step_count * self.env.step_dt,
            "phase": "release" if self.opened else "hold",
            "plug_pose_m_xyzw": plug.tolist(),
            "wrist_pose_m_xyzw": wrist.tolist(),
            "plug_position_in_wrist_m": relative_position.tolist(),
            "plug_quat_in_wrist_xyzw": relative_quat.tolist(),
            "robot_joint_position_rad": self.robot.data.joint_pos.torch[0].tolist(),
            "cable_points_m": c.points()[0].tolist(),
            "minimum_bend_radius_m": float(c.minimum_bend_radius[0]),
            "stretch_ratio": float(c.stretch()[0]),
            "guides_completed": int(c.completed[0]),
            "guides_retained": c.seated[0].tolist(),
            "skip": bool(c.skip[0]),
        }
        for field, key in (
            ("grasp_active", "grasp_active"),
            ("grasp_acquired", "grasp_acquired"),
            ("grasp_dropped", "grasp_dropped"),
            ("pad_normal_forces", "pad_normal_forces_n"),
            ("finger_contact_force", "finger_contact_force_n"),
            ("socket_contact_force", "socket_contact_force_n"),
            ("position_slip", "position_slip_m"),
            ("rotation_slip", "rotation_slip_rad"),
            ("contact_loss_time", "contact_loss_time_s"),
            ("acquisition_time", "acquisition_time_s"),
            ("tension", "net_finger_contact_force_n"),
            ("grasp_torque", "net_finger_contact_torque_nm"),
        ):
            row[key] = wp.to_torch(getattr(self.cable, field))[0].tolist()
        if hasattr(self.cable, "pad_contact_count"):
            row["pad_contact_count"] = wp.to_torch(self.cable.pad_contact_count)[0].tolist()
        return row

    def save_metadata(self) -> None:
        """Save the actual reset geometry and model properties in Newton SI units."""
        model = NewtonManager.get_model()
        bodies = self.cable.body_ids[0].tolist()
        shape_ids = [
            index
            for index, label in enumerate(model.shape_label)
            if "/Cable/" in label or ("/Robot/" in label and any(word in label for word in ("finger", "knuckle")))
        ]
        shapes = [{"index": index, "label": model.shape_label[index]} for index in shape_ids]
        for attr in (
            "shape_body",
            "shape_type",
            "shape_transform",
            "shape_scale",
            "shape_flags",
            "shape_material_mu",
            "shape_material_ke",
            "shape_material_kd",
        ):
            data = getattr(model, attr).numpy()
            for shape, index in zip(shapes, shape_ids, strict=True):
                shape[attr] = data[index].tolist()
        callbacks = getattr(NewtonManager, "_state_force_callbacks", ())
        metadata = {
            "argv": sys.argv,
            "task": self.env.spec.id if self.env.spec else self.args.variant,
            "seed": self.args.seed,
            "randomized": self.args.randomize,
            "physics_dt_s": self.env.physics_dt,
            "control_dt_s": self.env.step_dt,
            "physics_substeps": self.env.cfg.sim.physics.num_substeps,
            "cuda_graph": self.env.cfg.sim.physics.use_cuda_graph,
            "vbd_compliant_alm": getattr(
                self.env.cfg.sim.physics.solver_cfg.entries[1].solver_cfg, "rigid_compliant_alm", None
            ),
            "coupler_iterations": self.env.cfg.sim.physics.solver_cfg.iterations,
            "gripper_initial_position_rad": self.command.cfg.gripper_initial_position,
            "gripper_closed_position_rad": self.command.cfg.gripper_closed_position,
            "robot_joint_names": self.robot.joint_names,
            "body_labels": model.body_label,
            "body_mass_kg": model.body_mass.numpy().tolist(),
            "cable_body_ids": bodies,
            "cable_and_connector_total_mass_kg": float(model.body_mass.numpy()[bodies].sum()),
            "cable_body_inertia_kg_m2": model.body_inertia.numpy()[bodies].tolist(),
            "cable_body_flags": model.body_flags.numpy()[bodies].tolist(),
            "grasp_and_cable_shapes": shapes,
            "relevant_collision_filter_pairs": [
                list(pair) for pair in sorted(model.shape_collision_filter_pairs) if set(pair).intersection(shape_ids)
            ],
            "contact_grasp_limits": {
                name: getattr(self.cable.cfg, name)
                for name in (
                    "acquisition_timeout_s",
                    "contact_loss_timeout_s",
                    "minimum_pad_force",
                    "maximum_position_slip",
                    "maximum_rotation_slip",
                )
            },
            "registered_force_callbacks": [getattr(callback, "__qualname__", str(callback)) for callback in callbacks],
            "guide_positions_m": self.command.guide_positions[0].tolist(),
            "guide_yaws_rad": self.command.guide_yaws[0].tolist(),
            "initial": self.trace[0],
        }
        write_json(self.args.output_dir / "reset_model.json", metadata)

    def snapshot(self) -> None:
        """Record the current control-step state without advancing physics."""
        self.trace.append(self.measure())
        self.poses.append(NewtonManager.get_state().body_q.numpy().copy())

    def capture_frame(self, filename: str | None = None, *, video_frame: bool = True) -> None:
        """Optionally render the current Newton state at the fixed gripper close-up camera."""
        if self.visualizer is None or (self.writer is None and not (filename and self.args.images)):
            return
        self.visualizer._state = NewtonManager.get_state()
        pixels = self.visualizer.render_rgb_array()
        if filename and self.args.images:
            Image.fromarray(pixels).save(self.args.output_dir / filename)
        if self.writer is not None and video_frame:
            self.writer.send(pixels)

    def open_gripper(self) -> None:
        """Command open fingers [rad] through their actuators, leaving the arm target unchanged."""
        self.capture_frame("before_open.png", video_frame=False)
        self.opened = True
        self.open_step = self.step_count
        self.command.cfg.gripper_closed_position = self.args.open_position
        joint_ids = []
        values = []
        for index, name in enumerate(self.robot.joint_names):
            if "finger" in name or "knuckle" in name:
                sign = -1 if float(self.robot.data.default_joint_pos.torch[0, index]) < 0 else 1
                joint_ids.append(index)
                values.append(sign * self.args.open_position)
        targets = torch.tensor([values], device=self.env.device)
        self.robot.actuators.target_command.set_position_index(value=targets, joint_ids=joint_ids)

    def run(self) -> None:
        """Hold the arm, optionally open the fingers, and stop at the first native terminal."""
        action = torch.zeros((1, self.env.action_manager.total_action_dim), device=self.env.device)
        try:
            with torch.inference_mode():
                for _ in range(self.args.steps):
                    if self.args.open_after is not None and self.step_count == self.args.open_after:
                        self.open_gripper()
                    self.step_count += 1
                    _, _, terminated, truncated, _ = self.env.step(action)
                    if bool(terminated[0] | truncated[0]):
                        if self.terminal is None:
                            raise RuntimeError("Native episode ended without a captured terminal state.")
                        break
                    self.snapshot()
                    self.capture_frame()
            if self.terminal is None:
                self.capture_frame("final.png", video_frame=False)
        finally:
            if self.writer is not None:
                self.writer.close()
            self.env._reset_idx = self._original_reset
            self.save()

    def save(self) -> None:
        """Publish measurements and separate holding/release verdicts without treating routing as solved."""
        hold = [row for row in self.trace if row["phase"] == "hold"]
        release = [row for row in self.trace if row["phase"] == "release"]
        hold_tail = hold[-min(30, len(hold)) :]
        acquired = any(bool(row["grasp_acquired"]) for row in hold)
        completed_hold = hold[-1]["step"] >= (self.args.open_after or self.args.steps)
        hold_failed = self.terminal is not None and self.terminal["phase"] == "hold"
        hold_pass = (
            acquired
            and completed_hold
            and not hold_failed
            and all(bool(row["grasp_active"]) and not row["grasp_dropped"] for row in hold_tail)
        )
        release_detected = any(bool(row["grasp_dropped"]) for row in release)
        result = {
            "variant": self.args.variant,
            "probe": "stationary actuated finger contact hold followed by optional release",
            "routing_evaluation": False,
            "steps": self.step_count,
            "native_terminal": self.terminal,
            "additional_resets": 0,
            "hold_acquired": acquired,
            "hold_pass": hold_pass,
            "hold_peak_position_slip_m": max(row["position_slip_m"] for row in hold),
            "hold_peak_rotation_slip_rad": max(row["rotation_slip_rad"] for row in hold),
            "hold_tail_min_pad_force_n": np.min([row["pad_normal_forces_n"] for row in hold_tail], axis=0).tolist(),
            "release_command_step": self.open_step,
            "release_detected": release_detected if self.opened else None,
            "hold_then_release_pass": hold_pass and release_detected if self.args.open_after is not None else None,
        }
        write_json(self.args.output_dir / "trace.json", self.trace)
        write_json(self.args.output_dir / "result.json", result)
        np.savez_compressed(
            self.args.output_dir / "trajectory.npz",
            body_pose_m_xyzw=np.stack(self.poses),
            body_labels=np.array(NewtonManager.get_model().body_label),
            time_s=np.array([row["time_s"] for row in self.trace]),
        )
        concise = {key: value for key, value in result.items() if key != "native_terminal"}
        print("CONTACT_PROBE_RESULT " + json.dumps(concise, allow_nan=False), flush=True)


def main() -> None:
    """Launch one deterministic contact-grasp mechanical probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scurve", "insert"), default="scurve")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--randomize", action="store_true", help="Enable task randomization; disabled by default.")
    parser.add_argument("--gripper_initial_position", type=float, help="Initial finger driver angle [rad].")
    parser.add_argument("--gripper_closed_position", type=float, help="Holding finger driver target [rad].")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--open_after", type=int, help="Open fingers after this many control steps; otherwise hold.")
    parser.add_argument("--open_position", type=float, default=0.0, help="Release finger driver target [rad].")
    parser.add_argument("--no_graph", action="store_true", help="Disable CUDA graph capture for debugging.")
    parser.add_argument(
        "--vbd_compliant_alm", action="store_true", help="Test VBD compliant ALM in this development probe only."
    )
    parser.add_argument("--coupler_iterations", type=int, help="Override proxy relaxation passes for this probe.")
    parser.add_argument("--images", action="store_true", help="Save reset, pre-opening, and terminal close-up images.")
    parser.add_argument("--video", action="store_true", help="Save a close-up video at the simulation frame rate.")
    parser.add_argument("--show_gui", action="store_true", help="Show the Newton GL window.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.steps < 1 or (args.open_after is not None and not 0 < args.open_after < args.steps):
        parser.error("Require steps > 0 and, when specified, 0 < open_after < steps.")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("Video dimensions must be positive even integers.")
    if args.coupler_iterations is not None and args.coupler_iterations < 1:
        parser.error("coupler_iterations must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output_dir must be absent or empty; prior probe results are never overwritten.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = (
        "IsaacContrib-Route-USB-SCurve-Insert-Contact-UR10e"
        if args.variant == "insert"
        else "IsaacContrib-Route-USB-SCurve-Contact-UR10e"
    )
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.guide_count = 3
    cfg.commands.routing.randomize = args.randomize
    cfg.commands.routing.randomize_guides = args.randomize
    for name in ("gripper_initial_position", "gripper_closed_position"):
        if getattr(args, name) is not None:
            setattr(cfg.commands.routing, name, getattr(args, name))
    if args.gripper_initial_position is not None:
        cfg.commands.routing.gripper_initial_joint_positions = None
    cfg.sim.physics.use_cuda_graph = not args.no_graph
    if args.vbd_compliant_alm:
        cfg.sim.physics.solver_cfg.entries[1].solver_cfg.rigid_compliant_alm = True
    if args.coupler_iterations is not None:
        cfg.sim.physics.solver_cfg.iterations = args.coupler_iterations
    cfg.sim.visualizer_cfgs = []
    if args.images or args.video or args.show_gui:
        cfg.sim.physics.load_visual_shapes = True
        cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                window_width=args.width,
                window_height=args.height,
                focal_length=35.0,
                headless=False,
                enable_picking=False,
                enable_markers=False,
                enable_live_plots=False,
                streaming_view=False,
            )
        ]
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            probe = ContactProbe(env, args)
            print(f"CONTACT_PROBE_READY {args.output_dir.resolve()}", flush=True)
            probe.run()
        finally:
            env.close()


if __name__ == "__main__":
    main()
