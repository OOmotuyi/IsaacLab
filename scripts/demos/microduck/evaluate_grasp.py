# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate measured contact, lift, and balance for the object-conditioned grasp actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from grasp_checkpoint import make_grasp_runner
from tensordict import TensorDict

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.contrib.microduck.grasp_env_cfg import MicroDuckMarkerGraspEnvCfg


def main() -> None:
    """Evaluate a checkpoint at a fixed object height [m] for a duration [s]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--object_height", type=float, default=0.0395)
    parser.add_argument("--reach_height", type=float, help="Override the curriculum goal height [m] for diagnostics.")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--training_reset", action="store_true", help="Diagnostic: use the checkpoint's robot reset pose."
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument(
        "--sample_actions", action="store_true", help="Diagnostic: sample the actor's exploration noise."
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.duration <= 0:
        parser.error("num_envs and duration must be positive.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    cfg = MicroDuckMarkerGraspEnvCfg()
    if args.training_reset:
        import yaml

        # Read only scalar reset data; saved configs also contain Python-specific tags.
        saved = yaml.load((args.checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
        reset = saved["scene"]["robot"]["init_state"]
        cfg.scene.robot.init_state.pos = tuple(map(float, reset["pos"]))
        cfg.scene.robot.init_state.rot = tuple(map(float, reset["rot"]))
        cfg.scene.robot.init_state.joint_pos = {name: float(value) for name, value in reset["joint_pos"].items()}
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.commands.grasp.initial_object_height = args.object_height
    cfg.commands.grasp.initial_reach_height = args.object_height if args.reach_height is None else args.reach_height
    cfg.commands.grasp.curriculum = False
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = args.duration + 0.1
    cfg.sim.default_visualizer_cfg.focal_length = 20.0
    cfg.sim.default_visualizer_cfg.max_visible_envs = 1
    cfg.sim.default_visualizer_cfg.randomly_sample_visible_envs = False
    if args.offscreen:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.sim.physics.load_visual_shapes = True
        cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                headless=True, focal_length=20.0, max_visible_envs=1, randomly_sample_visible_envs=False
            )
        ]
    task_source = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib/microduck"
    source_paths = [
        task_source / name for name in ("grasp_contract.py", "grasp_env_cfg.py", "mdp/grasp.py", "microduck_env.py")
    ]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    with launch_simulation(cfg.sim, args):
        from isaaclab_tasks.contrib.microduck.microduck_env import MicroDuckGraspEnv

        env = MicroDuckGraspEnv(cfg)
        writer = None
        try:
            runner = make_grasp_runner()
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=env.device)
            std = runner.alg.actor.state_dict()["distribution.std_param"].to(env.device)
            obs, _ = env.reset(seed=args.seed)
            if args.record or args.preview:
                if not env.sim.visualizers:
                    raise ValueError("Recording or preview requires a visualizer.")
                from PIL import Image

                x, y, _ = env.scene["robot"].data.root_link_pos_w.torch[0].tolist()
                env.sim.set_camera_view((x + 0.42, y - 0.42, 0.30), (x + 0.05, y, 0.12))
                env.sim.render()
                if args.record:
                    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

                    frame = env.sim.visualizers[0].render_rgb_array()
                    args.record.parent.mkdir(parents=True, exist_ok=True)
                    writer = FFMPEG_VideoWriter(
                        str(args.record), (frame.shape[1], frame.shape[0]), fps=round(1 / env.step_dt), codec="libx264"
                    )
            grasp = env.command_manager.get_term("grasp")
            success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            opposed = torch.zeros_like(success)
            failed = torch.zeros_like(success)
            max_lift = torch.zeros(env.num_envs, device=env.device)
            min_error = torch.full_like(max_lift, float("inf"))
            min_root = torch.full_like(max_lift, float("inf"))
            trace = []
            start = time.perf_counter()
            with torch.inference_mode():
                for step in range(round(args.duration / env.step_dt)):
                    action = policy(TensorDict(obs, batch_size=[env.num_envs]))
                    if args.sample_actions:
                        action = action + torch.randn_like(action) * std
                    obs, _, terminated, truncated, _ = env.step(action)
                    grasp.refresh()
                    if writer is not None or args.preview and step == 25:
                        env.sim.render()
                        frame = env.sim.visualizers[0].render_rgb_array()
                        if writer is not None:
                            writer.write_frame(frame)
                        if args.preview and step == 25:
                            args.preview.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(frame).save(args.preview)
                    live = ~failed
                    success |= live & grasp.success_seen
                    opposed |= live & grasp.contacts.all(-1)
                    max_lift = torch.where(live, torch.maximum(max_lift, grasp.lift), max_lift)
                    min_error = torch.where(live, torch.minimum(min_error, grasp.distance), min_error)
                    root = env.scene["robot"].data.root_link_pos_w.torch - env.scene.env_origins
                    min_root = torch.where(live, torch.minimum(min_root, root[:, 2]), min_root)
                    failed |= terminated | truncated
                    if step % 10 == 0:
                        trace.append(
                            {
                                "time_s": step * env.step_dt,
                                "success_count": int(success.sum()),
                                "failure_count": int(failed.sum()),
                                "opposed_count": int(grasp.contacts.all(-1).sum()),
                                "stage_counts": torch.bincount(grasp.stage, minlength=4).cpu().tolist(),
                                "root_m": root[0].cpu().tolist(),
                                "root_quaternion_xyzw": env.scene["robot"]
                                .data.root_link_quat_w.torch[0]
                                .cpu()
                                .tolist(),
                                "joint_position_rad": env.scene["robot"].data.joint_pos.torch[0].cpu().tolist(),
                                "pinch_m": grasp.pinch[0].cpu().tolist(),
                                "goal_m": grasp.goal[0].cpu().tolist(),
                                "goal_error_m": float((grasp.goal[0] - grasp.pinch[0]).norm()),
                                "head_action_rad": action[0, 5:9].cpu().tolist(),
                                "raw_action": action[0].cpu().tolist(),
                                "object_m": env.scene["object"].data.root_link_pos_w.torch[0].cpu().tolist(),
                                "jaw_rad": float(env.scene["robot"].data.joint_pos.torch[0, grasp.jaw_ids[0]]),
                                "head_pose_m_xyzw": env.scene["robot"]
                                .data.body_link_pose_w.torch[0, grasp.head_id]
                                .cpu()
                                .tolist(),
                            }
                        )
            held_at_end = grasp.contacts.all(-1) & (grasp.lift_time >= 0.30)
            report = {
                "environment_source_sha256": source_hashes,
                "source_changed_during_evaluation": any(
                    hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
                    for path, digest in source_hashes.items()
                ),
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed,
                "num_envs": env.num_envs,
                "object_height_m": args.object_height,
                "reach_goal_height_m": cfg.commands.grasp.initial_reach_height,
                "duration_s": args.duration,
                "physics_dt_s": cfg.sim.dt,
                "control_dt_s": env.step_dt,
                "physics_substeps": cfg.sim.physics.num_substeps,
                "load_visual_shapes": cfg.sim.physics.load_visual_shapes,
                "friction_cone": cfg.sim.physics.solver_cfg.cone,
                "sample_actions": args.sample_actions,
                "wall_seconds": time.perf_counter() - start,
                "success_count": int(success.sum()),
                "stable_success_count": int((success & ~failed & held_at_end).sum()),
                "held_at_end_count": int((held_at_end & ~failed).sum()),
                "randomized_resets": False,
                "robot_reset": "training_configuration" if args.training_reset else "standing",
                "initial_root_pose_m_xyzw": list(cfg.scene.robot.init_state.pos) + list(cfg.scene.robot.init_state.rot),
                "environment_origins_m": env.scene.env_origins.cpu().tolist(),
                "trace_position_frames": {
                    "root_m": "environment",
                    "pinch_m": "world",
                    "goal_m": "world",
                    "object_m": "world",
                    "head_pose_m_xyzw": "world",
                },
                "joint_names": env.scene["robot"].joint_names,
                "opposed_contact_count": int(opposed.sum()),
                "failed_count": int(failed.sum()),
                "maximum_lift_m": max_lift.cpu().tolist(),
                "final_lift_m": grasp.lift.cpu().tolist(),
                "final_opposed_contacts": grasp.contacts.all(-1).cpu().tolist(),
                "minimum_reach_error_m": min_error.cpu().tolist(),
                "minimum_root_height_m": min_root.cpu().tolist(),
                "trace": trace,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in report.items()
                        if k not in {"trace", "maximum_lift_m", "minimum_reach_error_m", "minimum_root_height_m"}
                    },
                    indent=2,
                )
            )
        finally:
            if writer is not None:
                writer.close()
            env.close()


if __name__ == "__main__":
    main()
