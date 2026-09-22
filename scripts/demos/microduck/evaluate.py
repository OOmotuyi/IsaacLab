# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy trained in the native Isaac Lab MicroDuck environment."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.contrib.microduck.microduck_env_cfg import MicroDuckFlatEnvCfg


def main() -> None:
    """Run a fixed-command evaluation and optionally record the first robot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--velocity", type=float, nargs=3, default=(0.15, 0.0, 0.0))
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--offscreen", action="store_true", help="Render with a headless Newton GL visualizer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--randomized", action="store_true", help="Include training noise, randomization, and pushes.")
    parser.add_argument("--real_time", action="store_true", help="Pace the viewer to simulation time.")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.duration <= 0 or args.num_envs < 1:
        parser.error("duration and num_envs must be positive")

    cfg = MicroDuckFlatEnvCfg()
    if not args.randomized:
        cfg.play_mode()
    cfg.curriculum.walking = None
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 2.0
    cfg.episode_length_s = args.duration + 1.0
    cfg.commands.twist.ranges.lin_vel_x = (args.velocity[0],) * 2
    cfg.commands.twist.ranges.lin_vel_y = (args.velocity[1],) * 2
    cfg.commands.twist.ranges.ang_vel_z = (args.velocity[2],) * 2
    cfg.commands.twist.rel_standing_envs = 0.0
    cfg.commands.twist.turn_in_place_fraction = 0.0
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.reset_root.params["pose_range"] = {"z": (0.0, 0.01)}
    cfg.sim.default_visualizer_cfg.focal_length = 20.0
    cfg.sim.default_visualizer_cfg.max_visible_envs = 1
    cfg.sim.default_visualizer_cfg.randomly_sample_visible_envs = False

    if args.offscreen:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True)]

    with launch_simulation(cfg.sim, args):
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        from isaaclab_tasks.contrib.microduck.agents import MicroDuckPPORunnerCfg
        from isaaclab_tasks.contrib.microduck.microduck_env import MicroDuckEnv

        env = MicroDuckEnv(cfg)
        wrapper = RslRlVecEnvWrapper(env)
        agent_cfg = handle_deprecated_rsl_rl_cfg(MicroDuckPPORunnerCfg(), version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        observations = wrapper.get_observations()
        steps = round(args.duration / env.step_dt)
        failures = torch.zeros(env.num_envs, device=env.device)
        velocity_sum = torch.zeros(3, device=env.device)
        squared_error = torch.zeros(3, device=env.device)
        speed_sum = torch.zeros((), device=env.device)
        playback_samples = []
        heights = torch.zeros(env.num_envs, device=env.device)
        upright = torch.zeros(env.num_envs, device=env.device)
        target = torch.tensor(args.velocity, device=env.device)
        writer = None
        viewers = tuple(env.sim.visualizers)
        try:
            if args.record:
                if not env.sim.visualizers:
                    raise ValueError("--record requires --viz newton_gl or another local visualizer.")
                from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

                args.record.parent.mkdir(parents=True, exist_ok=True)
                frame = env.sim.visualizers[0].render_rgb_array()
                writer = FFMPEG_VideoWriter(
                    str(args.record), (frame.shape[1], frame.shape[0]), fps=round(1 / env.step_dt), codec="libx264"
                )
            with torch.inference_mode():
                completed_steps = 0
                start = time.perf_counter()
                for _ in range(steps):
                    if viewers and all(v.is_closed for v in viewers):
                        break
                    observations, _, dones, _ = wrapper.step(policy(observations))
                    policy.reset(dones)
                    failures += env.termination_manager.terminated.float()
                    data = env.scene["robot"].data
                    velocity = torch.cat((data.root_link_lin_vel_b.torch[:, :2], data.root_ang_vel_b.torch[:, 2:3]), -1)
                    velocity_sum += velocity.mean(0)
                    squared_error += (velocity - target).square().mean(0)
                    speed = data.root_link_lin_vel_w.torch[:, :2].norm(dim=-1)
                    speed_sum += speed.mean()
                    if args.record:
                        playback_samples.append(
                            {
                                "time_s": completed_steps * env.step_dt,
                                "simulation_time_s": (completed_steps + 1) * env.step_dt,
                                "command": args.velocity,
                                "body_twist": velocity[0].cpu().tolist(),
                                "ground_speed_m_s": float(speed[0]),
                            }
                        )
                    heights += data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                    upright += (data.projected_gravity_b.torch[:, 2] < -0.9).float()
                    if env.sim.visualizers:
                        x, y, z = data.root_link_pos_w.torch[0].cpu().tolist()
                        env.sim.set_camera_view((x + 0.6, y - 0.7, z + 0.35), (x, y, z + 0.02))
                        env.sim.render()
                        if writer:
                            if not env.sim.visualizers:
                                raise RuntimeError("The viewer closed before the recording completed.")
                            writer.write_frame(env.sim.visualizers[0].render_rgb_array())
                    completed_steps += 1
                    if args.real_time:
                        time.sleep(max(0.0, start + completed_steps * env.step_dt - time.perf_counter()))
            steps = completed_steps
            if not steps:
                raise RuntimeError("The viewer closed before any policy steps were simulated.")
            result = {
                "checkpoint": str(args.checkpoint.resolve()),
                "num_envs": env.num_envs,
                "seed": args.seed,
                "randomized": args.randomized,
                "duration_s": steps * env.step_dt,
                "command": args.velocity,
                "mean_body_twist": (velocity_sum / steps).cpu().tolist(),
                "twist_rmse": (squared_error / steps).sqrt().cpu().tolist(),
                "mean_ground_speed_m_s": float(speed_sum / steps),
                "mean_root_height_m": float(heights.mean() / steps),
                "upright_fraction": float(upright.mean() / steps),
                "survived_without_reset_fraction": float((failures == 0).float().mean()),
                "fall_resets": int(failures.sum()),
            }
            if args.record:
                result["playback_samples"] = playback_samples
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key != "playback_samples"}, indent=2))
        finally:
            if writer:
                writer.close()
            wrapper.close()


if __name__ == "__main__":
    main()
