# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate learned routing or a physics probe on a fixed number of new episodes."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import launch_simulation
from isaaclab.visualizers import WindowCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.contrib.usb_cable_routing_zigzag.env_cfg import UsbZigzagEnvCfg, UsbZigzagPPORunnerCfg
from isaaclab_tasks.contrib.usb_cable_routing_zigzag.mdp import cable_bend_angles

parser = argparse.ArgumentParser(description=__doc__)
controller = parser.add_mutually_exclusive_group(required=True)
controller.add_argument("--checkpoint")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--guides", type=int, default=3)
parser.add_argument("--seed", type=int, default=910001)
parser.add_argument("--output", type=Path, default=Path("logs/usb_cable_routing_zigzag/evaluation.json"))
controller.add_argument("--probe", choices=("zero", "scripted", "random"))
parser.add_argument("--viz", choices=("none", "newton_gl"), default="none")
parser.add_argument("--video", type=Path)
parser.add_argument("--no_graph", action="store_true")
parser.add_argument("--trace", action="store_true", help="Log cable state every 25 control steps.")
parser.add_argument("--export_policy", type=Path)
args = parser.parse_args()
if args.export_policy and not args.checkpoint:
    parser.error("--export_policy requires --checkpoint.")
if args.episodes < 1 or args.num_envs < 1 or not 1 <= args.guides <= 3:
    parser.error("Require positive episodes/envs and one to three guides.")

cfg = UsbZigzagEnvCfg()
cfg.scene.num_envs = min(args.num_envs, args.episodes)
cfg.seed = args.seed
cfg.commands.routing.curriculum = False
cfg.commands.routing.guide_count = args.guides
cfg.sim.physics.use_cuda_graph = not args.no_graph
if args.viz != "none" or args.video:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            eye=(1.25, -1.25, 1.05),
            lookat=(0.35, 0.04, 0.35),
            window=WindowCfg(size=(1280, 720)),
            headless=args.viz == "none",
        )
    ]
if args.video:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    cfg.video_recorders = [
        VideoRecorderCfg(
            output_dir=str(args.video.parent),
            output_filename_prefix=args.video.stem,
            video_length=800,
            step_offset=5,
        )
    ]

records = []
with launch_simulation(cfg):
    env = gym.make("IsaacContrib-Route-USB-Zigzag-UR10e", cfg=cfg).unwrapped
    agent_cfg = handle_deprecated_rsl_rl_cfg(UsbZigzagPPORunnerCfg(), version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        if args.export_policy:
            args.export_policy.mkdir(parents=True, exist_ok=True)
            runner.export_policy_to_jit(path=str(args.export_policy), filename="policy.pt")
    c = env.command_manager.get_term("routing")
    obs = wrapped.get_observations()
    initial = [None] * env.num_envs
    quotas = [args.episodes // env.num_envs + (i < args.episodes % env.num_envs) for i in range(env.num_envs)]
    completed_per_env = [0] * env.num_envs
    duration = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    try:
        with torch.inference_mode():
            while len(records) < args.episodes:
                before = time.monotonic()
                for i in range(env.num_envs):
                    if initial[i] is None and completed_per_env[i] < quotas[i]:
                        initial[i] = {
                            "stiffness_scale": float(c.material[i, 0]),
                            "friction": float(c.material[i, 1]),
                            "initial_centerline_m": c.points()[i].cpu().tolist(),
                            "guide_centers_m": c.guide_positions[i].cpu().tolist(),
                        }
                if args.probe == "scripted":
                    arm = env.action_manager.get_term("arm")
                    pos, _ = arm._compute_frame_pose()
                    tcp = pos + c.robot.data.root_pos_w.torch - env.scene.env_origins
                    delta = c.target() - tcp
                    action = delta / delta.norm(dim=-1, keepdim=True).clamp(min=0.06) * 0.35
                elif args.probe == "zero":
                    action = torch.zeros(env.num_envs, 3, device=env.device)
                elif args.probe == "random":
                    action = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
                else:
                    action = policy(obs)
                obs, reward, done, extra = wrapped.step(action)
                duration += 1
                if (args.probe or args.trace) and int(duration.max()) % 25 == 0:
                    print(
                        "STATE",
                        int(duration.max()),
                        "tip",
                        c.tip()[0].tolist(),
                        "guides",
                        c.completed.tolist(),
                        "stretch",
                        c.stretch().tolist(),
                        "force",
                        wp.to_torch(c.cable.tension).tolist(),
                        "action",
                        action[0].tolist(),
                        "tool_target",
                        (
                            env.action_manager.get_term("arm").processed_actions
                            + c.robot.data.root_pos_w.torch
                            - env.scene.env_origins
                        )[0].tolist(),
                        "lowering",
                        c.lowering.tolist(),
                        "target",
                        c.target()[0].tolist(),
                        "max_bend_rad",
                        cable_bend_angles(c.points()).max(dim=1).values.tolist(),
                        "near_guides",
                        c.points()[0][
                            (c.points()[0, :, 0, None] - c.guide_positions[0, None, :, 0]).abs().argmin(dim=0)
                        ].tolist(),
                        flush=True,
                    )
                for i in done.nonzero().flatten().tolist():
                    if completed_per_env[i] >= quotas[i]:
                        continue
                    flags = {
                        name: bool(env.termination_manager.get_term(name)[i])
                        for name in env.termination_manager.active_terms
                    }
                    records.append(
                        {
                            "configuration": i + completed_per_env[i] * env.num_envs,
                            "environment": i,
                            "success": flags["success"],
                            "terminations": flags,
                            "steps": int(duration[i]),
                            "max_joint_gap_ratio": float(c.last_episode_peaks[i, 0]),
                            "max_grasp_force_n": float(c.last_episode_peaks[i, 1]),
                            "collision_steps": int(c.last_episode_peaks[i, 2]),
                            "completed_guides": int(c.last_completed[i]),
                            "max_bend_angle_rad": float(c.last_peak_bend[i]),
                            "terminal_centerline_m": c.last_points[i].cpu().tolist(),
                            "retained_guides": c.last_seated[i].cpu().tolist(),
                            **initial[i],
                        }
                    )
                    completed_per_env[i] += 1
                    initial[i] = None
                    duration[i] = 0
                    print(f"EVALUATION {len(records)}/{args.episodes}: {flags}", flush=True)
                if args.viz != "none":
                    time.sleep(max(0, env.step_dt - (time.monotonic() - before)))
    finally:
        env.close()

records.sort(key=lambda record: record["configuration"])
assert [record["configuration"] for record in records] == list(range(args.episodes))
successes = sum(r["success"] for r in records)
n = len(records)
p = successes / n
z = 1.96
center = (p + z * z / (2 * n)) / (1 + z * z / n)
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
task_root = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/contrib"
source_hashes = {
    str(path.relative_to(task_root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for task_dir in ("usb_cable_routing", "usb_cable_routing_zigzag")
    for path in sorted((task_root / task_dir).rglob("*"))
    if path.suffix in (".py", ".usda")
}
result = {
    "task": "IsaacContrib-Route-USB-Zigzag-UR10e",
    "mode": args.probe or "ppo",
    "checkpoint": args.checkpoint,
    "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
    "seed": args.seed,
    "guide_count": args.guides,
    "episodes": n,
    "successes": successes,
    "success_rate": p,
    "wilson_95_interval": [center - half, center + half],
    "records": records,
    "command": sys.argv,
    "versions": {
        name: version(name) for name in ("isaaclab", "isaaclab-tasks", "newton", "warp-lang", "rsl-rl-lib", "torch")
    },
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": source_hashes,
    "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "robot_asset": cfg.scene.robot.spawn.usd_path,
    "robot_variant": cfg.scene.robot.spawn.variants,
    "num_envs": cfg.scene.num_envs,
    "control_dt_s": cfg.sim.dt * cfg.decimation,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
