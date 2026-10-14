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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


class _SampledPolicy(torch.nn.Module):
    """Export the trained Gaussian sampling distribution together with the mean actor."""

    def __init__(self, mean_policy: torch.nn.Module, std: torch.Tensor, insertion_precision: bool = False):
        super().__init__()
        self.mean_policy = mean_policy
        self.insertion_precision = insertion_precision
        self.register_buffer("std", std)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean = self.mean_policy(observation)
        scale = torch.ones_like(mean[:, :1])
        if self.insertion_precision:
            scale = torch.where(observation[:, 150] >= observation[:, 151], 0.1, 1.0)[:, None]
        return mean + torch.randn_like(mean) * self.std * scale


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="IsaacContrib-Route-USB-UR10e",
    choices=(
        "IsaacContrib-Route-USB-UR10e",
        "IsaacContrib-Route-USB-Zigzag-UR10e",
        "IsaacContrib-Route-USB-SCurve-UR10e",
        "IsaacContrib-Route-USB-SCurve-Insert-UR10e",
    ),
)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Registered PPO agent configuration.")
controller = parser.add_mutually_exclusive_group(required=True)
controller.add_argument("--checkpoint")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--guides", type=int, default=3)
parser.add_argument("--seed", type=int, default=900001)
parser.add_argument("--output", type=Path, default=Path("logs/usb_cable_routing/evaluation.json"))
controller.add_argument("--probe", choices=("zero", "scripted", "random"))
parser.add_argument("--viz", choices=("none", "newton_gl", "newton_rtx"), default="none")
parser.add_argument("--headless", action="store_true", help="Record the selected visualizer without a window.")
parser.add_argument("--video", type=Path)
parser.add_argument("--task_view", action="store_true", help="Frame the cable, guides and socket closely.")
parser.add_argument("--no_graph", action="store_true")
parser.add_argument("--trace", action="store_true", help="Log cable state every 25 control steps.")
parser.add_argument("--stochastic", action="store_true", help="Sample the trained Gaussian policy during evaluation.")
parser.add_argument("--export_policy", type=Path)
args = parser.parse_args()
if args.stochastic and not args.checkpoint:
    parser.error("--stochastic requires --checkpoint.")
if args.export_policy and not args.checkpoint:
    parser.error("--export_policy requires --checkpoint.")
if args.episodes < 1 or args.num_envs < 1 or not 1 <= args.guides <= 3:
    parser.error("Require positive episodes/envs and one to three guides.")

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = min(args.num_envs, args.episodes)
cfg.seed = args.seed
cfg.commands.routing.curriculum = False
cfg.commands.routing.guide_count = args.guides
cfg.sim.physics.use_cuda_graph = not args.no_graph
if args.viz != "none" or args.video:
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    cfg.sim.physics.load_visual_shapes = True
    visualizer_type = NewtonRTXVisualizerCfg if args.viz == "newton_rtx" else NewtonGLVisualizerCfg
    cfg.sim.visualizer_cfgs = [
        visualizer_type(
            eye=(0.94, 0.40, 0.72) if args.task_view else (1.45, 0.55, 1.20),
            lookat=(0.44, -0.09, 0.09) if args.task_view else (0.40, -0.35, 0.32),
            window_width=1280,
            window_height=720,
            headless=args.headless or args.viz == "none",
        )
    ]
    if args.viz == "newton_rtx":
        cfg.sim.visualizer_cfgs[0].render_settings = {
            "omni:rtx:waitForEvents": ("TokenArray", ["AllLoadingFinished", "OnlyOnFirstRequest"]),
            "omni:rtx:rendermode": ("Token", "PathTracing"),
            "omni:rtx:pt:samplesPerPixel": ("UInt", 8),
            "omni:rtx:pt:denoising:enabled": ("Bool", True),
            "omni:rtx:dlss:frameGeneration": ("Bool", False),
        }
if args.video:
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    cfg.video_recorders = [
        VideoRecorderCfg(
            output_dir=str(args.video.parent),
            output_filename_prefix=args.video.stem,
            video_length=math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) + 30,
            step_offset=5,
        )
    ]

records = []
with launch_simulation(cfg):
    env = gym.make(args.task, cfg=cfg).unwrapped
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, args.agent), version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        if args.stochastic:

            def policy(observations):
                return runner.alg.actor(observations, stochastic_output=True)

        if args.export_policy:
            args.export_policy.mkdir(parents=True, exist_ok=True)
            if args.stochastic:
                policy(wrapped.get_observations())
                exported = _SampledPolicy(
                    runner.alg.actor.as_jit().cpu(),
                    runner.alg.actor.output_std[0].detach().cpu(),
                    getattr(runner.alg.actor.distribution, "uses_insertion_precision", False),
                )
                torch.jit.script(exported).save(str(args.export_policy / "policy.pt"))
            else:
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
                        }
                if args.probe == "scripted":
                    arm = env.action_manager.get_term("arm")
                    pos, _ = arm._compute_frame_pose()
                    tcp = pos + c.robot.data.root_pos_w.torch - env.scene.env_origins
                    delta = c.target() - c.tip()
                    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                    action[:, :3] = delta / delta.norm(dim=-1, keepdim=True).clamp(min=0.06) * 0.20
                    if action.shape[1] == 6:
                        from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul

                        routed = c.completed >= c.guide_count
                        relative = wp.to_torch(c.cable.grasp_relative_quat)
                        flip = torch.tensor((0.0, 0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 4)
                        desired = quat_mul(quat_mul(c.socket_quat, flip), quat_inv(relative))
                        rotation = axis_angle_from_quat(quat_mul(desired, quat_inv(arm.target_quat)))
                        action[:, 3:] = torch.where(
                            routed[:, None], rotation.clamp(-0.01, 0.01) / arm.cfg.rotation_scale, 0.0
                        )
                elif args.probe == "zero":
                    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                elif args.probe == "random":
                    action = torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2 - 1
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
                            "minimum_wrist_height_m": float(c.last_arm_metrics[i, 0]),
                            "maximum_arm_speed_rad_s": float(c.last_arm_metrics[i, 1]),
                            "maximum_wrist_roll_rad": float(c.last_arm_metrics[i, 2]),
                            "minimum_joint_limit_margin_rad": float(c.last_arm_metrics[i, 3]),
                            "retained_guides": c.last_seated[i].cpu().tolist(),
                            **initial[i],
                        }
                    )
                    if hasattr(c, "last_minimum_bend_radius"):
                        records[-1]["minimum_bend_radius_m"] = float(c.last_minimum_bend_radius[i])
                    if hasattr(c, "last_depth"):
                        records[-1]["terminal_insertion_depth_m"] = float(c.last_depth[i])
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
    for path in sorted(task_root.glob("usb_cable_routing*/**/*"))
    if path.suffix in (".py", ".usda")
}
result = {
    "task": args.task,
    "visualizer": "newton_gl" if args.viz == "none" and args.video else args.viz,
    "mode": args.probe or ("residual_ppo" if "residual" in args.agent else "ppo"),
    "agent": args.agent,
    "stochastic_policy": args.stochastic,
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
    "robot_base_position_m": cfg.scene.robot.init_state.pos,
    "maximum_commanded_joint_speed_rad_s": cfg.actions.arm.maximum_joint_speed,
    "num_envs": cfg.scene.num_envs,
    "control_dt_s": cfg.sim.dt * cfg.decimation,
    "wrist_height_definition": "Height of the wrist_1_link frame origin above the tabletop; not surface clearance.",
    "collision_steps_definition": (
        "Control steps with registered robot-fixture contact candidates; not measured penetration."
    ),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
