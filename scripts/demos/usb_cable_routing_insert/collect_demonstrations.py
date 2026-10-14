# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay routing actions and collect physically checked USB-C insertion demonstrations.

This is a scripted data collector, not a trained-policy evaluation. Only trajectories
that pass the environment's complete success condition should initialize an actor.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.utils.math import axis_angle_from_quat, quat_inv, quat_mul

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.contrib.usb_cable_routing_insert.env_cfg import UsbInsertEnvCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing_actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()
    data = torch.load(args.routing_actions, map_location="cpu", weights_only=False)
    replay = data["actions"]
    if replay.ndim != 3 or replay.shape[-1] != 3:
        raise ValueError("Expected routing actions with shape [time, environment, 3].")
    cfg = UsbInsertEnvCfg()
    cfg.scene.num_envs = replay.shape[1]
    cfg.seed = args.seed
    cfg.commands.routing.guide_count = 3
    cfg.commands.routing.curriculum = False
    cfg.commands.routing.randomize = args.randomize
    records, observations, actions = [], [], []
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-SCurve-Insert-UR10e", cfg=cfg).unwrapped
        wrapped = RslRlVecEnvWrapper(env)
        obs = wrapped.get_observations()
        c = env.command_manager.get_term("routing")
        phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        hold = torch.tensor(cfg.commands.routing.seat_targets[-1], device=env.device)
        flip = torch.tensor((0.0, 0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 4)
        desired = quat_mul(c.socket_quat, flip)
        preinsert = c.socket_position + torch.tensor((-0.008, 0.0, 0.0), device=env.device)
        seated = c.socket_position + torch.tensor((0.006, 0.0, 0.0), device=env.device)
        try:
            with torch.inference_mode():
                for step in range(math.ceil(cfg.episode_length_s / env.step_dt)):
                    observations.append(obs["policy"].cpu().clone())
                    action = torch.zeros(env.num_envs, 6, device=env.device)
                    if step < len(replay):
                        action[:, :3] = replay[step].to(env.device)
                    action[c.completed >= 3] = 0
                    phase[c.route_mastered & (phase == 0) & ~finished] = 1
                    front, orientation = c.plug_pose()
                    error = axis_angle_from_quat(quat_mul(desired, quat_inv(orientation)))
                    phase[(phase == 1) & ((hold - c.tip()).norm(dim=-1) < 0.004)] = 2
                    phase[(phase == 2) & (error.norm(dim=-1) < 0.025)] = 3
                    aligned = (c.local_front[:, :2].abs() < 0.00008).all(-1) & (error.norm(dim=-1) < 0.02)
                    phase[(phase == 3) & c.inserting & aligned] = 4
                    move = torch.where((phase <= 2)[:, None], hold - c.tip(), preinsert - front)
                    move = torch.where((phase == 4)[:, None], seated - front, move)
                    active = phase > 0
                    action[active, :3] = (move / move.norm(dim=-1, keepdim=True).clamp(min=0.003) * 0.5)[active]
                    rotating = phase >= 2
                    action[rotating, 3:] = (error * 0.08 / cfg.actions.arm.rotation_scale).clamp(-0.3, 0.3)[rotating]
                    # A gentle lateral sweep avoids a tight transient bend behind the plug.
                    near = (phase >= 3) & (c.local_front[:, 2].abs() < 0.02)
                    action[near, :3] = (move / cfg.actions.arm.insertion_scale * 0.05).clamp(-0.15, 0.15)[near]
                    action[finished] = 0
                    actions.append(action.cpu().clone())
                    obs, _, done, _ = wrapped.step(action)
                    for index in (done & ~finished).nonzero().flatten().tolist():
                        flags = {
                            name: bool(env.termination_manager.get_term(name)[index])
                            for name in env.termination_manager.active_terms
                        }
                        records.append(
                            {
                                "environment": index,
                                "step": step,
                                "phase": int(phase[index]),
                                "success": flags["success"],
                                "completed": int(c.last_completed[index]),
                                "radius": float(c.last_minimum_bend_radius[index]),
                                "depth": float(c.last_depth[index]),
                                "force": float(c.last_episode_peaks[index, 1]),
                                "terminations": flags,
                            }
                        )
                        finished[index] = True
                        phase[index] = 0
                        print(json.dumps(records[-1]), flush=True)
                    if finished.all():
                        break
        finally:
            env.close()
    metadata = {
        "method": "physical routing replay followed by scripted pose feedback; not policy evaluation",
        "source": str(args.routing_actions),
        "source_sha256": hashlib.sha256(args.routing_actions.read_bytes()).hexdigest(),
        "seed": args.seed,
        "randomized": args.randomize,
        "episode_length_s": cfg.episode_length_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**metadata, "observations": torch.stack(observations), "actions": torch.stack(actions), "records": records},
        args.output,
    )
    args.output.with_suffix(".json").write_text(json.dumps({**metadata, "records": records}, indent=2) + "\n")
    print(f"Full physical successes: {sum(row['success'] for row in records)}/{len(records)}")


if __name__ == "__main__":
    main()
