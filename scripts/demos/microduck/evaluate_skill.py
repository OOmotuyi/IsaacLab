# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate frozen standing/pickup actors in the native Newton training task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.contrib.microduck.skill_env_cfg import MicroDuckPickupEnvCfg, MicroDuckStandEnvCfg


def main() -> None:
    """Compare checkpoint survival and measured phase-conditioned reaching error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=("stand", "pickup"), required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--duration", type=float, default=7.0)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = MicroDuckPickupEnvCfg() if args.skill == "pickup" else MicroDuckStandEnvCfg()
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration + 0.1
    with launch_simulation(cfg.sim, args):
        import torch

        from isaaclab.envs import ManagerBasedRLEnv

        from isaaclab_tasks.contrib.microduck.mdp.skills import skill_head_position
        from isaaclab_tasks.contrib.microduck.policy import MicroDuckSourceActor

        env = ManagerBasedRLEnv(cfg)
        source = MicroDuckSourceActor(cfg.actions.joint_pos.teacher_path).to(env.device)
        results = []
        try:
            for path in [None, *args.checkpoint]:
                if path is None:
                    actor = source
                else:
                    import copy

                    actor = copy.deepcopy(source.mlp)
                    state = torch.load(path, map_location=env.device, weights_only=False)["actor_state_dict"]
                    actor.load_state_dict(
                        {name.removeprefix("mlp."): value for name, value in state.items() if name.startswith("mlp.")}
                    )
                actor.eval()
                obs, _ = env.reset(seed=args.seed)
                failures = torch.zeros(env.num_envs, device=env.device)
                error_sum = torch.zeros(env.num_envs, device=env.device)
                low = torch.full((env.num_envs,), float("inf"), device=env.device)
                upright = torch.zeros(env.num_envs, device=env.device)
                deviations = torch.zeros(env.num_envs, device=env.device)
                samples = []
                start = time.perf_counter()
                steps = round(args.duration / env.step_dt)
                with torch.inference_mode():
                    for step in range(steps):
                        action = actor(obs["policy"])
                        deviations += (action - source(obs["policy"])).square().mean(-1)
                        obs, _, terminated, _, _ = env.step(action)
                        failures += terminated.float()
                        command = env.command_manager.get_term("twist")
                        position = skill_head_position(env)
                        error_sum += (position - command.reference_pose[:, :3]).square().sum(-1)
                        low = torch.minimum(low, position[:, 2])
                        upright += (env.scene["robot"].data.projected_gravity_b.torch[:, 2] < -0.7).float()
                        if step % 5 == 0:
                            samples.append(
                                {
                                    "time_s": (step + 1) * env.step_dt,
                                    "head_m": position[0].cpu().tolist(),
                                    "target_m": command.reference_pose[0, :3].cpu().tolist(),
                                    "phase_s": float(command.phase[0]),
                                }
                            )
                result = {
                    "checkpoint": str(path.resolve()) if path else "original MuJoCo ONNX actor",
                    "skill": args.skill,
                    "num_envs": env.num_envs,
                    "seed": args.seed,
                    "duration_s": args.duration,
                    "wall_seconds": time.perf_counter() - start,
                    "survived_fraction": float((failures == 0).float().mean()),
                    "fall_count": int(failures.sum()),
                    "head_target_rmse_m": float((error_sum.mean() / steps).sqrt()),
                    "mean_minimum_head_height_m": float(low.mean()),
                    "upright_fraction": float(upright.mean() / steps),
                    "action_change_rmse_rad": float((deviations.mean() / steps).sqrt()),
                    "samples": samples,
                }
                results.append(result)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
