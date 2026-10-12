# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare verified PPO initialization or export a genuinely updated cleanup skill."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.contrib.microduck.agents import MicroDuckScratchPPORunnerCfg, MicroDuckSkillPPORunnerCfg
from isaaclab_tasks.contrib.microduck.constants import JOINT_NAMES
from isaaclab_tasks.contrib.microduck.policy import MicroDuckSourceActor


def make_runner(*, from_scratch: bool = False) -> OnPolicyRunner:
    """Construct the actor, fresh critic, and optimizer on CPU without simulation."""
    observations = TensorDict({"policy": torch.zeros(1, 61), "critic": torch.zeros(1, 76)}, batch_size=[1])
    env = SimpleNamespace(num_envs=1, num_actions=14, cfg={}, get_observations=lambda: observations)
    config = handle_deprecated_rsl_rl_cfg(
        MicroDuckScratchPPORunnerCfg() if from_scratch else MicroDuckSkillPPORunnerCfg(), version("rsl-rl-lib")
    )
    return OnPolicyRunner(env, config.to_dict(), log_dir=None, device="cpu")


def export_scratch(checkpoint_path: Path, output: Path, skill: str) -> None:
    """Export a native scratch actor with its learned observation normalization."""
    experiment = json.loads((checkpoint_path.parents[3] / "experiment.json").read_text())
    if (
        experiment["initial_checkpoint"] is not None
        or experiment["teacher_policy"] is not None
        or experiment["source_action_prior"]
        or experiment["recorded_motion_reference"] is not None
    ):
        raise ValueError("Checkpoint provenance does not describe teacher-free scratch training.")
    env_config = (checkpoint_path.parent / "params/env.yaml").read_text()
    if "teacher_path: null" not in env_config or "source_prior: null" not in env_config:
        raise ValueError("Saved training configuration contains a teacher or source-action prior.")
    runner = make_runner(from_scratch=True)
    runner.load(str(checkpoint_path))
    actor = runner.get_inference_policy(device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    steps = [float(v["step"]) for v in checkpoint["optimizer_state_dict"]["state"].values() if "step" in v]
    if not steps or min(steps) <= 0:
        raise ValueError("Checkpoint has no recorded optimizer updates.")
    exported_actor = actor.as_onnx(verbose=False).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exported_actor,
        torch.zeros(1, 61),
        str(output),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    observations = actor.obs_normalizer.mean + actor.obs_normalizer.std * torch.randn(64, 61)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = actor(TensorDict({"policy": observations}, batch_size=[64])).numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations.numpy()])
    error = float(np.abs(expected - actual).max())
    if not np.isfinite(actual).all() or error > 2e-5:
        raise RuntimeError(f"Normalized scratch export mismatch: {error}")
    provenance = {
        "training_backend": "isaaclab_newton",
        "algorithm": "RSL-RL PPO",
        "initialization": "random",
        "teacher_policy": "none",
        "source_action_prior": "false",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "iteration": int(checkpoint["iter"]),
        "optimizer_steps": max(steps),
        "actor_normalization": "checkpoint empirical normalization included in ONNX",
        "export_parity_max_error_rad": error,
        "skill": skill,
        "command_contract": "phase_cos_sin_depth" if skill == "pickup" else "zero_command",
        "joint_names": ",".join(JOINT_NAMES),
        "action_scale": 1.0,
        "observation_names": (
            "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
        ),
        "cleanup_control": "scripted; sock grasp assisted",
    }
    model = onnx.load(str(output))
    onnx.helper.set_model_props(model, {key: str(value) for key, value in provenance.items()})
    onnx.save(model, str(output))
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def main() -> None:
    """Verify source parity and write a warm-start checkpoint or trained ONNX export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--skill", choices=("stand", "pickup"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.from_scratch:
        if args.source or args.checkpoint is None or args.skill is None:
            parser.error("Scratch export requires --checkpoint and --skill, without --source.")
        export_scratch(args.checkpoint, args.output, args.skill)
        return
    if args.source is None:
        parser.error("The original warm-start/export mode requires --source.")
    source = MicroDuckSourceActor(args.source)
    runner = make_runner()
    actor = runner.alg.actor
    actor.load_state_dict(source.folded_state_dict(), strict=False)
    # Sample around the source's training distribution, including its gravity/joint scales.
    observations = source.mean + source.scale * torch.randn(64, 61)
    session = ort.InferenceSession(str(args.source), providers=["CPUExecutionProvider"])
    expected = np.concatenate(
        [session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in observations.numpy()]
    )
    with torch.no_grad():
        actual = actor.mlp(observations).numpy()
    parity_error = float(np.abs(expected - actual).max())
    if parity_error > 2e-5:
        raise RuntimeError(f"Source-to-PPO initialization mismatch: {parity_error}")
    provenance = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "seed": args.seed,
        "source_parity_max_error_rad": parity_error,
        "actor_normalization": "source normalization folded into the first layer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        saved = runner.alg.save()
        saved.update(iter=0, infos=provenance)
        torch.save(saved, args.output)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        changes = [
            (actor.state_dict()[name] - value).abs().max().item() for name, value in source.folded_state_dict().items()
        ]
        optimizer_steps = [
            float(value["step"]) for value in checkpoint["optimizer_state_dict"]["state"].values() if "step" in value
        ]
        if not optimizer_steps or max(optimizer_steps) == 0 or max(changes) == 0:
            raise RuntimeError("Refusing to label an unchanged or unoptimized checkpoint as trained.")
        provenance.update(
            training_backend="isaaclab_newton",
            algorithm="RSL-RL PPO",
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            iteration=int(checkpoint["iter"]),
            optimizer_steps=max(optimizer_steps),
            max_actor_parameter_change=max(changes),
            cleanup_control="scripted; sock grasp assisted",
        )
        actor.eval()
        torch.onnx.export(
            actor.mlp,
            torch.zeros(1, 61),
            str(args.output),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
            dynamo=False,
        )
        exported = onnx.load(str(args.output))
        metadata = {item.key: item.value for item in onnx.load(str(args.source)).metadata_props}
        metadata.update({key: str(value) for key, value in provenance.items()})
        onnx.helper.set_model_props(exported, metadata)
        onnx.save(exported, str(args.output))
        output_session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            export_error = float(
                np.max(
                    np.abs(
                        output_session.run(None, {"obs": observations[:1].numpy()})[0]
                        - actor.mlp(observations[:1]).numpy()
                    )
                )
            )
        if export_error > 2e-5:
            raise RuntimeError(f"Trained ONNX export mismatch: {export_error}")
        provenance["export_parity_max_error_rad"] = export_error
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
