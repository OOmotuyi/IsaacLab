# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extend an Isaac-only standing checkpoint with object observations and a jaw output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
from rsl_rl.runners import OnPolicyRunner
from skill_checkpoint import make_runner
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.contrib.microduck.agents import MicroDuckGraspPPORunnerCfg
from isaaclab_tasks.contrib.microduck.constants import JOINT_NAMES


def make_grasp_runner() -> OnPolicyRunner:
    """Construct the 65-observation, 15-action actor and its 68-observation critic."""
    obs = TensorDict({"policy": torch.zeros(1, 65), "critic": torch.zeros(1, 68)}, batch_size=[1])
    env = SimpleNamespace(num_envs=1, num_actions=15, cfg={}, get_observations=lambda: obs)
    cfg = handle_deprecated_rsl_rl_cfg(MicroDuckGraspPPORunnerCfg(), version("rsl-rl-lib"))
    return OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device="cpu")


def initialize(standing: Path, output: Path, head_pitch_std: float = 0.03, body_std: float = 0.025) -> None:
    """Retain scratch-trained Isaac standing behavior while adding new task inputs."""
    source_manifest = json.loads((standing.parents[3] / "experiment.json").read_text())
    for name in ("initial_checkpoint", "teacher_policy", "recorded_motion_reference"):
        if source_manifest[name] is not None:
            raise ValueError("Standing source must have documented scratch-training provenance.")
    if source_manifest["source_action_prior"]:
        raise ValueError("Source actor used a pretrained action prior.")
    torch.manual_seed(42)
    old = make_runner(from_scratch=True)
    old.load(str(standing))
    old_actor = old.get_inference_policy(device="cpu")
    runner = make_grasp_runner()
    state = runner.alg.actor.state_dict()
    original = old.alg.actor.state_dict()
    for name, value in original.items():
        if name in state and state[name].shape == value.shape and name != "distribution.std_param":
            state[name].copy_(value)
    for name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        state[name][:, :48] = original[name][:, :48]
    # New measurements have fixed initial SI scales; existing normalization is retained.
    scales = torch.tensor([1.0] + [0.1] * 6 + [0.3] * 3 + [0.3, 1.0] + [1.0] * 5)
    state["obs_normalizer._std"][:, 48:] = scales
    state["obs_normalizer._var"][:, 48:] = scales.square()
    state["mlp.0.weight"].zero_()
    state["mlp.0.weight"][:, :48] = original["mlp.0.weight"][:, :48]
    with torch.no_grad():
        zero = old_actor.obs_normalizer(torch.zeros(1, 61))[0]
        state["mlp.0.bias"] = original["mlp.0.bias"] + original["mlp.0.weight"][:, 48:] @ zero[48:]
    state["mlp.6.weight"].zero_()
    state["mlp.6.weight"][:14] = original["mlp.6.weight"]
    state["mlp.6.bias"][:14] = original["mlp.6.bias"]
    state["mlp.6.bias"][14] = 1.0
    state["distribution.std_param"].fill_(body_std)
    state["distribution.std_param"][5:7] = head_pitch_std
    state["distribution.std_param"][14] = 0.60
    runner.alg.actor.load_state_dict(state)
    actor = runner.get_inference_policy(device="cpu")
    torch.manual_seed(42)
    obs = original["obs_normalizer._mean"] + original["obs_normalizer._std"] * torch.randn(64, 61)
    obs[:, 48:] = 0
    extended = torch.randn(64, 65)
    extended[:, :48] = obs[:, :48]
    with torch.inference_mode():
        expected = old_actor(TensorDict({"policy": obs}, batch_size=[64]))
        actual = actor(TensorDict({"policy": extended}, batch_size=[64]))
    error = float((actual[:, :14] - expected).abs().max())
    if error > 2e-5:
        raise RuntimeError(f"Standing extension changed body actions: {error}")
    provenance = {
        "initialization": "continued_isaac_scratch_standing",
        "standing_checkpoint": str(standing.resolve()),
        "standing_checkpoint_sha256": hashlib.sha256(standing.read_bytes()).hexdigest(),
        "mujoco_policy_weights_used": False,
        "teacher_policy": None,
        "body_action_parity_max_error_rad": error,
        "new_inputs": "object/goal state, marker-relative heading, jaw state, grasp stage",
        "new_output": "jaw",
        "critic_initialization": "random",
        "body_exploration_std_rad": body_std,
        "head_pitch_exploration_std_rad": head_pitch_std,
        "jaw_normalized_exploration_std": 0.60,
    }
    saved = runner.alg.save()
    saved.update(iter=0, infos=provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, output)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def initialize_geometric(record: Path, output: Path, body_std: float = 0.04) -> None:
    """Initialize a random actor with geometric body-offset biases [rad] from a JSON record."""
    geometry = json.loads(record.read_text())
    bias = torch.tensor(geometry["output_bias_rad"], dtype=torch.float32)
    if bias.shape != (14,) or not torch.isfinite(bias).all():
        raise ValueError("Geometric initialization requires fourteen finite body offsets [rad].")
    seed = int(geometry.get("seed", 42))
    torch.manual_seed(seed)
    runner = make_grasp_runner()
    state = runner.alg.actor.state_dict()
    state["mlp.6.weight"].zero_()
    state["mlp.6.bias"][:14] = bias
    state["mlp.6.bias"][14] = 1.0
    state["distribution.std_param"].fill_(body_std)
    state["distribution.std_param"][14] = 0.60
    runner.alg.actor.load_state_dict(state)
    provenance = {
        "initialization": "random_geometric_output_bias",
        "seed": seed,
        "initial_checkpoint": None,
        "teacher_policy": None,
        "mujoco_policy_weights_used": False,
        "critic_initialization": "random",
        "optimizer_initialization": "fresh",
        "geometric_pose_source": str(record.resolve()),
        "geometric_pose_source_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "geometric_reset_pose": geometry["geometric_reset_pose"],
        "output_bias_rad": bias.tolist(),
        "body_exploration_std_rad": body_std,
        "jaw_normalized_exploration_std": 0.60,
    }
    saved = runner.alg.save()
    saved.update(iter=0, infos=provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, output)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def verify_grasp_lineage(checkpoint: Path, *, _ancestors: tuple[Path, ...] = ()) -> dict:
    """Verify the run's initialization, source identity, and nonzero optimizer updates."""
    checkpoint = checkpoint.resolve()
    if checkpoint in _ancestors:
        raise ValueError("Cyclic grasp checkpoint lineage.")
    manifest_path = next(
        (parent / "experiment.json" for parent in checkpoint.parents if (parent / "experiment.json").is_file()), None
    )
    if manifest_path is None:
        raise ValueError("Grasp export requires an experiment.json with recorded launch and initialization.")
    experiment = json.loads(manifest_path.read_text())
    launch = None
    for value in experiment.values():
        if not isinstance(value, dict) or "command" not in value:
            continue
        command = value["command"]
        if (
            "--experiment_name" in command
            and Path(command[command.index("--experiment_name") + 1]).resolve() == checkpoint.parent.parent
        ):
            launch = value
            break
    if launch is None:
        raise ValueError("Checkpoint is not associated with a recorded training launch.")
    command = launch["command"]
    initial_path = Path(command[command.index("--checkpoint") + 1])
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    lineage = initial.get("infos") or {}
    if (
        lineage.get("initialization") not in {"continued_isaac_scratch_standing", "random_geometric_output_bias"}
        or lineage.get("mujoco_policy_weights_used") is not False
    ):
        lineage = verify_grasp_lineage(initial_path, _ancestors=(*_ancestors, checkpoint))
    if "parent_checkpoint" in lineage:
        parent = Path(lineage["parent_checkpoint"])
        if hashlib.sha256(parent.read_bytes()).hexdigest() != lineage["parent_checkpoint_sha256"]:
            raise ValueError("Grasp continuation source changed since initialization.")
    if lineage["initialization"] == "continued_isaac_scratch_standing":
        standing = Path(lineage["standing_checkpoint"])
        if hashlib.sha256(standing.read_bytes()).hexdigest() != lineage["standing_checkpoint_sha256"]:
            raise ValueError("Standing source checkpoint changed since initialization.")
    else:
        if lineage.get("initial_checkpoint") is not None or lineage.get("teacher_policy") is not None:
            raise ValueError("Random geometric initialization must not load a learned actor or teacher.")
        pose = Path(lineage["geometric_pose_source"])
        if hashlib.sha256(pose.read_bytes()).hexdigest() != lineage["geometric_pose_source_sha256"]:
            raise ValueError("Geometric initialization pose changed since training.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    steps = [float(v["step"]) for v in saved["optimizer_state_dict"]["state"].values() if "step" in v]
    change = max(
        float((value - initial["actor_state_dict"][name]).abs().max())
        for name, value in saved["actor_state_dict"].items()
        if name.startswith("mlp.")
    )
    if not steps or min(steps) <= 0 or change == 0:
        raise ValueError("Checkpoint has no verified actor optimization.")
    env_config = (checkpoint.parent / "params/env.yaml").read_text()
    if "initial_reach_height:" not in env_config:
        raise ValueError("Checkpoint predates the staged grasp observation contract.")
    contract = re.search(r"observation_contract:\s*[\"\']?(grasp_proprio_object_v\d+)", env_config)
    return {
        **lineage,
        "observation_contract": contract.group(1) if contract else "grasp_proprio_object_v2",
        "initial_checkpoint_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        "optimizer_steps": max(steps),
        "max_actor_parameter_change": change,
        "source_sha256": launch.get("source_sha256", {}),
        "lineage_checkpoints": [*lineage.get("lineage_checkpoints", []), str(initial_path.resolve())],
    }


def continue_training(
    checkpoint: Path, output: Path, body_std: float | None = None, head_pitch_std: float | None = None
) -> None:
    """Preserve a trained actor/critic while changing exploration [rad] with a fresh optimizer."""
    lineage = verify_grasp_lineage(checkpoint)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner = make_grasp_runner()
    runner.alg.actor.load_state_dict(source["actor_state_dict"])
    runner.alg.critic.load_state_dict(source["critic_state_dict"])
    state = runner.alg.actor.state_dict()
    std = state["distribution.std_param"]
    if body_std is not None:
        std[:14] = body_std
    if head_pitch_std is not None:
        std[5:7] = head_pitch_std
    runner.alg.actor.load_state_dict(state)
    changes = [
        float((value - source["actor_state_dict"][name]).abs().max())
        for name, value in state.items()
        if name != "distribution.std_param"
    ]
    if max(changes) != 0.0:
        raise RuntimeError("Exploration continuation changed the deterministic actor.")
    provenance = {
        **lineage,
        "parent_checkpoint": str(checkpoint.resolve()),
        "parent_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "critic_initialization": "continued_isaac_grasp",
        "optimizer_initialization": "fresh",
        "exploration_std": std.tolist(),
        "deterministic_actor_parameter_change": max(changes),
    }
    saved = runner.alg.save()
    saved.update(iter=source["iter"], infos=provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, output)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def export(checkpoint: Path, output: Path) -> None:
    """Export the learned 15-output actor with normalizers and checkpoint identity."""
    lineage = verify_grasp_lineage(checkpoint)
    runner = make_grasp_runner()
    runner.load(str(checkpoint))
    actor = runner.get_inference_policy(device="cpu").as_onnx(verbose=False).eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        torch.zeros(1, 65),
        str(output),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    obs = torch.randn(64, 65)
    with torch.inference_mode():
        expected = actor(obs).numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs.numpy()])
    error = float(np.abs(actual - expected).max())
    if error > 2e-5:
        raise RuntimeError(f"Grasp export mismatch: {error}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        "training_backend": "isaaclab_newton",
        "algorithm": "RSL-RL PPO",
        "initialization": lineage["initialization"],
        "mujoco_policy_weights_used": "false",
        "observation_contract": lineage["observation_contract"],
        "action_contract": "absolute_body_offsets_and_jaw",
        "joint_names": ",".join(JOINT_NAMES),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": str(saved["iter"]),
        "optimizer_steps": str(lineage["optimizer_steps"]),
        "initial_checkpoint_sha256": lineage["initial_checkpoint_sha256"],
        "max_actor_parameter_change": str(lineage["max_actor_parameter_change"]),
        "export_parity_max_error": str(error),
    }
    for key in ("standing_checkpoint_sha256", "geometric_pose_source_sha256"):
        if key in lineage:
            metadata[key] = lineage[key]
    model = onnx.load(str(output))
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, str(output))
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    """Prepare continued Isaac training or export its trained object-grasp actor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize_from", type=Path)
    parser.add_argument(
        "--initialize_geometric", type=Path, help="JSON geometric reset pose and body-offset bias [rad]."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--continue_from", type=Path)
    parser.add_argument("--body_std", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head_pitch_std", type=float)
    args = parser.parse_args()
    if (
        sum(
            value is not None
            for value in (args.initialize_from, args.initialize_geometric, args.checkpoint, args.continue_from)
        )
        != 1
    ):
        parser.error("Select --initialize_from, --initialize_geometric, --continue_from, or --checkpoint.")
    for name in ("body_std", "head_pitch_std"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive.")
    if args.initialize_from:
        initialize(
            args.initialize_from,
            args.output,
            0.03 if args.head_pitch_std is None else args.head_pitch_std,
            0.025 if args.body_std is None else args.body_std,
        )
    elif args.initialize_geometric:
        if args.head_pitch_std is not None:
            parser.error("Geometric initialization uses --body_std for all body joints.")
        initialize_geometric(args.initialize_geometric, args.output, 0.04 if args.body_std is None else args.body_std)
    elif args.continue_from:
        continue_training(args.continue_from, args.output, args.body_std, args.head_pitch_std)
    else:
        export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
