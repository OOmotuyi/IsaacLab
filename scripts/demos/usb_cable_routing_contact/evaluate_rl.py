# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate restricted recurrent contact policies once per explicit, isolated case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_STATE_PATH = Path(__file__).resolve().parents[1] / "usb_cable_routing_astra" / "benchmark_state.py"
_STATE_SPEC = importlib.util.spec_from_file_location("_contact_evaluation_state", _STATE_PATH)
_STATE = importlib.util.module_from_spec(_STATE_SPEC)
_STATE_SPEC.loader.exec_module(_STATE)


def main() -> None:
    """Run one fresh simulator process for each declared evaluation case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("scurve", "insert"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--policy", type=Path, help="Exported recurrent TorchScript policy.")
    artifact.add_argument("--checkpoint", type=Path, help="Native recurrent RSL-RL checkpoint.")
    parser.add_argument(
        "--training_config", type=Path, help="Saved params/agent.yaml; otherwise resolved near the artifact."
    )
    parser.add_argument(
        "--initial_reference_dir", type=Path, help="Require an exact match to CASE_ID/initial_state.json."
    )
    parser.add_argument("--worker_case_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = _STATE.load_evaluation_cases(args.manifest, args.variant)
    if args.worker_case_id is not None:
        selected = [case for case in cases if case["case_id"] == args.worker_case_id]
        if len(selected) != 1:
            parser.error("worker_case_id must identify exactly one selected case.")
        provenance = json.loads((args.output_dir / "provenance.json").read_text())
        kind = "policy" if args.policy else "checkpoint"
        if (
            provenance["variant"] != args.variant
            or provenance["manifest"] != str(args.manifest.resolve())
            or provenance["artifact_kind"] != kind
            or provenance["artifact"] != str(getattr(args, kind).resolve())
            or provenance["training_config"] != str(args.training_config.resolve())
            or provenance["initial_reference_dir"]
            != (str(args.initial_reference_dir.resolve()) if args.initial_reference_dir else None)
        ):
            parser.error("Worker arguments differ from the frozen evaluation provenance.")
        _assert_frozen(provenance)
        _run_worker(args, selected[0], provenance)
        return
    if args.output_dir.exists():
        parser.error("output_dir already exists; cases are never retried or overwritten.")
    kind = "policy" if args.policy else "checkpoint"
    artifact_path = getattr(args, kind).resolve()
    training_path = _resolve_training_config(artifact_path, args.training_config)
    training = _read_training_config(training_path)
    paths = {"manifest": args.manifest.resolve(), "artifact": artifact_path, "training_config": training_path}
    provenance = {
        "controller": "deterministic recurrent RL actor; physical contact grasp",
        "variant": args.variant,
        "artifact_kind": kind,
        "clip_actions": training["clip_actions"],
        "policy_observation_dim": 65 if args.variant == "insert" else 55,
        "num_envs": 1,
        "guide_count": 3,
        "randomize_guides": True,
        "randomize_cable": True,
        "case_count": len(cases),
        "process_isolation": "fresh_simulator_per_case",
        "recurrent_state": "reset before first action of each case",
        "attempts_per_case": 1,
        "initial_reference_dir": str(args.initial_reference_dir.resolve()) if args.initial_reference_dir else None,
        "command": sys.argv,
        "source_sha256": _source_hashes(),
        **{name: str(path) for name, path in paths.items()},
        **{f"{name}_sha256": _sha256(path) for name, path in paths.items()},
        "initial_reference_sha256": {},
    }
    if args.initial_reference_dir is not None:
        provenance["initial_reference_sha256"] = {
            str((args.initial_reference_dir / case["case_id"] / "initial_state.json").resolve()): _sha256(
                args.initial_reference_dir / case["case_id"] / "initial_state.json"
            )
            for case in cases
        }
    args.training_config = training_path
    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "provenance.json", provenance)
    records = []
    _write_json(args.output_dir / "progress.json", _progress(records, len(cases)))
    for case in cases:
        try:
            _assert_frozen(provenance)
            with (args.output_dir / f"{case['case_id']}.worker.log").open("x") as log:
                completed = subprocess.run(_worker_command(args, case), stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(
                    f"Worker exited with status {completed.returncode}; see {case['case_id']}.worker.log."
                )
            _assert_frozen(provenance)
            record = json.loads((args.output_dir / case["case_id"] / "result.json").read_text())
            if (
                any(record.get(key) != case[key] for key in ("case_id", "variant", "seed"))
                or record.get("status") != "completed"
                or type(record.get("success")) is not bool
            ):
                raise ValueError("Worker result lacks the assigned case's native terminal outcome.")
            records.append(record)
            _write_json(args.output_dir / "progress.json", _progress(records, len(cases)))
            print(json.dumps(record), flush=True)
        except Exception as error:
            failure = {**case, "status": "infrastructure_error", "error": str(error)}
            _write_json(args.output_dir / "infrastructure_error.json", failure)
            _write_json(args.output_dir / "progress.json", {**_progress(records, len(cases)), "error": failure})
            raise
    _write_json(args.output_dir / "evaluation.json", {**provenance, **_progress(records, len(cases))})


def _resolve_training_config(artifact: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for candidate in (
        artifact.parent / "params/agent.yaml",
        artifact.parent / "agent.yaml",
        artifact.parent.parent / "params/agent.yaml",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Supply --training_config with the policy's saved params/agent.yaml.")


def _read_training_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("obs_groups") != {"actor": ["policy"], "critic": ["policy"]}:
        raise ValueError("Training must use only the shared policy observation group for actor and critic.")
    if config.get("clip_actions") != 1.0 or config.get("actor", {}).get("class_name") != "RNNModel":
        raise ValueError("Require the contact task's generic recurrent actor and training action clipping of 1.0.")
    return config


def _load_policy(path: Path, kind: str, training: dict, observation_dim: int, action_dim: int, device: str):
    """Load actor-only inference without constructing a wrapper or resetting an environment."""
    import torch

    if kind == "policy":
        policy = torch.jit.load(str(path), map_location=device).eval()
    else:
        from rsl_rl.models import RNNModel
        from tensordict import TensorDict

        actor_cfg = dict(training["actor"])
        actor_cfg.pop("class_name")
        observations = TensorDict({"policy": torch.zeros(1, observation_dim, device=device)}, batch_size=[1])
        actor = RNNModel(observations, training["obs_groups"], "actor", action_dim, **actor_cfg)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        policy = actor.eval().as_jit().to(device).eval()
    if not callable(getattr(policy, "reset", None)):
        raise ValueError("Policy must provide reset() for recurrent episode state.")
    policy.reset()
    with torch.inference_mode():
        _policy_action(policy, torch.zeros(1, observation_dim, device=device), observation_dim, action_dim)
    policy.reset()
    return policy


def _policy_action(policy, observation, observation_dim: int, action_dim: int):
    import torch

    if observation.shape != (1, observation_dim) or not bool(torch.isfinite(observation).all()):
        raise ValueError("Policy observation has the wrong restricted shape or nonfinite values.")
    action = policy(observation)
    if (
        not isinstance(action, torch.Tensor)
        or action.shape != (1, action_dim)
        or not bool(torch.isfinite(action).all())
    ):
        raise ValueError("Policy produced invalid actions.")
    return action.clamp(-1.0, 1.0)


def _worker_command(args: argparse.Namespace, case: dict) -> list[str]:
    kind = "policy" if args.policy else "checkpoint"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--variant",
        args.variant,
        "--output_dir",
        str(args.output_dir.resolve()),
        f"--{kind}",
        str(getattr(args, kind).resolve()),
        "--training_config",
        str(args.training_config.resolve()),
        "--worker_case_id",
        case["case_id"],
    ]
    if args.initial_reference_dir is not None:
        command.extend(("--initial_reference_dir", str(args.initial_reference_dir.resolve())))
    return command


def _run_worker(args: argparse.Namespace, case: dict, provenance: dict) -> None:
    # Simulator and CUDA imports are confined to the isolated child process.
    import gymnasium as gym
    import torch
    from server import ContactBridge, build_task_cfg, task_id

    from isaaclab.app import launch_simulation

    case_dir = args.output_dir / case["case_id"]
    case_dir.mkdir()
    _write_json(case_dir / "case.json", {**provenance, **case})
    bridge = None
    try:
        cfg = build_task_cfg(args.variant, case["seed"])
        with launch_simulation(cfg):
            env = gym.make(task_id(args.variant), cfg=cfg).unwrapped
            original_reset = env._reset_idx
            try:
                policy = _load_policy(
                    Path(provenance["artifact"]),
                    provenance["artifact_kind"],
                    _read_training_config(Path(provenance["training_config"])),
                    provenance["policy_observation_dim"],
                    env.action_manager.total_action_dim,
                    env.device,
                )
                with torch.inference_mode():
                    env.reset(seed=case["seed"])
                    policy.reset()
                    initial = _STATE.capture_initial_state(env, case_dir / "initial_state.json")
                    if args.initial_reference_dir is not None:
                        reference = args.initial_reference_dir / case["case_id"] / "initial_state.json"
                        match = _STATE.assert_matching_initial_states(reference, initial)
                        _write_json(case_dir / "initial_state_match.json", match)
                    bridge = ContactBridge(
                        env,
                        case_dir,
                        args.variant,
                        case["seed"],
                        render_images=False,
                        controller=provenance["controller"],
                        rl_policy_loaded=True,
                        case_id=case["case_id"],
                    )
                    started = time.monotonic()
                    for _ in range(env.max_episode_length + 1):
                        action = _policy_action(
                            policy,
                            env.obs_buf["policy"],
                            provenance["policy_observation_dim"],
                            env.action_manager.total_action_dim,
                        )
                        if bridge.step(action):
                            break
                    if bridge.terminal is None:
                        raise RuntimeError("No native terminal outcome within the configured episode horizon.")
                    policy.reset()
                    bridge.save()
                    record = {
                        **case,
                        "status": "completed",
                        "success": bool(bridge.terminal["success"]),
                        "terminations": bridge.terminal["terminations"],
                        "steps": len(bridge.actions),
                        "simulation_time_s": bridge.terminal["time_s"],
                        "wall_time_s": time.monotonic() - started,
                        "physical_state_sha256": initial["physical_state_sha256"],
                        "policy_observation_sha256": initial["policy_observation_sha256"],
                        "episode_directory": str(bridge.episode_dir.resolve()),
                    }
                    _assert_frozen(provenance)
                    _write_json(case_dir / "result.json", record)
            finally:
                if bridge is not None:
                    bridge.save()
                env._reset_idx = original_reset
                env.close()
    except Exception as error:
        _write_json(
            case_dir / "infrastructure_error.json", {**case, "status": "infrastructure_error", "error": str(error)}
        )
        raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    task_root = _REPOSITORY_ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib"
    paths = list(task_root.glob("usb_cable_routing*/**/*.py")) + list(task_root.glob("usb_cable_routing*/**/*.usda"))
    paths.extend(
        (Path(__file__), Path(__file__).with_name("server.py"), _STATE_PATH, _STATE_PATH.with_name("server.py"))
    )
    return {str(path.relative_to(_REPOSITORY_ROOT)): _sha256(path) for path in sorted(paths)}


def _assert_frozen(provenance: dict) -> None:
    if _source_hashes() != provenance["source_sha256"]:
        raise RuntimeError("Evaluation sources changed after the run was frozen.")
    for name in ("manifest", "artifact", "training_config"):
        if _sha256(Path(provenance[name])) != provenance[f"{name}_sha256"]:
            raise RuntimeError(f"Frozen {name} changed during evaluation.")
    for name, expected in provenance["initial_reference_sha256"].items():
        if _sha256(Path(name)) != expected:
            raise RuntimeError("Paired initial-state reference changed during evaluation.")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _progress(records: list[dict], expected: int) -> dict:
    successes = sum(record["success"] for record in records)
    return {
        "expected_cases": expected,
        "completed_cases": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "complete": len(records) == expected,
        "success_rate": successes / len(records) if records else None,
        "records": records,
    }


if __name__ == "__main__":
    main()
