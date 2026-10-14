# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate fixed RL policies on explicit benchmark cases in isolated processes.

Each case receives a fresh simulator, one seeded reset, and one native rollout.
Policy actions go directly through the environment action manager.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmark_state import load_evaluation_cases

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_POLICY_ROOT = _REPOSITORY_ROOT / "logs/usb_posture_revision/final"
_VARIANTS = ("scurve", "insert")


def main() -> None:
    """Run every selected case once, using a fresh child process for each simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON object containing a cases list.")
    parser.add_argument("--variant", choices=_VARIANTS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="New directory for this controller and variant.")
    parser.add_argument(
        "--initial_reference_dir",
        type=Path,
        help="Optional directory containing CASE_ID/initial_state.json for strict matched-state checks.",
    )
    parser.add_argument("--worker_case_id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = load_evaluation_cases(args.manifest, args.variant)
    if args.worker_case_id is not None:
        selected = [case for case in cases if case["case_id"] == args.worker_case_id]
        if len(selected) != 1:
            parser.error("worker_case_id must identify exactly one case of the selected variant.")
        provenance = json.loads((args.output_dir / "provenance.json").read_text())
        if provenance["variant"] != args.variant or provenance["manifest"] != str(args.manifest.resolve()):
            parser.error("Worker arguments do not match the parent provenance.")
        _assert_frozen(provenance)
        _run_worker(args, selected[0], provenance)
        return
    if args.output_dir.exists():
        parser.error("output_dir already exists; completed or interrupted cases must not be rerun or overwritten.")
    policy_path = _POLICY_ROOT / args.variant / "policy.pt"
    checkpoint_path = _POLICY_ROOT / args.variant / "checkpoint.pt"
    for path in (policy_path, checkpoint_path):
        if not path.is_file():
            parser.error(f"Required frozen controller artifact is missing: {path}")
    provenance = {
        "controller": "fixed exported deterministic RL policy",
        "variant": args.variant,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "policy": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "stochastic_policy": False,
        "external_action_clipping": False,
        "num_envs": 1,
        "process_isolation": "fresh_simulator_per_case",
        "case_count": len(cases),
        "source_sha256": _source_hashes(),
    }
    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "provenance.json", provenance)
    _run_parent(args, cases, provenance)


def _run_parent(args: argparse.Namespace, cases: list[dict], provenance: dict) -> None:
    records = []
    _write_json(args.output_dir / "progress.json", _progress(records, len(cases)))
    for case in cases:
        log_path = args.output_dir / f"{case['case_id']}.worker.log"
        try:
            _assert_frozen(provenance)
            if (args.output_dir / case["case_id"]).exists():
                raise FileExistsError(f"Case directory already exists: {case['case_id']}")
            with log_path.open("x") as log:
                completed = subprocess.run(_worker_command(args, case), stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                raise RuntimeError(f"Worker exited with status {completed.returncode}; see {log_path}.")
            _assert_frozen(provenance)
            record = json.loads((args.output_dir / case["case_id"] / "result.json").read_text())
            if (
                any(record.get(key) != case[key] for key in ("case_id", "variant", "seed"))
                or record.get("status") != "completed"
                or type(record.get("success")) is not bool
            ):
                raise ValueError("Worker result does not match its assigned case or lacks a native outcome.")
            records.append(record)
            _write_json(args.output_dir / "progress.json", _progress(records, len(cases)))
            print(json.dumps(record), flush=True)
        except Exception as error:
            failure = {**case, "status": "infrastructure_error", "error": str(error), "worker_log": str(log_path)}
            _write_json(args.output_dir / "infrastructure_error.json", failure)
            _write_json(args.output_dir / "progress.json", {**_progress(records, len(cases)), "error": failure})
            raise
    _assert_frozen(provenance)
    _write_json(args.output_dir / "evaluation.json", {**provenance, **_progress(records, len(cases))})


def _worker_command(args: argparse.Namespace, case: dict) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--variant",
        args.variant,
        "--output_dir",
        str(args.output_dir.resolve()),
        "--worker_case_id",
        case["case_id"],
    ]
    if args.initial_reference_dir is not None:
        command.extend(("--initial_reference_dir", str(args.initial_reference_dir.resolve())))
    return command


def _run_worker(args: argparse.Namespace, case: dict, provenance: dict) -> None:
    # Only the isolated child imports simulator libraries or loads a policy.
    import gymnasium as gym
    import torch
    from benchmark_state import assert_matching_initial_states, capture_initial_state
    from server import DirectBridge, build_task_cfg, task_id

    from isaaclab.app import launch_simulation

    case_dir = args.output_dir / case["case_id"]
    case_dir.mkdir()
    _write_json(case_dir / "case.json", {**case, **provenance})
    bridge = None
    try:
        cfg = build_task_cfg(args.variant, case["seed"], render_images=True)
        with launch_simulation(cfg):
            env = gym.make(task_id(args.variant), cfg=cfg).unwrapped
            original_reset = env._reset_idx
            try:
                policy = torch.jit.load(provenance["policy"], map_location=env.device).eval()
                with torch.inference_mode():
                    # Model loading precedes the only explicit reset; no wrapper or warmup reset is used.
                    env.reset(seed=case["seed"])
                    initial = capture_initial_state(env, case_dir / "initial_state.json")
                    if args.initial_reference_dir is not None:
                        reference = args.initial_reference_dir / case["case_id"] / "initial_state.json"
                        match = assert_matching_initial_states(reference, initial)
                        _write_json(case_dir / "initial_state_match.json", match)
                    started = time.monotonic()
                    bridge = DirectBridge(
                        env,
                        case_dir,
                        args.variant,
                        case["seed"],
                        controller=provenance["controller"],
                        rl_policy_loaded=True,
                        render_images=False,
                        case_id=case["case_id"],
                    )
                    for _ in range(env.max_episode_length + 1):
                        action = policy(env.obs_buf["policy"])
                        expected_shape = (1, env.action_manager.total_action_dim)
                        if tuple(action.shape) != expected_shape or not bool(torch.isfinite(action).all()):
                            raise ValueError(f"Policy returned invalid actions for {case['case_id']}.")
                        if bridge.step(action):
                            break
                    if bridge.terminal is None:
                        raise RuntimeError("No native terminal outcome within the configured episode horizon.")
                    bridge.save()
                    terminal = bridge.terminal
                    record = {
                        **case,
                        "status": "completed",
                        "success": terminal["success"],
                        "terminations": terminal["terminations"],
                        "steps": len(bridge.actions),
                        "simulation_time_s": terminal["time_s"],
                        "wall_time_s": time.monotonic() - started,
                        "physical_state_sha256": initial["physical_state_sha256"],
                        "policy_observation_sha256": initial["policy_observation_sha256"],
                        "episode_directory": str(bridge.episode_dir.resolve()),
                    }
                    _assert_frozen(provenance)
                    _write_json(case_dir / "result.json", record)
                    print(json.dumps(record), flush=True)
            except Exception:
                if bridge is not None:
                    bridge.save()
                raise
            finally:
                env._reset_idx = original_reset
                env.close()
    except Exception as error:
        _write_json(
            case_dir / "infrastructure_error.json", {**case, "status": "infrastructure_error", "error": str(error)}
        )
        raise


def _assert_frozen(provenance: dict) -> None:
    if _source_hashes() != provenance["source_sha256"]:
        raise RuntimeError("Benchmark sources changed; this is not a frozen-source run.")
    for name in ("manifest", "policy", "checkpoint"):
        if _sha256(Path(provenance[name])) != provenance[f"{name}_sha256"]:
            raise RuntimeError(f"Frozen {name} changed during evaluation.")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    task_root = _REPOSITORY_ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib"
    paths = list(task_root.glob("usb_cable_routing*/**/*.py"))
    paths.extend(task_root.glob("usb_cable_routing*/**/*.usda"))
    paths.extend(Path(__file__).parent.glob("*.py"))
    return {str(path.relative_to(_REPOSITORY_ROOT)): _sha256(path) for path in sorted(paths)}


def _progress(records: list[dict], expected: int) -> dict:
    successes = sum(record["success"] for record in records)
    return {
        "expected_cases": expected,
        "completed_cases": len(records),
        "successes": successes,
        "failures": len(records) - successes,
        "complete": len(records) == expected,
        "records": records,
    }


if __name__ == "__main__":
    main()
