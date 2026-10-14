# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve one declared direct-controller case only after exact RL initial-state matching.

Run this CLI in a fresh process for each manifest case, with a shared per-variant
output directory. ``--controller_metadata`` is a private JSON artifact containing
nonempty ``controller``, ``model``, and ``prompt`` strings. Its complete bytes are
frozen, but model identity and model-context isolation are not verified here.
Only commands, responses, and ``public/observation.json`` are controller inputs;
the runner must separately prevent controller access to private artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server import ContactBridge

    from isaaclab.envs import ManagerBasedRLEnv

_COMMON_SPEC = importlib.util.spec_from_file_location(
    "_contact_direct_evaluation_common", Path(__file__).with_name("evaluate_rl.py")
)
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)


def main() -> None:
    """Launch one simulator and one attempt for an explicitly assigned matched case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("scurve", "insert"), required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--initial_reference_dir", type=Path, required=True)
    parser.add_argument("--controller_metadata", type=Path, required=True)
    args = parser.parse_args()
    case, provenance, case_dir = _prepare_case(args)
    try:
        # Keep simulator imports out of provenance preparation and contract tests.
        import gymnasium as gym
        import torch
        from server import ContactBridge, build_task_cfg, task_id

        from isaaclab.app import launch_simulation

        cfg = build_task_cfg(args.variant, case["seed"])
        with launch_simulation(cfg):
            env = gym.make(task_id(args.variant), cfg=cfg).unwrapped
            try:
                with torch.inference_mode():
                    record = _run_case(env, case, provenance, case_dir, ContactBridge)
            finally:
                env.close()
        _assert_frozen(provenance)
        _COMMON._write_json(case_dir / "result.json", record)
    except Exception as error:
        _record_error(case, case_dir, error)
        raise


def _prepare_case(args: argparse.Namespace) -> tuple[dict, dict, Path]:
    cases = _COMMON._STATE.load_evaluation_cases(args.manifest, args.variant)
    selected = [case for case in cases if case["case_id"] == args.case_id]
    if len(selected) != 1:
        raise ValueError("case_id must identify exactly one declared case of the selected variant.")
    case = selected[0]
    metadata = json.loads(args.controller_metadata.read_text())
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(name), str) or not metadata[name].strip()
        for name in ("controller", "model", "prompt")
    ):
        raise ValueError("controller_metadata requires nonempty controller, model, and prompt strings.")
    paths = {"manifest": args.manifest.resolve(), "controller_metadata": args.controller_metadata.resolve()}
    provenance = {
        "controller": metadata["controller"],
        "declared_model": metadata["model"],
        "model_identity_and_context_isolation": "not verified by this evaluator",
        "variant": args.variant,
        "policy_observation_dim": 65 if args.variant == "insert" else 55,
        "num_envs": 1,
        "guide_count": 3,
        "randomize_guides": True,
        "randomize_cable": True,
        "case_count": len(cases),
        "process_isolation": "one simulator case per CLI process",
        "attempts_per_case": 1,
        "initial_reference_dir": str(args.initial_reference_dir.resolve()),
        "source_sha256": _source_hashes(),
        **{name: str(path) for name, path in paths.items()},
        **{f"{name}_sha256": _COMMON._sha256(path) for name, path in paths.items()},
    }
    reference = (args.initial_reference_dir / case["case_id"] / "initial_state.json").resolve()
    reference_hash = _COMMON._sha256(reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Arguments or frozen sources differ from this benchmark's provenance.")
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError("An existing output directory must contain matching benchmark provenance.")
        with provenance_path.open("x") as stream:
            json.dump(provenance, stream, indent=2, allow_nan=False)
            stream.write("\n")
    case_dir = args.output_dir / case["case_id"]
    case_dir.mkdir()  # Claim once, including when startup or matching subsequently fails.
    provenance = {
        **provenance,
        "initial_reference": str(reference),
        "initial_reference_sha256": reference_hash,
    }
    _COMMON._write_json(case_dir / "case.json", {**provenance, **case, "command": sys.argv})
    return case, provenance, case_dir


def _source_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), _COMMON._STATE_PATH.with_name("client.py"))
    return {
        **_COMMON._source_hashes(),
        **{str(path.relative_to(_COMMON._REPOSITORY_ROOT)): _COMMON._sha256(path) for path in paths},
    }


def _assert_frozen(provenance: dict) -> None:
    if _source_hashes() != provenance["source_sha256"]:
        raise RuntimeError("Evaluation sources changed after the benchmark was frozen.")
    for name in ("manifest", "controller_metadata", "initial_reference"):
        if _COMMON._sha256(Path(provenance[name])) != provenance[f"{name}_sha256"]:
            raise RuntimeError(f"Frozen {name} changed during evaluation.")


def _run_case(env: ManagerBasedRLEnv, case: dict, provenance: dict, case_dir: Path, bridge_factory) -> dict:
    _assert_frozen(provenance)
    env.reset(seed=case["seed"])
    initial = _COMMON._STATE.capture_initial_state(env, case_dir / "initial_state.json")
    match = _COMMON._STATE.assert_matching_initial_states(provenance["initial_reference"], initial)
    _COMMON._write_json(case_dir / "initial_state_match.json", match)
    _assert_frozen(provenance)
    for name in ("commands", "responses"):
        (case_dir / name).mkdir()
    bridge = bridge_factory(
        env,
        case_dir,
        case["variant"],
        case["seed"],
        render_images=False,
        controller=provenance["controller"],
        rl_policy_loaded=False,
        case_id=case["case_id"],
    )
    started = time.monotonic()
    try:
        bridge.observation()
        print(f"CONTACT_ASTRA_BRIDGE_READY {case_dir.resolve()}", flush=True)
        _serve_commands(bridge, case_dir, provenance)
        if bridge.terminal is None:
            raise RuntimeError("Controller closed before a native terminal outcome; this attempt is incomplete.")
        _assert_frozen(provenance)
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
            "initial_state_verified": True,
            "episode_directory": str(bridge.episode_dir.resolve()),
        }
        return record
    finally:
        bridge.save()


def _serve_commands(bridge: ContactBridge, case_dir: Path, provenance: dict) -> None:
    while bridge.terminal is None:
        pending = [
            path
            for path in sorted((case_dir / "commands").glob("*.json"))
            if not (case_dir / "responses" / path.name).exists()
        ]
        if not pending:
            time.sleep(0.1)
            continue
        _assert_frozen(provenance)
        path = pending[0]
        bridge.sequence += 1
        request = None
        try:
            request = json.loads(path.read_text())
            result = bridge.execute(request)
        except (ValueError, TypeError, KeyError) as error:
            result = {"status": "error", "error": str(error)}
        result["request_id"] = path.stem
        _COMMON._write_json(case_dir / "responses" / path.name, result)
        with (case_dir / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps({"request": request, "response": result}) + "\n")
        print(f"COMMAND {path.stem} {result['status']}", flush=True)
        if result["status"] == "closed":
            return


def _record_error(case: dict, case_dir: Path, error: Exception) -> None:
    _COMMON._write_json(
        case_dir / "infrastructure_error.json",
        {**case, "status": "infrastructure_error", "completed": False, "error": str(error)},
    )


if __name__ == "__main__":
    main()
