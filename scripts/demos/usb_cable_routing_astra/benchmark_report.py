# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit and report the frozen 100-case-per-task RL versus direct Astra evaluation.

This offline reporter never loads a policy, starts a simulator, or submits commands.
Run from the repository, or place the script beside benchmark_state.py. Incomplete
runs retain their actual denominators; a missing trial is never scored as a failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_VARIANTS = ("scurve", "insert")
_CONTROLLERS = ("rl", "astra")
_EXPECTED_PER_TASK = 100
_MODEL = "gpt-6-astra"


def wilson_interval(successes: int, trials: int) -> list[float] | None:
    """Return the two-sided 95% Wilson interval, or None for no completed trials."""
    if not 0 <= successes <= trials:
        raise ValueError("Require 0 <= successes <= trials.")
    if not trials:
        return None
    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1 + z * z / trials
    center = (fraction + z * z / (2 * trials)) / denominator
    half_width = z / denominator * math.sqrt(fraction * (1 - fraction) / trials + z * z / (4 * trials**2))
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def exact_mcnemar(rl_only: int, astra_only: int) -> float:
    """Return the exact two-sided binomial McNemar p-value for discordant pairs."""
    if min(rl_only, astra_only) < 0:
        raise ValueError("Discordant counts must be nonnegative.")
    discordant = rl_only + astra_only
    numerator = sum(math.comb(discordant, count) for count in range(min(rl_only, astra_only) + 1))
    return min(1.0, 2 * numerator / 2**discordant)


def _read_json(path: Path) -> dict | list:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _object(path: Path) -> dict:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _repository(run_dir: Path) -> Path:
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents, *run_dir.parents):
        if (candidate / "scripts/demos/usb_cable_routing_astra/benchmark_state.py").is_file():
            return candidate
    raise ValueError("Cannot locate benchmark_state.py; run the reporter from the Isaac Lab repository.")


def _check_identity(value: dict, case: dict, *, episode: bool = False) -> None:
    for key in ("variant", "seed"):
        _require(type(value.get(key)) is type(case[key]) and value[key] == case[key], f"{key} disagrees with manifest")
    key = "evaluation_case_id" if episode else "case_id"
    _require(value.get(key) == case["case_id"], f"{key} disagrees with manifest")


def _empty_record(case: dict, controller: str) -> dict:
    return {
        **case,
        "controller": controller,
        "status": "not_started",
        "success": None,
        "observed_native_success": None,
        "steps": None,
        "simulation_time_s": None,
        "terminations": {},
        "initial_state": None,
        "physical_state_sha256": None,
        "policy_observation_sha256": None,
        "initial_state_match": None,
        "episode_directory": None,
        "errors": [],
    }


def _validate_terminal(episode: Path, case: dict, controller: str, initial: Path) -> dict:
    success = _object(episode / "success.json")
    _check_identity(success, case, episode=True)
    _require(success.get("rl_policy_loaded") is (controller == "rl"), "RL-loaded provenance flag is incorrect")
    _require(success.get("task_physics_or_success_modified") is False, "Task modification flag is missing or true")
    terminal = success.get("terminal")
    if terminal is None:
        return {"status": "incomplete"}
    _require(isinstance(terminal, dict), "Terminal must be an object or null")
    _require(type(terminal.get("success")) is bool, "Native success must be a boolean")
    terms = terminal.get("terminations")
    _require(
        isinstance(terms, dict) and terms and all(type(value) is bool for value in terms.values()),
        "Native terminations must be a nonempty dictionary of booleans",
    )
    _require(any(terms.values()), "Terminal has no native termination")
    _require(terms.get("success") is terminal["success"], "Success and native termination disagree")
    steps = success.get("policy_steps")
    _require(type(steps) is int and steps > 0, "Terminal policy_steps must be positive")
    _require(terminal.get("step") == steps, "Terminal step disagrees with action count")
    _require(success.get("recorded_states") == steps + 1, "Expected initial state plus every terminal control step")
    dt = success.get("simulation_dt_s")
    _require(type(dt) in (int, float) and dt > 0, "Invalid control interval")
    _require(
        math.isclose(terminal["time_s"], steps * dt, rel_tol=1e-12, abs_tol=1e-12), "Terminal time is inconsistent"
    )

    trace = _read_json(episode / "trace.json")
    _require(isinstance(trace, list) and len(trace) == steps + 1, "Trace count disagrees with native terminal")
    _require([row["step"] for row in trace] == list(range(steps + 1)), "Trace steps have an omission or duplicate")
    _require(all(terminal.get(key) == value for key, value in trace[-1].items()), "Terminal differs from final trace")
    initial_data = _object(initial)
    _require(success.get("task") == initial_data["task_id"], "Recorded task differs from initial-state task")
    _require(dt == initial_data["metadata"]["control_dt_s"], "Control interval differs from initial task configuration")
    with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory:
        poses = trajectory["body_pose_m_xyzw"]
        actions = trajectory["actions"]
        times = trajectory["time_s"]
        _require(poses.ndim == 3 and poses.shape[0] == steps + 1 and poses.shape[2] == 7, "Pose count/shape is invalid")
        _require(actions.ndim == 2 and len(actions) == steps, "Action count/shape is invalid")
        _require(np.isfinite(poses).all() and np.isfinite(actions).all(), "Trajectory contains nonfinite values")
        _require(
            np.array_equal(times, np.asarray([row["time_s"] for row in trace])), "Trajectory times differ from trace"
        )
        initial_pose = initial_data["arrays"]["newton_body_pose"]
        _require(
            np.array_equal(poses[0], np.asarray(initial_pose["values"], dtype=initial_pose["dtype"])),
            "First recorded Newton pose differs from the captured initial state",
        )
        _require(trajectory["body_labels"].tolist() == initial_data["metadata"]["body_labels"], "Body labels differ")
    return {
        "status": "completed",
        "success": terminal["success"],
        "observed_native_success": terminal["success"],
        "steps": steps,
        "simulation_time_s": terminal["time_s"],
        "terminations": terms,
    }


def _read_controller_record(case: dict, controller: str, directory: Path, episode: Path | None, compare_states) -> dict:
    record = _empty_record(case, controller)
    if not directory.exists():
        return record
    record["status"] = "incomplete"
    initial = directory / "initial_state.json"
    if initial.is_file():
        record["initial_state"] = str(initial)
        try:
            verified = compare_states(initial, initial)
            for key in ("physical_state_sha256", "policy_observation_sha256"):
                record[key] = verified[key]
        except (ValueError, TypeError, KeyError, OSError) as error:
            record["errors"].append(f"Invalid initial state: {error}")
    error_path = directory / "infrastructure_error.json"
    if error_path.is_file():
        try:
            failure = _object(error_path)
            _check_identity(failure, case)
            record["infrastructure_error"] = failure.get("error", "Unspecified infrastructure error")
        except (ValueError, TypeError, KeyError, OSError) as error:
            record["errors"].append(f"Malformed infrastructure error: {error}")
        record["status"] = "infrastructure_error"
    if episode is not None:
        record["episode_directory"] = str(episode)
        if (episode / "success.json").is_file():
            try:
                outcome = _validate_terminal(episode, case, controller, initial)
                record.update(outcome)
                if error_path.exists():
                    record["status"] = "infrastructure_error"
            except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
                record["errors"].append(f"Invalid native episode: {error}")
    if controller == "rl":
        result_path = directory / "result.json"
        try:
            episodes = list(directory.glob("episode_*"))
            _require(len(episodes) <= 1, "Multiple RL episodes violate the one-attempt rule")
            if result_path.is_file():
                result = _object(result_path)
                _check_identity(result, case)
                _require(result.get("status") == "completed", "RL result does not declare a native completion")
                _require(record["status"] == "completed", "RL result has no validated native terminal")
                for key in (
                    "success",
                    "terminations",
                    "steps",
                    "simulation_time_s",
                    "physical_state_sha256",
                    "policy_observation_sha256",
                ):
                    _require(result.get(key) == record[key], f"RL result disagrees with recorded {key}")
                _require(Path(result["episode_directory"]).resolve() == episode.resolve(), "RL episode path differs")
            elif record["status"] == "completed":
                # The terminal can be saved just before result.json during an active rollout.
                record["status"] = "incomplete"
        except (ValueError, TypeError, KeyError, OSError) as error:
            record["errors"].append(str(error))
    if record["errors"]:
        record["status"] = "invalid"
        record["success"] = None
    return record


def _audit_hashes(mapping: dict, root: Path, label: str, errors: list[str]) -> dict:
    _require(isinstance(mapping, dict) and mapping, f"{label} must contain file hashes")
    checked = {}
    for relative, expected in mapping.items():
        path = (root / relative).resolve()
        try:
            _require(path.is_relative_to(root.resolve()), f"{label}: path escapes repository: {relative}")
            actual = _sha256(path)
            _require(actual == expected, f"{label}: hash mismatch: {relative}")
            checked[relative] = actual
        except (ValueError, OSError) as error:
            errors.append(str(error))
    return checked


def _audit_provenance(run_dir: Path, repository: Path, manifest: dict, errors: list[str]) -> dict:
    manifest_hash = _sha256(run_dir / "manifest.json")
    prompt_hash = _sha256(run_dir / "astra_controller_prompt.md")
    _require(manifest.get("astra_prompt_sha256") == prompt_hash, "Astra prompt differs from the frozen case manifest")
    _require(manifest.get("model") == _MODEL, "Manifest model is not gpt-6-astra")
    provenance = {"manifest_sha256": manifest_hash, "astra_prompt_sha256": prompt_hash, "freeze_verified": False}
    freeze_path = run_dir / "freeze_manifest.json"
    freeze = None
    if freeze_path.is_file():
        freeze = _object(freeze_path)
        _require(freeze.get("manifest_sha256") == manifest_hash, "Frozen manifest hash changed")
        _require(freeze.get("astra_prompt_sha256") == prompt_hash, "Frozen prompt hash changed")
        _require(freeze.get("model") == _MODEL, "Frozen model identifier differs")
        count_before = len(errors)
        provenance["source_sha256"] = _audit_hashes(freeze.get("source_sha256"), repository, "source", errors)
        provenance["baseline_artifacts_sha256"] = _audit_hashes(
            freeze.get("baseline_artifacts_sha256"), repository, "baseline", errors
        )
        provenance["freeze_manifest_sha256"] = _sha256(freeze_path)
        provenance["freeze_verified"] = len(errors) == count_before
        provenance["protocol"] = freeze.get("protocol")
    provenance["rl"] = {}
    for variant in _VARIANTS:
        path = run_dir / "rl" / variant / "provenance.json"
        if not path.exists():
            continue
        info = _object(path)
        _require(info.get("manifest_sha256") == manifest_hash, f"{variant}: RL manifest hash differs")
        _require(info.get("variant") == variant, f"{variant}: RL provenance variant differs")
        _require(info.get("case_count") == _EXPECTED_PER_TASK, f"{variant}: RL case count differs")
        _require(info.get("num_envs") == 1, f"{variant}: RL was not evaluated with one environment")
        _require(info.get("stochastic_policy") is False, f"{variant}: RL deterministic policy flag is incorrect")
        _require(info.get("external_action_clipping") is False, f"{variant}: RL external clipping flag is incorrect")
        _audit_hashes(info.get("source_sha256"), repository, f"{variant} RL source", errors)
        if freeze is not None:
            for name, digest in info["source_sha256"].items():
                _require(freeze["source_sha256"].get(name) == digest, f"{variant}: RL source not frozen: {name}")
        for kind in ("policy", "checkpoint"):
            artifact = Path(info[kind]).resolve()
            _require(artifact.is_relative_to(repository), f"{variant}: {kind} is outside repository")
            _require(_sha256(artifact) == info[f"{kind}_sha256"], f"{variant}: {kind} hash changed")
            if freeze is not None:
                relative = str(artifact.relative_to(repository))
                _require(
                    freeze["baseline_artifacts_sha256"].get(relative) == info[f"{kind}_sha256"],
                    f"{variant}: {kind} was not frozen",
                )
        provenance["rl"][variant] = {"provenance_sha256": _sha256(path), **info}
    return provenance


def _audit_assignments(run_dir: Path, cases_by_id: dict, prompt_hash: str, errors: list[str]) -> dict:
    path = run_dir / "agent_assignments.json"
    if not path.is_file():
        return {}
    assignments = _object(path)
    names = set()
    for case_id, assignment in assignments.items():
        try:
            _require(case_id in cases_by_id, f"Unknown assigned case: {case_id}")
            case = cases_by_id[case_id]
            _require(isinstance(assignment, dict), "Assignment must be an object")
            for key in ("variant", "seed"):
                _require(type(assignment.get(key)) is type(case[key]) and assignment[key] == case[key], f"Wrong {key}")
            _require(assignment.get("model") == _MODEL, "Assignment model differs")
            _require(assignment.get("fork_turns") == "none", "Astra case did not receive a fresh context")
            _require(assignment.get("prompt_sha256") == prompt_hash, "Assignment prompt hash differs")
            name = assignment.get("agent_name")
            _require(isinstance(name, str) and name.startswith("/"), "Missing canonical agent name")
            _require(name not in names, f"Agent context reused: {name}")
            names.add(name)
            _require(bool(assignment.get("assigned_utc")), "Missing assignment timestamp")
            worker = Path(assignment["run_dir"]).resolve()
            episode = Path(assignment["episode_directory"]).resolve()
            _require(worker.parent == (run_dir / "astra").resolve(), "Worker path is outside official Astra directory")
            _require(episode.parent == worker and episode.name.startswith("episode_"), "Invalid assigned episode path")
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(f"Astra assignment {case_id}: {error}")
    return assignments


def _collect_astra_episodes(run_dir: Path, cases_by_id: dict, assignments: dict, errors: list[str]) -> dict:
    episodes = {}
    assigned_paths = {
        str(Path(value["episode_directory"]).resolve()): key
        for key, value in assignments.items()
        if isinstance(value, dict) and isinstance(value.get("episode_directory"), str)
    }
    for episode in sorted((run_dir / "astra").glob("*/episode_*")):
        if not episode.is_dir():
            continue
        path = episode / "success.json"
        try:
            case_id = (
                _object(path).get("evaluation_case_id")
                if path.is_file()
                else assigned_paths.get(str(episode.resolve()))
            )
            _require(case_id in cases_by_id, f"Unknown or missing evaluation_case_id in {episode}")
            _require(case_id not in episodes, f"Repeated Astra episode for {case_id}; retries are prohibited")
            episodes[case_id] = episode
            _require(case_id in assignments, f"Missing fresh-agent assignment for {case_id}")
            _require(
                assigned_paths.get(str(episode.resolve())) == case_id, f"Assignment episode path differs for {case_id}"
            )
        except (ValueError, TypeError, KeyError, OSError) as error:
            errors.append(str(error))
    return episodes


def _controller_summary(records: list[dict], expected: int, valid: bool) -> dict:
    counts = Counter(record["status"] for record in records)
    completed = [record for record in records if record["status"] == "completed"]
    successes = sum(record["success"] for record in completed)
    complete = len(completed) == expected and valid
    return {
        "expected": expected,
        "completed": len(completed),
        "successes": successes,
        "native_failures": len(completed) - successes,
        "infrastructure_errors": counts["infrastructure_error"],
        "incomplete": counts["incomplete"],
        "not_started": counts["not_started"],
        "invalid": counts["invalid"],
        "complete": complete,
        "final_success_rate": successes / expected if complete else None,
        "observed_success_fraction": successes / len(completed) if completed else None,
        "wilson95_completed_trials": wilson_interval(successes, len(completed)),
        "rate_scope": "final" if complete else "provisional completed cases only; may be selection-biased",
        "termination_counts": dict(
            Counter(name for record in completed for name, active in record["terminations"].items() if active)
        ),
    }


def build_report(run_dir: Path) -> dict:
    """Audit saved outcomes and return an explicit complete or partial benchmark report."""
    repository = _repository(run_dir)
    sys.path.insert(0, str(repository / "scripts/demos/usb_cable_routing_astra"))
    from benchmark_state import assert_matching_initial_states, load_evaluation_cases

    manifest_path = run_dir / "manifest.json"
    manifest = _object(manifest_path)
    cases = [case for variant in _VARIANTS for case in load_evaluation_cases(manifest_path, variant)]
    _require(manifest.get("cases_per_task") == _EXPECTED_PER_TASK, "Official manifest must declare 100 cases per task")
    for variant in _VARIANTS:
        _require(
            sum(case["variant"] == variant for case in cases) == _EXPECTED_PER_TASK, f"Expected 100 {variant} cases"
        )
    cases_by_id = {case["case_id"]: case for case in cases}
    errors = []
    warnings = []
    provenance = {}
    try:
        provenance = _audit_provenance(run_dir, repository, manifest, errors)
    except (ValueError, TypeError, KeyError, OSError) as error:
        errors.append(f"Provenance: {error}")
    if not (run_dir / "freeze_manifest.json").exists():
        warnings.append("No freeze_manifest.json: source and baseline freeze provenance is not verified.")
    prompt_hash = _sha256(run_dir / "astra_controller_prompt.md")
    assignments = _audit_assignments(run_dir, cases_by_id, prompt_hash, errors)
    astra_episodes = _collect_astra_episodes(run_dir, cases_by_id, assignments, errors)
    for path in (run_dir / "rl").glob("*/*/result.json"):
        variant, case_id = path.parent.parent.name, path.parent.name
        if case_id not in cases_by_id or cases_by_id[case_id]["variant"] != variant:
            errors.append(f"Result outside frozen manifest: {path}")
    records = []
    matching = {}
    for case in cases:
        case_id = case["case_id"]
        rl_dir = run_dir / "rl" / case["variant"] / case_id
        rl_episode = rl_dir / "episode_000" if (rl_dir / "episode_000").exists() else None
        rl = _read_controller_record(case, "rl", rl_dir, rl_episode, assert_matching_initial_states)
        episode = astra_episodes.get(case_id)
        astra = (
            _read_controller_record(case, "astra", episode, episode, assert_matching_initial_states)
            if episode
            else _empty_record(case, "astra")
        )
        if case_id in assignments and episode is None:
            astra["status"] = "incomplete"
        if rl["initial_state"] and astra["initial_state"]:
            try:
                match = assert_matching_initial_states(rl["initial_state"], astra["initial_state"])
                saved_match = _object(episode / "initial_state_comparison.json")
                _require(saved_match.get("matched") is True, "Astra initial matching report did not pass")
                for key in ("physical_state_sha256", "policy_observation_sha256", "state_sha256"):
                    _require(saved_match.get(key) == match[key], f"Saved match report has wrong {key}")
                matching[case_id] = match
                rl["initial_state_match"] = astra["initial_state_match"] = True
            except (ValueError, TypeError, KeyError, OSError) as error:
                rl["initial_state_match"] = astra["initial_state_match"] = False
                astra["errors"].append(f"Paired initial-state validation: {error}")
                astra["status"] = "invalid"
                astra["success"] = None
        elif astra["status"] == "completed":
            astra["errors"].append("Native terminal lacks the paired initial states")
            astra["status"] = "invalid"
            astra["success"] = None
        for record in (rl, astra):
            errors.extend(f"{record['controller']}/{case_id}: {error}" for error in record["errors"])
            records.append(record)

    for variant in _VARIANTS:
        if any(
            row["controller"] == "rl" and row["variant"] == variant and row["status"] != "not_started"
            for row in records
        ):
            if variant not in provenance.get("rl", {}):
                errors.append(f"{variant}: RL trials lack valid variant-level provenance")

    # A final driver aggregate confirms the source check at shutdown was reached.
    for variant in _VARIANTS:
        selected = [record for record in records if record["controller"] == "rl" and record["variant"] == variant]
        if all(record["status"] == "completed" for record in selected):
            try:
                aggregate = _object(run_dir / "rl" / variant / "evaluation.json")
                _require(aggregate.get("complete") is True, f"{variant}: RL driver did not mark completion")
                _require(
                    aggregate.get("completed_cases") == _EXPECTED_PER_TASK, f"{variant}: RL aggregate count differs"
                )
                _require(
                    aggregate.get("expected_cases") == _EXPECTED_PER_TASK,
                    f"{variant}: RL aggregate denominator differs",
                )
                aggregate_records = aggregate["records"]
                _require(len(aggregate_records) == _EXPECTED_PER_TASK, f"{variant}: RL aggregate record count differs")
                _require(
                    {row["case_id"] for row in aggregate_records} == {row["case_id"] for row in selected},
                    f"{variant}: RL aggregate omits or repeats cases",
                )
                _require(
                    aggregate.get("successes") == sum(row["success"] for row in selected),
                    f"{variant}: RL aggregate success count differs",
                )
            except (ValueError, TypeError, KeyError, OSError) as error:
                errors.append(str(error))

    variants = {}
    for variant in _VARIANTS:
        selected = [record for record in records if record["variant"] == variant]
        controllers = {
            controller: _controller_summary(
                [row for row in selected if row["controller"] == controller], _EXPECTED_PER_TASK, not errors
            )
            for controller in _CONTROLLERS
        }
        by_key = {(record["case_id"], record["controller"]): record for record in selected}
        table = {"both_success": 0, "rl_only_success": 0, "astra_only_success": 0, "both_failure": 0}
        for case in (case for case in cases if case["variant"] == variant):
            rl, astra = (by_key[case["case_id"], controller] for controller in _CONTROLLERS)
            if rl["status"] != "completed" or astra["status"] != "completed" or case["case_id"] not in matching:
                continue
            if rl["success"] and astra["success"]:
                cell = "both_success"
            elif rl["success"]:
                cell = "rl_only_success"
            elif astra["success"]:
                cell = "astra_only_success"
            else:
                cell = "both_failure"
            table[cell] += 1
        complete = (
            all(value["complete"] for value in controllers.values()) and sum(table.values()) == _EXPECTED_PER_TASK
        )
        variants[variant] = {
            "controllers": controllers,
            "paired_completed": sum(table.values()),
            "paired_expected": _EXPECTED_PER_TASK,
            "initial_states_matched": sum(
                case_id in matching for case_id, case in cases_by_id.items() if case["variant"] == variant
            ),
            "paired_table": table,
            "complete": complete,
            "mcnemar_exact_two_sided_p": exact_mcnemar(table["rl_only_success"], table["astra_only_success"])
            if complete
            else None,
        }
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "complete": all(value["complete"] for value in variants.values()),
        "valid": not errors,
        "variants": variants,
        "provenance": provenance,
        "agent_assignments_count": len(assignments),
        "protocol": {
            key: manifest.get(key)
            for key in (
                "attempt_rule",
                "model",
                "astra_context",
                "observation_scope",
                "randomization",
                "configuration_matching",
                "baseline",
            )
        },
        "interpretation": [
            "The existing native interfaces are compared; controller inputs are not identical.",
            "RL receives its fixed trained observation vector. Astra receives measured full-state JSON and "
            "optional images; physics pauses between Astra decisions.",
            "The insertion RL export is hybrid PPO routing plus programmed feedback and a learned PPO "
            "residual; it is not a pure end-to-end PPO baseline.",
            "Each controller receives one native episode per case, without retries or case replacement. Astra "
            "receives a new isolated model context and fixed prompt per case.",
            "Native timeout or task termination without success is a failure. Infrastructure errors and "
            "unfinished cases are reported separately and are not silently scored as failures.",
            "Partial completed-case fractions and Wilson intervals are provisional and can be "
            "selection-biased. Final rates and exact paired McNemar tests require all cases and valid audits.",
            "Wilson intervals describe per-task success proportions; exact McNemar tests compare discordant "
            "matched pairs without a continuity correction. The two task tests are reported without "
            "multiplicity adjustment.",
            "Assignment and source hashes audit recorded provenance; they are not independent cryptographic "
            "proof of model identity or prompt isolation.",
        ],
        "errors": errors,
        "warnings": warnings,
        "records": records,
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(run_dir: Path, summary: dict) -> None:
    """Write machine-readable results and a Markdown report without altering trial artifacts."""
    _atomic_text(run_dir / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fields = [
        "case_id",
        "variant",
        "seed",
        "controller",
        "status",
        "success",
        "observed_native_success",
        "steps",
        "simulation_time_s",
        "initial_state_match",
        "physical_state_sha256",
        "policy_observation_sha256",
        "episode_directory",
        "errors",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in summary["records"]:
        writer.writerow({**row, "errors": " | ".join(row["errors"])})
    _atomic_text(run_dir / "results.csv", stream.getvalue())
    lines = [
        "# RL versus direct Astra: matched USB cable tasks",
        "",
        f"Status: **{'complete' if summary['complete'] else 'incomplete'}**; "
        f"artifact audit: **{'passed' if summary['valid'] else 'failed'}**.",
        "",
        "Counts below use validated native terminal episodes. Missing cases are not failures. A provisional "
        "rate uses only completed cases, never the planned count of 100.",
        "",
        "| Task | Controller | Completed / expected | Successes | Native failures | Infrastructure errors | "
        "Incomplete | Not started | Invalid | Success fraction | Wilson 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for variant, result in summary["variants"].items():
        for controller, row in result["controllers"].items():
            fraction = row["observed_success_fraction"]
            rate = (
                "unavailable"
                if fraction is None
                else (
                    f"{row['successes']}/{row['completed']} ({fraction:.1%}); "
                    f"{'final' if row['complete'] else 'provisional'}"
                )
            )
            interval = row["wilson95_completed_trials"]
            ci = "unavailable" if interval is None else f"[{interval[0]:.1%}, {interval[1]:.1%}]"
            lines.append(
                f"| {variant} | {controller} | {row['completed']} / {row['expected']} | {row['successes']} | "
                f"{row['native_failures']} | {row['infrastructure_errors']} | {row['incomplete']} | "
                f"{row['not_started']} | {row['invalid']} | {rate} | {ci} |"
            )
    lines.extend(
        [
            "",
            "| Task | Matched initial states | Completed matched pairs | Both succeed | RL only | Astra only "
            "| Both fail | Exact McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for variant, row in summary["variants"].items():
        table = row["paired_table"]
        p_value = row["mcnemar_exact_two_sided_p"]
        p_text = "withheld until complete and valid" if p_value is None else f"{p_value:.6g}"
        lines.append(
            f"| {variant} | {row['initial_states_matched']} / 100 | {row['paired_completed']} / 100 | "
            f"{table['both_success']} | {table['rl_only_success']} | {table['astra_only_success']} | "
            f"{table['both_failure']} | {p_text} |"
        )
    lines.extend(["", *[paragraph + "\n" for paragraph in summary["interpretation"]]])
    lines.append(
        f"Source/baseline freeze verified: {summary['provenance'].get('freeze_verified', False)}. "
        f"Recorded Astra assignments: {summary['agent_assignments_count']} / 200.\n"
    )
    for label in ("errors", "warnings"):
        if summary[label]:
            lines.extend([f"{label.capitalize()}:", "", *[f"- {message}" for message in summary[label]], ""])
    lines.extend(
        ["Artifacts: [summary.json](summary.json), [results.csv](results.csv), [manifest.json](manifest.json).", ""]
    )
    _atomic_text(run_dir / "REPORT.md", "\n".join(lines))


def main() -> int:
    """Audit a benchmark directory and optionally require every frozen trial to be complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        summary = build_report(run_dir)
        write_report(run_dir, summary)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Benchmark report failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "valid": summary["valid"],
                "errors": len(summary["errors"]),
                "report": str(run_dir / "REPORT.md"),
            }
        )
    )
    if not summary["valid"]:
        return 2
    if args.require_complete and not summary["complete"]:
        print("Benchmark remains incomplete; no final comparison is claimed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
