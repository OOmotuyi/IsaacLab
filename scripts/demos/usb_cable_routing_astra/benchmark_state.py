# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture and compare actual initial states for paired Newton controller evaluations.

Capture after the final explicit seeded reset and before the first action. This module
never resets, steps, or recomputes observations. Equality is exact, including array
precision and ordering; a seed label alone is not considered evidence of a match.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_SCHEMA_VERSION = 1
_POLICY_ARRAY = "policy_observation"


def load_evaluation_cases(path: str | Path, variant: str) -> list[dict[str, str | int]]:
    """Validate a case manifest and return the selected variant in its declared order.

    Args:
        path: JSON manifest containing a cases list.
        variant: Task variant, either ``scurve`` or ``insert``.

    Returns:
        Cases containing an unchanged case identifier, variant, and integer seed.
        The caller enforces the required benchmark count; one-case smoke manifests are valid.

    Raises:
        ValueError: Cases have unsafe identifiers, duplicate identifiers or per-task seeds,
            invalid seed types/ranges, unsupported variants, or no selected cases.
    """
    variants = ("scurve", "insert")
    if variant not in variants:
        raise ValueError(f"Unknown evaluation variant: {variant}")
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("Manifest must be an object containing a cases list.")
    cases = []
    identifiers = set()
    seeds = set()
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Every manifest case must be an object.")
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for character in case_id
            )
        ):
            raise ValueError("case_id must contain only letters, digits, underscores, and hyphens.")
        if case_id in identifiers:
            raise ValueError(f"Duplicate case_id: {case_id}")
        identifiers.add(case_id)
        if case.get("variant") not in variants or type(case.get("seed")) is not int or not 0 <= case["seed"] < 2**32:
            raise ValueError(f"Invalid variant or seed for {case_id}; require an integer seed in [0, 2**32).")
        seed_key = (case["variant"], case["seed"])
        if seed_key in seeds:
            raise ValueError(f"Repeated seed within {case['variant']}: {case['seed']}")
        seeds.add(seed_key)
        if case["variant"] == variant:
            cases.append({key: case[key] for key in ("case_id", "variant", "seed")})
    if not cases:
        raise ValueError(f"Manifest contains no {variant} cases.")
    return cases


def capture_initial_state(env: ManagerBasedRLEnv, path: str | Path | None = None) -> dict[str, object]:
    """Capture initial physical state and the existing policy observation without advancing time.

    Args:
        env: Initialized, explicitly reset Newton environment with exactly one instance.
        path: Optional destination for an atomic JSON snapshot.

    Returns:
        JSON-compatible arrays, metadata, and separate physical/observation SHA-256 digests.
        Poses use position [m] and xyzw quaternion; velocities retain their native ordering.

    Raises:
        ValueError: The environment is not at the beginning of a single-instance episode,
            or a captured state array contains nonfinite values.
    """
    # Import only when capturing a live Newton environment; comparison needs no simulator.
    from isaaclab_newton.physics import NewtonManager

    if env.num_envs != 1 or np.any(_numpy(env.episode_length_buf) != 0):
        raise ValueError("Capture requires num_envs=1 immediately after reset and before any action.")
    command = env.command_manager.get_term("routing")
    arm = env.action_manager.get_term("arm")
    robot = command.robot
    cable = command.cable
    model = NewtonManager.get_model()
    state = NewtonManager.get_state()
    arrays = {}

    def add(name: str, value: object, units: str) -> None:
        arrays[name] = _array_record(_numpy(value), units)

    add("newton_body_pose", state.body_q, "position m; quaternion xyzw")
    add("newton_body_velocity", state.body_qd, "native spatial velocity: linear m/s; angular rad/s")
    add("robot_joint_position", robot.data.joint_pos, "rad")
    add("robot_joint_velocity", robot.data.joint_vel, "rad/s")
    add("robot_actuator_position_target", robot.actuators.target_command.position, "rad")
    add("cable_segment_pose", cable.data.segment_pose_w, "position m; quaternion xyzw")
    add("cable_segment_velocity", cable.data.segment_velocity_w, "native spatial velocity: linear m/s; angular rad/s")
    add("body_mass", model.body_mass, "kg")
    add("body_inertia", model.body_inertia, "kg m^2")
    add("body_center_of_mass", model.body_com, "m")
    dof_ids = _numpy(command.dof_ids).astype(np.int64)
    add("cable_dof_ids", dof_ids, "indices")
    add("cable_joint_stiffness", _numpy(model.joint_target_ke)[dof_ids], "native Newton joint stiffness")
    add("cable_joint_damping", _numpy(model.joint_target_kd)[dof_ids], "native Newton joint damping")
    add("shape_friction", model.shape_material_mu, "dimensionless")
    add("material_stiffness_scale_and_friction", command.material, "dimensionless")
    add("grasp_relative_quaternion", cable.grasp_relative_quat, "xyzw")
    add("grasp_active", cable.grasp_active, "boolean")
    add("grasp_force", cable.tension, "N")
    add("grasp_torque", cable.grasp_torque, "N m")
    add("guide_positions", command.guide_positions, "m")
    add("guide_yaws", command.guide_yaws, "rad")
    add("environment_origins", env.scene.env_origins, "m")

    measured_position, measured_quaternion = arm._compute_frame_pose()
    valid = _numpy(arm._target_valid).astype(bool)
    add("cartesian_target_valid", valid, "boolean")
    add(
        "effective_cartesian_position_target",
        np.where(valid[:, None], _numpy(arm._processed_actions), _numpy(measured_position)),
        "m in robot root frame",
    )
    if hasattr(arm, "target_quat"):
        quaternion = np.where(valid[:, None], _numpy(arm.target_quat), _numpy(measured_quaternion))
    else:
        quaternion = _numpy(command.tool_quat if command.home_q is not None else measured_quaternion)
    add("effective_cartesian_quaternion_target", quaternion, "xyzw in robot root frame")
    for name in (
        "completed",
        "guide_count",
        "dwell",
        "success_dwell",
        "lowering",
        "seated",
        "skip",
        "inserting",
        "entered",
        "wrong_entry",
        "route_dwell",
        "route_mastered",
        "ready_to_align",
        "ready_to_approach",
    ):
        if hasattr(command, name):
            add(f"task_{name}", getattr(command, name), "count or flag")
    for name in ("lift_targets", "seat_targets", "socket_position"):
        if hasattr(command, name):
            add(f"task_{name}", getattr(command, name), "m")
    if hasattr(command, "socket_quat"):
        add("task_socket_quaternion", command.socket_quat, "xyzw")
    add(_POLICY_ARRAY, env.obs_buf["policy"], "native observation scaling")
    metadata = _configuration_value(
        {
            "task_id": env.spec.id if env.spec is not None else type(env.cfg).__qualname__,
            "num_envs": env.num_envs,
            "physics_dt_s": env.physics_dt,
            "control_dt_s": env.step_dt,
            "episode_length_s": env.cfg.episode_length_s,
            "body_labels": list(model.body_label),
            "shape_labels": list(model.shape_label),
            "joint_labels": list(model.joint_label),
            "robot_joint_names": list(robot.joint_names),
            "physics_cfg": env.cfg.sim.physics.to_dict(),
            "action_cfg": arm.cfg.to_dict(),
            "command_cfg": command.cfg.to_dict(),
            "termination_cfg": env.cfg.terminations.to_dict(),
            "cable_material_cfg": env.cfg.scene.cable.spawn.physics_material.to_dict(),
        }
    )
    snapshot = {
        "schema_version": _SCHEMA_VERSION,
        "task_id": metadata["task_id"],
        "metadata": metadata,
        "arrays": arrays,
    }
    snapshot.update(_snapshot_digests(snapshot))
    if path is not None:
        _write_snapshot(Path(path), snapshot)
    return snapshot


def save_initial_state(env: ManagerBasedRLEnv, path: str | Path) -> dict[str, object]:
    """Capture and atomically save an initial-state snapshot without advancing simulation."""
    return capture_initial_state(env, path)


def assert_matching_initial_states(
    reference: dict[str, object] | str | Path,
    candidate: dict[str, object] | str | Path,
) -> dict[str, object]:
    """Require exactly matching physical arrays, task settings, and initial policy observations.

    Args:
        reference: Reference snapshot dictionary or path to its JSON file.
        candidate: Candidate snapshot dictionary or path to its JSON file.

    Returns:
        Compact successful matching report suitable for inclusion in evaluation results.

    Raises:
        ValueError: Snapshot integrity, schema, physical state, or observations do not match.
    """
    first, second = _load_verified(reference), _load_verified(candidate)
    if first["state_sha256"] != second["state_sha256"]:
        changed = []
        if first["metadata"] != second["metadata"]:
            changed.append("metadata")
        names = first["arrays"].keys() | second["arrays"].keys()
        for name in sorted(names):
            a, b = first["arrays"].get(name), second["arrays"].get(name)
            if a is None or b is None or a["sha256"] != b["sha256"]:
                changed.append(name)
        raise ValueError("Initial configurations do not match exactly: " + ", ".join(changed))
    return {
        "matched": True,
        "comparison": "exact array dtype, shape, values, units, labels, task settings and initial policy observation",
        "task_id": first["task_id"],
        "physical_state_sha256": first["physical_state_sha256"],
        "policy_observation_sha256": first["policy_observation_sha256"],
        "state_sha256": first["state_sha256"],
        "array_count": len(first["arrays"]),
    }


def _numpy(value: object) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def _array_record(value: np.ndarray, units: str) -> dict[str, object]:
    if value.dtype.kind not in "bifu" or not np.isfinite(value).all():
        raise ValueError("Initial-state arrays must contain finite numeric or boolean values.")
    value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
    header = {"dtype": value.dtype.str, "shape": list(value.shape), "units": units}
    digest = hashlib.sha256(_canonical_json(header) + b"\0" + value.tobytes()).hexdigest()
    return {**header, "values": value.tolist(), "sha256": digest}


def _configuration_value(value: object) -> object:
    if isinstance(value, Enum):
        return _configuration_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, np.generic):
        return _configuration_value(value.item())
    if isinstance(value, dict):
        return {str(key): _configuration_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_configuration_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if hasattr(value, "to_dict"):
        return _configuration_value(value.to_dict())
    raise TypeError(f"Unsupported initial-state configuration value: {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _snapshot_digests(snapshot: dict[str, object]) -> dict[str, str]:
    physical = {
        "schema_version": snapshot["schema_version"],
        "task_id": snapshot["task_id"],
        "metadata": snapshot["metadata"],
        "arrays": {name: item["sha256"] for name, item in snapshot["arrays"].items() if name != _POLICY_ARRAY},
    }
    physical_hash = hashlib.sha256(_canonical_json(physical)).hexdigest()
    observation_hash = snapshot["arrays"][_POLICY_ARRAY]["sha256"]
    combined = hashlib.sha256(_canonical_json([physical_hash, observation_hash])).hexdigest()
    return {
        "physical_state_sha256": physical_hash,
        "policy_observation_sha256": observation_hash,
        "state_sha256": combined,
    }


def _load_verified(value: dict[str, object] | str | Path) -> dict[str, object]:
    snapshot = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported initial-state snapshot schema.")
    try:
        for name, record in snapshot["arrays"].items():
            array = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
            if (
                list(array.shape) != record["shape"]
                or _array_record(array, record["units"])["sha256"] != record["sha256"]
            ):
                raise ValueError(f"Initial-state snapshot array integrity failed: {name}")
        for name, digest in _snapshot_digests(snapshot).items():
            if snapshot.get(name) != digest:
                raise ValueError(f"Initial-state snapshot integrity failed: {name}")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete initial-state snapshot.") from error
    return snapshot


def _write_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".initial-state-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_json(snapshot) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
