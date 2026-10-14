# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send one explicitly chosen action to the paused Newton simulation bridge."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


def main() -> int:
    """Submit one request or read the latest observation, without choosing any robot action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="Directory of a running bridge server.")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--request", help="JSON object describing one explicitly chosen operation.")
    operation.add_argument("--observe", action="store_true", help="Read observation.json without advancing physics.")
    parser.add_argument("--timeout_s", type=float, default=120.0, help="Response wait limit [s].")
    parser.add_argument(
        "--request_id", type=uuid.UUID, help="Reuse this UUID to wait for a previously submitted request."
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout_s must be finite and positive.")
    if args.observe and args.request_id is not None:
        parser.error("--request_id applies only to --request.")
    try:
        if args.observe:
            print(json.dumps(_read_object(args.run_dir / "observation.json"), indent=2, allow_nan=False))
            return 0
        request = json.loads(args.request)
        _validate_request(request)
        if not (args.run_dir / "commands").is_dir() or not (args.run_dir / "responses").is_dir():
            raise ValueError(f"Bridge is not ready at {args.run_dir}: commands/ and responses/ must exist.")
        request_id = str(args.request_id or uuid.uuid4())
        reservation_dir = args.run_dir / "client_requests"
        reservation_dir.mkdir(exist_ok=True)
        reservation = reservation_dir / f"{request_id}.json"
        command = args.run_dir / "commands" / f"{request_id}.json"
        response_path = args.run_dir / "responses" / f"{request_id}.json"
        print(f"Request ID: {request_id}", file=sys.stderr, flush=True)
        if reservation.exists():
            if _read_object(reservation) != request:
                raise ValueError("This request ID belongs to a different request; it cannot be reused.")
        elif command.exists() or response_path.exists():
            raise ValueError("Request ID already exists without a matching client record; choose an unused UUID.")
        elif _publish_once(reservation, request):
            if not _publish_once(command, request):
                raise ValueError("Command ID already exists; it was not replaced or submitted again.")
        elif _read_object(reservation) != request:
            raise ValueError("Concurrent submission used this request ID for a different request.")
        # A consumed command may still be executing. The permanent reservation prevents resubmission.
        deadline = time.monotonic() + args.timeout_s
        while not response_path.exists():
            if time.monotonic() >= deadline:
                print(
                    f"Timed out waiting for {response_path}. The command may still be executing. "
                    f"Resume with the same --request and --request_id {request_id}; do not create a new request.",
                    file=sys.stderr,
                )
                return 124
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        response = _read_object(response_path)
        print(json.dumps(response, indent=2, allow_nan=False))
        return 1 if response.get("error") or response.get("ok") is False else 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _publish_once(path: Path, value: dict[str, object]) -> bool:
    """Publish a complete JSON file atomically without replacing a previously submitted operation."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".request-", delete=False) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_request(request: object) -> None:
    if not isinstance(request, dict):
        raise ValueError("--request must be a JSON object.")
    operation = request.get("op")
    if operation not in ("observe", "move", "hold", "reset", "close"):
        raise ValueError("Request op must be observe, move, hold, reset, or close.")
    fields = {"op", "rationale"}
    if operation in ("observe", "move"):
        fields.add("camera")
        if request.get("camera", "routing") not in ("routing", "socket"):
            raise ValueError("camera must be routing or socket.")
    if operation in ("move", "hold", "reset"):
        if not isinstance(request.get("rationale"), str) or not request["rationale"].strip():
            raise ValueError(f"{operation} requires an explicit, nonempty model rationale.")
    if operation in ("move", "hold"):
        fields.add("steps")
        steps = request.get("steps")
        if type(steps) is not int or not 1 <= steps <= 90:
            raise ValueError("steps must be an integer from 1 to 90.")
    if operation == "reset":
        fields.add("seed")
        if "seed" in request and (type(request["seed"]) is not int or request["seed"] < 0):
            raise ValueError("seed must be a nonnegative integer.")
    if operation == "move":
        fields.update(
            (
                "frame",
                "target_position_m",
                "target_quat_xyzw",
                "speed_m_s",
                "angular_speed_rad_s",
                "stop_on_stage_change",
            )
        )
        if request.get("frame") not in ("tcp", "tip", "plug_front"):
            raise ValueError("move frame must be tcp, tip, or plug_front.")
        _validate_vector(request.get("target_position_m"), 3, "target_position_m")
        if "target_quat_xyzw" in request:
            _validate_vector(request["target_quat_xyzw"], 4, "target_quat_xyzw")
            if not math.isclose(math.hypot(*request["target_quat_xyzw"]), 1.0, abs_tol=0.001):
                raise ValueError("target_quat_xyzw must have unit norm within 0.001.")
        for name, maximum, default in (("speed_m_s", 0.15, 0.10), ("angular_speed_rad_s", 0.5, 0.30)):
            value = request.get(name, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be finite and in (0, {maximum}].")
        if "stop_on_stage_change" in request and type(request["stop_on_stage_change"]) is not bool:
            raise ValueError("stop_on_stage_change must be a boolean.")
    unknown = request.keys() - fields
    if unknown:
        raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}.")
    json.dumps(request, allow_nan=False)


def _validate_vector(value: object, length: int, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)
    ):
        raise ValueError(f"{name} must contain {length} finite numbers.")


if __name__ == "__main__":
    sys.exit(main())
