# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract the phase-conditioned reaching reference from a measured cleanup trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def build_reference(trace: Path, output: Path) -> None:
    """Write beak position [m], gravity direction, and root height [m] by phase [s]."""
    poses = defaultdict(list)
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row["policy"] != "alpha_ground_pick":
            continue
        phase = round((math.atan2(row["command"][1], row["command"][0]) % (2 * math.pi)) * 2 / math.pi, 2)
        root = np.asarray(row["root_pose"])
        rotation = Rotation.from_quat(root[3:])
        yaw = math.atan2(2 * (root[6] * root[5] + root[3] * root[4]), 1 - 2 * (root[4] ** 2 + root[5] ** 2))
        local = Rotation.from_euler("z", -yaw).apply(np.asarray(row["mouth_position_m"]) - [root[0], root[1], 0])
        gravity = rotation.inv().apply([0, 0, -1])
        poses[phase].append(np.r_[local, gravity, root[2]])
    if len(poses) < 2:
        raise ValueError("The trace must contain a measured pickup cycle.")
    phases = sorted(poses)
    result = {
        "phase_s": phases,
        "reference": [np.mean(poses[phase], axis=0).tolist() for phase in phases],
        "columns": [
            "head_forward_m",
            "head_lateral_m",
            "head_height_m",
            "gravity_x",
            "gravity_y",
            "gravity_z",
            "root_height_m",
        ],
        "source_trace": str(trace.resolve()),
        "source_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "source_policy": "alpha_ground_pick",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/microduck/assets/pickup_reference.json"))
    args = parser.parse_args()
    build_reference(args.trace, args.output)
