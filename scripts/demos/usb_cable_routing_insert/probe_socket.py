# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe actual plug/socket collision geometry with Newton VBD, independently of routing."""

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from isaaclab_tasks.contrib.usb_cable_routing.assets import SEGMENTS
from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import (
    INSERTION_DEPTH,
    PLUG_FRONT_OFFSET,
    PREINSERT_GAP,
    SOCKET_QUAT,
    _fit_insert_connector,
)


@wp.kernel
def drive_plug(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    com: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.mat33),
    body: int,
    target: wp.vec3,
    target_q: wp.quat,
    dt: float,
    forces: wp.array(dtype=wp.spatial_vector),
):
    q = wp.transform_get_rotation(pose[body])
    target_com = target + wp.quat_rotate(target_q, com[body])
    current_com = wp.transform_point(pose[body], com[body])
    f = 200.0 * (target_com - current_com) - 0.2 * wp.spatial_top(velocity[body])
    error = target_q * wp.quat_inverse(q)
    if error[3] < 0.0:
        error = -error
    axis = wp.vec3(error[0], error[1], error[2])
    angle = 2.0 * wp.atan2(wp.length(axis), wp.max(error[3], 1.0e-8))
    local_error = wp.quat_rotate_inv(q, wp.normalize(axis) * angle)
    omega = wp.quat_rotate_inv(q, wp.spatial_bottom(velocity[body]))
    kp, kd = 0.15, 0.0001
    lhs = inertia[body] + wp.diag(wp.vec3(kd * dt + kp * dt * dt))
    torque = wp.quat_rotate(q, wp.inverse(lhs) * (inertia[body] * (kp * local_error - (kd + kp * dt) * omega)))
    forces[body] = wp.spatial_vector(f, torque)


def build_probe(lateral_offset: float):
    """Build a socket and the identical runtime plug; lateral offset is in [m]."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_shape_cfg.ke = 20000.0
    builder.default_shape_cfg.kd = 20.0
    builder.default_shape_cfg.mu = 0.40
    builder.default_shape_cfg.gap = 0.002
    builder.begin_world()
    socket = (
        Path(__file__).resolve().parents[3]
        / "source/isaaclab_tasks/isaaclab_tasks/contrib/usb_cable_routing_insert/assets/usb_c_socket.usda"
    )
    builder.add_usd(str(socket))
    # The common connector hook expects cable body labels; inactive bodies carry no shapes.
    for i in range(SEGMENTS):
        plug = builder.add_body(
            xform=wp.transform(
                (-PREINSERT_GAP - PLUG_FRONT_OFFSET, lateral_offset, 0.0)
                if i == SEGMENTS - 1
                else (-2.0 - i, 0.0, 0.0),
                wp.quat(*SOCKET_QUAT),
            ),
            mass=0.021 / SEGMENTS,
            inertia=wp.mat33(1e-8, 0, 0, 0, 1e-8, 0, 0, 0, 1e-8),
            is_kinematic=i < SEGMENTS - 1,
            label=f"/Probe/Cable/segment_{i}",
        )
    _fit_insert_connector(builder, 0, None, None)
    builder.end_world()
    builder.color()
    model = builder.finalize(device="cuda:0")
    solver = SolverVBD(model, iterations=12, rigid_body_contact_buffer_size=1024)
    return model, solver, plug


def run_probe(lateral_offset: float = 0.0) -> dict:
    """Drive the physical plug toward 6 mm depth; report actual seating and blocking [m]."""
    model, solver, plug = build_probe(lateral_offset)
    state, next_state = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = 1 / 960
    samples = []
    for i in range(1200):
        desired_depth = -0.008 + 0.014 * min(i / 800, 1.0)
        state.clear_forces()
        wp.launch(
            drive_plug,
            1,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.body_inertia,
                plug,
                wp.vec3(desired_depth - PLUG_FRONT_OFFSET, lateral_offset, 0.0),
                wp.quat(*SOCKET_QUAT),
                dt,
                state.body_f,
            ],
            device=model.device,
        )
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, dt)
        state, next_state = next_state, state
        if i % 32 == 0 or i == 1199:
            pose = state.body_q.numpy()[plug]
            front = wp.transform_point(wp.transform(*pose[:3], *pose[3:]), wp.vec3(0, 0, PLUG_FRONT_OFFSET))
            samples.append([i * dt, *list(front)])
    last = np.array(samples[-1][1:])
    return {
        "controller": "isolated compliant pose drive; no robot policy or routed cable",
        "lateral_target_m": lateral_offset,
        "target_depth_m": INSERTION_DEPTH,
        "final_depth_m": float(last[0]),
        "final_lateral_error_m": float(np.linalg.norm(last[1:])),
        "trajectory_time_xyz": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/usb_cable_routing_insert/socket_probe.json"))
    args = parser.parse_args()
    results = {"aligned": run_probe(), "misaligned": run_probe(0.0015)}
    aligned, blocked = results["aligned"], results["misaligned"]
    assert 0.0055 <= aligned["final_depth_m"] <= 0.0068 and aligned["final_lateral_error_m"] < 0.00012
    assert blocked["final_depth_m"] < 0.001 and blocked["final_lateral_error_m"] > 0.001
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "trajectory_time_xyz"} for k, v in results.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
