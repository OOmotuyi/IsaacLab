# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check cable-length feasibility and robot clearance at reset and at the insertion endpoint.

This is a geometric endpoint check, not a collision-free trajectory or a routing demonstration.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from isaaclab_tasks.contrib.usb_cable_routing.assets import ANCHOR, CABLE_LENGTH, SEGMENT_LENGTH, SEGMENTS
from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import (
    GRASP_LOCAL_Z,
    INSERTION_DEPTH,
    PLUG_FRONT_OFFSET,
    SOCKET_FACE,
)
from isaaclab_tasks.contrib.usb_cable_routing_insert.env_cfg import UsbInsertEnvCfg
from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import select_socket_orientation
from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import GUIDE_YAWS, GUIDES
from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import cable_bend_radii, guide_crossings
from isaaclab_tasks.contrib.usb_cable_routing_scurve.parameters import MIN_BEND_RADIUS


def cable_length_check() -> dict:
    """Measure a smooth centerline through the guide passages to the seated cable tip [m]."""
    pieces = [np.array([ANCHOR, np.array(ANCHOR) + [SEGMENT_LENGTH, 0, 0]])]
    previous, axis = pieces[-1][-1], np.array([1.0, 0, 0])
    t = np.linspace(0, 1, 600)[:, None]

    def span(start, end, first_axis, last_axis, handle):
        return (
            (1 - t) ** 3 * start
            + 3 * t * (1 - t) ** 2 * (start + handle * first_axis)
            + 3 * t * t * (1 - t) * (end - handle * last_axis)
            + t**3 * end
        )

    for guide, yaw in zip(GUIDES, GUIDE_YAWS):
        center = np.array(guide) + [0, 0, 0.014]
        following = np.array([np.cos(yaw), np.sin(yaw), 0])
        entry, end = center - 0.015 * following, center + 0.015 * following
        pieces.append(span(previous, entry, axis, following, np.linalg.norm(entry - previous) * 0.20))
        pieces.append(np.array([entry, end]))
        previous, axis = end, following
    end = np.array(SOCKET_FACE) + [INSERTION_DEPTH - PLUG_FRONT_OFFSET + SEGMENT_LENGTH / 2, 0, 0]
    pieces.append(span(previous, end, axis, np.array([1.0, 0, 0]), 0.030))
    path = np.concatenate(pieces)
    arc = np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    samples = np.stack([np.interp(np.linspace(0, arc[-1], SEGMENTS + 1), arc, path[:, i]) for i in range(3)], axis=-1)
    points = torch.tensor(samples)[None]
    radius = float(cable_bend_radii(points).min())
    crossings = guide_crossings(points, torch.tensor([GUIDES]), torch.tensor([GUIDE_YAWS]))
    assert arc[-1] < CABLE_LENGTH and radius >= MIN_BEND_RADIUS and crossings.all()
    return {
        "smooth_path_length_m": float(arc[-1]),
        "remaining_length_m": float(CABLE_LENGTH - arc[-1]),
        "minimum_discrete_bend_radius_m": radius,
        "all_guide_passages_crossed": bool(crossings.all()),
        "note": "Geometric length feasibility; not a settled VBD cable or a routing rollout.",
    }


def robot_clearance(env: ManagerBasedRLEnv, *, cable: bool = False, penetration_tolerance: float = 1e-6) -> dict:
    """Check robot contact separation [m] against fixtures or cable, with penetration tolerance [m]."""
    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    command = env.command_manager.get_term("routing")
    kinds = command.cable._shape_kind.numpy()
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = (contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy())
    points = (contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy())
    margins = (contacts.rigid_contact_margin0.numpy(), contacts.rigid_contact_margin1.numpy())
    normals = contacts.rigid_contact_normal.numpy()
    bodies, poses = model.shape_body.numpy(), state.body_q.numpy()
    cable_bodies = set(command.cable.body_ids.cpu().flatten().tolist())
    pairs = []
    for i in range(count):
        first, second = shapes[0][i], shapes[1][i]
        if cable:
            selected = (kinds[first] == 1 and bodies[second] in cable_bodies) or (
                kinds[second] == 1 and bodies[first] in cable_bodies
            )
        else:
            selected = {int(kinds[first]), int(kinds[second])} == {1, 2}
        if not selected:
            continue
        world_points = []
        for side, shape in enumerate((first, second)):
            body = bodies[shape]
            point = points[side][i]
            world_points.append(
                np.array(wp.transform_point(wp.transform(*poses[body]), wp.vec3(point))) if body >= 0 else point
            )
        separation = float(np.dot(world_points[1] - world_points[0], normals[i]) - margins[0][i] - margins[1][i])
        pairs.append((model.shape_label[first], model.shape_label[second], separation))
    penetrations = [pair for pair in pairs if pair[2] < -penetration_tolerance]
    assert not penetrations, penetrations
    return {
        "penetrating_pairs": len(penetrations),
        "maximum_penetration_m": max((max(0.0, -p[2]) for p in pairs), default=0.0),
        "minimum_reported_separation_m": min((p[2] for p in pairs), default=None),
        "near_contact_pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp_steps", type=int, default=120, help="Zero-action grasp validation steps.")
    parser.add_argument("--output", type=Path, default=Path("logs/usb_cable_routing_insert/reachability.json"))
    args = parser.parse_args()
    result = {"cable_geometry": cable_length_check(), "kinematic_endpoint_check_only": True}
    cfg = UsbInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.routing.randomize = False
    cfg.seed = 43
    with launch_simulation(cfg):
        env = gym.make("IsaacContrib-Route-USB-SCurve-Insert-UR10e", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            limits = robot.data.joint_pos_limits.torch
            joints = robot.data.joint_pos.torch
            assert ((joints >= limits[..., 0] - 1e-5) & (joints <= limits[..., 1] + 1e-5)).all()
            result["reset_joint_limits_satisfied"] = True
            result["reset_clearance"] = robot_clearance(env)
            result["reset_cable_clearance"] = robot_clearance(env, cable=True)
            peak_penetration = 0.0
            for _ in range(args.grasp_steps):
                obs, reward, done, timeout, _ = env.step(torch.zeros(1, 6, device=env.device))
                assert not (done | timeout).any()
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                clearance = robot_clearance(env, cable=True, penetration_tolerance=0.0002)
                peak_penetration = max(peak_penetration, clearance["maximum_penetration_m"])
            result["grasp_validation"] = {
                "steps": args.grasp_steps,
                "maximum_cable_robot_penetration_m": peak_penetration,
                "penetration_tolerance_m": 0.0002,
            }
            command = env.command_manager.get_term("routing")
            action = env.action_manager.get_term("arm")
            robot = command.robot
            pose = wp.to_torch(NewtonManager.get_state_0().body_q)
            wrist = pose[wp.to_torch(command.cable._wrist_ids).long(), 3:]
            plug = command.plug_pose()[1]
            axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
            result["grasp_validation"]["wrist_plug_axis_dot"] = float(
                (quat_apply(wrist, axis) * quat_apply(plug, axis)).sum()
            )
            target = command.socket_position.clone()
            target[:, 0] += INSERTION_DEPTH - PLUG_FRONT_OFFSET + GRASP_LOCAL_Z
            origin = robot.data.root_pos_w.torch - env.scene.env_origins
            flip = select_socket_orientation(quat_mul(quat_inv(command.socket_quat), plug), wrist, command.socket_quat)
            goal_quat = quat_mul(
                quat_mul(command.socket_quat, flip), quat_inv(wp.to_torch(command.cable.grasp_relative_quat))
            )
            # Continue from the same measured wrist-above-hand branch used at reset.
            joints = command.home_q.clone()
            robot.write_joint_position_to_sim_index(position=joints)
            action._ik_controller.set_command(torch.cat((target - origin, goal_quat), dim=-1))
            for _ in range(120):
                env.sim.forward()
                robot.update(0)
                position, orientation = action._compute_frame_pose()
                joints = robot.data.joint_pos.torch.clone()
                next_joints = action._ik_controller.compute(
                    position, orientation, action._compute_frame_jacobian(), joints[:, action._joint_ids]
                )
                joints[:, action._joint_ids] += (next_joints - joints[:, action._joint_ids]).clamp(-0.10, 0.10)
                robot.write_joint_position_to_sim_index(position=joints)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints))
            env.sim.forward()
            robot.update(0)
            position, orientation = action._compute_frame_pose()
            error = float((position + origin - target).norm())
            angle = float(2 * torch.acos((orientation * goal_quat).sum(-1).abs().clamp(max=1)))
            assert error < 0.001 and angle < np.deg2rad(2)
            assert ((joints >= limits[..., 0] - 1e-5) & (joints <= limits[..., 1] + 1e-5)).all()
            result["insertion_joint_limits_satisfied"] = True
            result.update(
                position_error_m=error,
                orientation_error_rad=angle,
                joint_positions_rad=joints.cpu().tolist(),
                insertion_clearance=robot_clearance(env),
            )
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Cable length, reset clearance and insertion endpoint checks passed: {args.output}")


if __name__ == "__main__":
    main()
