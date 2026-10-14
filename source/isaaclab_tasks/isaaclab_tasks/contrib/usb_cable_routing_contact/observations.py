# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Shared controller observations without full cable geometry or evaluator flags.

Connector and fixture poses currently emulate an ideal pose estimator using the
simulator. They are not camera measurements. Keep this limitation explicit until
an image-based estimator replaces their source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CONTROLLER_FIELDS = (
    "arm_joint_position_rad",
    "arm_joint_velocity_rad_s",
    "gripper_drive_position_rad",
    "gripper_drive_velocity_rad_s",
    "tcp_position_m",
    "tcp_quat_xyzw",
    "commanded_tcp_position_m",
    "commanded_tcp_quat_xyzw",
    "connector_position_m",
    "connector_quat_xyzw",
    "guide_positions_m",
    "guide_yaw_sin_cos",
    "requested_guide_count",
    "time_remaining_s",
    "last_action",
)
INSERTION_FIELDS = ("socket_position_m", "socket_quat_xyzw")


def controller_measurements(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Return the shared RL/direct inputs, with environment batch as the first axis.

    Position and velocity quantities use [m], [rad], [rad/s], and time uses [s].
    Quaternion order is xyzw. Fixture and connector poses are ideal simulated pose
    estimates, not measured camera outputs. Neither cable interior geometry nor
    evaluator-derived progress, grasp, material, bend, stretch or force data is read.
    """
    command = env.command_manager.get_term("routing")
    robot = command.robot
    arm = env.action_manager.get_term("arm")
    tcp, tcp_quat = arm._compute_frame_pose()
    root = robot.data.root_pos_w.torch - env.scene.env_origins
    tcp = tcp + root
    target = torch.where(arm._target_valid[:, None], arm.processed_actions + root, tcp)
    target_quat = arm.target_quat if hasattr(arm, "target_quat") else command.tool_quat
    target_quat = torch.where(arm._target_valid[:, None], target_quat, tcp_quat)
    drive = robot.joint_names.index("finger_joint")
    connector = command.cable.data.segment_pose_w.torch[:, -1]
    remaining = (env.max_episode_length_s - env.episode_length_buf * env.step_dt).clamp(min=0)
    result = {
        "arm_joint_position_rad": robot.data.joint_pos.torch[:, command.arm_joint_ids],
        "arm_joint_velocity_rad_s": robot.data.joint_vel.torch[:, command.arm_joint_ids],
        "gripper_drive_position_rad": robot.data.joint_pos.torch[:, drive : drive + 1],
        "gripper_drive_velocity_rad_s": robot.data.joint_vel.torch[:, drive : drive + 1],
        "tcp_position_m": tcp,
        "tcp_quat_xyzw": tcp_quat,
        "commanded_tcp_position_m": target,
        "commanded_tcp_quat_xyzw": target_quat,
        "connector_position_m": connector[:, :3] - env.scene.env_origins,
        "connector_quat_xyzw": connector[:, 3:],
        "guide_positions_m": command.guide_positions,
        "guide_yaw_sin_cos": torch.stack((command.guide_yaws.sin(), command.guide_yaws.cos()), dim=-1),
        "requested_guide_count": command.guide_count[:, None].float(),
        "time_remaining_s": remaining[:, None],
        "last_action": env.action_manager.action,
    }
    if hasattr(command, "socket_position"):
        result["socket_position_m"] = command.socket_position
        result["socket_quat_xyzw"] = command.socket_quat
    return result


def contact_policy_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Flatten shared inputs into [N, 55] routing or [N, 65] insertion observations."""
    measurements = controller_measurements(env)
    fields = CONTROLLER_FIELDS + (INSERTION_FIELDS if "socket_position_m" in measurements else ())
    return torch.cat([measurements[name].flatten(1) for name in fields], dim=-1)
