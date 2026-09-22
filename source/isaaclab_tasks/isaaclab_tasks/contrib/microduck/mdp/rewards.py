# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck velocity objectives, preserving the upstream error definitions."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def track_velocity(env: ManagerBasedRLEnv, angular: bool = False, variance: float = 0.1) -> torch.Tensor:
    """Reward commanded velocity, including uncommanded vertical/roll/pitch motion."""
    robot = env.scene["robot"]
    command = env.command_manager.get_command("twist")
    if angular:
        actual = robot.data.root_ang_vel_b.torch
        error = actual[:, :2].square().sum(-1) + (actual[:, 2] - command[:, 2]).square()
    else:
        actual = robot.data.root_link_lin_vel_b.torch
        error = (actual[:, :2] - command[:, :2]).square().sum(-1) + actual[:, 2].square()
    return torch.exp(-error / variance)


def upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward gravity alignment using the source variance of 0.05."""
    return torch.exp(-env.scene["robot"].data.projected_gravity_b.torch[:, :2].square().sum(-1) / 0.05)


def posture(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward leg pose with separate standing/walking joint tolerances [rad]."""
    controller = env.action_manager.get_term("joint_pos").controller
    error = env.scene["robot"].data.joint_pos.torch[:, controller.joint_ids] - controller.home
    command = env.command_manager.get_command("twist")
    standing = command[:, :2].norm(dim=-1) + command[:, 2].abs() < 0.01
    stand_std = error.new_tensor([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
    walk_std = error.new_tensor([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
    std = torch.where(standing[:, None], stand_std, walk_std)
    error = error[:, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]]
    return torch.exp(-(error / std).square().mean(-1))


def head_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward each head joint independently with tolerance 0.5 [rad]."""
    controller = env.action_manager.get_term("joint_pos").controller
    q = env.scene["robot"].data.joint_pos.torch[:, controller.joint_ids]
    error = q[:, 5:9] - controller.home[:, 5:9] - env.command_manager.get_command("head_pose")
    return torch.exp(-error.square() / 0.25).mean(-1)


def invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate nonfinite articulation states before the next policy evaluation."""
    robot = env.scene["robot"]
    return ~(
        torch.isfinite(robot.data.joint_pos.torch).all(-1)
        & torch.isfinite(robot.data.joint_vel.torch).all(-1)
        & torch.isfinite(robot.data.root_state_w.torch).all(-1)
    )


class MicroDuckHeadBias(ManagerTermBase):
    """Penalize persistent head tracking error with a one-second moving average."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.error = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.error[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        controller = env.action_manager.get_term("joint_pos").controller
        measured = env.scene["robot"].data.joint_pos.torch[:, controller.joint_ids][:, 5:9]
        error = measured - controller.home[:, 5:9] - env.command_manager.get_command("head_pose")
        self.error.lerp_(error, min(1.0, env.step_dt))
        return -self.error.abs().mean(-1)


def body_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize trunk roll/pitch angular velocity [rad/s] in the world frame."""
    return env.scene["robot"].data.root_ang_vel_w.torch[:, :2].square().sum(-1)


def angular_momentum(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize whole-robot angular momentum about its center of mass [kg m²/s]."""
    data = env.scene["robot"].data
    mass = data.body_mass.torch[..., None]
    position = data.body_com_pos_w.torch
    velocity = data.body_com_lin_vel_w.torch
    center = (mass * position).sum(1, keepdim=True) / mass.sum(1, keepdim=True)
    orbital = torch.cross(position - center, mass * velocity, dim=-1)
    # Newton stores inertia about each COM in the link frame.
    quat = data.body_link_quat_w.torch
    omega = quat_apply_inverse(quat, data.body_com_ang_vel_w.torch)
    inertia = data.body_inertia.torch.reshape(*mass.shape[:2], 3, 3)
    spin = quat_apply(quat, (inertia @ omega[..., None]).squeeze(-1))
    return (orbital + spin).sum(1).square().sum(-1)


def self_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize any robot/robot contact, matching the source's single contact slot."""
    return env.action_manager.get_term("joint_pos").controller.self_collision()
