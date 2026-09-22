# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foot-site kinematics and gait objectives for MicroDuck."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply


def foot_state(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source foot-site position [m] and linear velocity [m/s]."""
    robot = env.scene["robot"]
    if not hasattr(env, "_microduck_foot_ids"):
        env._microduck_foot_ids = robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)[0]
        env._microduck_contact_ids = env.scene["feet"].find_bodies(["ankle_left", "ankle_right"], preserve_order=True)[
            0
        ]
    ids = env._microduck_foot_ids
    offset = robot.data.body_link_pos_w.torch.new_tensor([0.0, -0.0238146, -0.0140852])
    local = offset.expand(env.num_envs, 2, 3)
    world = quat_apply(robot.data.body_link_quat_w.torch[:, ids], local)
    position = robot.data.body_link_pos_w.torch[:, ids] + world
    velocity = robot.data.body_link_lin_vel_w.torch[:, ids] + torch.cross(
        robot.data.body_link_ang_vel_w.torch[:, ids], world, dim=-1
    )
    return position, velocity


def foot_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return privileged foot heights [m], air times [s], contacts, and forces [N]."""
    position, _ = foot_state(env)
    sensor = env.scene["feet"].data
    ids = env._microduck_contact_ids
    force = sensor.net_forces_w.torch[:, ids]
    return torch.cat(
        (
            position[:, :, 2] - env.scene.env_origins[:, 2:3],
            sensor.current_air_time.torch[:, ids],
            (force.norm(dim=-1) > 0.0).float(),
            force.flatten(1),
        ),
        dim=-1,
    )


def moving(env: ManagerBasedRLEnv) -> torch.Tensor:
    command = env.command_manager.get_command("twist")
    return (command[:, :2].norm(dim=-1) + command[:, 2].abs() > 0.01).float()


def air_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward feet that stay airborne for 0.125–0.300 [s]."""
    foot_state(env)
    duration = env.scene["feet"].data.current_air_time.torch[:, env._microduck_contact_ids]
    return ((duration > 0.125) & (duration < 0.300)).float().sum(-1) * moving(env)


def clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize foot height error [m], weighted by horizontal speed [m/s]."""
    position, velocity = foot_state(env)
    height = position[:, :, 2] - env.scene.env_origins[:, 2:3]
    return ((height - 0.02).abs() * velocity[:, :, :2].norm(dim=-1)).sum(-1) * moving(env)


def slip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize squared horizontal foot speed [m²/s²] while contacting the ground."""
    _, velocity = foot_state(env)
    force = env.scene["feet"].data.net_forces_w.torch[:, env._microduck_contact_ids]
    contact = force.norm(dim=-1) > 0.0
    return (velocity[:, :, :2].square().sum(-1) * contact).sum(-1) * moving(env)


class MicroDuckSwingHeight(ManagerTermBase):
    """Measure maximum swing clearance and score it once on touchdown."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peak = torch.zeros((env.num_envs, 2), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.peak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        position, _ = foot_state(env)
        height = position[:, :, 2] - env.scene.env_origins[:, 2:3]
        sensor = env.scene["feet"]
        ids = env._microduck_contact_ids
        in_air = sensor.data.current_air_time.torch[:, ids] > 0.0
        self.peak[:] = torch.where(in_air, torch.maximum(self.peak, height), self.peak)
        landed = sensor.compute_first_contact(env.step_dt)[:, ids].bool()
        cost = ((self.peak / 0.02 - 1.0).square() * landed).sum(-1) * moving(env)
        self.peak[landed] = 0.0
        return cost
