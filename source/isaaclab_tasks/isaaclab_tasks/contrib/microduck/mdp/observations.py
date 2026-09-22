# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrated and delayed MicroDuck actor observations."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils.math import quat_apply_inverse


def joint_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read the fourteen encoders relative to home, including calibration bias [rad]."""
    controller = env.action_manager.get_term("joint_pos").controller
    value = env.scene["robot"].data.joint_pos.torch[:, controller.joint_ids] - controller.home
    if hasattr(env, "_microduck_encoder_bias"):
        value = value + env._microduck_encoder_bias
    return value


class MicroDuckDelayedObservation(ManagerTermBase):
    """Keep one previous policy sample, with independent per-environment IMU delays."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.kind = cfg.params["kind"]
        dim = 14 if self.kind == "joint_velocity" else 3
        self.previous = torch.zeros((env.num_envs, dim), device=env.device)
        self.output = self.previous.clone()
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.lag = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = -1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, kind: str, random_delay: bool = True) -> torch.Tensor:
        robot = env.scene["robot"]
        if kind == "joint_velocity":
            ids = env.action_manager.get_term("joint_pos").controller.joint_ids
            value = robot.data.joint_vel.torch[:, ids]
        else:
            value = (
                robot.data.root_ang_vel_b.torch if kind == "angular_velocity" else robot.data.projected_gravity_b.torch
            )
            if hasattr(env, "_microduck_imu_quat"):
                value = quat_apply_inverse(env._microduck_imu_quat, value)
        step = getattr(env, "common_step_counter", 0)
        if step != self.last_step:
            if kind != "joint_velocity":
                if step % 64 == 0:
                    self.lag[:] = torch.rand(env.num_envs, device=env.device) < 0.5 if random_delay else False
            self.output[:] = torch.where(self.lag[:, None], self.previous, value)
            self.previous.copy_(value)
            self.last_step = step
        self.output[self.fresh] = value[self.fresh]
        self.previous[self.fresh] = value[self.fresh]
        self.fresh.zero_()
        return self.output


def linear_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return root-link linear velocity in the robot frame [m/s]."""
    return env.scene["robot"].data.root_link_lin_vel_b.torch
