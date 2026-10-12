# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck sensor calibration and perturbation events."""

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis


def sensor_calibration(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Sample encoder biases [rad] and IMU mounting error up to 6 [deg]."""
    if not hasattr(env, "_microduck_encoder_bias"):
        env._microduck_encoder_bias = torch.zeros((env.num_envs, 14), device=env.device)
        env._microduck_imu_quat = torch.zeros((env.num_envs, 4), device=env.device)
        env._microduck_imu_quat[:, 3] = 1.0
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env._microduck_encoder_bias[env_ids] = torch.empty((len(env_ids), 14), device=env.device).uniform_(-0.015, 0.015)
    axis = torch.randn((len(env_ids), 3), device=env.device)
    axis /= axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = torch.empty(len(env_ids), device=env.device).uniform_(0.0, math.radians(6.0))
    env._microduck_imu_quat[env_ids] = quat_from_angle_axis(angle, axis)


def velocity_push(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Add a planar velocity impulse [-0.3, 0.3] [m/s] to the current velocity."""
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    velocity = robot.data.root_link_vel_w.torch[env_ids].clone()
    velocity[:, :2] += torch.empty((len(env_ids), 2), device=env.device).uniform_(-0.3, 0.3)
    robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


class MicroDuckCenterOfMass(ManagerTermBase):
    """Sample independent link COM offsets [m] from the unrandomized asset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.params["asset_cfg"].name]
        self.default = self.robot.data.body_com_pos_b.torch.clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        ranges: tuple[float, float, float],
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        body_ids = asset_cfg.body_ids
        if isinstance(body_ids, slice):
            body_ids = torch.arange(self.robot.num_bodies, device=env.device)[body_ids]
        else:
            body_ids = torch.tensor(body_ids, device=env.device)
        offsets = torch.empty((len(env_ids), len(body_ids), 3), device=env.device).uniform_(-1.0, 1.0)
        offsets *= offsets.new_tensor(ranges)
        self.robot.set_coms_index(
            coms=self.default[env_ids[:, None], body_ids] + offsets, env_ids=env_ids, body_ids=body_ids
        )
