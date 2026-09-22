# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck command sampling."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CommandTerm, CommandTermCfg


class MicroDuckVelocityCommand(UniformVelocityCommand):
    """Include the upstream 15% turn-in-place command bucket."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        turn = torch.rand(len(env_ids), device=self.device) < self.cfg.turn_in_place_fraction
        turn_ids = env_ids[turn & ~self.is_standing_env[env_ids]]
        self.vel_command_b[turn_ids, :2] = 0.0
        magnitude = torch.empty(len(turn_ids), device=self.device).uniform_(0.4, 1.0)
        sign = 2 * torch.randint(0, 2, (len(turn_ids),), device=self.device) - 1
        self.vel_command_b[turn_ids, 2] = magnitude * sign * self.cfg.ranges.ang_vel_z[1]


class MicroDuckPoseCommand(CommandTerm):
    """Sample independent pose offsets [m or rad, depending on command element]."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, len(cfg.ranges)), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        for index, limits in enumerate(self.cfg.ranges):
            self._command[env_ids, index] = torch.empty(len(env_ids), device=self.device).uniform_(*limits)

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass
