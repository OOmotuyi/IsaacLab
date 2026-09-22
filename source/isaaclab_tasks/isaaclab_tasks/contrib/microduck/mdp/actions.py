# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched BAM servo actions with randomized supply and communication delay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm

from .controller import MicroDuckController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..microduck_env_cfg import MicroDuckActionCfg


class MicroDuckAction(ActionTerm):
    """Convert policy offsets [rad] into delayed BAM motor efforts [N·m]."""

    def __init__(self, cfg: MicroDuckActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.controller = MicroDuckController(
            self._asset,
            json.loads(Path(cfg.motor_parameters).read_text()),
            training_model=True,
            voltage_limit=max(cfg.battery_range),
        )
        self.controller.battery.uniform_(*cfg.battery_range)
        self.controller.sag.uniform_(*cfg.sag_range)
        self._raw_actions = torch.zeros_like(self.controller.home)
        self._processed_actions = self.controller.home.clone()
        self._history = self.controller.home.unsqueeze(0).repeat(cfg.delay_steps[1] + 1, 1, 1)
        self._cursor = 0
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(self.controller.home + self.cfg.scale * actions)
        self.controller.last_action.copy_(actions)

    def apply_actions(self) -> None:
        self._history[self._cursor] = self._processed_actions
        self.controller.target.copy_(self._history[(self._cursor - self._lag) % len(self._history), self._env_indices])
        self.controller.apply()
        self._cursor = (self._cursor + 1) % len(self._history)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset only selected motor histories and sample per-episode motor parameters."""
        env_ids = slice(None) if env_ids is None else env_ids
        self.controller.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self.controller.home[env_ids]
        self._history[:, env_ids] = self.controller.home[env_ids]
        count = self._raw_actions[env_ids].shape[0]
        self._lag[env_ids] = torch.randint(
            self.cfg.delay_steps[0], self.cfg.delay_steps[1] + 1, (count,), device=self.device
        )
        self.controller.friction_scale[env_ids] = torch.empty((count, 14), device=self.device).uniform_(
            *self.cfg.friction_range
        )
