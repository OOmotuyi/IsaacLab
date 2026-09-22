# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-conditioned standing and pickup objectives for native Newton PPO training."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from ..policy import MicroDuckSourceActor
from .actions import MicroDuckAction


class MicroDuckSkillCommand(CommandTerm):
    """Supply the original three-value phase command and a measured pose reference."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.style = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.jaw_target = torch.full((self.num_envs, 1), math.pi / 6, device=self.device)
        self.reference_pose = torch.tensor([0.075, 0, 0.215, 0, 0, -1, 0.116], device=self.device).repeat(
            self.num_envs, 1
        )
        if cfg.skill == "pickup":
            reference = json.loads(Path(cfg.reference_path).read_text())
            self.times = torch.tensor(reference["phase_s"], device=self.device)
            self.poses = torch.tensor(reference["reference"], device=self.device)
            if not bool((self.times[1:] > self.times[:-1]).all()) or self.poses.shape != (len(self.times), 7):
                raise ValueError("Pickup reference requires increasing phase times and seven pose values.")
        self.metrics["head_error_m"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.style[env_ids] = torch.rand(len(env_ids), device=self.device) < 0.5
        self.phase[env_ids] = 0
        self._command[env_ids] = 0
        self.jaw_target[env_ids] = math.pi / 6
        if self.cfg.skill == "pickup":
            self._command[env_ids, 0] = 1
            self.reference_pose[env_ids] = self.poses[0]

    def _update_metrics(self) -> None:
        self.metrics["head_error_m"].copy_((skill_head_position(self._env) - self.reference_pose[:, :3]).norm(dim=-1))

    def _update_command(self) -> None:
        if self.cfg.skill == "stand":
            return
        elapsed = self._env.episode_length_buf * self._env.step_dt
        marker_phase = elapsed.clamp_max(1.4) + (elapsed - 2.4).clamp_min(0)
        sock_phase = torch.where(elapsed < 1.7, elapsed.clamp_max(0.6), 2.0 + 0.5 * (elapsed - 1.7))
        self.phase.copy_(torch.where(self.style, sock_phase, marker_phase).clamp_max(2.8))
        self._command[:, 0] = torch.cos(self.phase * (math.pi / 2))
        self._command[:, 1] = torch.sin(self.phase * (math.pi / 2))
        upper = torch.searchsorted(self.times, self.phase.contiguous()).clamp(1, len(self.times) - 1)
        lower = upper - 1
        blend = ((self.phase - self.times[lower]) / (self.times[upper] - self.times[lower])).clamp(0, 1)
        self.reference_pose.copy_(torch.lerp(self.poses[lower], self.poses[upper], blend[:, None]))
        closed = (elapsed >= torch.where(self.style, 0.7, 1.5)) & (elapsed < 4.5)
        hold = torch.where(self.style, 0.18, -math.pi / 36)
        self.jaw_target[:, 0] = torch.where(closed, hold, math.pi / 6)


class MicroDuckSkillAction(MicroDuckAction):
    """Apply body actions and scripted jaw targets with the cleanup servo contract."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.controller.training_model = False
        self.teacher = MicroDuckSourceActor(cfg.teacher_path).to(self.device).eval()
        self.teacher_action = torch.zeros_like(self._raw_actions)
        self.jaw_ids = self._asset.find_joints("mouth")[0]
        self._env = env

    def process_actions(self, actions: torch.Tensor) -> None:
        commands = self._env.command_manager
        self.controller.command.copy_(
            torch.cat([commands.get_command(name) for name in ("twist", "head_pose", "body_pose")], -1)
        )
        with torch.no_grad():
            self.teacher_action.copy_(self.teacher(self.controller.observations()))
        super().process_actions(actions)

    def apply_actions(self) -> None:
        super().apply_actions()
        self._asset.actuators.target_command.set_position_index(
            value=self._env.command_manager.get_term("twist").jaw_target, joint_ids=self.jaw_ids
        )


def skill_head_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return beak forward/lateral displacement and height above ground [m]."""
    robot = env.scene["robot"]
    if not hasattr(env, "_microduck_skill_head"):
        env._microduck_skill_head = robot.find_bodies("jaw_soft")[0][0]
    index = env._microduck_skill_head
    offset = robot.data.root_link_pos_w.torch.new_tensor([[-0.00809334, 0, -0.0777383]]).expand(env.num_envs, -1)
    point = robot.data.body_link_pos_w.torch[:, index] + quat_apply(robot.data.body_link_quat_w.torch[:, index], offset)
    origin = robot.data.root_link_pos_w.torch.clone()
    origin[:, 2] = env.scene.env_origins[:, 2]
    return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w.torch), point - origin)


def skill_head_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the measured phase-conditioned beak target [m]."""
    target = env.command_manager.get_term("twist").reference_pose[:, :3]
    error = (skill_head_position(env) - target) / target.new_tensor([0.03, 0.02, 0.015])
    return torch.exp(-error.square().mean(-1))


def skill_balance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward phase-conditioned trunk orientation, allowing the pickup lean."""
    target = env.command_manager.get_term("twist").reference_pose[:, 3:6]
    return torch.exp(-(env.scene["robot"].data.projected_gravity_b.torch - target).square().sum(-1) / 0.05)


def skill_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward phase-conditioned trunk height [m]."""
    target = env.command_manager.get_term("twist").reference_pose[:, 6]
    actual = env.scene["robot"].data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((actual - target) / 0.015).square())


def skill_source_prior(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Regularize body actions toward the released actor during PPO fine-tuning."""
    action = env.action_manager.get_term("joint_pos")
    return 1.0 - (action.raw_actions - action.teacher_action).square().mean(-1) / 0.0025


def skill_stationary(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Discourage uncommanded trunk translation [m/s] and yaw motion [rad/s]."""
    data = env.scene["robot"].data
    error = data.root_link_lin_vel_b.torch.square().sum(-1) + 0.1 * data.root_ang_vel_b.torch[:, 2].square()
    return torch.exp(-error / 0.05)
