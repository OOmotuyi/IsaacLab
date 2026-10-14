# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guide-specific progress for a physically bent cable; no prescribed cable motion."""

import torch
import warp as wp

from isaaclab.managers import CommandTerm

from isaaclab_tasks.contrib.usb_cable_routing.mdp import (
    RoutingAction,
    RoutingCommand,
    cartesian_progress,
    guide_crossings,
    routing_observation,
)

from .env_cfg import GUIDES


def cable_bend_angles(points: torch.Tensor) -> torch.Tensor:
    """Return unsigned angles between neighboring centerline segments [rad], shape [N, S-1]."""
    direction = torch.nn.functional.normalize(points[:, 1:] - points[:, :-1], dim=-1)
    cosine = (direction[:, 1:] * direction[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosine)


class ZigzagRoutingAction(RoutingAction):
    """Cartesian IK with sufficient lateral workspace to approach alternating guides."""

    def process_actions(self, actions: torch.Tensor):
        pos, _ = self._compute_frame_pose()
        previous = torch.where(self._target_valid[:, None], self._processed_actions, pos).clone()
        super().process_actions(actions)
        origin = self._asset.data.root_pos_w.torch - self._env.scene.env_origins
        # Extend only the lateral workspace; preserve the existing IK and other limits.
        y = previous[:, 1] + actions[:, 1].clamp(-1.0, 1.0) * self.cfg.scale
        self._processed_actions[:, 1] = (y + origin[:, 1]).clamp(-0.38, 0.22) - origin[:, 1]
        c = self._env.command_manager.get_term("routing")
        self._ik_controller.set_command(torch.cat((self._processed_actions, c.tool_quat), dim=-1))


class ZigzagRoutingCommand(RoutingCommand):
    """Require ordered crossings and retained bends, with one clearance phase per guide."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.guide_positions[:] = torch.tensor(GUIDES, device=self.device)
        self.lift_targets = torch.tensor(cfg.lift_targets, device=self.device)
        self.seat_targets = torch.tensor(cfg.seat_targets, device=self.device)
        self.metrics["max_bend_angle"] = torch.zeros(self.num_envs, device=self.device)
        self.last_points = torch.zeros_like(self.points())
        self.points_at_last_step = self.points().clone()
        self.peak_bend = torch.zeros(self.num_envs, device=self.device)
        self.last_peak_bend = torch.zeros_like(self.peak_bend)

    def target(self) -> torch.Tensor:
        """Return the current free-end shaping target [m], shape [N, 3]."""
        indices = torch.minimum(self.completed, self.guide_count - 1)
        return torch.where(self.lowering[:, None], self.seat_targets[indices], self.lift_targets[indices])

    def refresh(self):
        if self._last_step == self._env.common_step_counter:
            return
        self._last_step = self._env.common_step_counter
        tip = self.tip()
        self.progress[:] = cartesian_progress(self.previous_tip, tip, self.target())
        self.previous_tip[:] = tip
        self.episode_peaks[:, 0] = torch.maximum(self.episode_peaks[:, 0], self.stretch())
        self.episode_peaks[:, 1] = torch.maximum(self.episode_peaks[:, 1], wp.to_torch(self.cable.tension))
        self.episode_peaks[:, 2] += wp.to_torch(self.cable.collisions)
        points = self.points()
        self.points_at_last_step[:] = points
        self.peak_bend[:] = torch.maximum(self.peak_bend, cable_bend_angles(points).max(dim=1).values)
        ids = torch.arange(self.num_envs, device=self.device)
        index = torch.minimum(self.completed, self.guide_count - 1)
        guide = self.guide_positions[ids, index]
        nearest = (points[..., 0] - guide[:, None, 0]).abs().argmin(dim=1)
        crossing = points[ids, nearest]
        clear = (crossing[:, 2] > 0.060) & ((crossing[:, 1] - guide[:, 1]).abs() < 0.019)
        self.lowering |= clear & ((tip - self.target()).norm(dim=-1) < 0.030)
        self.seated[:] = guide_crossings(points, self.guide_positions)
        current = self.seated[ids, index] & (self.completed < self.guide_count)
        self.dwell[:] = torch.where(current, self.dwell + 1, 0)
        advance = self.dwell >= 5
        self.bonus[:] = advance.float()
        later = torch.arange(3, device=self.device)[None] > self.completed[:, None]
        active = torch.arange(3, device=self.device)[None] < self.guide_count[:, None]
        self.skip |= (self.seated & later & active & ~current[:, None]).any(dim=1)
        self.completed += advance.long()
        self.lowering[advance] = self.completed[advance] >= self.guide_count[advance]
        self.dwell[advance] = 0
        all_retained = (self.seated | ~active).all(dim=1)
        valid = all_retained & (self.completed >= self.guide_count) & ~self.skip
        valid &= wp.to_torch(self.cable.grasp_active).bool() & (self.stretch() < 0.10)
        self.success_dwell[:] = torch.where(valid, self.success_dwell + 1, 0)

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_points[env_ids] = self.points_at_last_step[env_ids]
        self.last_peak_bend[env_ids] = self.peak_bend[env_ids]
        self.last_episode_peaks[env_ids] = self.episode_peaks[env_ids]
        self.last_arm_metrics[env_ids] = self.episode_arm_metrics[env_ids]
        self.last_completed[env_ids] = self.completed[env_ids]
        self.last_seated[env_ids] = self.seated[env_ids]
        if hasattr(self._env, "termination_manager"):
            done = self._env.termination_manager.dones[env_ids]
            # Parallel episodes from the previous stage must not promote the new stage.
            eligible = done & (self.guide_count[env_ids] == self.stage)
            self._episodes += int(eligible.sum())
            self._successes += int((eligible & self.curriculum_success()[env_ids]).sum())
        if self.cfg.curriculum and self._episodes >= self.cfg.curriculum_episodes:
            if self._successes / self._episodes >= self.cfg.curriculum_success_rate and self.stage < 3:
                self.stage += 1
                print(
                    f"[USB zigzag curriculum] advanced to {self.stage} guides "
                    f"after {self._successes}/{self._episodes} current-stage successes",
                    flush=True,
                )
            self._episodes = self._successes = 0
        result = CommandTerm.reset(self, env_ids)
        self.peak_bend[env_ids] = 0
        return result

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["max_bend_angle"][:] = cable_bend_angles(self.points()).max(dim=1).values


def zigzag_observation(env) -> torch.Tensor:
    """Expose cable/robot state and the staggered guide positions [m] to the policy."""
    command = env.command_manager.get_term("routing")
    return torch.cat((routing_observation(env), command.guide_positions.flatten(1)), dim=-1)


def bend_penalty(env) -> torch.Tensor:
    """Penalize sharp changes above 0.65 [rad] between adjacent 15 mm cable segments."""
    command = env.command_manager.get_term("routing")
    return (cable_bend_angles(command.points()) - 0.65).clamp(min=0).sum(dim=1)


def lost_crossings(seated: torch.Tensor, completed: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Count previously completed active guides whose cable crossings have been lost."""
    index = torch.arange(seated.shape[1], device=seated.device)[None]
    earlier = (index < completed[:, None]) & (index < count[:, None])
    return (earlier & ~seated).sum(dim=1).float()


def lost_guide_penalty(env) -> torch.Tensor:
    """Penalize undoing an earlier bend while approaching the next guide."""
    command = env.command_manager.get_term("routing")
    command.refresh()
    return lost_crossings(command.seated, command.completed, command.guide_count)
