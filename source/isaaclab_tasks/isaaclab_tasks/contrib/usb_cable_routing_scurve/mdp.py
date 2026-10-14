# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crossings and collisions expressed in each upright guide's local passage frame."""

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import ModelFlags

from isaaclab_tasks.contrib.usb_cable_routing.assets import CABLE_DIAMETER
from isaaclab_tasks.contrib.usb_cable_routing.mdp import cartesian_progress, routing_observation
from isaaclab_tasks.contrib.usb_cable_routing_zigzag.mdp import ZigzagRoutingCommand, cable_bend_angles

from .assets import (
    CHANNEL_WIDTH,
    CRADLE_HEIGHT,
    ENTRY_WIDTH,
    GUIDE_YAWS,
    GUIDES,
    LIP_UNDERSIDE,
    PASSAGE_LENGTH,
    WALL_TOP,
)
from .parameters import DYNAMIC_FRICTION, MIN_BEND_RADIUS


def cable_bend_radii(points: torch.Tensor) -> torch.Tensor:
    """Estimate local centerline radii [m], shape [N, S-1], from adjacent chord directions.

    Uses mean neighboring chord length divided by twice the half-angle sine. This is
    exact for equal chords on a circle. Straight joints have infinite radius;
    degenerate chords have zero radius and therefore fail the task limit.
    """
    chord = points[:, 1:] - points[:, :-1]
    length = chord.norm(dim=-1)
    direction = chord / length.clamp(min=1e-12)[..., None]
    twice_sine = (direction[:, 1:] - direction[:, :-1]).norm(dim=-1)
    radius = 0.5 * (length[:, 1:] + length[:, :-1]) / twice_sine.clamp(min=1e-12)
    radius = torch.where(twice_sine > 1e-7, radius, torch.inf)
    return torch.where((length[:, 1:] > 1e-9) & (length[:, :-1] > 1e-9), radius, 0.0)


def guide_local_points(points: torch.Tensor, guides: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Transform centerlines [m] into guide-local coordinates, shape [N, G, P, 3]."""
    delta = points[:, None] - guides[:, :, None]
    cosine, sine = yaws.cos()[..., None], yaws.sin()[..., None]
    x = cosine * delta[..., 0] + sine * delta[..., 1]
    y = -sine * delta[..., 0] + cosine * delta[..., 1]
    return torch.stack((x, y, delta[..., 2]), dim=-1)


def guide_crossings(points: torch.Tensor, guides: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Require a radius-aware cable crossing at both rotated guide faces [m, rad]."""
    local = guide_local_points(points, guides, yaws)
    a, b = local[:, :, :-1], local[:, :, 1:]
    difference = b - a
    axial = difference[..., 0]
    seated = torch.ones(points.shape[0], guides.shape[1], dtype=torch.bool, device=points.device)
    for face in (-PASSAGE_LENGTH / 2, PASSAGE_LENGTH / 2):
        t = (face - a[..., 0]) / torch.where(axial.abs() > 1e-8, axial, torch.ones_like(axial))
        crossing = a + t[..., None] * difference
        inside = (t >= 0) & (t <= 1) & (axial.abs() > 1e-8)
        inside &= crossing[..., 1].abs() < (CHANNEL_WIDTH - CABLE_DIAMETER) / 2
        inside &= crossing[..., 2] >= CRADLE_HEIGHT + CABLE_DIAMETER / 2 - 0.0005
        inside &= crossing[..., 2] <= LIP_UNDERSIDE - CABLE_DIAMETER / 2
        seated &= inside.any(dim=-1)
    return seated


def cable_guide_distance(points: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return shortest distance [m] from each cable polyline [N, P, 3] to a guide target [N, 3]."""
    start, direction = points[:, :-1], points[:, 1:] - points[:, :-1]
    t = ((goal[:, None] - start) * direction).sum(-1) / direction.square().sum(-1).clamp(min=1e-12)
    nearest = start + t.clamp(0.0, 1.0)[..., None] * direction
    return (nearest - goal[:, None]).norm(dim=-1).amin(dim=1)


class SCurveRoutingCommand(ZigzagRoutingCommand):
    """Retain the existing ordered curriculum while using rotated channel geometry."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.guide_positions[:] = torch.tensor(GUIDES, device=self.device)
        self.guide_yaws = torch.tensor(GUIDE_YAWS, device=self.device).repeat(self.num_envs, 1)
        self.cable_progress = torch.zeros(self.num_envs, device=self.device)
        self.minimum_bend_radius = cable_bend_radii(self.points()).amin(dim=1)
        self.episode_minimum_bend_radius = self.minimum_bend_radius.clone()
        self.last_minimum_bend_radius = self.minimum_bend_radius.clone()
        self.metrics["minimum_bend_radius_m"] = self.minimum_bend_radius.clone()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        # Keep the deterministic nominal contact value explicit for this task.
        if not self.cfg.randomize:
            self.material[env_ids, 1] = DYNAMIC_FRICTION
            model = NewtonManager.get_model()
            for env_id in env_ids.tolist():
                wp.to_torch(model.shape_material_mu)[self.cable_shapes[env_id]] = DYNAMIC_FRICTION
            NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        self.minimum_bend_radius[env_ids] = cable_bend_radii(self.points()[env_ids]).amin(dim=1)
        self.episode_minimum_bend_radius[env_ids] = self.minimum_bend_radius[env_ids]
        self.points_at_last_step[env_ids] = self.points()[env_ids]
        self.cable_progress[env_ids] = 0

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_minimum_bend_radius[env_ids] = self.episode_minimum_bend_radius[env_ids]
        return super().reset(env_ids)

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["minimum_bend_radius_m"][:] = self.minimum_bend_radius

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
        ids = torch.arange(self.num_envs, device=self.device)
        index = torch.minimum(self.completed, self.guide_count - 1)
        goal = self.guide_positions[ids, index].clone()
        goal[:, 2] += torch.where(self.lowering, CRADLE_HEIGHT + CABLE_DIAMETER / 2 + 0.005, WALL_TOP + 0.012)
        # Evaluate both states against the same goal so phase switches cannot earn progress.
        self.cable_progress[:] = cable_guide_distance(self.points_at_last_step, goal) - cable_guide_distance(
            points, goal
        )
        self.points_at_last_step[:] = points
        self.minimum_bend_radius[:] = cable_bend_radii(points).amin(dim=1)
        self.episode_minimum_bend_radius[:] = torch.minimum(self.episode_minimum_bend_radius, self.minimum_bend_radius)
        self.peak_bend[:] = torch.maximum(self.peak_bend, cable_bend_angles(points).max(dim=1).values)
        ids = torch.arange(self.num_envs, device=self.device)
        index = torch.minimum(self.completed, self.guide_count - 1)
        local = guide_local_points(points, self.guide_positions, self.guide_yaws)[ids, index]
        crossing = local[ids, local[..., 0].abs().argmin(dim=1)]
        clear = (crossing[:, 2] > WALL_TOP + 0.006) & (crossing[:, 1].abs() < (ENTRY_WIDTH - CABLE_DIAMETER) / 2)
        self.lowering |= clear
        self.seated[:] = guide_crossings(points, self.guide_positions, self.guide_yaws)
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
        valid = (self.seated | ~active).all(dim=1) & (self.completed >= self.guide_count) & ~self.skip
        valid &= wp.to_torch(self.cable.grasp_active).bool() & (self.stretch() < 0.10)
        valid &= self.minimum_bend_radius >= MIN_BEND_RADIUS - 1e-7
        self.success_dwell[:] = torch.where(valid, self.success_dwell + 1, 0)


def scurve_observation(env) -> torch.Tensor:
    """Expose cable state and guide positions [m] and yaw sine/cosine to PPO."""
    command = env.command_manager.get_term("routing")
    return torch.cat(
        (
            routing_observation(env),
            command.guide_positions.flatten(1),
            command.guide_yaws.sin(),
            command.guide_yaws.cos(),
        ),
        dim=-1,
    )


def collision_penalty(env) -> torch.Tensor:
    """Penalize measured contacts and tool incursions into the rotated upright walls."""
    command = env.command_manager.get_term("routing")
    command.refresh()
    pos, _ = env.action_manager.get_term("arm")._compute_frame_pose()
    tcp = pos + command.robot.data.root_pos_w.torch - env.scene.env_origins
    local = guide_local_points(tcp[:, None], command.guide_positions, command.guide_yaws).squeeze(2)
    active = torch.arange(3, device=env.device)[None] < command.guide_count[:, None]
    wall = (local[..., 0].abs() < PASSAGE_LENGTH / 2 + 0.012) & (local[..., 1].abs() > CHANNEL_WIDTH / 2 - 0.007)
    wall &= (local[..., 1].abs() < CHANNEL_WIDTH / 2 + 0.018) & (local[..., 2] < WALL_TOP + 0.015) & active
    return torch.maximum(
        wp.to_torch(command.cable.collisions).float(), wall.any(dim=1).float() + (tcp[:, 2] < 0.022).float()
    )


def cable_overbent(env) -> torch.Tensor:
    """Reject bends below the 15 mm task limit, with 0.1 micrometer numerical tolerance."""
    command = env.command_manager.get_term("routing")
    command.refresh()
    return command.minimum_bend_radius < MIN_BEND_RADIUS - 1e-7


def bend_penalty(env) -> torch.Tensor:
    """Penalize curvature within 25 percent of the minimum allowed radius [m]."""
    command = env.command_manager.get_term("routing")
    radius = cable_bend_radii(command.points())
    return (1.25 * MIN_BEND_RADIUS / radius.clamp(min=1e-6) - 1.0).clamp(min=0).sum(dim=1)


def guide_approach_reward(env) -> torch.Tensor:
    """Reward cable approach to the active opening [m/s], retaining socket progress after routing."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    progress = torch.where(c.completed < c.guide_count, 5.0 * c.cable_progress + 0.1 * c.progress, c.progress)
    return progress / env.step_dt
