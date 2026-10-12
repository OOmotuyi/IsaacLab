# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared object-grasp phase and observation contract for training and bedroom playback."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat


def grasp_lift_target(start_height: torch.Tensor, lift_distance: float = 0.05) -> torch.Tensor:
    """Return object-center goals [m] that clear the 0.13 m bedroom bin rim."""
    target = torch.maximum(start_height + lift_distance, start_height.new_tensor(0.20)).clamp_max(0.225)
    return torch.maximum(target, start_height + 0.01)


def grasp_object_heading(object_quaternion: torch.Tensor) -> torch.Tensor:
    """Return the horizontal approach heading [rad] perpendicular to the marker axis."""
    lateral = quat_apply(
        object_quaternion, object_quaternion.new_tensor([[0, 1, 0]]).expand(len(object_quaternion), -1)
    )
    return torch.atan2(-lateral[:, 0], lateral[:, 1])


def grasp_aperture_alignment(
    head_quaternion: torch.Tensor, object_quaternion: torch.Tensor, goal_height: torch.Tensor
) -> torch.Tensor:
    """Score marker-aligned jaw orientation for goals above the floor [m]."""
    heading = grasp_object_heading(object_quaternion)
    yaw = quat_from_euler_xyz(torch.zeros_like(heading), torch.zeros_like(heading), heading)
    forward = quat_apply_inverse(
        yaw, quat_apply(head_quaternion, head_quaternion.new_tensor([[0, 0, -1]]).expand(len(heading), -1))
    )
    upper = quat_apply_inverse(
        yaw, quat_apply(head_quaternion, head_quaternion.new_tensor([[1, 0, 0]]).expand(len(heading), -1))
    )
    pitch = (torch.pi / 2) * ((0.215 - goal_height) / 0.175).clamp(0, 1).sqrt()
    target_forward = torch.stack((pitch.cos(), torch.zeros_like(pitch), -pitch.sin()), -1)
    target_upper = torch.stack((pitch.sin(), torch.zeros_like(pitch), pitch.cos()), -1)
    error = (forward - target_forward).square().sum(-1) + (upper - target_upper).square().sum(-1)
    return torch.exp(-error)


class MicroDuckGraspState:
    """Track approach, contact, and lift from physical poses and velocities."""

    def __init__(
        self,
        count: int,
        device: str | torch.device,
        initial_height: float,
        reach_height: float,
        lift_distance: float = 0.05,
    ) -> None:
        """Allocate per-environment state; all heights and distances use [m]."""
        self.stage = torch.zeros(count, dtype=torch.long, device=device)
        self.start_height = torch.full((count,), initial_height, device=device)
        self.reach_height = torch.full_like(self.start_height, max(initial_height, reach_height))
        self.goal = torch.zeros((count, 3), device=device)
        self.approach_time = torch.zeros(count, device=device)
        self.contact_time = self.approach_time.clone()
        self.settle_time = self.approach_time.clone()
        self.lift_time = self.approach_time.clone()
        self.lost_contact_time = self.approach_time.clone()
        self.lift_goal_height = self.approach_time.clone()
        self.grasp_start_height = self.approach_time.clone()
        self.reach_time = self.approach_time.clone()
        self.success_seen = torch.zeros(count, dtype=torch.bool, device=device)
        self.reach_seen = self.success_seen.clone()
        self.contacts = torch.zeros((count, 2), dtype=torch.bool, device=device)
        self.lift_distance = lift_distance
        self.object_quaternion = torch.zeros((count, 4), device=device)
        self.object_quaternion[:, 3] = 1
        self.pinch_local = self.goal.new_tensor([[-0.016, 0, -0.063]]).expand(count, -1).clone()
        self.grasp_local = self.pinch_local.clone()
        self.object_local_previous = self.pinch_local.clone()
        self.has_local_sample = torch.zeros(count, dtype=torch.bool, device=device)

    def reset_progress(self, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Clear only selected environments' phase, dwell timers [s], and success."""
        for name in (
            "stage",
            "approach_time",
            "contact_time",
            "settle_time",
            "has_local_sample",
            "lift_time",
            "lost_contact_time",
            "lift_goal_height",
            "grasp_start_height",
            "reach_time",
            "success_seen",
            "reach_seen",
        ):
            getattr(self, name)[env_ids] = 0
        self.contacts[env_ids] = False

    def update_progress(
        self,
        head: torch.Tensor,
        object_position: torch.Tensor,
        object_velocity: torch.Tensor,
        origins: torch.Tensor,
        dt: float | torch.Tensor,
        object_quaternion: torch.Tensor | None = None,
    ) -> None:
        """Advance using head pose [m, xyzw], object state [m, m/s], contacts, and time [s]."""
        if object_quaternion is not None:
            self.object_quaternion.copy_(object_quaternion)
        object_local = quat_apply_inverse(head[:, 3:], object_position - head[:, :3])
        step_dt = torch.as_tensor(dt, device=head.device, dtype=head.dtype)
        relative_speed = (object_local - self.object_local_previous).norm(dim=-1) / step_dt.clamp_min(1e-9)
        settled = self.contacts.all(-1) & self.has_local_sample & (relative_speed < 0.003)
        self.settle_time.copy_(torch.where(settled, self.settle_time + dt, 0))
        self.object_local_previous.copy_(object_local)
        self.has_local_sample[:] = True
        self.lift = object_position[:, 2] - origins[:, 2] - self.start_height
        holding = self.stage >= 2
        self.lost_contact_time.copy_(torch.where(holding & ~self.contacts.all(-1), self.lost_contact_time + dt, 0))
        lost = holding & (self.lost_contact_time >= 0.16)
        self.stage[lost] = 0
        self.approach_time[lost] = 0
        reference = torch.where((self.stage >= 2)[:, None], self.grasp_local, self.pinch_local)
        self.pinch = head[:, :3] + quat_apply(head[:, 3:], reference)
        self.distance = (self.pinch - object_position).norm(dim=-1)
        self.goal.copy_(object_position)
        self.goal[:, 2] = torch.maximum(self.goal[:, 2], origins[:, 2] + self.reach_height)
        final_reach = self.reach_height <= self.start_height + 0.001
        # Approach along the beak aperture before closing around the marker.
        exit_axis = quat_apply(head[:, 3:], head.new_tensor([[0, 0, -1]]).expand(len(head), -1))
        self.goal -= ((self.stage == 0) & final_reach)[:, None] * exit_axis * 0.018
        self.goal_distance = (self.goal - self.pinch).norm(dim=-1)
        tolerance = torch.where(self.reach_height > self.start_height + 0.04, 0.020, 0.012)
        orientation_height = torch.where(final_reach, object_position[:, 2], self.goal[:, 2]) - origins[:, 2]
        alignment = grasp_aperture_alignment(head[:, 3:], self.object_quaternion, orientation_height)
        aligned = alignment > torch.where(final_reach, 0.995, 0.9)
        self.reach_time.copy_(torch.where((self.goal_distance < tolerance) & aligned, self.reach_time + dt, 0))
        self.reach_seen |= self.reach_time >= 0.20
        self.approach_time.copy_(
            torch.where((self.goal_distance < 0.012) & final_reach & aligned, self.approach_time + dt, 0)
        )
        self.stage[(self.stage == 1) & (self.distance > 0.035) & ~self.contacts.any(-1)] = 0
        self.stage[(self.stage == 0) & (self.approach_time >= 0.06)] = 1
        self.contact_time.copy_(torch.where(self.contacts.all(-1), self.contact_time + dt, 0))
        acquired = (self.stage <= 1) & (self.contact_time >= 0.10) & (self.settle_time >= 0.10)
        self.lift_goal_height[acquired] = object_position[acquired, 2]
        self.stage[acquired] = 2
        # Measure the settled grip frame once; this changes no object or joint state.
        self.grasp_local[acquired] = object_local[acquired]
        self.grasp_start_height[acquired] = object_position[acquired, 2] - origins[acquired, 2]
        self.pinch[acquired] = object_position[acquired]
        self.distance[acquired] = 0
        # Holding requires real opposed contacts, vertical clearance, and a slow object.
        required_lift = grasp_lift_target(self.start_height, self.lift_distance) - self.start_height
        stable = self.contacts.all(-1) & (self.lift >= required_lift - 0.002)
        stable &= object_velocity.norm(dim=-1) < 0.12
        self.lift_time.copy_(torch.where(stable, self.lift_time + dt, 0))
        self.success_seen |= self.lift_time >= 0.30
        self.stage[self.lift_time >= 0.30] = 3
        self.goal[self.stage == 1] = object_position[self.stage == 1]
        lift_height = grasp_lift_target(self.start_height, self.lift_distance)
        # Advance at 40 mm/s, with at most 10 mm lead over the measured object.
        lifted_goal = torch.minimum(
            torch.maximum(self.lift_goal_height, object_position[:, 2]) + 0.04 * dt,
            torch.minimum(origins[:, 2] + lift_height, object_position[:, 2] + 0.01),
        )
        self.lift_goal_height.copy_(torch.where(self.stage >= 2, lifted_goal, self.lift_goal_height))
        self.goal[:, 2] = torch.where(self.stage >= 2, self.lift_goal_height, self.goal[:, 2])
        self.goal_distance = (self.goal - self.pinch).norm(dim=-1)

    def features(
        self,
        root_quaternion: torch.Tensor,
        object_position: torch.Tensor,
        object_velocity: torch.Tensor,
        jaw_position: torch.Tensor,
        jaw_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Return sixteen features in yaw coordinates with distances [m], angles [rad], and rates [m/s, rad/s]."""
        orientation = yaw_quat(root_quaternion)
        robot_forward = quat_apply(orientation, orientation.new_tensor([[1, 0, 0]]).expand(len(orientation), -1))
        heading_error = grasp_object_heading(self.object_quaternion) - torch.atan2(
            robot_forward[:, 1], robot_forward[:, 0]
        )
        heading_error = torch.atan2(heading_error.sin(), heading_error.cos())
        return torch.cat(
            (
                quat_apply_inverse(orientation, self.goal - self.pinch),
                quat_apply_inverse(orientation, object_position - self.pinch),
                quat_apply_inverse(orientation, object_velocity),
                jaw_position,
                jaw_velocity,
                torch.nn.functional.one_hot(self.stage, 4).float(),
                heading_error[:, None],
            ),
            dim=-1,
        )
