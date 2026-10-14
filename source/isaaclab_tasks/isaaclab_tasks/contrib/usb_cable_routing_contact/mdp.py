# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-grasp reset, guide randomization and measured socket-load success."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.usb_cable_routing.assets import ANCHOR, CABLE_LENGTH, SEGMENT_LENGTH
from isaaclab_tasks.contrib.usb_cable_routing_insert.assets import (
    INSERTION_DEPTH,
    PLUG_FRONT_OFFSET,
    PREINSERT_GAP,
)
from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import (
    InsertionCommand,
    insertion_errors,
    insertion_seated,
)
from isaaclab_tasks.contrib.usb_cable_routing_scurve.assets import GUIDE_YAWS, GUIDES
from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import SCurveRoutingCommand
from isaaclab_tasks.contrib.usb_cable_routing_scurve.parameters import MIN_BEND_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def transformed_guide_targets(targets: torch.Tensor, positions: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Transform nominal targets [m] with guide translations [m] and yaw [rad].

    Args:
        targets: Nominal shaping targets, shape [3, 3].
        positions: Sampled guide centers, shape [N, 3, 3].
        yaws: Sampled guide yaw angles, shape [N, 3].

    Returns:
        Shaping targets in environment coordinates [m], shape [N, 3, 3].
    """
    nominal = torch.tensor(GUIDES, dtype=positions.dtype, device=positions.device)
    angle = yaws - torch.tensor(GUIDE_YAWS, dtype=yaws.dtype, device=yaws.device)
    offset = targets - nominal
    result = positions.clone()
    result[..., 0] += angle.cos() * offset[:, 0] - angle.sin() * offset[:, 1]
    result[..., 1] += angle.sin() * offset[:, 0] + angle.cos() * offset[:, 1]
    result[..., 2] += offset[:, 2]
    return result


def randomize_guide_poses(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Sample horizontal guide positions [m] and yaw [rad] at reset, preserving slack.

    Upright orientation and cradle height are fixed. Infeasible guide-center
    polylines are rejected before any controller acts; both controllers use the
    same seeded sampling procedure.
    """
    command = env.command_manager.get_term("routing")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    count = len(env_ids)
    nominal = torch.tensor(GUIDES, device=env.device)
    nominal_yaw = torch.tensor(GUIDE_YAWS, device=env.device)
    positions = nominal.expand(count, -1, -1).clone()
    yaws = nominal_yaw.expand(count, -1).clone()
    pending = torch.arange(count, device=env.device)
    for _ in range(32):
        if command.cfg.randomize_guides:
            positions[pending, :, :2] = (
                nominal[None, :, :2]
                + (2.0 * torch.rand(len(pending), 3, 2, device=env.device) - 1.0) * command.cfg.guide_xy_jitter
            )
            yaws[pending] = (
                nominal_yaw
                + (2.0 * torch.rand(len(pending), 3, device=env.device) - 1.0) * command.cfg.guide_yaw_jitter
            )
        seat = transformed_guide_targets(command.seat_targets, positions[pending], yaws[pending])
        guide_centers = positions[pending].clone()
        guide_centers[..., 2] += 0.012
        anchor = torch.tensor(ANCHOR, device=env.device).expand(len(pending), 1, 3)
        route = torch.cat((anchor, guide_centers, seat[:, -1:]), dim=1)
        length = (route[:, 1:] - route[:, :-1]).norm(dim=-1).sum(dim=-1)
        if hasattr(command, "socket_position"):
            ids = env_ids[pending]
            socket_offset = torch.zeros(len(pending), 3, device=env.device)
            socket_offset[:, 2] = INSERTION_DEPTH - (PLUG_FRONT_OFFSET - SEGMENT_LENGTH / 2)
            socket_tail = command.socket_position[ids] + quat_apply(command.socket_quat[ids], socket_offset)
            inserted = torch.cat((anchor, guide_centers, socket_tail[:, None]), dim=1)
            inserted_length = (inserted[:, 1:] - inserted[:, :-1]).norm(dim=-1).sum(dim=-1)
            length = torch.maximum(length, inserted_length)
        valid = length <= CABLE_LENGTH - command.cfg.minimum_layout_slack
        valid &= (positions[pending, 1:, 0] - positions[pending, :-1, 0]).amin(dim=-1) > 0.10
        pending = pending[~valid]
        if len(pending) == 0:
            break
        if not command.cfg.randomize_guides:
            raise ValueError("The nominal contact-task layout does not satisfy its slack limit.")
    if len(pending):
        raise ValueError("Unable to sample guide poses satisfying the cable slack limit.")
    command.guide_positions[env_ids] = positions
    command.guide_yaws[env_ids] = yaws
    command.contact_lift_targets[env_ids] = transformed_guide_targets(command.lift_targets, positions, yaws)
    command.contact_seat_targets[env_ids] = transformed_guide_targets(command.seat_targets, positions, yaws)


def socket_force_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize measured connector/socket contact force above 2 N after routing."""
    command = env.command_manager.get_term("routing")
    command.refresh()
    force = wp.to_torch(command.cable.socket_contact_force)
    return (command.completed >= command.guide_count).float() * (force - 2.0).clamp(min=0)


class _ContactCommandMixin:
    """Shared physical closure and per-environment shaping targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contact_lift_targets = self.lift_targets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.contact_seat_targets = self.seat_targets.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.gripper_joint_ids = [
            i for i, name in enumerate(self.robot.joint_names) if "finger" in name or "knuckle" in name
        ]
        default = self.robot.data.default_joint_pos.torch[:, self.gripper_joint_ids]
        self._gripper_sign = torch.where(default < 0, -1.0, 1.0)

    def routing_target(self) -> torch.Tensor:
        """Return the current randomized guide shaping target [m], shape [N, 3]."""
        index = torch.minimum(self.completed, self.guide_count - 1)
        ids = torch.arange(self.num_envs, device=self.device)
        return torch.where(
            self.lowering[:, None], self.contact_seat_targets[ids, index], self.contact_lift_targets[ids, index]
        )

    def _solve_home(self) -> None:
        super()._solve_home()
        initial_positions = self.cfg.gripper_initial_joint_positions
        if initial_positions is None:
            self.home_q[:, self.gripper_joint_ids] = self._gripper_sign * self.cfg.gripper_initial_position
            return
        names = [self.robot.joint_names[index] for index in self.gripper_joint_ids]
        if set(initial_positions) != set(names):
            raise ValueError("The pre-held reset requires one initial position for every gripper joint.")
        positions = torch.tensor([initial_positions[name] for name in names], device=self.device)
        if not torch.isfinite(positions).all():
            raise ValueError("Initial gripper positions must be finite.")
        self.home_q[:, self.gripper_joint_ids] = positions

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        closing = self._gripper_sign[env_ids] * self.cfg.gripper_closed_position
        self.robot.set_joint_position_target_index(target=closing, joint_ids=self.gripper_joint_ids, env_ids=env_ids)
        for index in range(3):
            guide = self._env.scene[f"guide{index + 1}"]
            pose = guide.data.default_root_state.torch[env_ids, :7].clone()
            pose[:, :3] = self.guide_positions[env_ids, index] + self._env.scene.env_origins[env_ids]
            if index >= self.stage:
                pose[:, 2] -= 1.0
            half_yaw = self.guide_yaws[env_ids, index] / 2
            pose[:, 3:5] = 0
            pose[:, 5], pose[:, 6] = half_yaw.sin(), half_yaw.cos()
            guide.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        # Joint-state writes must reach body poses before recording the physical grasp frame.
        self._env.sim.forward()
        self.robot.update(0)
        self.episode_arm_metrics[env_ids] = 0
        self.episode_arm_metrics[env_ids, 0] = torch.inf
        self.episode_arm_metrics[env_ids, 3] = torch.inf
        self.update_arm_metrics()
        self.cable.reset_contact_grasp(env_ids)
        action = self._env.action_manager.get_term("arm")
        if hasattr(action, "target_quat"):
            _, wrist = action._compute_frame_pose()
            action.target_quat[env_ids] = wrist[env_ids]


class ContactSCurveCommand(_ContactCommandMixin, SCurveRoutingCommand):
    """S-curve routing with verified physical acquisition and retention."""

    def target(self) -> torch.Tensor:
        if not hasattr(self, "contact_lift_targets"):
            return super().target()
        return self.routing_target()

    def refresh(self) -> None:
        super().refresh()
        acquired = wp.to_torch(self.cable.grasp_acquired).bool()
        self.success_dwell[:] = torch.where(acquired, self.success_dwell, 0)


class ContactInsertCommand(_ContactCommandMixin, InsertionCommand):
    """Physical-grasp insertion with separate socket and gripping force checks."""

    def target(self) -> torch.Tensor:
        if not hasattr(self, "contact_lift_targets"):
            return super().target()
        goal = self.routing_target()
        if not hasattr(self, "socket_position"):
            return goal
        depth = torch.where(self.inserting, INSERTION_DEPTH, -PREINSERT_GAP)
        offset = torch.zeros_like(goal)
        offset[:, 2] = depth - (PLUG_FRONT_OFFSET - SEGMENT_LENGTH / 2)
        insertion_goal = self.socket_position + quat_apply(self.socket_quat, offset)
        staged = torch.where(self.ready_to_approach[:, None], insertion_goal, self.contact_seat_targets[:, -1])
        return torch.where(self.route_mastered[:, None], staged, goal)

    def refresh(self) -> None:
        if self._last_step == self._env.common_step_counter:
            return
        previous_dwell = self.success_dwell.clone()
        old_depth, old_angle = self.local_front[:, 2].clone(), self.angle_error.clone()
        SCurveRoutingCommand.refresh(self)
        local, angle = insertion_errors(*self.plug_pose(), self.socket_position, self.socket_quat)
        self.local_front[:], self.angle_error[:] = local, angle
        active = torch.arange(3, device=self.device)[None] < self.guide_count[:, None]
        routed = (self.completed >= self.guide_count) & (self.seated | ~active).all(dim=1) & ~self.skip
        retained = routed & wp.to_torch(self.cable.grasp_active).bool() & wp.to_torch(self.cable.grasp_acquired).bool()
        retained &= (self.stretch() < 0.10) & (self.minimum_bend_radius >= MIN_BEND_RADIUS - 1e-7)
        self.route_dwell[:] = torch.where(retained, self.route_dwell + 1, 0)
        self.route_mastered |= self.route_dwell >= self.cfg.hold_steps
        self.ready_to_align |= self.route_mastered & (
            (self.tip() - self.contact_seat_targets[:, -1]).norm(dim=-1) < 0.004
        )
        self.ready_to_approach |= self.ready_to_align & (angle < 0.025)
        self.angular_progress[:] = torch.where(retained & self.ready_to_align, old_angle - angle, 0.0)
        aligned = (local[:, :2].abs() < 0.0005).all(dim=1) & (angle < math.radians(4.0))
        self.inserting |= retained & aligned & (local[:, 2] < -0.0005) & (local[:, 2] >= -0.012)
        self.entered |= self.inserting & aligned & (local[:, 2] >= 0) & retained
        self.wrong_entry |= (local[:, 2] > 0) & (local[:, 2] < 0.012) & (local[:, :2].abs() < 0.006).all(1) & ~routed
        self.depth_progress[:] = torch.where(retained & aligned, local[:, 2] - old_depth, 0.0)
        valid = retained & self.entered & insertion_seated(local, angle)
        valid &= wp.to_torch(self.cable.socket_contact_force) < 2.0
        self.success_dwell[:] = torch.where(valid, previous_dwell + 1, 0)
