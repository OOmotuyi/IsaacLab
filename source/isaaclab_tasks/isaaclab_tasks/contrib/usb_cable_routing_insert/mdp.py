# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Six-axis manipulation and measured USB-C insertion after ordered routing."""

import math

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_inv, quat_mul

from isaaclab_tasks.contrib.usb_cable_routing.assets import SEGMENT_LENGTH
from isaaclab_tasks.contrib.usb_cable_routing.mdp import RoutingAction, initial_centerline
from isaaclab_tasks.contrib.usb_cable_routing_scurve.mdp import (
    SCurveRoutingCommand,
    scurve_observation,
)
from isaaclab_tasks.contrib.usb_cable_routing_scurve.parameters import MIN_BEND_RADIUS

from .assets import (
    INSERT_TOOL_QUAT,
    INSERTION_DEPTH,
    PLUG_FRONT_OFFSET,
    PREINSERT_GAP,
    SOCKET_FACE,
    SOCKET_QUAT,
)


def initial_insertion_centerline() -> np.ndarray:
    """Return the shared slack centerline with an upright held endpoint [m]."""
    return initial_centerline()


def insertion_errors(
    front: torch.Tensor,
    orientation: torch.Tensor,
    socket_position: torch.Tensor,
    socket_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return plug nose coordinates [m] in the socket frame and reversible angular error [rad]."""
    local = quat_apply(quat_inv(socket_quat), front - socket_position)
    flip = torch.zeros_like(socket_quat)
    flip[:, 2] = 1.0
    alternative = quat_mul(socket_quat, flip)
    cosine = torch.maximum((orientation * socket_quat).sum(-1).abs(), (orientation * alternative).sum(-1).abs())
    angle = 2 * torch.acos(cosine.clamp(0.0, 1.0))
    return local, angle


def select_socket_orientation(
    relative_plug_quat: torch.Tensor, wrist_quat: torch.Tensor, socket_quat: torch.Tensor
) -> torch.Tensor:
    """Choose a USB-C symmetry with short rotation and a downward-facing gripper approach.

    Quaternions use xyzw ordering. The clearance cost rejects a palm beneath the
    low socket by penalizing upward tool approach more strongly than a half-turn.
    """
    identity = torch.zeros_like(relative_plug_quat)
    identity[:, 3] = 1.0
    flipped = torch.zeros_like(relative_plug_quat)
    flipped[:, 2] = 1.0
    candidates = torch.stack((identity, flipped), dim=1)
    grasp = quat_mul(quat_inv(wrist_quat), quat_mul(socket_quat, relative_plug_quat))
    desired_wrist = quat_mul(
        quat_mul(socket_quat[:, None].expand_as(candidates), candidates),
        quat_inv(grasp)[:, None].expand_as(candidates),
    )
    axis = torch.zeros_like(desired_wrist[..., :3])
    axis[..., 2] = 1.0
    approach = quat_apply(desired_wrist, axis)
    rotation = 2 * torch.acos((candidates * relative_plug_quat[:, None]).sum(-1).abs().clamp(max=1))
    cost = rotation + 10.0 * (approach[..., 2] + 0.1).clamp(min=0)
    return torch.where((cost[:, 0] < cost[:, 1])[:, None], identity, flipped)


def insertion_seated(local: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Accept centered USB-C seating at 5.5–6.8 mm depth and at most 2 degrees error."""
    return (
        (local[:, 0].abs() <= 0.00015)
        & (local[:, 1].abs() <= 0.00012)
        & (local[:, 2] >= INSERTION_DEPTH - 0.0005)
        & (local[:, 2] <= INSERTION_DEPTH + 0.0008)
        & (angle <= math.radians(2.0))
    )


class InsertionAction(RoutingAction):
    """Translate and rotate the held connector with bounded Cartesian IK increments."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Retain the parent's public position target used by resets and observations.
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 3] = 1.0

    @property
    def action_dim(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        pos, quat = self._compute_frame_pose()
        command = self._env.command_manager.get_term("routing")
        command.cable.collisions.zero_()
        target = torch.where(self._target_valid[:, None], self._processed_actions, pos)
        orientation = torch.where(self._target_valid[:, None], self.target_quat, quat)
        scale = torch.where(command.completed >= command.guide_count, self.cfg.insertion_scale, self.cfg.scale)
        target += actions[:, :3].clamp(-1.0, 1.0) * scale[:, None]
        origin = self._asset.data.root_pos_w.torch - self._env.scene.env_origins
        lower = torch.tensor((0.17, -0.38, 0.045), device=self.device) - origin
        upper = torch.tensor((0.80, 0.22, 0.35), device=self.device) - origin
        self._processed_actions[:] = torch.maximum(torch.minimum(target, upper), lower)
        rotation = actions[:, 3:].clamp(-1.0, 1.0) * self.cfg.rotation_scale
        if self.cfg.lock_wrist_while_routing:
            rotation *= command.ready_to_align[:, None]
        angle = rotation.norm(dim=-1)
        delta = quat_from_angle_axis(angle, rotation / angle.clamp(min=1e-8)[:, None])
        self.target_quat[:] = quat_mul(delta, orientation)
        self.target_quat[:] /= self.target_quat.norm(dim=-1, keepdim=True)
        self._target_valid[:] = True
        self._ik_controller.set_command(torch.cat((self._processed_actions, self.target_quat), dim=-1))


class InsertionCommand(SCurveRoutingCommand):
    """Track route retention, pre-insertion alignment, entry and seated dwell."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.tool_quat[:] = torch.tensor(INSERT_TOOL_QUAT, device=self.device)
        self.socket_position = torch.tensor(SOCKET_FACE, device=self.device).repeat(self.num_envs, 1)
        self.socket_quat = torch.tensor(SOCKET_QUAT, device=self.device).repeat(self.num_envs, 1)
        self.inserting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ready_to_align = torch.zeros_like(self.inserting)
        self.ready_to_approach = torch.zeros_like(self.inserting)
        self.entered = torch.zeros_like(self.inserting)
        self.wrong_entry = torch.zeros_like(self.inserting)
        self.local_front = torch.zeros(self.num_envs, 3, device=self.device)
        self.angle_error = torch.zeros(self.num_envs, device=self.device)
        self.angular_progress = torch.zeros(self.num_envs, device=self.device)
        self.route_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.route_mastered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.depth_progress = torch.zeros(self.num_envs, device=self.device)
        self.last_depth = torch.zeros(self.num_envs, device=self.device)
        self.last_inserted = torch.zeros_like(self.inserting)
        self.metrics["insertion_depth_m"] = self.last_depth.clone()
        self.metrics["alignment_error_rad"] = self.last_depth.clone()
        self.metrics["route_mastered"] = self.last_depth.clone()

    def plug_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the free connector's shell nose position [m] and orientation (xyzw)."""
        pose = self.cable.data.segment_pose_w.torch[:, -1]
        offset = torch.tensor((0.0, 0.0, PLUG_FRONT_OFFSET), device=self.device).expand(self.num_envs, 3)
        return pose[:, :3] + quat_apply(pose[:, 3:], offset) - self._env.scene.env_origins, pose[:, 3:]

    def target(self) -> torch.Tensor:
        goal = super().target()
        if not hasattr(self, "socket_position"):
            return goal
        depth = torch.where(self.inserting, INSERTION_DEPTH, -PREINSERT_GAP)
        offset = torch.zeros_like(goal)
        offset[:, 2] = depth - (PLUG_FRONT_OFFSET - SEGMENT_LENGTH / 2)
        insertion_goal = self.socket_position + quat_apply(self.socket_quat, offset)
        settle = self.seat_targets[-1].expand_as(goal)
        staged = torch.where(self.ready_to_approach[:, None], insertion_goal, settle)
        return torch.where(self.route_mastered[:, None], staged, goal)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        self.inserting[env_ids] = self.entered[env_ids] = self.wrong_entry[env_ids] = False
        local, angle = insertion_errors(*self.plug_pose(), self.socket_position, self.socket_quat)
        self.local_front[env_ids], self.angle_error[env_ids] = local[env_ids], angle[env_ids]
        self.depth_progress[env_ids] = self.angular_progress[env_ids] = 0
        self.route_dwell[env_ids] = 0
        self.route_mastered[env_ids] = False
        self.ready_to_align[env_ids] = self.ready_to_approach[env_ids] = False

    def refresh(self):
        if self._last_step == self._env.common_step_counter:
            return
        previous_dwell = self.success_dwell.clone()
        old_depth = self.local_front[:, 2].clone()
        old_angle = self.angle_error.clone()
        super().refresh()
        local, angle = insertion_errors(*self.plug_pose(), self.socket_position, self.socket_quat)
        self.local_front[:], self.angle_error[:] = local, angle
        active = torch.arange(3, device=self.device)[None] < self.guide_count[:, None]
        routed = (self.completed >= self.guide_count) & (self.seated | ~active).all(dim=1) & ~self.skip
        retained = routed & wp.to_torch(self.cable.grasp_active).bool() & (self.stretch() < 0.10)
        retained &= self.minimum_bend_radius >= MIN_BEND_RADIUS - 1e-7
        self.route_dwell[:] = torch.where(retained, self.route_dwell + 1, 0)
        self.route_mastered |= self.route_dwell >= self.cfg.hold_steps
        self.ready_to_align |= self.route_mastered & ((self.tip() - self.seat_targets[-1]).norm(dim=-1) < 0.004)
        self.ready_to_approach |= self.ready_to_align & (angle < 0.025)
        self.angular_progress[:] = torch.where(retained & self.ready_to_align, old_angle - angle, 0.0)
        aligned = (local[:, :2].abs() < 0.0005).all(dim=1) & (angle < math.radians(4.0))
        self.inserting |= routed & aligned & (local[:, 2] < -0.0005) & (local[:, 2] >= -0.012)
        self.entered |= self.inserting & aligned & (local[:, 2] >= 0) & routed
        self.wrong_entry |= (local[:, 2] > 0) & (local[:, 2] < 0.012) & (local[:, :2].abs() < 0.006).all(1) & ~routed
        self.depth_progress[:] = torch.where(routed & aligned, local[:, 2] - old_depth, 0.0)
        valid = routed & self.entered & insertion_seated(local, angle)
        valid &= wp.to_torch(self.cable.grasp_active).bool() & (self.stretch() < 0.10)
        valid &= (self.minimum_bend_radius >= MIN_BEND_RADIUS - 1e-7) & (wp.to_torch(self.cable.tension) < 2.0)
        self.success_dwell[:] = torch.where(valid, previous_dwell + 1, 0)

    def curriculum_success(self) -> torch.Tensor:
        """Advance routing difficulty from measured route retention, separately from insertion success."""
        return self.route_mastered

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_depth[env_ids] = self.local_front[env_ids, 2]
        self.last_inserted[env_ids] = self.success_dwell[env_ids] >= self.cfg.hold_steps
        return super().reset(env_ids)

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["insertion_depth_m"][:] = self.local_front[:, 2]
        self.metrics["alignment_error_rad"][:] = self.angle_error
        self.metrics["route_mastered"][:] = self.route_mastered.float()


def insertion_observation(env) -> torch.Tensor:
    """Add socket-frame plug pose, grasp loads, action orientation, and insertion phase."""
    c = env.command_manager.get_term("routing")
    front, quat = c.plug_pose()
    local, _ = insertion_errors(front, quat, c.socket_position, c.socket_quat)
    return torch.cat(
        (
            scurve_observation(env),
            local,
            quat_mul(quat_inv(c.socket_quat), quat),
            env.action_manager.get_term("arm").target_quat,
            c.inserting[:, None].float(),
            c.entered[:, None].float(),
            wp.to_torch(c.cable.tension)[:, None],
            wp.to_torch(c.cable.grasp_torque)[:, None],
            c.ready_to_align[:, None].float(),
            c.ready_to_approach[:, None].float(),
        ),
        dim=-1,
    )


def angular_progress_reward(env) -> torch.Tensor:
    """Reward measured angular error reduction [rad/s] after safe route retention."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    return c.angular_progress / env.step_dt


def alignment_reward(env) -> torch.Tensor:
    """Reward close, correctly oriented approach only after routing is complete."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    return (c.completed >= c.guide_count).float() * torch.exp(
        -c.local_front[:, :2].norm(dim=-1) / 0.005 - c.angle_error / 0.2
    )


def insertion_progress(env) -> torch.Tensor:
    """Reward forward depth progress [m/s] after aligned routing."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    return c.depth_progress / env.step_dt


def premature_entry(env) -> torch.Tensor:
    """Reject socket entry before the cable has completed all active guides in order."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    return c.wrong_entry


def insertion_force_penalty(env) -> torch.Tensor:
    """Penalize grasp reaction force above 2 N near the insertion phase."""
    c = env.command_manager.get_term("routing")
    return (c.completed >= c.guide_count).float() * (wp.to_torch(c.cable.tension) - 2.0).clamp(min=0)


def curriculum_routing_complete(env) -> torch.Tensor:
    """Finish one/two-guide training stages; full evaluation still requires routing plus insertion."""
    c = env.command_manager.get_term("routing")
    c.refresh()
    return c.route_mastered & (c.guide_count < 3) & c.cfg.curriculum
