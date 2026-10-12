# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-relative grasp observations, contact-gated rewards, and reset curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTerm, CommandTerm

from ..constants import MARKER_SUPPORT_Y
from ..grasp_contract import MicroDuckGraspState, grasp_aperture_alignment, grasp_lift_target
from .controller import MicroDuckController

if TYPE_CHECKING:
    from ..grasp_env_cfg import MicroDuckGraspActionCfg, MicroDuckGraspCommandCfg


class MicroDuckGraspAction(ActionTerm):
    """Command fourteen absolute joint offsets and the articulated jaw [rad]."""

    def __init__(self, cfg: MicroDuckGraspActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.controller = MicroDuckController(self._asset, json.loads(Path(cfg.motor_parameters).read_text()))
        self._raw = torch.zeros((self.num_envs, 15), device=self.device)
        self._target = self.controller.home.clone()
        self.jaw_ids = self._asset.find_joints("mouth")[0]
        self.jaw_target = torch.full((self.num_envs, 1), 0.5235987756, device=self.device)

    @property
    def action_dim(self) -> int:
        return 15

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.cat((self._target, self.jaw_target), -1)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw.copy_(actions)
        self._target.copy_(self.controller.home + actions[:, :14])
        self.jaw_target[:, 0] = 0.2181661565 + 0.3054326191 * actions[:, 14].clamp(-1, 1)

    def apply_actions(self) -> None:
        self.controller.target.copy_(self._target)
        self.controller.apply()
        self._asset.actuators.target_command.set_position_index(value=self.jaw_target, joint_ids=self.jaw_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.controller.reset(env_ids)
        self._raw[env_ids] = 0
        self._target[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self.controller.joint_ids]
        self.jaw_target[env_ids] = 0.5235987756


class MicroDuckGraspCommand(CommandTerm, MicroDuckGraspState):
    """Track actual approach, opposed contact, and sustained object lift."""

    def __init__(self, cfg: MicroDuckGraspCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.object = env.scene["object"]
        self.head_id = self.robot.find_bodies("jaw_soft")[0][0]
        self.feet = self.robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)[0]
        self.jaw_ids = self.robot.find_joints("mouth")[0]
        self.neck_id = self.robot.find_joints("neck_pitch")[0][0]
        self.neck_error = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros((self.num_envs, 16), device=self.device)
        MicroDuckGraspState.__init__(
            self, self.num_envs, self.device, cfg.initial_object_height, cfg.initial_reach_height, cfg.lift_distance
        )
        self.height_level = cfg.initial_object_height
        self.reach_level = max(cfg.initial_object_height, cfg.initial_reach_height)
        self.recent_reach = 0.0
        self.recent_success = 0.0
        self.recent_episodes = 0
        self.last_step = -1
        self._progress_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_step = torch.full_like(self._progress_step, -1)
        self.geom_masks = None
        for name in (
            "success",
            "opposed_contact",
            "contact_dwell_s",
            "lifting_fraction",
            "jaw_angle_rad",
            "object_lift_m",
            "grasp_lift_m",
            "reach_error_m",
            "goal_error_m",
            "jaw_orientation",
            "neck_error_rad",
            "object_height_m",
            "reach_curriculum_height_m",
            "reach_curriculum_success",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.reset_progress(env_ids)
        step = getattr(self._env, "common_step_counter", 0)
        self._progress_step[env_ids] = step
        self._reset_step[env_ids] = step
        self.last_step = -1

    def _update_metrics(self) -> None:
        self.refresh()
        self.metrics["success"].copy_(self.success_seen.float())
        self.metrics["opposed_contact"].copy_(self.contacts.all(-1).float())
        self.metrics["contact_dwell_s"].copy_(self.contact_time)
        self.metrics["lifting_fraction"].copy_((self.stage >= 2).float())
        self.metrics["jaw_angle_rad"].copy_(self.robot.data.joint_pos.torch[:, self.jaw_ids[0]])
        self.metrics["object_lift_m"].copy_(self.lift)
        self.metrics["grasp_lift_m"].copy_(
            torch.where(self.stage >= 2, self.lift + self.start_height - self.grasp_start_height, 0.0)
        )
        self.metrics["reach_error_m"].copy_(self.distance)
        self.metrics["goal_error_m"].copy_(self.goal_distance)
        self.metrics["jaw_orientation"].copy_(grasp_jaw_orientation(self._env))
        self.metrics["neck_error_rad"].copy_(self.neck_error.abs())
        self.metrics["object_height_m"].copy_(self.start_height)
        self.metrics["reach_curriculum_height_m"].copy_(self.reach_height)
        self.metrics["reach_curriculum_success"].copy_(self.reach_seen.float())

    def _update_command(self) -> None:
        self.refresh()
        data = self.robot.data
        self._command.copy_(
            self.features(
                data.root_link_quat_w.torch,
                self.object.data.root_link_pos_w.torch,
                self.object.data.root_link_lin_vel_w.torch,
                data.joint_pos.torch[:, self.jaw_ids],
                data.joint_vel.torch[:, self.jaw_ids],
            )
        )

    def refresh(self) -> None:
        """Read current contact and pose state once per control step [s]."""
        step = getattr(self._env, "common_step_counter", 0)
        if self.last_step == step:
            return
        self.last_step = step
        data = self.robot.data
        head = data.body_link_pose_w.torch[:, self.head_id]
        self._read_contacts()
        self.contacts[self._reset_step == step] = False
        # Partial resets refresh observations without advancing other worlds twice.
        dt = (self._progress_step < step).float() * self._env.step_dt
        self.update_progress(
            head,
            self.object.data.root_link_pos_w.torch,
            self.object.data.root_link_lin_vel_w.torch,
            self._env.scene.env_origins,
            dt,
            self.object.data.root_link_quat_w.torch,
        )
        height = self.goal[:, 2] - self._env.scene.env_origins[:, 2]
        depth = ((0.215 - height) / 0.175).clamp(0, 1)
        self.neck_error.copy_(data.joint_pos.torch[:, self.neck_id] - (0.27 - 1.65 * depth))
        self._progress_step.fill_(step)

    def _read_contacts(self) -> None:
        solver = self._env.action_manager.get_term("joint_pos").controller.solver
        if self.geom_masks is None:
            names = [solver.mj_model.geom(i).name for i in range(solver.mj_model.ngeom)]
            masks = [
                ["Object" in name for name in names],
                ["upper_mouth_pad" in name for name in names],
                ["lower_mouth_" in name for name in names],
            ]
            self.geom_masks = torch.tensor(masks, device=self.device)
            if not self.geom_masks.any(-1).all():
                raise ValueError("Object and both jaw contact geometries are required.")
        contact = solver.mjw_data.contact
        geometries = wp.to_torch(contact.geom)
        world = wp.to_torch(contact.worldid).long()
        valid = torch.arange(len(geometries), device=self.device) < wp.to_torch(solver.mjw_data.nacon)[0]
        valid &= (geometries >= 0).all(-1) & (wp.to_torch(contact.dist) <= 0.0005)
        lookup = self.geom_masks[:, geometries.clamp_min(0)]
        has_object = lookup[0].any(-1)
        counts = torch.zeros((self.num_envs, 2), device=self.device)
        for i in range(2):
            hit = valid & has_object & lookup[i + 1].any(-1)
            counts[:, i].scatter_add_(0, world.clamp(0, self.num_envs - 1), hit.float())
        self.contacts.copy_(counts > 0)


def reset_grasp(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset real object/support poses [m] and the nominal standing robot state."""
    command = env.command_manager.get_term("grasp")
    if command.cfg.curriculum and command.last_step >= 0:
        survived = ~env.termination_manager.terminated[env_ids]
        held_success = command.contacts[env_ids].all(-1) & (command.lift_time[env_ids] >= 0.30)
        command.recent_success += float((held_success & survived).sum())
        # A transient pass followed by a fall must not advance the reach curriculum.
        steady_reach = (command.reach_time[env_ids] >= 0.20) & survived
        steady_reach &= command.neck_error[env_ids].abs() < 0.30
        command.recent_reach += float(steady_reach.sum())
        command.recent_episodes += len(env_ids)
        if command.recent_episodes >= max(256, env.num_envs):
            rate = command.recent_success / command.recent_episodes
            if command.reach_level > command.height_level + 1e-5:
                if command.recent_reach / command.recent_episodes >= 0.65:
                    command.reach_level = max(command.height_level, command.reach_level - 0.005)
            elif rate >= 0.65:
                if command.height_level > command.cfg.minimum_object_height + 1e-5:
                    command.height_level = max(command.cfg.minimum_object_height, command.height_level - 0.01)
            command.recent_success = 0.0
            command.recent_reach = 0.0
            command.recent_episodes = 0
    command.start_height[env_ids] = command.height_level
    command.reach_height[env_ids] = command.reach_level
    origin = env.scene.env_origins[env_ids]
    support_height = env.scene["support"].cfg.spawn.size[2]
    if command.height_level < support_height + 0.0095 - 1e-6:
        raise ValueError("Marker support would intersect the ground at this object height.")
    support_offset = -0.0095 - support_height / 2
    # Elevated curriculum resets must clear the open lower jaw at standing height.
    object_x = 0.068 + max(0.0, command.height_level - 0.18) * 1.1
    for name, y_offset, z_offset in (
        ("object", 0.0, 0.0),
        ("support", MARKER_SUPPORT_Y[0], support_offset),
        ("support_other", MARKER_SUPPORT_Y[1], support_offset),
    ):
        asset = env.scene[name]
        state = asset.data.default_root_state.torch[env_ids].clone()
        state[:, :3] = origin + state.new_tensor([object_x, y_offset, command.height_level + z_offset])
        state[:, 7:] = 0
        asset.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=env_ids)
    robot = env.scene["robot"]
    root = robot.data.default_root_state.torch[env_ids].clone()
    root[:, :3] += origin
    positions = robot.data.default_joint_pos.torch[env_ids].clone()
    root[:, 7:] = 0
    robot.write_root_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim_index(position=positions, velocity=torch.zeros_like(positions), env_ids=env_ids)


def grasp_jaw_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Align the aperture with the marker axis, with the upper pad above the lower pad."""
    command = env.command_manager.get_term("grasp")
    data = env.scene["robot"].data
    head = data.body_link_quat_w.torch[:, command.head_id]
    final_reach = command.reach_height <= command.start_height + 0.001
    height = torch.where(
        final_reach, command.start_height + command.lift, command.goal[:, 2] - env.scene.env_origins[:, 2]
    )
    score = grasp_aperture_alignment(head, command.object_quaternion, height)
    # Small roll errors collide the long marker with the pads before its center is aligned.
    return torch.where(final_reach, score.pow(10), score)


def grasp_invalid_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate invalid robot or object states before another policy evaluation."""
    from .rewards import invalid_state

    return invalid_state(env) | ~torch.isfinite(env.scene["object"].data.root_state_w.torch).all(-1)


def grasp_reward(env: ManagerBasedRLEnv, kind: str) -> torch.Tensor:
    """Return a measured objective, with zero shaping on invalid terminal states."""
    value = _grasp_reward_value(env, kind)
    # The separate termination penalty remains active; invalid episodes are failures.
    return torch.where(env.termination_manager.get_term("invalid"), 0.0, value)


def _grasp_reward_value(env: ManagerBasedRLEnv, kind: str) -> torch.Tensor:
    command = env.command_manager.get_term("grasp")
    command.refresh()
    data = env.scene["robot"].data
    if kind == "approach":
        error = (command.pinch - command.goal).norm(dim=-1)
        # Physical approach and grasp share a fixed object-centering objective across phases.
        final_reach = command.reach_height <= command.start_height + 0.001
        error = torch.where(final_reach | (command.stage >= 2), command.distance, error)
        scale = torch.where(final_reach, 0.01, 0.05)
        distance = 1.0 - torch.tanh(error / scale)
        return distance * (0.1 + 0.9 * grasp_jaw_orientation(env))
    if kind == "orientation":
        return grasp_jaw_orientation(env)
    if kind == "alignment":
        return torch.exp(-(command.distance / 0.015).square())
    if kind == "jaw":
        jaw = data.joint_pos.torch[:, command.jaw_ids[0]]
        close = (command.stage >= 2) | command.contacts.all(-1)
        close |= (command.stage == 1) & (command.distance < 0.012)
        target = torch.where(close, -0.0872664626, 0.5235987756)
        # A held marker physically blocks full closure; reward the squeeze command.
        commanded = env.action_manager.get_term("joint_pos").jaw_target[:, 0]
        value = torch.where(close, commanded, jaw)
        score = (1.0 - (value - target).abs() / 0.6108652382).clamp(0, 1)
        # Open-jaw credit belongs to pregrasp, not hovering outside the insertion threshold.
        return torch.where((command.stage == 1) & ~close, 0.0, score)
    if kind == "contacts":
        return command.contacts.float().mean(-1)
    if kind == "lift":
        start = torch.where(command.stage >= 2, command.grasp_start_height, command.start_height)
        required = (grasp_lift_target(command.start_height, command.cfg.lift_distance) - start).clamp_min(0.01)
        lifted = command.lift + command.start_height - start
        return (lifted / required).clamp(0, 1) * command.contacts.all(-1)
    if kind == "success":
        return (command.lift_time >= 0.30).float()
    if kind == "neck":
        return -(command.neck_error / 0.5).square()
    if kind == "balance":
        masses = data.body_mass.torch
        center = (data.body_com_pos_w.torch * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
        feet = data.body_link_pos_w.torch[:, command.feet].mean(1)
        error = (center[:, :2] - feet[:, :2]) / center.new_tensor([0.035, 0.04])
        return torch.exp(-error.square().sum(-1))
    if kind in {"posture", "height", "anchor"}:
        goal_height = command.goal[:, 2] - env.scene.env_origins[:, 2]
        depth = ((0.215 - goal_height) / 0.175).clamp(0, 1)
        if kind == "posture":
            pitch = 0.55 * depth.pow(1.3)
            target = torch.stack((pitch.sin(), torch.zeros_like(pitch), -pitch.cos()), -1)
            # Keep a useful learning signal before the robot has learned to lean.
            return -(data.projected_gravity_b.torch - target).square().sum(-1)
        if kind == "height":
            height = data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
            return torch.exp(-((height - (0.118 - 0.030 * depth.square())) / 0.03).square())
        root_xy = data.root_link_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        return torch.exp(-(root_xy / 0.05).square().sum(-1))
    if kind == "movement":
        return data.root_ang_vel_b.torch.square().sum(-1) + 0.05 * data.joint_vel.torch.square().sum(-1)
    raise ValueError(f"Unknown grasp reward: {kind}")


def grasp_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End fallen, escaped, or dropped-object attempts using geometry [m]."""
    robot = env.scene["robot"].data
    command = env.command_manager.get_term("grasp")
    root = robot.root_link_pos_w.torch - env.scene.env_origins
    obj = env.scene["object"].data.root_link_pos_w.torch - env.scene.env_origins
    return (root[:, 2] < 0.055) | (root[:, :2].norm(dim=-1) > 0.30) | (obj[:, 2] < command.start_height - 0.025)
