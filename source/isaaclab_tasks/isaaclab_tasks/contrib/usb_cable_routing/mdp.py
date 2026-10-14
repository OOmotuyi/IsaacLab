# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical cable routing, Cartesian actions, and success-driven curriculum."""

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton import ModelFlags

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from .assets import (
    ANCHOR,
    ARM_JOINTS,
    CABLE_LENGTH,
    GRASP_LOCAL_Z,
    GUIDES,
    HOME_ARM_JOINT_POS,
    SEGMENT_LENGTH,
    SEGMENTS,
    START,
    START_DIRECTION,
    TOOL_QUAT,
)


def guide_crossings(points: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
    """Test cable crossings inside both faces of each open guide [m], returning [N, G] flags."""
    a, b = points[:, :-1], points[:, 1:]
    seated = torch.ones((points.shape[0], guides.shape[1]), dtype=torch.bool, device=points.device)
    for dx in (-0.009, 0.009):
        plane = guides[:, :, None, 0] + dx
        xa, xb = a[:, None, :, 0], b[:, None, :, 0]
        t = (plane - xa) / torch.where((xb - xa).abs() > 1e-8, xb - xa, torch.ones_like(xa))
        yz = a[:, None, :, 1:] + t[..., None] * (b - a)[:, None, :, 1:]
        inside = (
            ((xa - plane) * (xb - plane) <= 0)
            & ((xb - xa).abs() > 1e-8)
            & ((yz[..., 0] - guides[:, :, None, 1]).abs() < 0.019)
            & (yz[..., 1] > 0.0115)
            & (yz[..., 1] < 0.047)
        )
        seated &= inside.any(dim=-1)
    return seated


def cartesian_progress(previous: torch.Tensor, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return distance reduction toward one target [m], without a stationary reward."""
    return (previous - target).norm(dim=-1) - (current - target).norm(dim=-1)


def initial_centerline() -> np.ndarray:
    """Sample a slack 600 mm reset cable in front of the arm with an upright held end [m]."""
    anchor, tip = np.array(ANCHOR), np.array(START)
    direction = np.array(START_DIRECTION)
    end = tip - SEGMENT_LENGTH * direction
    start = anchor + (SEGMENT_LENGTH, 0, 0)
    t = np.linspace(0, 1, 4001)[:, None]
    lo, hi = 0.0, 0.5
    for _ in range(45):
        bulge = (lo + hi) / 2
        p1, p2, p3 = start + (0.08, 0, 0), (start + end) / 2 - (0, bulge, 0), end - 0.10 * direction
        curve = (
            (1 - t) ** 4 * start
            + 4 * t * (1 - t) ** 3 * p1
            + 6 * t * t * (1 - t) ** 2 * p2
            + 4 * t**3 * (1 - t) * p3
            + t**4 * end
        )
        arc = np.r_[0.0, np.linalg.norm(np.diff(curve, axis=0), axis=1).cumsum()]
        if arc[-1] > CABLE_LENGTH - 2 * SEGMENT_LENGTH:
            hi = bulge
        else:
            lo = bulge
    samples = np.stack([np.interp(np.linspace(0, arc[-1], SEGMENTS - 1), arc, curve[:, i]) for i in range(3)], axis=-1)
    return np.vstack((anchor, samples, tip)).astype(np.float32)


class RoutingAction(DifferentialInverseKinematicsAction):
    """Three bounded Cartesian increments; IK holds the gripper orientation fixed."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._target_valid[env_ids] = False

    def apply_actions(self) -> None:
        """Apply IK while bounding commanded joint increments [rad] and respecting position limits."""
        super().apply_actions()
        current = self._asset.data.joint_pos.torch[:, self._joint_ids]
        desired = self._asset.actuators.target_command.position.torch[:, self._joint_ids]
        maximum_step = self.cfg.maximum_joint_speed * self._env.physics_dt
        target = current + (desired - current).clamp(-maximum_step, maximum_step)
        limits = self._asset.data.joint_pos_limits.torch[:, self._joint_ids]
        target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)
        self._env.command_manager.get_term("routing").update_arm_metrics()

    @property
    def action_dim(self):
        return 3

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        pos, quat = self._compute_frame_pose()
        command = self._env.command_manager.get_term("routing")
        command.cable.collisions.zero_()
        target = torch.where(self._target_valid[:, None], self._processed_actions, pos)
        target = target + actions.clamp(-1.0, 1.0) * self.cfg.scale
        self._target_valid[:] = True
        origin = self._asset.data.root_pos_w.torch - self._env.scene.env_origins
        lower = torch.tensor((0.17, -0.25, self.cfg.minimum_tcp_height), device=self.device) - origin
        upper = torch.tensor((0.80, 0.22, 0.35), device=self.device) - origin
        target = torch.maximum(torch.minimum(target, upper), lower)
        self._processed_actions[:] = target
        desired_quat = command.tool_quat if command.home_q is not None else quat
        self._ik_controller.set_command(torch.cat((target, desired_quat), dim=-1))


class RoutingCommand(CommandTerm):
    """Track retained, ordered cable crossings and increase the number of physical guides."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cable = env.scene["cable"]
        self.robot = env.scene["robot"]
        self.arm_joint_ids, _ = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.wrist_body_ids, _ = self.robot.find_bodies(["wrist_1_link", "wrist_2_link"], preserve_order=True)
        self.stage = cfg.guide_count
        self.home_q = None
        self.lowering = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.tool_quat = torch.tensor(TOOL_QUAT, device=self.device).repeat(self.num_envs, 1)
        self.guide_positions = torch.tensor(GUIDES, device=self.device).repeat(self.num_envs, 1, 1)
        self.completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.guide_count = torch.full_like(self.completed, self.stage)
        self.dwell = torch.zeros_like(self.completed)
        self.success_dwell = torch.zeros_like(self.completed)
        self.seated = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
        self.bonus = torch.zeros(self.num_envs, device=self.device)
        self.skip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.material = torch.ones(self.num_envs, 2, device=self.device)
        self.episode_peaks = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_episode_peaks = torch.zeros_like(self.episode_peaks)
        self.episode_arm_metrics = torch.zeros(self.num_envs, 4, device=self.device)
        self.episode_arm_metrics[:, (0, 3)] = torch.inf
        self.last_arm_metrics = self.episode_arm_metrics.clone()
        self.last_completed = torch.zeros_like(self.completed)
        self.last_seated = torch.zeros_like(self.seated)
        self.previous_tip = torch.tensor(START, device=self.device).repeat(self.num_envs, 1)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self._episodes = self._successes = 0
        self._last_step = -1
        for name in (
            "guide_count",
            "completed_guides",
            "success",
            "max_stretch",
            "grasp_force",
            "minimum_wrist_height_m",
            "maximum_arm_speed_rad_s",
            "maximum_wrist_roll_rad",
            "joint_limit_margin_rad",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        model = NewtonManager.get_model()
        joints = model.joint_label
        starts = model.joint_qd_start.numpy()
        rows = []
        for env_id in range(self.num_envs):
            indices = [j for j, label in enumerate(joints) if f"/env_{env_id}/Cable/" in label]
            rows.append([k for j in indices for k in range(int(starts[j]), int(starts[j + 1]))])
        self.dof_ids = torch.tensor(rows, device=self.device, dtype=torch.long)
        self.base_stiffness = wp.to_torch(model.joint_target_ke)[self.dof_ids].clone()
        shape_body = wp.to_torch(model.shape_body).long()
        self.cable_shapes = []
        for ids in self.cable.body_ids:
            self.cable_shapes.append(torch.isin(shape_body, ids).nonzero().flatten())

    @property
    def command(self):
        return self.target() - self.tip()

    def points(self):
        pose = self.cable.data.segment_pose_w.torch
        axis = quat_apply(
            pose[..., 3:].reshape(-1, 4),
            torch.tensor((0, 0, SEGMENT_LENGTH / 2), device=self.device).expand(self.num_envs * SEGMENTS, 3),
        )
        axis = axis.reshape(self.num_envs, SEGMENTS, 3)
        return (
            torch.cat((pose[:, :1, :3] - axis[:, :1], pose[..., :3] + axis), dim=1)
            - self._env.scene.env_origins[:, None]
        )

    def tip(self):
        return self.points()[:, -1]

    def target(self):
        indices = self.completed.clamp(max=2)
        goal = self.guide_positions[torch.arange(self.num_envs, device=self.device), indices].clone()
        goal[:, 1] = ANCHOR[1] + (goal[:, 1] - ANCHOR[1]) * (0.73 - ANCHOR[0]) / (goal[:, 0] - ANCHOR[0])
        goal[:, 0] = torch.where(self.lowering, 0.775, 0.685)
        goal[:, 2] = torch.where(self.lowering, 0.038, 0.35)
        return goal

    def stretch(self):
        pose = self.cable.data.segment_pose_w.torch
        axis = quat_apply(
            pose[..., 3:].reshape(-1, 4),
            torch.tensor((0, 0, SEGMENT_LENGTH / 2), device=self.device).expand(self.num_envs * SEGMENTS, 3),
        )
        axis = axis.reshape(self.num_envs, SEGMENTS, 3)
        gaps = (pose[:, 1:, :3] - axis[:, 1:]) - (pose[:, :-1, :3] + axis[:, :-1])
        return gaps.norm(dim=-1).max(dim=1).values / SEGMENT_LENGTH

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
        nearest = (points[..., 0] - GUIDES[0][0]).abs().argmin(dim=1)
        first_crossing = points[torch.arange(self.num_envs, device=self.device), nearest]
        clear = (first_crossing[:, 2] > 0.060) & ((first_crossing[:, 1] - GUIDES[0][1]).abs() < 0.015)
        self.lowering |= clear & ((self.tip() - self.target()).norm(dim=-1) < 0.025)
        self.seated[:] = guide_crossings(self.points(), self.guide_positions)
        ids = torch.arange(self.num_envs, device=self.device)
        index = self.completed.clamp(max=2)
        current = self.seated[ids, index] & (self.completed < self.guide_count)
        self.dwell[:] = torch.where(current, self.dwell + 1, 0)
        advance = self.dwell >= 5
        self.bonus[:] = advance.float()
        # A later guide may only be visited after its predecessor has been seated.
        later = torch.arange(3, device=self.device)[None] > self.completed[:, None]
        active = torch.arange(3, device=self.device)[None] < self.guide_count[:, None]
        self.skip |= (self.seated & later & active & ~current[:, None]).any(dim=1)
        self.completed += advance.long()
        # A valid alternative route can seat a guide without visiting the lift waypoint.
        self.lowering |= self.completed > 0
        self.dwell[advance] = 0
        all_retained = (self.seated | ~active).all(dim=1)
        valid = all_retained & (self.completed >= self.guide_count) & ~self.skip
        valid &= wp.to_torch(self.cable.grasp_active).bool() & (self.stretch() < 0.10)
        self.success_dwell[:] = torch.where(valid, self.success_dwell + 1, 0)

    def curriculum_success(self) -> torch.Tensor:
        """Return episode mastery flags used only for guide-count advancement."""
        return self.success_dwell >= self.cfg.hold_steps

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.last_episode_peaks[env_ids] = self.episode_peaks[env_ids]
        self.last_arm_metrics[env_ids] = self.episode_arm_metrics[env_ids]
        self.last_completed[env_ids] = self.completed[env_ids]
        self.last_seated[env_ids] = self.seated[env_ids]
        if hasattr(self._env, "termination_manager"):
            done = self._env.termination_manager.dones[env_ids]
            self._episodes += int(done.sum())
            self._successes += int((done & self.curriculum_success()[env_ids]).sum())
        if self.cfg.curriculum and self._episodes >= self.cfg.curriculum_episodes:
            rate = self._successes / self._episodes
            if rate >= self.cfg.curriculum_success_rate and self.stage < 3:
                self.stage += 1
                print(
                    f"[USB curriculum] advanced to {self.stage} guides "
                    f"after {self._successes}/{self._episodes} successes",
                    flush=True,
                )
            self._episodes = self._successes = 0
        return super().reset(env_ids)

    def _solve_home(self):
        action = self._env.action_manager.get_term("arm")
        q = self.robot.data.default_joint_pos.torch.clone()
        # This measured branch keeps wrist_1 above the hand throughout lowering.
        q[:, action._joint_ids] = torch.tensor(HOME_ARM_JOINT_POS, device=self.device)
        self.robot.write_joint_position_to_sim_index(position=q)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        target = torch.tensor(START, device=self.device).repeat(self.num_envs, 1)
        target += torch.tensor(START_DIRECTION, device=self.device) * (GRASP_LOCAL_Z - SEGMENT_LENGTH / 2)
        root = self.robot.data.root_pos_w.torch - self._env.scene.env_origins
        desired = torch.cat((target - root, self.tool_quat), dim=-1)
        action._ik_controller.set_command(desired)
        for _ in range(160):
            self._env.sim.forward()
            self.robot.update(0)
            pos, quat = action._compute_frame_pose()
            q = self.robot.data.joint_pos.torch.clone()
            next_q = action._ik_controller.compute(pos, quat, action._compute_frame_jacobian(), q[:, action._joint_ids])
            q[:, action._joint_ids] += (next_q - q[:, action._joint_ids]).clamp(-0.10, 0.10)
            self.robot.write_joint_position_to_sim_index(position=q)
        self._env.sim.forward()
        self.robot.update(0)
        pos, quat = action._compute_frame_pose()
        if (pos - desired[:, :3]).norm(dim=-1).max() > 0.003:
            raise RuntimeError("Cable-task pre-grasp IK did not converge.")
        if (quat * self.tool_quat).sum(-1).abs().min() < 0.999:
            raise RuntimeError("Cable-task pre-grasp orientation did not converge.")
        self.home_q = q.detach().clone()

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.home_q is None:
            self._solve_home()
        self.robot.write_joint_position_to_sim_index(position=self.home_q[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.home_q[env_ids]), env_ids=env_ids)
        self.robot.actuators.target_command.set_position_index(value=self.home_q[env_ids], env_ids=env_ids)
        action = self._env.action_manager.get_term("arm")
        root = self.robot.data.root_pos_w.torch[env_ids] - self._env.scene.env_origins[env_ids]
        action.processed_actions[env_ids] = torch.tensor(START, device=self.device) - root
        action.processed_actions[env_ids] += torch.tensor(START_DIRECTION, device=self.device) * (
            GRASP_LOCAL_Z - SEGMENT_LENGTH / 2
        )
        action._target_valid[env_ids] = True
        count = len(env_ids)
        points = torch.tensor(initial_centerline(), device=self.device).repeat(count, 1, 1)
        if self.cfg.randomize:
            jitter = (torch.rand(count, 1, 2, device=self.device) * 2 - 1) * self.cfg.position_jitter
            envelope = torch.sin(torch.linspace(0, torch.pi, SEGMENTS + 1, device=self.device))[None, :, None]
            envelope[:, :2] = 0
            points[..., :2] += jitter * envelope
        direction = points[:, 1:] - points[:, :-1]
        direction /= direction.norm(dim=-1, keepdim=True)
        # Quaternion rotating the canonical segment z-axis onto its centerline tangent.
        quat = torch.stack(
            (-direction[..., 1], direction[..., 0], torch.zeros_like(direction[..., 0]), 1 + direction[..., 2]), dim=-1
        )
        quat /= quat.norm(dim=-1, keepdim=True)
        position = (points[:, 1:] + points[:, :-1]) / 2 + self._env.scene.env_origins[env_ids, None]
        pose = torch.cat((position, quat), dim=-1)
        self.cable.write_segment_pose_to_sim_index(segment_pose=pose, env_ids=env_ids)
        self.cable.write_segment_velocity_to_sim_index(
            segment_velocity=torch.zeros(count, SEGMENTS, 6, device=self.device), env_ids=env_ids
        )
        body_pose = wp.to_torch(NewtonManager.get_state_0().body_q)
        wrist = body_pose[wp.to_torch(self.cable._wrist_ids).long()[env_ids], 3:]
        end = body_pose[self.cable.body_ids[env_ids, -1], 3:]
        wp.to_torch(self.cable.grasp_relative_quat)[env_ids] = quat_mul(quat_inv(wrist), end)
        wp.to_torch(self.cable.grasp_torque)[env_ids] = 0
        if hasattr(action, "target_quat"):
            action.target_quat[env_ids] = wrist
        wp.to_torch(self.cable.grasp_active)[env_ids] = 1
        wp.to_torch(self.cable.tension)[env_ids] = 0
        wp.to_torch(self.cable.collisions)[env_ids] = 0
        self.guide_count[env_ids] = self.stage
        for index in range(3):
            obj = self._env.scene[f"guide{index + 1}"]
            state = obj.data.default_root_state.torch[env_ids].clone()
            state[:, :3] = self.guide_positions[env_ids, index] + self._env.scene.env_origins[env_ids]
            if index >= self.stage:
                state[:, 2] -= 1.0
            obj.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
        model = NewtonManager.get_model()
        scales = torch.ones(count, device=self.device)
        mu = torch.full_like(scales, 0.4)
        if self.cfg.randomize:
            lo, hi = self.cfg.stiffness_range
            scales = lo + (hi - lo) * torch.rand(count, device=self.device)
            lo, hi = self.cfg.friction_range
            mu = lo + (hi - lo) * torch.rand(count, device=self.device)
        self.material[env_ids] = torch.stack((scales, mu), dim=-1)
        wp.to_torch(model.joint_target_ke)[self.dof_ids[env_ids]] = self.base_stiffness[env_ids] * scales[:, None]
        for local, env_id in enumerate(env_ids.tolist()):
            wp.to_torch(model.shape_material_mu)[self.cable_shapes[env_id]] = mu[local]
        NewtonManager.add_model_change(ModelFlags.JOINT_DOF_PROPERTIES)
        NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        self.episode_peaks[env_ids] = 0
        self.episode_arm_metrics[env_ids] = 0
        self.episode_arm_metrics[env_ids, 0] = torch.inf
        self.episode_arm_metrics[env_ids, 3] = torch.inf
        self.update_arm_metrics()
        self.previous_tip[env_ids] = torch.tensor(START, device=self.device)
        self.progress[env_ids] = 0
        self.lowering[env_ids] = False
        self.completed[env_ids] = 0
        self.dwell[env_ids] = 0
        self.success_dwell[env_ids] = 0
        self.skip[env_ids] = False
        self.bonus[env_ids] = 0
        self.seated[env_ids] = False

    def update_arm_metrics(self) -> None:
        """Track wrist height [m], arm speed [rad/s], grasp roll and joint-limit margin [rad]."""
        if self.home_q is None:
            return
        joints = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        speed = self.robot.data.joint_vel.torch[:, self.arm_joint_ids].abs().amax(dim=-1)
        limits = self.robot.data.joint_pos_limits.torch[:, self.arm_joint_ids]
        margin = torch.minimum(joints - limits[..., 0], limits[..., 1] - joints).amin(dim=-1)
        height = self.robot.data.body_link_pos_w.torch[:, self.wrist_body_ids[0], 2] - self._env.scene.env_origins[:, 2]
        roll = (joints[:, 5] - self.home_q[:, self.arm_joint_ids[5]]).abs()
        self.episode_arm_metrics[:, 0] = torch.minimum(self.episode_arm_metrics[:, 0], height)
        self.episode_arm_metrics[:, 1] = torch.maximum(self.episode_arm_metrics[:, 1], speed)
        self.episode_arm_metrics[:, 2] = torch.maximum(self.episode_arm_metrics[:, 2], roll)
        self.episode_arm_metrics[:, 3] = torch.minimum(self.episode_arm_metrics[:, 3], margin)

    def _update_metrics(self):
        self.metrics["minimum_wrist_height_m"][:] = self.episode_arm_metrics[:, 0]
        self.metrics["maximum_arm_speed_rad_s"][:] = self.episode_arm_metrics[:, 1]
        self.metrics["maximum_wrist_roll_rad"][:] = self.episode_arm_metrics[:, 2]
        self.metrics["joint_limit_margin_rad"][:] = self.episode_arm_metrics[:, 3]
        self.metrics["guide_count"][:] = self.guide_count
        self.metrics["completed_guides"][:] = self.completed
        self.metrics["success"][:] = (self.success_dwell >= self.cfg.hold_steps).float()
        self.metrics["max_stretch"][:] = self.stretch()
        self.metrics["grasp_force"][:] = wp.to_torch(self.cable.tension)

    def _update_command(self):
        self.refresh()


def _term(env):
    term = env.command_manager.get_term("routing")
    term.refresh()
    return term


def routing_observation(env):
    """Robot state, cable centerline, active guide, stage, and grasp state."""
    c = env.command_manager.get_term("routing")
    q = c.robot.data.joint_pos.torch
    qd = c.robot.data.joint_vel.torch
    return torch.cat(
        (
            q,
            qd * 0.1,
            c.points().flatten(1),
            c.command,
            c.completed[:, None] / 3,
            c.guide_count[:, None] / 3,
            wp.to_torch(c.cable.grasp_active)[:, None].float(),
            c.lowering[:, None].float(),
            env.action_manager.get_term("arm").processed_actions,
        ),
        dim=-1,
    )


def approach_reward(env):
    """Distance reduction per control interval [m/s]; stationary poses earn zero."""
    return _term(env).progress / env.step_dt


def guide_reward(env):
    """One reward pulse for each newly retained crossing in order."""
    return _term(env).bonus


def routing_success(env):
    """Require all active guides to retain the cable simultaneously for a dwell period."""
    c = _term(env)
    return c.success_dwell >= c.cfg.hold_steps


def collision_penalty(env):
    """Penalize tool incursions into the table or guide walls using their collision geometry."""
    c = _term(env)
    pos, _ = env.action_manager.get_term("arm")._compute_frame_pose()
    tcp = pos + c.robot.data.root_pos_w.torch - env.scene.env_origins
    delta = tcp[:, None] - c.guide_positions
    active = torch.arange(3, device=env.device)[None] < c.guide_count[:, None]
    wall = (delta[..., 0].abs() < 0.025) & (delta[..., 1].abs() > 0.014)
    wall &= (delta[..., 1].abs() < 0.048) & (tcp[:, None, 2] < 0.065) & active
    return torch.maximum(wp.to_torch(c.cable.collisions).float(), wall.any(dim=1).float() + (tcp[:, 2] < 0.022).float())


def stretch_penalty(env):
    """Penalize rod joint extension/shear beyond two percent of a segment length."""
    return (_term(env).stretch() - 0.02).clamp(min=0)


def cable_overstretched(env):
    """Terminate when any joint gap exceeds 30 percent of one segment length."""
    c = _term(env)
    return (c.stretch() > 0.30) & (env.episode_length_buf > 5)


def cable_dropped(env):
    """A released grasp, nonfinite cable, or cable below the tabletop fails."""
    c = _term(env)
    p = c.points()
    return (
        ~wp.to_torch(c.cable.grasp_active).bool()
        | ~torch.isfinite(p).all(dim=(1, 2))
        | (p[..., 2].min(dim=1).values < -0.025)
    )


def skipped_guide(env):
    """Report an out-of-order guide crossing."""
    return _term(env).skip


def arm_posture_penalty(env) -> torch.Tensor:
    """Penalize folded wrist angles and unnecessary cable-grasp roll [rad squared]."""
    c = _term(env)
    joints = c.robot.data.joint_pos.torch[:, c.arm_joint_ids]
    folded = (joints[:, 3].abs() - torch.pi / 2).clamp(min=0).square()
    rolled = ((joints[:, 5] - c.home_q[:, c.arm_joint_ids[5]]).abs() - torch.pi).clamp(min=0).square()
    return folded + rolled


def arm_clearance_penalty(env) -> torch.Tensor:
    """Penalize the wrist entering a 140 mm clearance band above the table [m]."""
    c = _term(env)
    height = c.robot.data.body_link_pos_w.torch[:, c.wrist_body_ids[0], 2] - env.scene.env_origins[:, 2]
    return (0.14 - height).clamp(min=0)


def unsafe_arm_posture(env) -> torch.Tensor:
    """Reject wrist_1 height below 90 mm [m] or arm joints outside their hard limits [rad]."""
    c = _term(env)
    joints = c.robot.data.joint_pos.torch[:, c.arm_joint_ids]
    limits = c.robot.data.joint_pos_limits.torch[:, c.arm_joint_ids]
    invalid = ((joints < limits[..., 0] - 1e-3) | (joints > limits[..., 1] + 1e-3)).any(dim=-1)
    height = c.robot.data.body_link_pos_w.torch[:, c.wrist_body_ids[0], 2] - env.scene.env_origins[:, 2]
    return invalid | (height < 0.09) | ~torch.isfinite(joints).all(dim=-1)
