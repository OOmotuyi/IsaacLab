# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Cartesian control independent of hidden cable routing and insertion progress."""

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from isaaclab_tasks.contrib.usb_cable_routing_insert.mdp import InsertionAction


class ContactInsertionAction(InsertionAction):
    """Execute six bounded pose increments with a constant translation scale [m].

    Rotation permission and translation gain do not depend on evaluator flags.
    The controller must choose small increments for precision insertion itself.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        pos, quat = self._compute_frame_pose()
        self._env.scene["cable"].collisions.zero_()
        target = torch.where(self._target_valid[:, None], self._processed_actions, pos)
        orientation = torch.where(self._target_valid[:, None], self.target_quat, quat)
        target = target + actions[:, :3].clamp(-1.0, 1.0) * self.cfg.scale
        origin = self._asset.data.root_pos_w.torch - self._env.scene.env_origins
        lower = torch.tensor((0.17, -0.38, 0.045), device=self.device) - origin
        upper = torch.tensor((0.80, 0.22, 0.35), device=self.device) - origin
        self._processed_actions[:] = torch.maximum(torch.minimum(target, upper), lower)
        rotation = actions[:, 3:].clamp(-1.0, 1.0) * self.cfg.rotation_scale
        angle = rotation.norm(dim=-1)
        delta = quat_from_angle_axis(angle, rotation / angle.clamp(min=1e-8)[:, None])
        self.target_quat[:] = quat_mul(delta, orientation)
        self.target_quat[:] /= self.target_quat.norm(dim=-1, keepdim=True)
        self._target_valid[:] = True
        self._ik_controller.set_command(torch.cat((self._processed_actions, self.target_quat), dim=-1))
