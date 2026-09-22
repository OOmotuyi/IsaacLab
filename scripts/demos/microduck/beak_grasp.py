# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-frame head kinematics for the prototype MicroDuck beak."""

from __future__ import annotations

import mujoco
import numpy as np


class BeakGraspController:
    """Center the beak laterally while preserving the trained pickup pitch."""

    def __init__(self, model: mujoco.MjModel, dof_ids: np.ndarray, home: np.ndarray):
        self.model = model
        self.data = mujoco.MjData(model)
        self.joints = model.dof_jntid[dof_ids[5:9]]
        self.addresses = model.jnt_qposadr[self.joints]
        self.body_id = int(model.jnt_bodyid[self.joints[3]])
        self.home = home[5:9]
        self.local_point: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.maximum_frame_error = 0.0

    def _position(self) -> np.ndarray:
        return self.data.xpos[self.body_id] + self.data.xmat[self.body_id].reshape(3, 3) @ self.local_point

    def apply(
        self,
        qpos: np.ndarray,
        action: np.ndarray,
        measured_point: np.ndarray,
        grasp_point: np.ndarray,
    ) -> None:
        """Update the head yaw target [rad] to center the measured pinch point [m].

        The actual physics state is read-only; only the commanded robot joint
        targets change. Cloth vertices are never attached or repositioned.
        """
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        if self.local_point is None:
            self.local_point = rotation.T @ (measured_point - self.data.xpos[self.body_id])
        error = float(np.linalg.norm(self._position() - measured_point))
        self.maximum_frame_error = max(self.maximum_frame_error, error)
        if error > 0.001:
            raise RuntimeError(f"Head kinematic frame mismatch: {error:.6f} m")

        limits = self.model.jnt_range[self.joints]
        self.data.qpos[self.addresses] = np.clip(self.home + action[0, 5:9], limits[:, 0], limits[:, 1])
        mujoco.mj_kinematics(self.model, self.data)
        point = self._position()
        self.target = point.copy()
        self.target[1] = grasp_point[1]
        joint = self.joints[2]
        derivative = np.cross(self.data.xaxis[joint], point - self.data.xanchor[joint])[1]
        if abs(derivative) > 0.01:
            correction = np.clip((grasp_point[1] - point[1]) / derivative, -0.3, 0.3)
            angle = np.clip(self.data.qpos[self.addresses[2]] + correction, limits[2, 0], limits[2, 1])
            action[0, 7] = angle - self.home[2]
