# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture actual rigid-body and cloth states for offline rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import Articulation, RigidObject


class CleanupRecording:
    """Optional 50 Hz world-pose [m, xyzw] and cloth-vertex [m] recording."""

    def __init__(self, enabled: bool, body_names: list[str]):
        self.enabled = enabled
        self.body_names = body_names
        self.times: list[float] = []
        self.bodies: list[np.ndarray] = []
        self.markers: list[np.ndarray] = []
        self.cloth: list[np.ndarray] = []
        self.assistance: list[bool] = []

    def append(
        self, time: float, robot: Articulation, marker: RigidObject, nodes: torch.Tensor, assistance_active: bool
    ) -> None:
        """Copy the measured state at simulation time [s], only when enabled."""
        if not self.enabled:
            return
        self.times.append(time)
        self.bodies.append(robot.data.body_link_pose_w.torch[0].cpu().numpy().copy())
        self.markers.append(marker.data.root_link_pose_w.torch[0].cpu().numpy().copy())
        self.cloth.append(nodes.cpu().numpy().copy())
        self.assistance.append(assistance_active)

    def save(self, directory: Path) -> None:
        """Save recorded states without changing or smoothing the simulation."""
        if self.enabled:
            np.savez_compressed(
                directory / "render_states.npz",
                time_s=self.times,
                body_names=self.body_names,
                body_pose_m_xyzw=self.bodies,
                marker_pose_m_xyzw=self.markers,
                cloth_points_m=self.cloth,
                assistance_active=self.assistance,
            )
