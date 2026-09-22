# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical target and measurements for the released mouth-to-ground motion."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.math import quat_apply


class PickupTarget:
    """Optional free rigid block; no attachment or grasp assistance is applied."""

    def __init__(self, robot: Articulation, source_xml: Path, enabled: bool):
        """Spawn a 0.025 × 0.025 × 0.030 [m], 0.010 [kg] block when enabled."""
        self.robot = robot
        self.object = None
        self._jaw_id = None
        self._start = None
        self._max_rise = 0.0
        self._min_distance = math.inf
        if not enabled:
            return
        site = ET.parse(source_xml).find(".//body[@name='jaw_soft']/site[@name='mouth_tip']")
        if site is None:
            raise ValueError(f"No jaw_soft/mouth_tip site in {source_xml}")
        self._mouth_offset = tuple(float(value) for value in site.attrib["pos"].split())
        self.object = RigidObject(
            RigidObjectCfg(
                prim_path="/World/PickupTarget",
                spawn=sim_utils.CuboidCfg(
                    size=(0.025, 0.025, 0.030),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.010),
                    physics_material=RigidBodyMaterialBaseCfg(static_friction=0.7, dynamic_friction=0.7),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.16, 0.015)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.079, 0.006, 0.0155)),
            )
        )

    def reset(self) -> None:
        """Place the free block at the measured mouth approach point [m] for this episode."""
        if self.object is None:
            return
        pose = self.robot.data.root_link_pose_w.torch[0]
        x, y, _, qx, qy, qz, qw = pose.tolist()
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        # Measured from the released ground-pick rollout, relative to the initial base.
        self._start = pose.new_tensor(
            [
                x + math.cos(yaw) * 0.079 - math.sin(yaw) * 0.006,
                y + math.sin(yaw) * 0.079 + math.cos(yaw) * 0.006,
                0.0155,
            ]
        )
        root_pose = torch.cat((self._start, pose.new_tensor([0.0, 0.0, 0.0, 1.0]))).unsqueeze(0)
        self.object.write_root_pose_to_sim_index(root_pose=root_pose)
        self.object.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((1, 6)))
        self.object.reset()
        self._jaw_id = self.robot.find_bodies("jaw_soft")[0][0]

    def update(self, dt: float) -> None:
        """Refresh object state after a physics step of duration dt [s]."""
        if self.object is not None:
            self.object.update(dt)

    def sample(self) -> dict[str, list[float] | float] | None:
        """Measure world positions, mouth-to-center distance, and center rise [m]."""
        if self.object is None:
            return None
        position = self.object.data.root_link_pos_w.torch[0]
        data = self.robot.data
        mouth = data.body_link_pos_w.torch[:, self._jaw_id] + quat_apply(
            data.body_link_quat_w.torch[:, self._jaw_id],
            position.new_tensor(self._mouth_offset).unsqueeze(0),
        )
        if not torch.isfinite(torch.cat((position, mouth[0]))).all():
            raise RuntimeError("Non-finite pickup target or mouth state.")
        distance = float((mouth[0] - position).norm())
        rise = float(position[2] - self._start[2])
        self._max_rise = max(self._max_rise, rise)
        self._min_distance = min(self._min_distance, distance)
        return {
            "position_m": position.tolist(),
            "mouth_position_m": mouth[0].tolist(),
            "mouth_to_center_distance_m": distance,
            "center_rise_m": rise,
        }

    def metrics(self) -> dict[str, object] | None:
        """Report object motion [m]; center rise alone does not establish a grasp."""
        state = self.sample()
        if state is None:
            return None
        return {
            "size_m": [0.025, 0.025, 0.030],
            "mass_kg": 0.010,
            "initial_position_m": self._start.tolist(),
            "final_position_m": state["position_m"],
            "maximum_center_rise_m": self._max_rise,
            "minimum_mouth_to_center_distance_m": self._min_distance,
            "grasp_assistance": False,
        }
