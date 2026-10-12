# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MicroDuck environments with task-specific collision filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.cloner import newton_builder_world_hook
from newton import ModelBuilder, ShapeFlags

from isaaclab.envs import ManagerBasedRLEnv

from .microduck_env_cfg import MicroDuckFlatEnvCfg

if TYPE_CHECKING:
    from .grasp_env_cfg import MicroDuckMarkerGraspEnvCfg


class MicroDuckEnv(ManagerBasedRLEnv):
    """Preserve the walking MJCF's separate foot/floor and self-collision groups."""

    def __init__(self, cfg: MicroDuckFlatEnvCfg, **kwargs):
        self._last_shape_count = 0
        self._ground_shapes: list[int] = []
        with newton_builder_world_hook(self._filter_world_collisions):
            super().__init__(cfg, **kwargs)

    def _filter_world_collisions(
        self, builder: ModelBuilder, world: int, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        del position, quaternion
        collision_shapes = []
        for index in range(self._last_shape_count, builder.shape_count):
            if not builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES:
                continue
            if builder.shape_world[index] == -1:
                self._ground_shapes.append(index)
            elif builder.shape_world[index] == world:
                collision_shapes.append(index)
        self._last_shape_count = builder.shape_count
        feet = [index for index in collision_shapes if "_foot_collision/" in builder.shape_label[index]]
        internal = [index for index in collision_shapes if index not in feet]
        if len(feet) != 2 or len(internal) != 3:
            raise ValueError(
                f"Expected two feet and three self-only walking colliders; found {len(feet)} and {len(internal)}."
            )
        # MJCF feet/floor use contype=conaffinity=1; legs and battery use 2.
        # Their mutual contacts must be filtered while each group can collide internally.
        for index in internal:
            for other in (*self._ground_shapes, *feet):
                builder.shape_collision_filter_pairs.append((min(index, other), max(index, other)))


class MicroDuckGraspEnv(ManagerBasedRLEnv):
    """Exclude floor contacts with the immovable marker-support fixtures."""

    def __init__(self, cfg: MicroDuckMarkerGraspEnvCfg, **kwargs):
        self._last_shape_count = 0
        self._ground_shapes: list[int] = []
        with newton_builder_world_hook(self._filter_support_floor):
            super().__init__(cfg, **kwargs)

    def _filter_support_floor(
        self, builder: ModelBuilder, world: int, position: np.ndarray, quaternion: np.ndarray
    ) -> None:
        del world, position, quaternion
        supports = []
        for index in range(self._last_shape_count, builder.shape_count):
            if not builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES:
                continue
            label = builder.shape_label[index]
            # A single flat-imported world may assign -1 to every shape's world index.
            if "GroundPlane" in label:
                self._ground_shapes.append(index)
            elif "/Support" in label:
                supports.append(index)
        self._last_shape_count = builder.shape_count
        if len(supports) != 2 or not self._ground_shapes:
            raise ValueError("Marker grasping requires two support colliders and a ground plane.")
        for support in supports:
            for ground in self._ground_shapes:
                builder.shape_collision_filter_pairs.append((min(support, ground), max(support, ground)))
