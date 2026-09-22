# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step-based schedules from the upstream MicroDuck walking task."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


def walking_schedule(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    step_offset: int = 0,
    action_rate_scale: float = 1.0,
) -> dict[str, float]:
    """Update smoothing, standing probability, head range, and head-bias weight.

    Args:
        env: Training environment.
        env_ids: Reset environment indices; this schedule applies globally.
        step_offset: Previously completed control steps when continuing a checkpoint.
        action_rate_scale: Multiplier for the scheduled action-change penalty.
    """
    del env_ids
    step = env.common_step_counter + step_offset
    action_weight = -0.1
    for iteration, weight in ((500, -0.2), (750, -0.4), (1000, -0.6), (1250, -0.8), (1500, -1.0)):
        if step > iteration * 24:
            action_weight = weight
    action_weight *= action_rate_scale
    action_cfg = env.reward_manager.get_term_cfg("action_rate")
    action_cfg.weight = action_weight
    env.reward_manager.set_term_cfg("action_rate", action_cfg)

    standing = 0.02
    for iteration, fraction in ((500, 0.05), (750, 0.1), (1000, 0.15), (1500, 0.2), (2000, 0.25)):
        if step >= iteration * 24:
            standing = fraction
    env.command_manager.get_term("twist").cfg.rel_standing_envs = standing

    head_range = (0.05, 0.05, 0.07, 0.015)
    for iteration, ranges in (
        (500, (0.17, 0.17, 0.21, 0.047)),
        (1000, (0.39, 0.39, 0.49, 0.11)),
        (1500, (0.72, 0.72, 0.91, 0.20)),
        (2000, (1.10, 1.10, 1.40, 0.31)),
    ):
        if step >= iteration * 24:
            head_range = ranges
    env.command_manager.get_term("head_pose").cfg.ranges = tuple((-value, value) for value in head_range)

    bias_weight = 0.0
    for iteration, weight in ((600, 1.0), (1000, 2.0), (1500, 3.0)):
        if step > iteration * 24:
            bias_weight = weight
    bias_cfg = env.reward_manager.get_term_cfg("head_bias")
    bias_cfg.weight = bias_weight
    env.reward_manager.set_term_cfg("head_bias", bias_cfg)
    return {"action_rate_weight": action_weight, "standing_fraction": standing, "head_bias_weight": bias_weight}
